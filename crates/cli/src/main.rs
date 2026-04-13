//! WAFER CLI: Interactive REPL, AOT compiler, and WASM runner for WAFER Forth.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use clap::{Parser, Subcommand};
use wafer_core::export::{ExportConfig, export_module, serialize_metadata};
use wafer_core::outer::ForthVM;
use wafer_core::runner::{run_precompiled_bytes, run_wasm_file};
use wafer_core::runtime_native::NativeRuntime;

/// 8-byte magic trailer identifying a native WAFER executable.
const NATIVE_MAGIC: &[u8; 8] = b"WAFEREXE";
/// Size of the trailer: `payload_len`(8) + `metadata_len`(8) + magic(8).
const TRAILER_SIZE: u64 = 24;

/// WAFER: WebAssembly Forth Engine in Rust
#[derive(Parser, Debug)]
#[command(name = "wafer", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Forth source file to execute (when no subcommand is given)
    file: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Compile a Forth source file to a standalone WASM module
    Build {
        /// Input Forth source file
        file: String,

        /// Output file (default: input stem + .wasm, or no extension with --native)
        #[arg(short, long)]
        output: Option<String>,

        /// Entry-point word name (default: MAIN, or top-level execution)
        #[arg(long)]
        entry: Option<String>,

        /// Also generate a JS loader and HTML page for browser execution
        #[arg(long)]
        js: bool,

        /// Produce a standalone native executable (AOT-compiled)
        #[arg(long)]
        native: bool,
    },

    /// Run a pre-compiled WASM module
    Run {
        /// .wasm file to execute
        file: String,
    },
}

fn main() -> anyhow::Result<()> {
    // Check for embedded payload before CLI parsing. If this binary has
    // an appended WASM payload (produced by --native), run it directly.
    if let Some(output) = check_embedded_payload()? {
        if !output.is_empty() {
            print!("{output}");
        }
        return Ok(());
    }

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Build {
            file,
            output,
            entry,
            js,
            native,
        }) => cmd_build(&file, output.as_deref(), entry, js, native),

        Some(Commands::Run { file }) => cmd_run(&file),

        None => cmd_eval_or_repl(cli.file.as_deref()),
    }
}

/// Check if this executable has an appended WAFER payload.
///
/// Returns `Some(output)` if a payload was found and executed,
/// `None` if this is a normal wafer binary.
fn check_embedded_payload() -> anyhow::Result<Option<String>> {
    let Ok(exe_path) = std::env::current_exe() else {
        return Ok(None);
    };
    let Ok(mut file) = std::fs::File::open(&exe_path) else {
        return Ok(None);
    };
    let file_len = file.seek(SeekFrom::End(0))?;
    if file_len < TRAILER_SIZE {
        return Ok(None);
    }

    // Read the 24-byte trailer.
    file.seek(SeekFrom::End(-(TRAILER_SIZE as i64)))?;
    let mut trailer = [0u8; TRAILER_SIZE as usize];
    file.read_exact(&mut trailer)?;

    // Check magic.
    if &trailer[16..24] != NATIVE_MAGIC {
        return Ok(None);
    }

    let payload_len = u64::from_le_bytes(trailer[0..8].try_into().unwrap());
    let metadata_len = u64::from_le_bytes(trailer[8..16].try_into().unwrap());

    // Read the payload and metadata.
    let data_start = file_len - TRAILER_SIZE - metadata_len - payload_len;
    file.seek(SeekFrom::Start(data_start))?;

    let mut payload = vec![0u8; payload_len as usize];
    file.read_exact(&mut payload)?;

    let mut metadata_bytes = vec![0u8; metadata_len as usize];
    file.read_exact(&mut metadata_bytes)?;

    let metadata_json = String::from_utf8_lossy(&metadata_bytes);
    let output = run_precompiled_bytes(&payload, &metadata_json)?;
    Ok(Some(output))
}

/// `wafer build program.fth -o program.wasm [--native]`
fn cmd_build(
    file: &str,
    output: Option<&str>,
    entry: Option<String>,
    js: bool,
    native: bool,
) -> anyhow::Result<()> {
    let source = std::fs::read_to_string(file)?;

    let mut vm = ForthVM::<NativeRuntime>::new()?;
    vm.set_recording(true);
    vm.evaluate(&source)?;

    // Print any side-effect output from evaluation.
    let eval_output = vm.take_output();
    if !eval_output.is_empty() {
        print!("{eval_output}");
    }

    let config = ExportConfig { entry_word: entry };
    let (wasm_bytes, metadata) = export_module(&mut vm, &config)?;

    // Determine output path.
    let out_path = match output {
        Some(p) => p.to_string(),
        None => {
            let stem = Path::new(file)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("out");
            if native {
                stem.to_string()
            } else {
                format!("{stem}.wasm")
            }
        }
    };

    if native {
        build_native(&wasm_bytes, &metadata, &out_path)?;
    } else {
        std::fs::write(&out_path, &wasm_bytes)?;
        let word_count = vm.ir_words().len();
        let host_count = metadata.host_functions.len();
        eprintln!(
            "Wrote {out_path} ({} bytes, {word_count} words, {host_count} host functions)",
            wasm_bytes.len()
        );
    }

    if js {
        let out = Path::new(&out_path);
        let wasm_filename = out
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("out.wasm");
        let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        let dir = out.parent().unwrap_or_else(|| Path::new("."));

        let js_path = dir.join(format!("{stem}.js"));
        let html_path = dir.join(format!("{stem}.html"));
        let js_filename = format!("{stem}.js");

        let js_code = wafer_core::js_loader::generate_js_loader(wasm_filename, &metadata);
        let html_code = wafer_core::js_loader::generate_html_page(wasm_filename, &js_filename);

        std::fs::write(&js_path, &js_code)?;
        std::fs::write(&html_path, &html_code)?;
        eprintln!("Wrote {} and {}", js_path.display(), html_path.display());
    }

    Ok(())
}

/// Build a native executable by appending AOT-compiled WASM to the wafer binary.
fn build_native(
    wasm_bytes: &[u8],
    metadata: &wafer_core::export::ExportMetadata,
    out_path: &str,
) -> anyhow::Result<()> {
    // AOT precompile the WASM to native code.
    let mut config = wasmtime::Config::new();
    config.cranelift_nan_canonicalization(false);
    let engine = wasmtime::Engine::new(&config)?;
    let precompiled = engine.precompile_module(wasm_bytes)?;

    // Read the current wafer binary.
    let self_exe = std::env::current_exe()?;
    let self_bytes = std::fs::read(&self_exe)?;

    // Serialize metadata.
    let metadata_json = serialize_metadata(metadata);
    let metadata_bytes = metadata_json.as_bytes();

    // Assemble: wafer binary + precompiled payload + metadata + trailer.
    let mut out = Vec::with_capacity(
        self_bytes.len() + precompiled.len() + metadata_bytes.len() + TRAILER_SIZE as usize,
    );
    out.extend_from_slice(&self_bytes);
    out.extend_from_slice(&precompiled);
    out.extend_from_slice(metadata_bytes);
    out.extend_from_slice(&(precompiled.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(NATIVE_MAGIC);

    std::fs::write(out_path, &out)?;

    // Make executable on Unix.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(out_path, std::fs::Permissions::from_mode(0o755))?;
    }

    eprintln!(
        "Wrote {out_path} ({:.1} MB native executable, {:.0} KB precompiled WASM)",
        out.len() as f64 / 1_048_576.0,
        precompiled.len() as f64 / 1024.0
    );

    Ok(())
}

/// `wafer run program.wasm`
fn cmd_run(file: &str) -> anyhow::Result<()> {
    let output = run_wasm_file(file)?;
    if !output.is_empty() {
        print!("{output}");
    }
    Ok(())
}

/// `wafer` (REPL) or `wafer program.fth` (evaluate and exit)
fn cmd_eval_or_repl(file: Option<&str>) -> anyhow::Result<()> {
    let mut vm = ForthVM::<NativeRuntime>::new()?;

    match file {
        Some(file) => {
            let source = std::fs::read_to_string(file)?;
            vm.evaluate(&source)?;
            let output = vm.take_output();
            if !output.is_empty() {
                print!("{output}");
            }
        }
        None => {
            if !stdin_is_tty() {
                // Non-interactive: read all of stdin and evaluate
                let mut input = String::new();
                Read::read_to_string(&mut std::io::stdin(), &mut input)?;
                for line in input.lines() {
                    match vm.evaluate(line) {
                        Ok(()) => {
                            let output = vm.take_output();
                            if !output.is_empty() {
                                print!("{output}");
                            }
                        }
                        Err(e) => {
                            eprintln!("Error: {e}");
                        }
                    }
                }
            } else {
                // Interactive REPL
                println!(
                    "WAFER v{} - WebAssembly Forth Engine in Rust",
                    env!("CARGO_PKG_VERSION")
                );
                println!("Type BYE to exit.");

                let mut rl = rustyline::DefaultEditor::new()?;
                loop {
                    let prompt = if vm.is_compiling() { "  ] " } else { "> " };
                    match rl.readline(prompt) {
                        Ok(line) => {
                            let trimmed = line.trim();
                            if trimmed.eq_ignore_ascii_case("BYE") {
                                break;
                            }
                            let _ = rl.add_history_entry(&line);
                            match vm.evaluate(&line) {
                                Ok(()) => {
                                    let output = vm.take_output();
                                    // PAGE (form feed) clears the terminal
                                    if output.contains('\x0C') {
                                        print!("\x1b[2J\x1b[H");
                                    }
                                    let output = output.replace('\x0C', "");
                                    if !vm.is_compiling() {
                                        // Move cursor back up to end of input line so
                                        // output appears inline, like traditional Forth:
                                        //   > 2 2 + . 4  ok
                                        let col = prompt.len() + line.len() + 1;
                                        print!("\x1b[A\x1b[{col}G {output} ok");
                                        println!();
                                    } else if !output.is_empty() {
                                        print!("{output}");
                                    }
                                }
                                Err(e) => {
                                    eprintln!("Error: {e}");
                                }
                            }
                        }
                        Err(
                            rustyline::error::ReadlineError::Interrupted
                            | rustyline::error::ReadlineError::Eof,
                        ) => {
                            break;
                        }
                        Err(e) => {
                            eprintln!("Readline error: {e}");
                            break;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Check if stdin is a terminal (TTY).
fn stdin_is_tty() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal()
}
