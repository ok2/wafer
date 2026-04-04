//! WAFER CLI: Interactive REPL, AOT compiler, and WASM runner for WAFER Forth.

use std::path::Path;

use clap::{Parser, Subcommand};
use wafer_core::export::{ExportConfig, export_module};
use wafer_core::outer::ForthVM;
use wafer_core::runner::run_wasm_file;

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

        /// Output .wasm file (default: input with .wasm extension)
        #[arg(short, long)]
        output: Option<String>,

        /// Entry-point word name (default: MAIN, or top-level execution)
        #[arg(long)]
        entry: Option<String>,

        /// Also generate a JS loader and HTML page for browser execution
        #[arg(long)]
        js: bool,
    },

    /// Run a pre-compiled WASM module
    Run {
        /// .wasm file to execute
        file: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Build {
            file,
            output,
            entry,
            js,
        }) => cmd_build(&file, output.as_deref(), entry, js),

        Some(Commands::Run { file }) => cmd_run(&file),

        None => cmd_eval_or_repl(cli.file.as_deref()),
    }
}

/// `wafer build program.fth -o program.wasm`
fn cmd_build(
    file: &str,
    output: Option<&str>,
    entry: Option<String>,
    js: bool,
) -> anyhow::Result<()> {
    let source = std::fs::read_to_string(file)?;

    let mut vm = ForthVM::new()?;
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
            format!("{stem}.wasm")
        }
    };

    std::fs::write(&out_path, &wasm_bytes)?;

    let word_count = vm.ir_words().len();
    let host_count = metadata.host_functions.len();
    eprintln!(
        "Wrote {out_path} ({} bytes, {word_count} words, {host_count} host functions)",
        wasm_bytes.len()
    );

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
    let mut vm = ForthVM::new()?;

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
                std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)?;
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
                                    if !output.is_empty() {
                                        print!("{output}");
                                    }
                                    if !vm.is_compiling() {
                                        println!(" ok");
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
