//! WAFER CLI: Interactive REPL and AOT compiler for WAFER Forth.

use clap::Parser;
use wafer_core::outer::ForthVM;

/// WAFER: WebAssembly Forth Engine in Rust
#[derive(Parser, Debug)]
#[command(name = "wafer", version, about)]
struct Cli {
    /// Forth source file to execute
    file: Option<String>,

    /// Compile all words into a single optimized WASM module
    #[arg(long)]
    consolidate: bool,

    /// Output file for consolidated WASM (requires --consolidate)
    #[arg(short, long)]
    output: Option<String>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let mut vm = ForthVM::new()?;

    match cli.file {
        Some(ref file) => {
            let source = std::fs::read_to_string(file)?;
            vm.evaluate(&source)?;
            let output = vm.take_output();
            if !output.is_empty() {
                print!("{output}");
            }
        }
        None => {
            // Check if stdin is a pipe (not a TTY)
            if !atty_is_tty() {
                // Non-interactive: read all of stdin and evaluate
                let mut input = String::new();
                std::io::Read::read_to_string(&mut std::io::stdin(), &mut input)?;
                // Evaluate line-by-line to handle multi-line input
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
fn atty_is_tty() -> bool {
    use std::io::IsTerminal;
    std::io::stdin().is_terminal()
}
