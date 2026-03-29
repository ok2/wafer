//! WAFER CLI: Interactive REPL and AOT compiler for WAFER Forth.

use clap::Parser;

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

    match cli.file {
        Some(ref _file) => {
            // TODO: Step 9 - Load and execute Forth file
            eprintln!("WAFER: file execution not yet implemented");
        }
        None => {
            // TODO: Step 9 - Interactive REPL
            println!(
                "WAFER v{} - WebAssembly Forth Engine in Rust",
                env!("CARGO_PKG_VERSION")
            );
            println!("Type BYE to exit.");
            eprintln!("REPL not yet implemented");
        }
    }

    Ok(())
}
