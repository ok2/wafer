# WAFER Project Conventions

## What is WAFER?
WAFER (WebAssembly Forth Engine in Rust) is an optimizing Forth 2012 compiler targeting WebAssembly.

## Architecture
- Rust kernel (~35 primitives) + Forth standard library (everything else in .fth files)
- IR-based compilation pipeline: Forth -> IR -> type inference -> optimize -> WASM codegen
- Multi-typed stack: use WASM's native typed stack when types are known via inference, fall back to linear memory for dynamic/polymorphic cases
- Subroutine threading via WASM function tables
- JIT mode: per-word WASM modules + shared function table
- Consolidation mode: recompile all words into single optimized WASM module

## Code Style
- `cargo fmt` and `cargo clippy` must pass with no warnings
- Every public function needs a doc comment
- Every module needs unit tests
- Use `thiserror` for error types in core crate, `anyhow` for CLI crate
- Prefer returning `Result` over panicking

## Testing (Critical)
- **Specs-driven TDD**: Every feature starts with its failing test, then implementation
- Run `cargo test --workspace` before committing
- Forth 2012 compliance: `cargo test --test compliance`
- Property-based tests with `proptest` for numeric operations and optimizer correctness
- Snapshot tests with `insta` for IR and WASM output
- 100% compliance is mandatory for each implemented word set before moving on
- Never break existing compliance tests

## Forth Source (.fth files)
- One file per word set in `forth/`
- Document each word with standard stack effect notation: `( before -- after )`
- Maximize words written in Forth, minimize Rust primitives
- Boot order: boot.fth -> core.fth -> core_ext.fth -> ... -> prelude.fth

## Key Principles
1. Maximize Forth, minimize Rust (self-hosting goal)
2. Correctness first, performance second
3. Test-driven: if it's not tested, it doesn't work
4. Every word set at 100% compliance before moving to the next
