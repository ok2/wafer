# WAFER Project Conventions

## What is WAFER?

WAFER (WebAssembly Forth Engine in Rust) is an optimizing Forth 2012 compiler targeting WebAssembly. Currently a working Forth system with 200+ words, JIT compilation, 12 word sets at 100% compliance, and a full optimization pipeline (peephole, constant folding, inlining, strength reduction, DCE, tail calls, stack-to-local promotion with loop/IF support, self-recursive direct calls, consolidation). Beats gforth on all benchmarks in release mode.

## Architecture

- Each Forth word compiles to its own WASM module via `wasm-encoder`
- Modules share memory, globals (dsp/rsp), and a function table via wasmtime imports
- IR-based compilation: Forth -> `Vec<IrOp>` -> WASM codegen -> wasmtime instantiation
- Dictionary: linked-list in a `Vec<u8>` buffer simulating WASM linear memory
- Primitives: either IR-based (compiled to WASM) or host functions (Rust closures in wasmtime)

## Key Files

- `crates/core/src/outer.rs` -- ForthVM: the main runtime, outer interpreter, compiler, all primitives
- `crates/core/src/codegen.rs` -- IR-to-WASM translation, module generation, wasmtime execution tests
- `crates/core/src/dictionary.rs` -- Dictionary data structure with create/find/reveal
- `crates/core/src/ir.rs` -- IrOp enum (the intermediate representation)
- `crates/core/src/memory.rs` -- Memory layout constants (stack regions, dictionary base, etc.)
- `crates/core/src/optimizer.rs` -- IR optimization passes (peephole, fold, inline, DCE, etc.)
- `crates/core/src/config.rs` -- WaferConfig: unified optimization configuration
- `crates/core/src/consolidate.rs` -- Consolidation recompiler (single-module direct calls)
- `crates/cli/src/main.rs` -- CLI REPL with rustyline

## Adding a New Word

**IR primitive** (simple stack/arithmetic/logic -- preferred when possible):

```rust
self.register_primitive("WORD_NAME", false, vec![IrOp::Dup, IrOp::Mul])?;
```

**Host function** (needs Rust logic -- I/O, dictionary manipulation, complex stack access):

```rust
let func = Func::new(&mut self.store, func_type.clone(), move |mut caller, _params, _results| {
    // manipulate memory/globals directly
    Ok(())
});
self.register_host_primitive("WORD_NAME", false, func)?;
```

**Special interpreter token** (defining words like VARIABLE, CONSTANT, CREATE):
Handle in `interpret_token_immediate()` or `compile_token()` as a special case.

## Code Style

- `cargo fmt --all` and `cargo clippy --workspace` must pass with no warnings
- Every public function needs a doc comment
- Use `thiserror` for error types in core crate, `anyhow` for CLI
- Prefer returning `Result` over panicking

## Testing

- Run `cargo test --workspace` before committing (currently 427 unit + 1 benchmark + 11 compliance + 11 comparison)
- Forth 2012 compliance: `cargo test -p wafer-core --test compliance`
- Cross-engine comparison (vs gforth): `cargo test -p wafer-core --test comparison`
- Performance benchmarks (release mode): `cargo test -p wafer-core --test comparison -- --nocapture --ignored`
- Test helper in outer.rs: `eval_output("forth code")` returns printed output as String
- Test helper: `eval_stack("forth code")` returns data stack as Vec<i32>

## Key Principles

1. Correctness first, performance second
2. Maximize Forth, minimize Rust (self-hosting goal -- not yet started)
3. Test-driven: if it's not tested, it doesn't work
4. Every word set at 100% compliance before moving to the next
5. Never break existing tests
