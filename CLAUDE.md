# WAFER Project Conventions

## What is WAFER?

WAFER (WebAssembly Forth Engine in Rust) is an optimizing Forth 2012 compiler targeting WebAssembly. Currently a working Forth system with 200+ words, JIT compilation, 12 word sets at 100% compliance, and a full optimization pipeline (peephole, constant folding, inlining, strength reduction, DCE, tail calls, stack-to-local promotion with loop/IF support, self-recursive direct calls, consolidation). Beats gforth on all benchmarks in release mode. Includes a browser-based REPL via wasm-pack.

## Architecture

- Each Forth word compiles to its own WASM module via `wasm-encoder`
- Modules share memory, globals (dsp/rsp/fsp), and a function table via runtime imports
- IR-based compilation: Forth -> `Vec<IrOp>` -> WASM codegen -> runtime instantiation
- Dictionary: linked-list in a `Vec<u8>` buffer simulating WASM linear memory
- Primitives: either IR-based (compiled to WASM) or host functions (closures via `HostFn`)
- **Runtime trait**: `ForthVM<R: Runtime>` is generic over the execution backend
  - `NativeRuntime` (wasmtime) -- CLI, tests, AOT compilation
  - `WebRuntime` (js-sys) -- browser REPL via wasm-pack

## Crate Structure

- `crates/core` -- compiler, optimizer, codegen, dictionary, runtime traits, outer interpreter
- `crates/cli` -- CLI REPL with rustyline, `wafer build`/`wafer run` commands
- `crates/web` -- browser REPL (wasm-bindgen entry point, WebRuntime, HTML/CSS/JS frontend)

## Key Files

- `crates/core/src/outer.rs` -- `ForthVM<R: Runtime>`: outer interpreter, compiler, all primitives
- `crates/core/src/runtime.rs` -- `Runtime` + `HostAccess` traits (execution backend abstraction)
- `crates/core/src/runtime_native.rs` -- `NativeRuntime`: wasmtime implementation (behind `native` feature)
- `crates/core/src/codegen.rs` -- IR-to-WASM translation, module generation
- `crates/core/src/dictionary.rs` -- Dictionary data structure with create/find/reveal
- `crates/core/src/ir.rs` -- IrOp enum (the intermediate representation)
- `crates/core/src/memory.rs` -- Memory layout constants (stack regions, dictionary base, etc.)
- `crates/core/src/optimizer.rs` -- IR optimization passes (peephole, fold, inline, DCE, etc.)
- `crates/core/src/config.rs` -- WaferConfig: unified optimization configuration
- `crates/core/src/consolidate.rs` -- Consolidation recompiler (single-module direct calls)
- `crates/core/boot.fth` -- Bootstrap Forth definitions loaded at startup
- `crates/cli/src/main.rs` -- CLI REPL with rustyline
- `crates/web/src/lib.rs` -- `WaferRepl` wasm-bindgen entry point
- `crates/web/src/runtime_web.rs` -- `WebRuntime`: browser WebAssembly API via js-sys
- `crates/web/www/` -- Frontend (index.html, style.css, app.js)

## Feature Flags (wafer-core)

- `default = ["native"]` -- includes wasmtime, NativeRuntime, runner, export, etc.
- `native` -- enables `dep:wasmtime` and all native-only modules
- No features -- pure Rust only (dictionary, IR, optimizer, codegen, outer interpreter). Used by `wafer-web`.

## Adding a New Word

**IR primitive** (simple stack/arithmetic/logic -- preferred when possible):

```rust
self.register_primitive("WORD_NAME", false, vec![IrOp::Dup, IrOp::Mul])?;
```

**Host function** (needs Rust logic -- I/O, dictionary manipulation, complex stack access):

```rust
let shared_state = Arc::clone(&self.some_field);
let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
    let sp = ctx.get_dsp();
    let val = ctx.mem_read_i32(sp);
    // ... logic using ctx for memory/global access ...
    ctx.set_dsp(sp + CELL_SIZE);
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

- Run `cargo test --workspace` before committing (currently 431 unit + 1 benchmark + 11 compliance + 9 comparison)
- Forth 2012 compliance: `cargo test -p wafer-core --test compliance`
- Cross-engine comparison (vs gforth): `cargo test -p wafer-core --test comparison`
- Performance benchmarks (release mode): `cargo test -p wafer-core --test comparison -- --nocapture --ignored`
- Test helper in outer.rs: `eval_output("forth code")` returns printed output as String
- Test helper: `eval_stack("forth code")` returns data stack as Vec<i32>

## Web REPL

- Build: `cd crates/web && wasm-pack build --target web --out-dir www/pkg`
- Serve: `python3 -m http.server -d crates/web/www 8080`
- Open: `http://localhost:8080/`
- Dev build (faster, unoptimized): `wasm-pack build --target web --dev --out-dir www/pkg`

## Key Principles

1. Correctness first, performance second
2. Maximize Forth, minimize Rust (self-hosting goal -- not yet started)
3. Test-driven: if it's not tested, it doesn't work
4. Every word set at 100% compliance before moving to the next
5. Never break existing tests
