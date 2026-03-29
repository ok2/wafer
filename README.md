# WAFER

**WebAssembly Forth Engine in Rust**

An optimizing Forth 2012 compiler targeting WebAssembly.

## Goals

- **Full Forth 2012 compliance** -- all word sets, 100% test suite pass rate
- **Optimizing compiler** -- constant folding, inlining, peephole optimization, stack-to-local promotion
- **Multi-typed stack** -- type inference uses WASM's native typed stack when possible, memory fallback for dynamic cases
- **Self-hosting** -- minimal Rust kernel (~35 primitives), everything else written in WAFER Forth
- **Dual mode** -- JIT for interactive development, consolidation recompilation into a single optimized WASM module

## Architecture

```
Forth Source -> Outer Interpreter -> IR (with type inference) -> Optimization -> WASM Codegen
```

- **Subroutine threading** via WASM function tables
- **JIT mode**: each new word compiles to a separate WASM module
- **Consolidation mode**: recompile all words into a single module with direct calls
- **IR-based pipeline** enables optimization passes before WASM emission

## Building

```bash
cargo build --workspace
```

## Running

```bash
# Interactive REPL
cargo run -p wafer

# Run a Forth file
cargo run -p wafer -- file.fth

# Compile to optimized WASM
cargo run -p wafer -- --consolidate file.fth -o output.wasm
```

## Testing

```bash
# All tests
cargo test --workspace

# Forth 2012 compliance dashboard
cargo test --test compliance

# Benchmarks
cargo bench --workspace
```

## Project Structure

```
crates/
  core/       wafer-core: compiler, IR, optimizer, codegen
  cli/        wafer: CLI REPL and AOT compiler
  web/        wafer-web: browser bindings (planned)
forth/        Standard library written in WAFER Forth
tests/        Integration tests and Forth 2012 compliance suite
```

## Compliance Status

Targeting 100% Forth 2012 compliance, tested with [Gerry Jackson's forth2012-test-suite](https://github.com/gerryjackson/forth2012-test-suite).

| Word Set | Status |
|----------|--------|
| Core | Pending |
| Core Extensions | Pending |
| Block | N/A |
| Double-Number | Pending |
| Exception | Pending |
| Facility | Pending |
| File-Access | Pending |
| Floating-Point | Pending |
| Locals | Pending |
| Memory-Allocation | Pending |
| Programming-Tools | Pending |
| Search-Order | Pending |
| String | Pending |
| Extended-Character | Pending |

## License

MIT OR Apache-2.0
