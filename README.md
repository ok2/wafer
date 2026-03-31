# WAFER

**WebAssembly Forth Engine in Rust**

An optimizing Forth 2012 compiler targeting WebAssembly.

## Status

WAFER is a working Forth system. It JIT-compiles each word definition to a separate WASM module and executes via `wasmtime`. 272 tests passing (261 unit + 11 compliance), **0 errors on all 11 tested Forth 2012 word sets**.

**Working features:**

- Colon definitions with full control flow (IF/ELSE/THEN, DO/LOOP/+LOOP, BEGIN/UNTIL, BEGIN/WHILE/REPEAT)
- 130+ words: stack, arithmetic, comparison, logic, memory, I/O, defining words, system, exceptions, double-cell, strings
- Recursion (RECURSE), nested control structures, loop counters (I, J)
- VARIABLE, CONSTANT, CREATE, DOES>
- Number bases (HEX, DECIMAL), number prefixes ($hex, #dec, %bin)
- Pictured numeric output (<# # #S #> HOLD SIGN)
- Comments (backslash, parentheses), string output (." ...)
- Interactive REPL with line editing

**Example session:**

```forth
: FIB DUP 2 < IF DROP 1 ELSE DUP 1 - RECURSE SWAP 2 - RECURSE + THEN ;
: FIBS 0 DO I FIB . LOOP ;
12 FIBS CR    \ prints: 1 1 2 3 5 8 13 21 34 55 89 144

VARIABLE COUNTER  0 COUNTER !
: BUMP COUNTER @ 1 + COUNTER ! ;
BUMP BUMP BUMP COUNTER @ .  \ prints: 3
```

## Goals

- **Full Forth 2012 compliance** -- all word sets, 100% test suite pass rate
- **Optimizing compiler** -- constant folding, inlining, peephole optimization, stack-to-local promotion
- **Multi-typed stack** -- type inference uses WASM's native typed stack when possible
- **Self-hosting** -- minimal Rust kernel (~35 primitives), everything else in WAFER Forth
- **Consolidation mode** -- recompile all JIT words into a single optimized WASM module

## Architecture

```
Forth Source -> Outer Interpreter -> IR -> [Optimize] -> WASM Codegen (wasm-encoder)
                                                              |
                                                    wasmtime instantiation
                                                    (shared memory + table)
```

- **Subroutine threading** via WASM function tables and `call_indirect`
- **JIT mode**: each new word compiles to a separate WASM module linked to shared memory/globals/table
- **IR-based pipeline** enables future optimization passes before WASM emission
- **Dictionary**: linked-list word headers in simulated linear memory

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

# Pipe input
echo ': SQUARE DUP * ; 7 SQUARE .' | cargo run -p wafer
```

## Testing

```bash
# All tests (185 currently passing)
cargo test --workspace

# Forth 2012 compliance dashboard
cargo test -p wafer-core --test compliance

# Lints
cargo clippy --workspace
```

## Project Structure

```
crates/
  core/       wafer-core: dictionary, IR, codegen (wasm-encoder), outer interpreter
  cli/        wafer: CLI REPL and file execution (wasmtime, rustyline)
  web/        wafer-web: browser bindings (planned)
forth/        Standard library in WAFER Forth (planned, currently stubs)
tests/        Forth 2012 compliance suite (gerryjackson/forth2012-test-suite submodule)
```

## Implemented Words

### Core (Forth 2012 Section 6.1) -- In Progress

| Category     | Words                                                                                                           |
| ------------ | --------------------------------------------------------------------------------------------------------------- |
| Stack        | `DUP DROP SWAP OVER ROT NIP TUCK 2DUP 2DROP 2SWAP 2OVER ?DUP PICK DEPTH`                                        |
| Arithmetic   | `+ - * / MOD /MOD NEGATE ABS MIN MAX 1+ 1- 2* 2/ */ */MOD M* UM* UM/MOD FM/MOD SM/REM S>D <# # #S #> HOLD SIGN` |
| Comparison   | `= <> < > U< 0= 0< 0<> 0> WITHIN`                                                                               |
| Logic        | `AND OR XOR INVERT LSHIFT RSHIFT`                                                                               |
| Memory       | `@ ! C@ C! +! 2@ 2! HERE ALLOT , C, CELLS CELL+ CHARS CHAR+ ALIGNED ALIGN MOVE FILL CMOVE CMOVE>`               |
| Control      | `IF ELSE THEN DO LOOP +LOOP I J UNLOOP LEAVE BEGIN UNTIL WHILE REPEAT RECURSE EXIT`                             |
| Defining     | `: ; VARIABLE CONSTANT CREATE DOES> IMMEDIATE`                                                                  |
| I/O          | `. U. .S CR EMIT SPACE SPACES TYPE ." S" ACCEPT`                                                                |
| Return stack | `>R R> R@`                                                                                                      |
| System       | `EXECUTE ' CHAR [CHAR] ['] DECIMAL HEX BASE STATE >IN >BODY ENVIRONMENT? SOURCE ABORT TRUE FALSE BL`            |
| Compiler     | `LITERAL POSTPONE [ ] EVALUATE ABORT"`                                                                          |
| Parsing      | `WORD FIND COUNT >NUMBER`                                                                                       |

### Not Yet Implemented

11 word sets at 100% compliance: Core, Core Ext, Core Plus, Exception, Double-Number, String, Search-Order, Memory-Allocation, Programming-Tools, Facility, Locals. 130+ words including VALUE, DEFER, CASE, DOES>, CATCH/THROW, double-cell arithmetic, string operations.

## Compliance Status

Targeting 100% Forth 2012 compliance via [Gerry Jackson's test suite](https://github.com/gerryjackson/forth2012-test-suite).

| Word Set           | Status                            |
| ------------------ | --------------------------------- |
| Core               | **100%** (0 errors on test suite) |
| Core Extensions    | **100%** (0 errors on test suite) |
| Double-Number      | **100%** (0 errors on test suite) |
| Exception          | **100%** (0 errors on test suite) |
| Facility           | **100%** (0 errors on test suite) |
| File-Access        | Pending (requires WASI)           |
| Floating-Point     | Pending                           |
| Locals             | **100%** (0 errors on test suite) |
| Memory-Allocation  | **100%** (0 errors on test suite) |
| Programming-Tools  | **100%** (0 errors on test suite) |
| Search-Order       | **100%** (0 errors on test suite) |
| String             | **100%** (0 errors on test suite) |
| Extended-Character | Pending                           |

## License

MIT OR Apache-2.0
