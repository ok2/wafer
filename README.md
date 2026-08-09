# WAFER

**WebAssembly Forth Engine in Rust**

An optimizing Forth 2012 compiler targeting WebAssembly. WAFER JIT-compiles each word definition to a separate WASM module and executes it via [wasmtime](https://wasmtime.dev/) (CLI) or the browser's WebAssembly API (web REPL).

## Highlights

- **200+ words** across 12 Forth 2012 word sets, all at **100% compliance**
- **Optimizing compiler** with 6 IR passes + stack-to-local promotion (loops + IF) + consolidation
- **Faster than gforth** on all benchmarks in release mode (2-10x faster)
- **JIT compilation** — each `:` definition compiles to its own WASM module
- **Self-recursive direct calls** — RECURSE compiles to native `call` instead of `call_indirect`
- **Typed calling convention** — a word with a statically known stack effect passes its stack items as WASM values, so a call keeps them in registers instead of round-tripping through memory
- **Consolidation mode** — recompile all words into a single optimized WASM module
- **Interactive REPL** with line editing (rustyline)
- **Browser REPL** — runs entirely in the browser via wasm-pack + js-sys
- **Runtime abstraction** — `ForthVM<R: Runtime>` is generic over execution backend (wasmtime or browser)

## Installation

Requires [Rust](https://www.rust-lang.org/tools/install) 1.85+ (edition 2024).

```bash
cargo install --git https://github.com/ok2/wafer.git wafer
```

This installs the `wafer` binary to `~/.cargo/bin/`.

To install from a local checkout:

```bash
cargo install --path crates/cli
```

## Usage

```bash
# Interactive REPL (type BYE to exit)
wafer

# Run a Forth file
wafer program.fth

# Pipe input
echo ': SQUARE DUP * ; 7 SQUARE .' | wafer

# Consolidation: recompile all words into a single optimized WASM module
wafer --consolidate program.fth

# Consolidation with WASM output
wafer --consolidate -o output.wasm program.fth
```

**Example REPL session:**

```forth
: FIB DUP 2 < IF DROP 1 ELSE DUP 1 - RECURSE SWAP 2 - RECURSE + THEN ;
: FIBS 0 DO I FIB . LOOP ;
12 FIBS CR    \ prints: 1 1 2 3 5 8 13 21 34 55 89 144

VARIABLE COUNTER  0 COUNTER !
: BUMP COUNTER @ 1 + COUNTER ! ;
BUMP BUMP BUMP COUNTER @ .  \ prints: 3
```

## Building from source

```bash
git clone --recurse-submodules https://github.com/ok2/wafer.git
cd wafer
cargo build --workspace --release
```

If you already cloned without `--recurse-submodules`, fetch the Forth 2012 test suite with:

```bash
git submodule update --init
```

## Performance

WAFER beats gforth (the GNU Forth reference implementation) on all benchmarks in release mode, and is within
reach of SwiftForth `sf64`, which compiles to native code:

```
Benchmark                   WAFER     CONSOL     gforth       sf64    WAFER/gf   WAFER/sf
Fibonacci(25)                 378        359       3238        296       0.11x      1.21x
Factorial(12)x10K             335        320        633        183       0.51x      1.75x
GCD-bench(500)                 14         15         29         31       0.48x      0.45x
NestedLoops(50)                72         69        698        207       0.10x      0.33x
Collatz(2K)                   994        997       3940        668       0.25x      1.49x
```

Times in microseconds. WAFER/gf < 1.0 means WAFER is faster. CONSOL = after `CONSOLIDATE`.

A word whose stack effect is statically known gets a **typed entry point**: its stack items travel in and out
as WASM values instead of through the memory data stack, so cranelift keeps them in registers across a call
the way a native Forth keeps TOS in one. The word also keeps a `( -- )` wrapper, which is what the function
table, `EXECUTE` and the outer interpreter reach, so nothing about the memory ABI changes from the outside.
Call-heavy code is what this pays for -- Fibonacci went from 4.3x slower than `sf64` to 1.2x. Set
`WAFER_TYPED_CALLS=0` to fall back to the memory-stack convention.

## Testing

```bash
# All tests (~620 currently passing)
cargo test --workspace

# Forth 2012 compliance suite
cargo test -p wafer-core --test compliance

# Cross-engine comparison (WAFER vs gforth, requires gforth)
cargo test -p wafer-core --test comparison -- --nocapture --ignored

# Optimization benchmark report (WAFER-internal)
cargo test -p wafer-core --test benchmark_report -- --nocapture --ignored

# Lints
cargo clippy --workspace
```

## Architecture

```
Forth Source -> Outer Interpreter -> IR -> [Optimize] -> WASM Codegen (wasm-encoder)
                                                              |
                                                    Runtime trait instantiation
                                                    (shared memory + table)
                                                         /           \
                                              NativeRuntime      WebRuntime
                                              (wasmtime)         (js-sys)
```

- **Runtime abstraction**: `ForthVM<R: Runtime>` separates the compiler from the execution engine
  - `NativeRuntime` — wasmtime-based, for CLI, tests, and AOT compilation
  - `WebRuntime` — browser WebAssembly API via js-sys, for the browser REPL
- **Subroutine threading** via WASM function tables (`call_indirect` for cross-word, direct `call` for self-recursion)
- **JIT mode**: each new word compiles to a separate WASM module linked to shared memory/globals/table
- **IR-based pipeline** with 6 optimization passes (peephole, constant folding, strength reduction, DCE, tail call detection, inlining) plus stack-to-local promotion (with loop and IF/ELSE support), DO/LOOP index locals, and consolidation
- **Dictionary**: linked-list word headers in simulated linear memory

## Project Structure

```
crates/
  core/       wafer-core: dictionary, IR, codegen, optimizer, outer interpreter, Runtime trait
  cli/        wafer: CLI REPL, file execution, consolidation
  web/        wafer-web: browser REPL (wasm-bindgen + WebRuntime + HTML/CSS/JS frontend)
tests/        Forth 2012 compliance suite (git submodule)
```

## Forth 2012 Compliance

Tested against [Gerry Jackson's Forth 2012 test suite](https://github.com/gerryjackson/forth2012-test-suite). 12 of 14 word sets pass at 100%.

| Word Set           | Status                                  |
| ------------------ | --------------------------------------- |
| Core               | **100%** (0 errors)                     |
| Core Plus          | **100%** (0 errors)                     |
| Core Extensions    | **100%** (0 errors)                     |
| Double-Number      | **100%** (0 errors)                     |
| Exception          | **100%** (0 errors)                     |
| Facility           | **100%** (0 errors)                     |
| Floating-Point     | **100%** (0 errors)                     |
| Locals             | **100%** (0 errors)                     |
| Memory-Allocation  | **100%** (0 errors)                     |
| Programming-Tools  | **100%** (0 errors)                     |
| Search-Order       | **100%** (0 errors)                     |
| String             | **100%** (0 errors)                     |
| File-Access        | Not started (requires WASI integration) |
| Extended-Character | Not started                             |

## Implemented Words

Over 200 words are implemented across the following categories:

| Category     | Words                                                                                                           |
| ------------ | --------------------------------------------------------------------------------------------------------------- |
| Stack        | `DUP DROP SWAP OVER ROT NIP TUCK 2DUP 2DROP 2SWAP 2OVER ?DUP PICK DEPTH`                                        |
| Arithmetic   | `+ - * / MOD /MOD NEGATE ABS MIN MAX 1+ 1- 2* 2/ */ */MOD M* UM* UM/MOD FM/MOD SM/REM S>D <# # #S #> HOLD SIGN` |
| Comparison   | `= <> < > U< 0= 0< 0<> 0> WITHIN`                                                                               |
| Logic        | `AND OR XOR INVERT LSHIFT RSHIFT`                                                                               |
| Memory       | `@ ! C@ C! +! 2@ 2! HERE ALLOT , C, CELLS CELL+ CHARS CHAR+ ALIGNED ALIGN MOVE FILL CMOVE CMOVE>`               |
| Control      | `IF ELSE THEN DO LOOP +LOOP I J UNLOOP LEAVE BEGIN UNTIL WHILE REPEAT RECURSE EXIT`                             |
| Defining     | `: ; VARIABLE CONSTANT VALUE CREATE DOES> IMMEDIATE DEFER`                                                      |
| I/O          | `. U. .S CR EMIT SPACE SPACES TYPE ." S" ACCEPT`                                                                |
| Return stack | `>R R> R@`                                                                                                      |
| System       | `EXECUTE ' CHAR [CHAR] ['] DECIMAL HEX BASE STATE >IN >BODY ENVIRONMENT? SOURCE ABORT TRUE FALSE BL`            |
| Compiler     | `LITERAL POSTPONE [ ] EVALUATE ABORT"`                                                                          |
| Parsing      | `WORD FIND COUNT >NUMBER`                                                                                       |
| Exceptions   | `CATCH THROW`                                                                                                   |
| Double-cell  | `D+ D- D. D.R DNEGATE DABS D= D< D0= D0< D>S 2CONSTANT 2VARIABLE 2LITERAL M+ M*/`                               |
| Strings      | `COMPARE SEARCH SLITERAL REPLACES SUBSTITUTE UNESCAPE`                                                          |
| Floating-Pt  | `F+ F- F* F/ FABS FNEGATE FSQRT FSIN FCOS FTAN FEXP FLOG FMIN FMAX` and 55+ more                                |
| Case         | `CASE OF ENDOF ENDCASE`                                                                                         |
| Tools        | `WORDS SEE SEE-IR HELP INCLUDE INCLUDED .S F.S ? DUMP MARKER REMEMBER EMPTY GILD BYE`                           |

## Web REPL

Build and run the browser-based REPL:

```bash
cd crates/web
wasm-pack build --target web --out-dir www/pkg
python3 -m http.server -d www 8080
# Open http://localhost:8080/
```

## Roadmap

- **File-Access word set** — requires WASI integration for file I/O
- **Extended-Character word set** — Unicode support
- **Self-hosting** — minimal Rust kernel (~35 primitives), everything else in Forth

## License

MIT OR Apache-2.0
