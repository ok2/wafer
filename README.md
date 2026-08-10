# WAFER

**WebAssembly Forth Engine in Rust**

An optimizing Forth 2012 compiler targeting WebAssembly. WAFER JIT-compiles each word definition to a separate WASM module and executes it via [wasmtime](https://wasmtime.dev/) (CLI) or the browser's WebAssembly API (web REPL).

## Highlights

- **200+ words** across 12 Forth 2012 word sets, all at **100% compliance**
- **Optimizing compiler** with 6 IR passes + stack-to-local promotion (per region, so a hot loop keeps its registers even inside a word that does I/O; `DO` and `BEGIN` loops alike) + consolidation
- **Faster than gforth** on every benchmark, and past SwiftForth `sf64` -- a native-code compiler -- on four of five
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

WAFER beats gforth (the GNU Forth reference implementation) on every benchmark, and SwiftForth
`sf64` -- which compiles to native code -- on four of the five.

Measured with all three engines running **native x86-64**, on an idle 16-vCPU Xeon Platinum 8124M
@ 3.0 GHz (median of three runs):

```
Benchmark                   WAFER     gforth       sf64    WAFER/gf   WAFER/sf
Fibonacci(25)                 411       3221        355       0.13x      1.16x
Factorial(12)x100K            994       7141       3058       0.14x      0.33x
GCD-bench(20K)               1591       3211       2423       0.50x      0.66x
NestedLoops(50)x1K            889       6824       2342       0.13x      0.38x
Collatz(2K)                   391       3981       1659       0.10x      0.24x
```

Times in microseconds; WAFER is the better of the JIT and `CONSOLIDATE` runs. Below 1.0 means WAFER
is faster. Fibonacci is the one WAFER loses: it is one call per node with no loop to promote, and
`sf64` keeps its stack in registers across a call the way only a native code generator can.
Fibonacci, GCD and Collatz held to within 2% across the three runs; Factorial and NestedLoops are
softer, since `sf64` varied by half there, but they are wide wins either way.

`just bench-compare` on the development machine (M1 Ultra, arm64) reports different numbers, and
they flatter WAFER:

```
Benchmark                   WAFER     CONSOL     gforth       sf64    WAFER/gf   WAFER/sf
Fibonacci(25)                 237        242       3340        287       0.07x      0.83x
Factorial(12)x100K            480        479       6109       1594       0.08x      0.30x
GCD-bench(20K)                549        541       1830        797       0.30x      0.68x
NestedLoops(50)x1K            501        509       7092       1898       0.07x      0.26x
Collatz(2K)                   196        190       3955        633       0.05x      0.30x
```

The only SwiftForth build for macOS is x86-64 under Rosetta 2, while WAFER and gforth are native
arm64 -- so that `sf64` column is native against emulated. The gap is not small, and it lands
exactly where it matters: Fibonacci reads 0.83x there and 1.16x when neither engine is emulated.
Treat the arm64 table as what the regression limits in `comparison.rs` are calibrated against, and
the x86-64 table as what to believe about the engines.

A caveat applies to both: `sf64` uses 64-bit cells to WAFER's 32-bit, so WAFER does less work per
operation.

A word whose stack effect is statically known gets a **typed entry point**: its stack items travel in and out
as WASM values instead of through the memory data stack, so cranelift keeps them in registers across a call
the way a native Forth keeps TOS in one. The word also keeps a `( -- )` wrapper, which is what the function
table, `EXECUTE` and the outer interpreter reach, so nothing about the memory ABI changes from the outside.
Call-heavy code is what this pays for -- Fibonacci went from 4.3x slower than `sf64` to 1.2x. Set
`WAFER_TYPED_CALLS=0` to fall back to the memory-stack convention.

Recursive words then get one more thing: their base-case guard is tested at the **call site**, so a
leaf of the recursion costs a comparison instead of a call. `: FIB DUP 2 < IF EXIT THEN ... RECURSE`
compiles its `RECURSE` as `DUP 2 < IF ELSE RECURSE THEN`, which is what the callee would have done
on entry anyway. Half of fib's nodes are leaves, and that is worth 1.4x.

## Testing

```bash
# All tests (~635 currently passing)
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
- **IR-based pipeline** with 6 optimization passes (peephole, constant folding, strength reduction, DCE, tail call detection, inlining) plus per-region stack-to-local promotion (DO and BEGIN loops, IF/ELSE), DO/LOOP index locals, typed entry points for words with a known stack effect, self-guard expansion, and consolidation
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
