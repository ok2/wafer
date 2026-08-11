# WAFER

**WebAssembly Forth Engine in Rust**

An optimizing Forth 2012 compiler targeting WebAssembly. WAFER JIT-compiles each word definition to a separate WASM module and executes it via [wasmtime](https://wasmtime.dev/) (CLI) or the browser's WebAssembly API (web REPL).

## Highlights

- **200+ words** across 12 Forth 2012 word sets, all at **100% compliance**
- **Optimizing compiler** with 6 IR passes + stack-to-local promotion (per region, so a hot loop keeps its registers even inside a word that does I/O; `DO` and `BEGIN` loops alike) + consolidation
- **Faster than gforth** on every benchmark, and past SwiftForth `sf64` -- a native-code compiler -- on five of six
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

WAFER beats gforth (the GNU Forth reference implementation) on every benchmark by 3-20x, and
SwiftForth `sf64` -- which compiles to native code -- on five of the six. Fibonacci is the one it
loses: one call per node, no loop to promote, and `sf64` keeps its stack in registers across a call
the way only a native code generator can.

Measured on the development machine (M1 Ultra, arm64), median of three reports:

```
Benchmark                   WAFER     CONSOL     gforth       sf64    WAFER/gf   WAFER/sf
Fibonacci(33)               11307      11407     157001      13053       0.07x      0.87x
Factorial(12)x2M             9639       9599     123950      32091       0.08x      0.30x
GCD-bench(400K)             11662      11580      38580      17001       0.30x      0.68x
NestedLoops(50)x20K          8920       9852     140518      36828       0.06x      0.24x
CrossCalls(3M)              10883       3769      87691       8240       0.04x      0.46x
Collatz(2K)x50               8838       8715     189903      28657       0.05x      0.30x
```

Times in microseconds; the ratios use the better of `WAFER` and `CONSOL`. Below 1.0 means WAFER is
faster.

**The `sf64` column here flatters WAFER, and by enough to change an answer.** The only SwiftForth
build for macOS is x86-64 running under Rosetta 2, while WAFER and gforth are native arm64 -- so
that column compares native code against emulated code, and the penalty falls hardest on the
call-heavy benchmark. Measured with all three engines native on x86-64 (Xeon Platinum 8124M,
Ubuntu 22.04; two reports agreed within 1%), Fibonacci reads **1.21x** where the table above says
0.87x; the other five keep their wins. That native comparison is what the "five of six" above
rests on:

```
Benchmark                   WAFER     CONSOL     gforth       sf64    WAFER/gf   WAFER/sf
Fibonacci(33)               19512      19511     129784      16076       0.15x      1.21x
Factorial(12)x2M            22532      16601     137168      57986       0.12x      0.29x
GCD-bench(400K)             34216      34089      66595      51680       0.51x      0.66x
NestedLoops(50)x20K         10729      17827     126687      40469       0.08x      0.27x
CrossCalls(3M)              20457       7412      81303      29264       0.09x      0.25x
Collatz(2K)x50              18686      17328     188592      80857       0.09x      0.21x
```

A second caveat holds on any host: `sf64` uses 64-bit cells to WAFER's 32-bit, so WAFER does less
work per operation.

`CrossCalls` is the only benchmark with a cross-word call left in its hot loop -- the other five
have their callee inlined away or are self-recursive -- so it is the only one that measures what
`CONSOLIDATE` does, and there it is worth 2.9x. `NestedLoops` goes the other way: `CONSOLIDATE`
makes it 1.1x _slower_ on the M1 and 1.7x on x86-64 -- not worse code but worse luck. Both paths
emit identical WASM for the hot word; the delta is where the machine code lands. A tight loop
pays for straddling an instruction-fetch window (16 bytes on the M1, 32 on Skylake, where a fused
branch crossing the boundary drops the loop out of the uop cache -- the JCC erratum), Cranelift
does not align loop headers, and dead prologue bytes in the per-word JIT module happen to shift
its loops into luckier spots. Details in
[docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md#8-consolidation).

Every benchmark is sized to run about 10 ms. Not for the usual reason -- the timing wrapper already
excludes start-up and compilation -- but to keep a comfortable margin over timer resolution and
first-iteration effects without pushing the report past a minute. gforth is 3-20x slower than
WAFER, so it sets the wall clock.

A word whose stack effect is statically known gets a **typed entry point**: its stack items travel in and out
as WASM values instead of through the memory data stack, so cranelift keeps them in registers across a call
the way a native Forth keeps TOS in one. The word also keeps a `( -- )` wrapper, which is what the function
table, `EXECUTE` and the outer interpreter reach, so nothing about the memory ABI changes from the outside.
Only a caller inside the same module can use the fast entry -- `RECURSE` in the JIT path, every resolvable
call after `CONSOLIDATE` -- so that is exactly when it is emitted. Set `WAFER_TYPED_CALLS=0` to fall back.

Recursive words then get one more thing: their base-case guard is tested at the **call site**, so a
leaf of the recursion costs a comparison instead of a call. `: FIB DUP 2 < IF EXIT THEN ... RECURSE`
compiles its `RECURSE` as `DUP 2 < IF ELSE RECURSE THEN`, which is what the callee would have done
on entry anyway. Half of fib's nodes are leaves, and that is worth 1.4x.

## Testing

Everything below has a `just` target; the raw command is given where it is worth
knowing what the target does.

```bash
just test            # all tests (~638 currently passing)
just compliance      # Forth 2012 compliance suite
just clippy          # lints
just fmt             # formatting check (Rust + Markdown)
just ci              # everything CI runs
```

Benchmarks are separate, because they are `#[ignore]`d -- they take minutes, and
a debug build would measure nothing useful:

```bash
just bench-compare       # WAFER vs gforth vs SwiftForth, the table in Performance
just bench-opts          # WAFER against its own optimization settings
just bench               # criterion micro-benchmarks
just compare-correctness # same three engines, compared on output instead of time
```

`bench-compare` needs `gforth` and `sf64` on `PATH` -- a missing engine drops its
column rather than failing. Each number in it is the best of three processes, and
each process reports the mean of its three fastest of seven timed repetitions:
benchmark noise is one-sided, so the fastest runs are the honest ones, and only a
fresh process resamples core placement and code layout. Run it on an idle
machine; a busy one produced 20-79% run-to-run spread where an idle one gives
1-6%.

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
