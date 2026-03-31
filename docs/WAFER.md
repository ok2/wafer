# How WAFER Works

WAFER (WebAssembly Forth Engine in Rust) is a Forth 2012 compiler that JIT-compiles each Forth word to its own WebAssembly module and executes it via [wasmtime](https://wasmtime.dev/). This document describes concretely what happens at each step -- from startup through word definition to execution. For background on the Forth language itself, see [FORTH.md](FORTH.md).

## Project Layout

```
crates/
  core/src/
    outer.rs          ForthVM: outer interpreter, compiler, all primitives
    codegen.rs        IR-to-WASM translation, module generation
    dictionary.rs     Dictionary (linked list in Vec<u8>)
    ir.rs             IrOp enum -- the intermediate representation
    memory.rs         Memory layout constants (addresses, sizes)
    error.rs          Error types
  cli/src/
    main.rs           CLI REPL (rustyline), file execution
  web/src/
    lib.rs            Browser bindings (planned)
forth/
    prelude.fth       Standard library loader (planned)
tests/
    forth2012-test-suite/   Forth 2012 compliance test suite (submodule)
```

The entire compiler and runtime lives in `outer.rs` (~5200 lines). Codegen is in `codegen.rs` (~1500 lines). Everything else is supporting infrastructure.

## What Happens When You Start WAFER

Running `cargo run -p wafer` or the `wafer` binary triggers this sequence:

### 1. ForthVM::new()

`ForthVM::new()` in `outer.rs` creates the shared WebAssembly infrastructure:

```
wasmtime Engine          Compilation engine (shared across all modules)
wasmtime Store           Runtime state container
Linear Memory            16 pages (1 MiB), expandable to 256 pages (16 MiB)
Global: DSP              Data stack pointer, initialized to 0x1540
Global: RSP              Return stack pointer, initialized to 0x2540
Function Table           256 funcref entries (grows as needed)
```

A host function is created in Rust and made available as a WASM import:

- **emit** -- takes an `i32` character code, appends it to the output buffer

This is the only host function that compiled WASM modules import directly. Other I/O words like `.` (dot) are registered as host primitives that live in the function table but are implemented entirely in Rust.

### 2. register_primitives()

For each of the ~80 built-in words (DUP, DROP, +, -, @, !, IF, DO, EMIT, ...), the system:

1. Creates a HIDDEN entry in the dictionary
2. Compiles the word's IR body to a WASM module via `compile_word()`
3. Instantiates the module with wasmtime, linking it to the shared memory, globals, and table
4. Installs the compiled function in the function table at the word's index
5. Reveals the dictionary entry (removes HIDDEN flag)

After this, the dictionary contains all built-in words and the function table has a compiled WASM function for each one.

### 3. REPL loop

The CLI enters a `rustyline` loop. The prompt is `>` in interpret mode and `]` in compile mode. Each line is passed to `vm.evaluate()`, which feeds it to the outer interpreter.

## Memory Layout

All stacks and data structures live in a single WASM linear memory. Addresses are fixed at compile time in `memory.rs`:

```
Address   Region                 Size       Notes
-------   ------                 ----       -----
0x0000    System variables       64 B       STATE, BASE, >IN, HERE, LATEST, HLD, ...
0x0040    Input buffer           1024 B     Current source line being parsed
0x0440    PAD                    256 B      Scratch area for string formatting
0x0540    Data stack             4096 B     1024 cells, grows downward
0x1540    Return stack           4096 B     1024 cells, grows downward
0x2540    Float stack            2048 B     256 doubles, grows downward
0x2D40    Dictionary             variable   Linked list of word headers, grows upward
  ...
0x10000   User data space        variable   VARIABLE, CREATE, ALLOT data, grows upward
```

The data stack pointer (DSP) starts at 0x1540 (the top of the data stack region) and decrements by 4 bytes for each push. The return stack pointer (RSP) starts at 0x2540 and works the same way.

## What Happens When You Type `5 3 + .`

The outer interpreter in `evaluate()` tokenizes the input by whitespace and processes each token:

### Token: `5`

1. `dictionary.find("5")` -- not found
2. `parse_number("5")` -- succeeds, returns 5
3. `push_data_stack(5)`:
   - Read DSP global (0x1540)
   - Decrement by 4 → 0x153C
   - Write `5` to memory at 0x153C
   - Update DSP to 0x153C

### Token: `3`

Same path. Stack is now:

```
0x1538: 3   ← DSP (top of stack)
0x153C: 5
```

### Token: `+`

1. `dictionary.find("+")` -- found, returns WordId and function table index
2. `execute_word(word_id)`:
   - Look up function reference in `table[word_id]`
   - Call it via wasmtime: `func.call(&mut store, &[], &mut [])`
3. The compiled WASM function for `+` executes:
   - Load value at DSP (3), increment DSP
   - Load value at DSP (5), increment DSP
   - Compute 5 + 3 = 8
   - Decrement DSP, store 8

Stack is now: `0x153C: 8 ← DSP`

### Token: `.`

1. `dictionary.find(".")` -- found (host function primitive)
2. `execute_word(word_id)` -- calls the Rust `dot` closure:
   - Reads value at DSP (8), increments DSP
   - Formats as `"8 "` and appends to the output buffer
3. Back in the REPL, output buffer is printed: `8  ok`

## What Happens When You Define a Word

### `: square dup * ;`

#### Token: `:`

The colon handler in `interpret_token()`:

1. Reads the next token from input: `"square"`
2. `dictionary.create("square", false)` -- creates a new HIDDEN entry, returns `WordId(N)` where N is the next function table index
3. Clears `compiling_ir` (the IR accumulator)
4. Sets `state = -1` (compile mode)
5. Prompt changes to `]`

#### Token: `dup` (in compile mode)

1. `dictionary.find("dup")` -- found, not IMMEDIATE
2. Since we are compiling and the word is not immediate, append `IrOp::Call(dup_word_id)` to `compiling_ir`

#### Token: `*` (in compile mode)

1. `dictionary.find("*")` -- found, not IMMEDIATE
2. Append `IrOp::Call(mul_word_id)` to `compiling_ir`

#### Token: `;`

The semicolon handler triggers `finish_colon_def()`:

**Step 1 -- Take the IR:**

```rust
compiling_ir = [IrOp::Call(WordId(0)),   // DUP
                IrOp::Call(WordId(6))]   // *
```

(Actual WordId values depend on registration order.)

**Step 2 -- Codegen:**

`compile_word("square", &ir, &config)` in `codegen.rs` generates a complete WASM module:

```wasm
;; Type section
(type $void (func))
(type $i32  (func (param i32)))

;; Import section -- shared resources
(import "env" "emit"   (func $emit (type $i32)))
(import "env" "memory" (memory 16))
(import "env" "dsp"    (global $dsp (mut i32)))
(import "env" "rsp"    (global $rsp (mut i32)))
(import "env" "table"  (table 256 funcref))

;; Function section
(func $fn (type $void)
  ;; Call DUP via function table
  (i32.const 0)                          ;; DUP's table index
  (call_indirect (type $void) (table 0))
  ;; Call * via function table
  (i32.const 6)                          ;; MUL's table index
  (call_indirect (type $void) (table 0))
)

;; Export section
(export "fn" (func $fn))

;; Element section -- install in table
(elem (i32.const N) func $fn)            ;; N = square's WordId
```

This module is generated as raw WASM bytecode using `wasm-encoder`, not as text. The pseudocode above is for illustration.

**Step 3 -- Instantiate and install:**

```
Module::new(&engine, &wasm_bytes)        Parse WASM
Instance::new(&store, &module, &[        Link to shared resources:
    emit_func,                             - emit host function
    memory,                                - shared linear memory
    dsp,                                   - data stack pointer global
    rsp,                                   - return stack pointer global
    table,                                 - shared function table
])
table.set(N, exported_fn)                Install in function table
```

**Step 4 -- Reveal:**

`dictionary.reveal()` removes the HIDDEN flag from "square". The word is now visible to `FIND` and can be called.

**Step 5 -- Return to interpret mode:**

`state = 0`, prompt returns to `>`.

### What happens when you then type `7 square .`

- `7` -- pushed to data stack (same as before)
- `square` -- `dictionary.find("SQUARE")` returns WordId(N), `execute_word(N)` calls the compiled function, which in turn calls DUP and * via `call_indirect`, leaving 49 on the stack
- `.` -- pops 49, prints `49`

Output: `49  ok`

## The Compilation Pipeline

```
Forth source
     │
     ▼
Outer Interpreter (outer.rs)
  Tokenizes by whitespace, looks up each token in dictionary.
  In interpret mode: executes words, pushes numbers.
  In compile mode: builds Vec<IrOp>.
     │
     ▼
IR (ir.rs)
  ~50 operation types: PushI32, Dup, Add, Call(WordId),
  If { then_body, else_body }, DoLoop { body }, Exit, ...
  Control flow is nested: If/DoLoop/BeginUntil contain Vec<IrOp> bodies.
     │
     ▼
Codegen (codegen.rs)
  compile_word() walks the IR and emits WASM instructions via wasm-encoder.
  Each word becomes a standalone WASM module that imports the shared resources.
  The function is exported as "fn" and placed in the table via the element section.
     │
     ▼
Wasmtime
  Module::new() parses and validates the WASM.
  Instance::new() links imports and instantiates.
  The exported function is extracted and stored in the function table.
     │
     ▼
Function Table
  Every word has a unique index. Calling a word = call_indirect with that index.
  All modules share the same table, memory, and globals.
```

## How Words Call Each Other

Every word -- primitive or user-defined -- gets a unique `WordId`, which is its index in the shared function table.

When the compiler encounters a word reference during compilation, it emits:

```wasm
(i32.const <word_id>)                     ;; push the function table index
(call_indirect (type $void) (table 0))    ;; indirect call through the table
```

At runtime, wasmtime resolves the table entry and calls the target function. Because all functions share the same memory, globals, and table, state passes between words through the data stack in linear memory. There are no function parameters or return values at the WASM level -- everything goes through the stack.

This is subroutine threading: each word is a subroutine, and calling a word is an indirect function call.

## Dictionary Structure

The dictionary is a linked list stored in a `Vec<u8>` buffer on the Rust side (in the `Dictionary` struct). Each entry has this layout:

```
Offset  Size    Contents
------  ----    --------
+0      4       Link: address of previous dictionary entry (0 = end of list)
+4      1       Flags: IMMEDIATE (0x80) | HIDDEN (0x40) | name_length (0-31)
+5      N       Name: N bytes, stored in uppercase
+5+N    0-3     Padding to 4-byte alignment
        4       Code field: function table index (WordId)
        ...     Parameter field: used by VARIABLE, CONSTANT, DOES>, etc.
```

**FIND** walks the list backward from `LATEST` (the most recent entry). For each entry:

1. Skip if HIDDEN flag is set
2. Compare name length, then bytes (case-insensitive)
3. If match: return the entry address, WordId (from code field), and IMMEDIATE flag
4. Otherwise follow the link to the previous entry

This means later definitions shadow earlier ones, and words being compiled (HIDDEN) are invisible to FIND until `;` reveals them.

## Primitives: Two Kinds

### IR Primitives

Most built-in words are defined as IR sequences and compiled to WASM like any user-defined word:

```rust
self.register_primitive("DUP",  false, vec![IrOp::Dup])?;
self.register_primitive("+",    false, vec![IrOp::Add])?;
self.register_primitive("OVER", false, vec![IrOp::Over])?;
self.register_primitive("ABS",  false, vec![IrOp::Dup, IrOp::ZeroLt,
    IrOp::If { then_body: vec![IrOp::Negate], else_body: None }])?;
```

The codegen translates each `IrOp` to inline WASM instructions. For example, `IrOp::Dup` becomes:

```wasm
;; peek: read top-of-stack without popping
(global.get $dsp)
(i32.load)              ;; TOS value now on WASM operand stack

;; push_via_local: store value back to a new stack slot
(local.set 0)           ;; save to scratch local
(global.get $dsp)       ;; dsp -= 4 (grow stack)
(i32.const 4)
(i32.sub)
(global.set $dsp)
(global.get $dsp)       ;; mem[dsp] = saved value
(local.get 0)
(i32.store)
```

### Host Primitives

Words that need access to Rust-side state (output buffer, dictionary, user variables) are implemented as Rust closures and placed directly in the function table:

```rust
let func = Func::new(&mut store, void_type, move |mut caller, _, _| {
    // Read/write memory and globals via wasmtime's Caller API
    Ok(())
});
self.register_host_primitive(".", false, func)?;
```

Examples: `.` (print number), `HERE` (return user data pointer), `WORDS` (list dictionary), `TYPE` (print string from memory).

Note: `EMIT` is an IR primitive -- it compiles to WASM code that calls the imported `emit` host function. This is different from host primitives like `.`, which are Rust closures placed directly in the function table.

## What WAFER Does NOT Create on Disk

WAFER generates all WASM modules in memory. No `.wasm` files are written to disk. No caches, no configuration files, no persistent state. Every time you start WAFER, it rebuilds everything from scratch.

The `--consolidate` CLI flag is reserved for a planned feature: compiling all words into a single optimized WASM module for ahead-of-time deployment. This is not yet implemented.

## Running the Tests

```bash
cargo test --workspace              # All unit tests (~220)
cargo test -p wafer-core --test compliance   # Forth 2012 compliance suite
cargo run -p wafer -- file.fth      # Execute a Forth source file
echo '5 3 + .' | cargo run -p wafer # Pipe input (non-interactive)
```

Test helpers in `outer.rs` for writing Rust-side tests:

```rust
assert_eq!(eval_output("5 3 + ."), "8 ");
assert_eq!(eval_stack("1 2 3"), vec![1, 2, 3]);
```
