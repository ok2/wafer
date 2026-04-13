# WAFER Codebase Reading Order

Optimal sequence for learning the entire system. Each step builds on the previous.

---

## Phase 1: Mental Model Foundation

### 1. `crates/core/src/memory.rs` (148 lines)

**Read first.** Defines the physical memory map — every address, every region. You'll reference these constants everywhere else.

- Key insight: stacks grow DOWN, dictionary grows UP
- Memorize: DATA_STACK_TOP=0x1600, DICTIONARY_BASE=0x2D40
- System variables at offset 0: STATE, BASE, >IN, HERE, LATEST, SOURCE-ID, #TIB, HLD, LEAVE-FLAG
- Notice how regions are laid out to never overlap (verified by compile-time assertions)

### 2. `crates/core/src/ir.rs` (259 lines)

**The central data structure.** Every Forth word compiles to `Vec<IrOp>`. This is the language the optimizer speaks and the codegen consumes.

- ~70 variants across 10 categories
- Pay attention to control-flow variants: `If`, `DoLoop`, `BeginUntil`, `BeginWhileRepeat`, `BeginDoubleWhileRepeat` — they contain nested `Vec<IrOp>` bodies (tree structure, not flat)
- `Call(WordId)` and `TailCall(WordId)` — how words reference each other
- Float ops are separate from integer ops (separate stack)
- `IrWord` struct: name + body + is_immediate

### 3. `crates/core/src/error.rs` (84 lines)

**Quick read.** 15 error variants. Note `Throw(i32)` for the Exception word set and `Abort(String)` for ABORT".

### 4. `crates/core/src/config.rs` (61 lines)

**Quick read.** 7 optimization flags in two tiers: OptConfig (IR-level) and CodegenOpts (codegen-level). Default = all enabled.

---

## Phase 2: Data Structures

### 5. `crates/core/src/dictionary.rs` (906 lines)

**How words live in memory.** The dictionary is a linked list stored in a `Vec<u8>` that simulates WASM linear memory.

- Entry format: link(4) + flags(1) + name(N) + padding + code_field(4)
- Flags byte: IMMEDIATE=0x80, HIDDEN=0x40, LENGTH_MASK=0x1F
- `create()` writes the entry, starts HIDDEN; `reveal()` removes HIDDEN flag
- `find()`: fast path via HashMap index, fallback via linked-list walk
- Wordlist support: `current_wid`, `search_order`, `find_in_wid()`
- `DictionaryState` for MARKER save/restore
- Read every test — they document exact behavior

---

## Phase 3: The Pipeline

### 6. `crates/core/src/optimizer.rs` (1013 lines)

**IR transformations.** Read the `optimize()` function first to see the pass ordering, then each pass.

- `peephole()`: pattern-match adjacent ops. ~15 patterns. Runs to fixpoint. Study each match arm.
- `constant_fold()`: evaluate PushI32+PushI32+BinaryOp at compile time. Also unary and float.
- `strength_reduce()`: multiply by power-of-2 → shift. 0 compare → ZeroEq/ZeroLt.
- `dce()`: eliminate dead branches (constant condition), truncate after Exit.
- `inline()`: replace Call(id) with body if ≤8 ops, non-recursive, no Exit, no ForthLocals. `detailcall()` converts TailCall back to Call.
- `tail_call_detect()`: last Call → TailCall if return stack balanced. Recurses into If branches.
- Key: `apply_to_bodies()` — every pass recurses into control-flow nested bodies.

### 7. `crates/core/src/codegen.rs` (4205 lines) — **The Big One**

**IR → WASM translation.** Read in order:

1. **Constants** (lines 1-80): import indices, type indices, DSP/RSP/FSP globals, memory alignment
2. **Helper functions** (lines 80-210): `dsp_dec/inc`, `push_via_local`, `pop`, `peek`, `dsp_writeback/reload`, `rpush_via_local`, `rpop`
3. **Float helpers** (lines 225-330): `fsp_dec/inc`, `fpush_via_local`, `fpop`, `fpeek`, `emit_float_binary/unary/cmp`
4. **`emit_op()`** (line 344+): the giant match — each IrOp variant → WASM instructions. This is the heart.
5. **`compile_word()`**: builds the WASM module structure (imports, types, functions, element section)
6. **`compile_consolidated_module()`**: multi-function module for CONSOLIDATE/export
7. **Stack-to-local promotion**: analysis pass that replaces memory stack operations with WASM locals

Key patterns to understand:

- DSP cached in local 0: read from global at function entry, write back before calls and at exit
- Scratch locals at SCRATCH_BASE(1): used as temporaries for stack manipulation
- `EmitCtx`: carries f64 locals, Forth local base, loop local base, self_word_id for recursion
- DO/LOOP: index+limit in WASM locals when possible (fast path), fallback to return stack

---

## Phase 4: The Runtime Abstraction

### 8. `crates/core/src/runtime.rs` (152 lines)

**NEW: Read this before outer.rs.** Defines two traits:

- `Runtime` — abstraction over WASM execution backend (memory, globals, table, module instantiation, host function registration)
- `HostAccess` — memory/global ops available to host function callbacks
- `HostFn = Box<dyn Fn(&mut dyn HostAccess) -> Result<()>>` — runtime-agnostic host function type
- Key insight: ForthVM is now `ForthVM<R: Runtime>`, completely decoupled from wasmtime

### 8b. `crates/core/src/runtime_native.rs` (328 lines)

**NativeRuntime**: wasmtime implementation of Runtime trait.

- `CallerHostAccess` wraps wasmtime `Caller` to implement `HostAccess`
- `NativeRuntime` owns Engine, Store, Memory, Table, Globals
- `register_host_func`: creates a wasmtime `Func` that bridges `HostFn` → wasmtime callback
- Study how `instantiate_and_install` provides the 6 imports

### 9. `crates/core/src/outer.rs` — ForthVM struct (lines 1-240)

**Read the struct definition carefully.** ~35 fields. Group them mentally:

- Runtime: `rt: R` (generic over Runtime trait — no more direct wasmtime fields)
- Compilation state: state, compiling_name, compiling_ir, control_stack, compiling_word_id, compiling_locals
- Output: output (Arc<Mutex<String>>)
- Dictionary bridge: dictionary, user_here, here_cell, base_cell
- Word metadata: ir_bodies, host_word_names, word_pfa_map, does_definitions
- Shared state for host functions: pending_define, pending_actions, pending_does_patch, throw_code, word_lookup
- Configuration: config, batch_mode, deferred_ir
- Export support: toplevel_ir, recording_toplevel
- Advanced: marker_states, conditional_skip_depth, next_block_label, substitutions, search_order, next_wid

### 10. `crates/core/src/outer.rs` — new() and primitive registration

**How the VM boots.** Read:

- `new_with_config()`: creates `R::new()` runtime, then calls `register_primitives()` and loads boot.fth
- `register_primitive()`: creates dictionary entry → optimizes IR → compiles to WASM → `rt.instantiate_and_install()`
- `register_host_primitive()`: creates dictionary entry → `rt.register_host_func()` with HostFn closure
- `register_primitives()`: ~130 words registered in batch_mode, then `compile_batch()`
- Each host function: study 5-10 representative ones to understand the pattern

### 11. `crates/core/src/outer.rs` — Outer interpreter loop

**The main loop.** Read:

- `evaluate()`: sets up input buffer, calls `interpret_token()` in a loop
- `interpret_token()`: conditional compilation, `:` handling, `]` handling, dispatch to compile/interpret mode
- `interpret_token_immediate()`: string literals, dictionary lookup, execute found word, parse number
- `compile_token()`: POSTPONE, string literals, control-flow words (IF/ELSE/THEN/DO/LOOP/BEGIN/WHILE/REPEAT/AGAIN/UNTIL/CASE/OF/ENDOF/ENDCASE), dictionary lookup, compile Call(id), parse number → PushI32
- `finish_colon_def()`: optimize → codegen → install

### 12. `crates/core/src/outer.rs` — Control flow compilation

**Most complex part.** 13 `ControlEntry` variants. Understand:

- `ControlEntry::If { then_body }` → pushed when IF seen, then_body accumulates until ELSE or THEN
- `ControlEntry::Do { body }` → pushed by DO, body accumulates until LOOP/+LOOP
- `ControlEntry::Begin { body }` → pushed by BEGIN, resolved by UNTIL/AGAIN/WHILE
- `ControlEntry::BeginWhile { test, body }` → WHILE splits Begin into test + body
- `ControlEntry::Case/Of` → CASE/OF/ENDOF/ENDCASE pattern
- `ControlEntry::QDo` → ?DO (conditional entry)
- `ControlEntry::Ahead` → AHEAD (unconditional forward branch)
- CS-PICK and CS-ROLL: advanced control-flow manipulation for tools word set

---

## Phase 5: Self-Hosting

### 13. `crates/core/boot.fth` (307 lines)

**Forth replaces Rust.** 7 phases of definitions that replace host functions with compiled Forth.

- Phase 1: Stack/memory (DEPTH, PICK, 2OVER, FILL, MOVE, /STRING, -TRAILING)
- Phase 2: Double-cell arithmetic (D+, DNEGATE, D-, DABS, D0=, D0<, D=, D<, DU<)
- Phase 3: Mixed arithmetic (SM/REM, FM/MOD, */, _/MOD) — built on M_ and UM/MOD host primitives
- Phase 4: HERE, ALLOT, comma, C-comma, ALIGN — magic numbers for sysvar offsets
- Phase 5: I/O and pictured numeric output (TYPE, SPACES, <# HOLD # #S #> . U. .R U.R D. D.R)
- Phase 6: DEFER support (DEFER!, DEFER@)
- Phase 7: String operations, SOURCE, FALIGNED, etc.
- Key insight: why Forth not Rust? Self-hosting goal + compiled Forth with direct calls beats host function dispatch

---

## Phase 6: Production Features

### 14. `crates/core/src/consolidate.rs` (169 lines)

**Quick read.** Mostly tests. Real logic is in `codegen::compile_consolidated_module()` and `outer::ForthVM::consolidate()`. Understand the concept: merge all JIT modules into one, replacing call_indirect with direct call.

### 15. `crates/core/src/export.rs` (409 lines)

**wafer build pipeline.** Entry point resolution (--entry > MAIN > top-level), IR collection, memory snapshot, metadata embedding in custom section.

### 16. `crates/core/src/runner.rs` (402 lines)

**Standalone execution.** Creates the 6 imports from scratch, registers host function stubs for known words (., TYPE, SPACES, .S, M*, UM*, UM/MOD, DEPTH). Shows the minimal set needed to run exported modules.

### 17. `crates/cli/src/main.rs` (354 lines)

**CLI ties it together.** Three modes: REPL (rustyline), file evaluation, subcommands (build, run). Native executable trick: append AOT payload + "WAFEREXE" trailer to binary.

### 18. `crates/web/src/lib.rs` (56 lines)

**Browser entry point.** `WaferRepl` struct with `#[wasm_bindgen]`:

- `new()` → `ForthVM::<WebRuntime>::new()`
- `evaluate(input)` → returns output string
- `data_stack()`, `is_compiling()`, `reset()`

### 19. `crates/web/src/runtime_web.rs` (542 lines)

**WebRuntime**: browser implementation of Runtime trait.

- Uses `js_sys::WebAssembly` for module instantiation
- `WebHostAccess`: implements HostAccess via `js_sys` typed arrays
- Memory access through `Int32Array`/`Uint8Array` views on `WebAssembly.Memory.buffer`
- Closures kept alive via `_closures: Vec<JsValue>` to prevent GC

### 20. `crates/web/www/` (727 lines)

**Frontend**: app.js (terminal emulation, stack display), index.html, style.css.

---

## Phase 7: Testing

### 21. Unit tests (embedded in each source file)

Re-read each file's `#[cfg(test)] mod tests`. They document edge cases and expected behavior.

### 22. `crates/core/tests/compliance.rs`

Forth 2012 compliance infrastructure: boot_with_prerequisites, run_suite, 11 word set tests.

### 23. `crates/core/tests/comparison.rs`

Cross-engine benchmarks vs gforth. Performance validation.
