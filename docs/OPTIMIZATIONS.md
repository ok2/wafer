# Optimizations

WAFER's compilation pipeline has a dedicated optimization stage between IR construction and WASM codegen:

```
Forth Source -> Outer Interpreter -> Vec<IrOp> -> [Optimizer] -> WASM Codegen -> wasmtime
```

The optimizer (`crates/core/src/optimizer.rs`) transforms `Vec<IrOp> -> Vec<IrOp>` through composable passes. A separate consolidation step (`crates/core/src/consolidate.rs`) can merge all JIT-compiled words into a single WASM module for cross-word optimization.

This document describes every optimization that makes sense for WAFER, why it matters, and whether it exists yet.

## Status Summary

| #  | Optimization             | Level        | Status      | Impact  |
| -- | ------------------------ | ------------ | ----------- | ------- |
| 1  | Stack-to-Local Promotion | Codegen      | Phase 1     | Highest |
| 2  | Peephole Optimization    | IR pass      | Done        | High    |
| 3  | Constant Folding         | IR pass      | Done        | High    |
| 4  | Inlining                 | IR pass      | Done        | High    |
| 5  | Strength Reduction       | IR pass      | Done        | Medium  |
| 6  | Dead Code Elimination    | IR pass      | Done        | Medium  |
| 7  | Tail Call Optimization   | IR + Codegen | Done        | Medium  |
| 8  | Consolidation            | Architecture | Done        | High    |
| 9  | Compound IR Operations   | IR + Codegen | Done        | Medium  |
| 10 | Codegen Improvements     | Codegen      | Done        | Medium  |
| 11 | wasmtime Configuration   | Runtime      | Done        | Low     |
| 12 | Dictionary Hash Index    | Runtime      | Done        | Low     |
| 13 | Startup Batching         | Architecture | Not started | Low     |
| 14 | Float / Double-Cell      | Codegen      | Not started | Future  |

## 1. Stack-to-Local Promotion

**Status: Phase 1 done.** Straight-line words (no control flow, calls, or I/O) use WASM locals instead of memory stack. Stack manipulation ops (Swap, Rot, Nip, Tuck, Dup, Drop) emit zero WASM instructions. Switchable via `WaferConfig::codegen.stack_to_local_promotion`.

### The Problem

WAFER simulates the Forth data stack in WASM linear memory. Every push and pop goes through a global stack pointer (`dsp`) and a memory load/store. A simple `DUP *` (square the top of stack) compiles to roughly 30 WASM instructions:

```wasm
;; DUP: peek top, push copy
global.get $dsp
i32.load              ;; peek
local.set 0
global.get $dsp
i32.const 4
i32.sub
global.set $dsp       ;; dsp_dec
global.get $dsp
local.get 0
i32.store             ;; push copy

;; MUL: pop two, multiply, push result
global.get $dsp
i32.load
local.set 0           ;; pop first
global.get $dsp
i32.const 4
i32.add
global.set $dsp
global.get $dsp
i32.load
local.set 1           ;; pop second
global.get $dsp
i32.const 4
i32.add
global.set $dsp
local.get 1
local.get 0
i32.mul               ;; the actual multiply
local.set 2
global.get $dsp
i32.const 4
i32.sub
global.set $dsp
global.get $dsp
local.get 2
i32.store             ;; push result
```

With stack-to-local promotion, the same `DUP *` becomes:

```wasm
local.get 0           ;; dup: read value
local.get 0           ;; dup: second copy
i32.mul               ;; multiply
local.set 0           ;; store result
```

That is a **~7x reduction** in instruction count.

### How It Works

When the compiler can statically determine the types and lifetimes of values on the stack, it maps them to WASM locals instead of memory. The `StackType` enum and `StackEffect` struct in `types.rs` already define the type system. What is missing:

1. **Stack-effect inference**: walk the IR and compute the type/local assignment for each stack slot at each point
2. **Dual-mode codegen**: emit local-based code when types are known, fall back to memory-based code at type boundaries (calls to unknown words, EXECUTE, etc.)
3. **Spill/reload at boundaries**: when calling another word, flush locals back to the memory stack (the callee expects a memory-based stack), then reload after return

### Where It Lives

- Type definitions: `crates/core/src/types.rs` (exists)
- Inference pass: new code in `optimizer.rs` or a dedicated `promote.rs`
- Codegen integration: `crates/core/src/codegen.rs` `emit_op()` needs a second code path

## 2. Peephole Optimization

**Status: Done.** Implemented in `optimizer.rs::peephole()`. Runs as fixpoint loop.

A peephole optimizer scans adjacent IR operations and replaces recognized patterns with cheaper equivalents. This is the lowest-effort, highest-return IR pass because Forth's postfix style generates many redundant sequences.

### Patterns

| Pattern              | Replacement                        | Savings                    |
| -------------------- | ---------------------------------- | -------------------------- |
| `PushI32(n), Drop`   | _(remove both)_                    | 1 push + 1 pop             |
| `Dup, Drop`          | _(remove both)_                    | 1 peek+push + 1 pop        |
| `Swap, Swap`         | _(remove both)_                    | 2x(2 pops + 2 pushes)      |
| `Swap, Drop`         | `Nip`                              | 1 pop                      |
| `Over, Over`         | `TwoDup` (new)                     | 1 peek+push                |
| `Drop, Drop`         | `TwoDrop` (new)                    | 1 dsp adjustment           |
| `PushI32(0), Add`    | _(remove both)_                    | 1 push + 1 pop + add       |
| `PushI32(0), Or`     | _(remove both)_                    | same                       |
| `PushI32(-1), And`   | _(remove both)_                    | same                       |
| `PushI32(1), Add`    | `Inc` (new or codegen special)     | avoids pushing constant    |
| `PushI32(1), Sub`    | `Dec` (new or codegen special)     | same                       |
| `ZeroEq, ZeroEq`     | _(remove both)_ for boolean inputs | 2 comparisons              |
| `DivMod, Swap, Drop` | `Div` (new or codegen special)     | avoids computing remainder |
| `DivMod, Drop`       | `Mod` (new or codegen special)     | avoids computing quotient  |

### Implementation

A single function `fn peephole(ops: Vec<IrOp>) -> Vec<IrOp>` that makes repeated passes until no more patterns match. Recurse into control flow bodies (If/DoLoop/Begin*).

## 3. Constant Folding

**Status: Done.** Implemented in `optimizer.rs::constant_fold()`. Folds binary and unary ops on known constants.

When both operands of an operation are compile-time constants, compute the result at compile time.

### Examples

```
; Before:   5 3 +       ->  IR: PushI32(5), PushI32(3), Add
; After:                 ->  IR: PushI32(8)

; Before:   0 0=         ->  IR: PushI32(0), ZeroEq
; After:                 ->  IR: PushI32(-1)

; Before:   7 NEGATE     ->  IR: PushI32(7), Negate
; After:                 ->  IR: PushI32(-7)

; Before:   4 3 < IF ... ->  IR: PushI32(4), PushI32(3), Lt, If{...}
; After:                 ->  IR: PushI32(0), If{...}
;   (then DCE removes the dead branch entirely)
```

Constant folding composes with inlining: after inlining a word, new folding opportunities appear. Run folding after every inlining pass.

### Implementation

A function `fn constant_fold(ops: Vec<IrOp>) -> Vec<IrOp>` that simulates a compile-time stack of known constants and replaces foldable sequences. Must handle all arithmetic, comparison, logic, and unary operations in `IrOp`.

## 4. Inlining

**Status: Done.** Implemented in `optimizer.rs::inline()`. Inlines word bodies <= 8 ops, non-recursive. TailCall converted back to Call when inlining. IR bodies stored in `ForthVM::ir_bodies`.

Replace `Call(WordId)` with the callee's IR body, avoiding the `call_indirect` overhead and enabling further optimization of the combined code.

### Why It Matters

Every call in WAFER is `call_indirect` through a function table. This is slower than a direct `call` and prevents the WASM engine from optimizing across call boundaries. Inlining eliminates the call entirely and exposes the callee's operations to peephole, constant folding, and stack-to-local promotion.

### Example

```forth
: SQUARE  DUP * ;
: MAIN    5 SQUARE 3 SQUARE + ;
```

Before inlining, MAIN's IR:

```
PushI32(5), Call(SQUARE), PushI32(3), Call(SQUARE), Add
```

After inlining SQUARE:

```
PushI32(5), Dup, Mul, PushI32(3), Dup, Mul, Add
```

After constant folding:

```
PushI32(25), PushI32(9), Add
```

After more folding:

```
PushI32(34)
```

### Requirements

- Store each word's IR body in the dictionary (currently discarded after compilation)
- Inline policy: inline when body size is below a threshold (e.g., 8 IR ops)
- Do not inline recursive words (detect cycles)
- Do not inline words with side effects that depend on call context (rare)
- Re-run peephole and constant folding after inlining

## 5. Strength Reduction

**Status: Done.** Implemented in `optimizer.rs::strength_reduce()`. Power-of-2 multiply to shift, zero comparisons to ZeroEq/ZeroLt.

Replace expensive operations with cheaper equivalents when one operand is a known constant.

### Patterns

| Pattern                | Replacement                     | Why                            |
| ---------------------- | ------------------------------- | ------------------------------ |
| `PushI32(2^n), Mul`    | `PushI32(n), Lshift`            | shift is 1 cycle vs multiply   |
| `PushI32(2^n), DivMod` | `PushI32(n), Rshift` (unsigned) | shift vs divide                |
| `PushI32(1), Lshift`   | `Dup, Add`                      | add is often faster than shift |
| `PushI32(0), Gt`       | `ZeroGt` (if added)             | avoids pushing constant        |
| `PushI32(0), Eq`       | `ZeroEq`                        | already exists as IR op        |
| `PushI32(0), Lt`       | `ZeroLt`                        | already exists as IR op        |

The most common case is `CELLS` which is defined as `PushI32(4), Mul`. Strength reduction turns this into `PushI32(2), Lshift`.

## 6. Dead Code Elimination

**Status: Done.** Implemented in `optimizer.rs::dce()`. Truncates after Exit, eliminates constant-conditional branches.

Remove IR operations that can never execute or whose results are never used.

### Cases

1. **Unreachable code**: anything after `Exit` in a linear sequence
2. **Constant conditionals**: `PushI32(0), If { then, else }` -- keep only `else`; `PushI32(non-zero), If { then, else }` -- keep only `then`
3. **Push-then-drop**: `PushI32(n), Drop` -- remove both (also caught by peephole, but DCE handles it for non-adjacent cases when intervening ops are also dead)
4. **Empty control structures**: `If { [], None }` -- remove the entire If

DCE should run after constant folding, since folding can create new constant conditionals.

## 7. Tail Call Optimization

**Status: Done.** `optimizer.rs::tail_call_detect()` converts the last `Call` to `TailCall` when return stack is balanced. Codegen emits `call_indirect + return`.

### What Exists

The codegen for `TailCall` emits:

```wasm
i32.const <word_id>
call_indirect (type $void) (table 0)
return
```

This is semantically a tail call -- the current frame returns immediately after the callee. True WASM tail calls (`return_call_indirect`) are a WASM proposal not yet standard, so this is the best available approximation. It does not eliminate the call frame, but it does skip any cleanup code after the call site.

### What Is Missing

The compiler (`outer.rs`) needs to detect tail position: when the last operation before `;` (or before `Exit`) is a `Call`, convert it to `TailCall`. For `RECURSE` in tail position, this enables tail-recursive patterns like:

```forth
: GCD  ( a b -- gcd )  ?DUP IF TUCK MOD RECURSE THEN ;
```

Detection rule: if the last IR op in a word body (or in a branch of an `If`) is `Call(id)`, and there are no pending return-stack items (`>R` without matching `R>`), replace with `TailCall(id)`.

## 8. Consolidation

**Status: Done.** `CONSOLIDATE` word recompiles all IR-based words into a single WASM module with direct `call` instructions. Implemented in `codegen.rs::compile_consolidated_module()` and `outer.rs::consolidate()`.

### The Idea

After interactive development, `CONSOLIDATE` recompiles all defined words into a **single WASM module**. This enables:

1. **Direct calls**: `call_indirect` through the function table becomes `call` to a known function index. Direct calls are faster and allow the WASM engine's optimizer (Cranelift) to see through them.
2. **Cross-word inlining by Cranelift**: with all functions in one module, Cranelift can inline small functions during its own optimization passes.
3. **Single instantiation**: one `Module::new()` + `Instance::new()` instead of N separate ones.
4. **Shared locals optimization**: Cranelift can allocate registers across the entire module.

### Design

- Collect all word IR bodies (requires storing them -- see Inlining prerequisite)
- Generate one WASM module with N internal functions
- Each function corresponds to a word, using direct `call` to siblings
- Re-populate the function table with the new module's exports
- The `compile_core_module()` stub in `codegen.rs` is the entry point

### Two Modes

| Mode          | When                    | Properties                                       |
| ------------- | ----------------------- | ------------------------------------------------ |
| JIT (current) | Interactive development | Per-word modules, `call_indirect`, fast redefine |
| Consolidated  | After `CONSOLIDATE`     | Single module, direct `call`, no redefine        |

## 9. Compound IR Operations

**Status: Done.** `TwoDup` and `TwoDrop` IrOp variants with optimized codegen. Peephole converts `Over, Over -> TwoDup` and `Drop, Drop -> TwoDrop`.

Some common multi-op sequences have more efficient WASM implementations than emitting each op individually.

### Candidates

**`2DUP` (currently `Over, Over`)**

Current codegen: two separate Over implementations, each doing a peek + push. The compound version reads `dsp` once and copies two cells:

```wasm
;; compound 2DUP
global.get $dsp
i32.load offset=0       ;; b (top)
local.set 0
global.get $dsp
i32.load offset=4       ;; a (second)
local.set 1
global.get $dsp
i32.const 8
i32.sub
global.set $dsp         ;; one dsp adjustment instead of two
global.get $dsp
local.get 1
i32.store offset=0      ;; push a
global.get $dsp
local.get 0
i32.store offset=4      ;; push b (adjusted offset)
```

**`2DROP` (currently `Drop, Drop`)**

Instead of two separate `dsp += 4`, emit one `dsp += 8`.

**`NipNip`** -- drop two items below top. Common after double-cell operations.

**`IncFetch` / `FetchInc`** -- `1+ @` or `@ 1+`, common loop patterns.

These can be added as new `IrOp` variants recognized by peephole and emitted by codegen with specialized WASM sequences.

## 10. Codegen Improvements

**Status: Done (DSP caching).** `$dsp` cached in WASM local 0, written back before calls and at function exit. Commutative optimization and loop index in local are future work.

These are improvements within `codegen.rs` `emit_op()` that do not require new IR operations.

### Global Caching

Cache `dsp` in a WASM local at function entry, write it back at function exit and before/after calls. This eliminates repeated `global.get $dsp` / `global.set $dsp` pairs within a function body:

```wasm
;; function entry
global.get $dsp
local.set $cached_dsp

;; ... use local.get/set $cached_dsp throughout ...

;; before call
local.get $cached_dsp
global.set $dsp
call_indirect ...
global.get $dsp
local.set $cached_dsp

;; function exit
local.get $cached_dsp
global.set $dsp
```

Globals in WASM are effectively memory accesses. Locals are register-allocated by Cranelift. This alone could cut 30-40% of global access instructions.

### Commutative Operand Stack Usage

For commutative operations (Add, Mul, And, Or, Xor, Eq, NotEq), the current codegen pops both operands into locals, then pushes them back onto the WASM operand stack for the operation. Instead, leave them on the operand stack:

```wasm
;; current: pop a to local, pop b to local, push both, operate
;; better for commutative ops:
global.get $dsp
i32.load              ;; a on wasm stack
global.get $dsp
i32.const 4
i32.add
i32.load              ;; b on wasm stack
i32.add               ;; result on wasm stack
;; ... store result
```

### Loop Index in Local

`DO...LOOP` currently stores the loop index and limit on the return stack (in memory). Keep them in WASM locals for the duration of the loop body. This makes `I` (read loop index) a simple `local.get` instead of a memory load from the return stack.

## 11. wasmtime Configuration

**Status: Done.** NaN canonicalization disabled, module caching enabled via `cache_config_load_default()`.

### Available Knobs

| Setting                                  | Current         | Recommended | Effect                                 |
| ---------------------------------------- | --------------- | ----------- | -------------------------------------- |
| `Config::cranelift_opt_level`            | Speed (default) | Speed       | Already optimal for JIT                |
| `Config::cranelift_nan_canonicalization` | true            | false       | Skip NaN fixup (no floats yet)         |
| `Config::parallel_compilation`           | true            | true        | Already optimal                        |
| Module caching                           | none            | file-based  | Cache compiled modules across sessions |
| Epoch interruption                       | none            | enable      | Protect against infinite loops         |

Module caching is the most impactful: `wasmtime::Config::cache_config_load_default()` enables disk-based caching of compiled WASM, so restarting WAFER with the same definitions does not re-invoke Cranelift.

## 12. Dictionary Hash Index

**Status: Done.** `HashMap<String, (u32, u32, bool)>` in Dictionary struct. `find()` uses O(1) hash lookup with linked-list fallback. Updated on `reveal()` and `toggle_immediate()`.

The dictionary lookup (`dictionary.rs` `find()`) walks a linked list from the most recent entry backward, comparing names character by character. After registering 80+ primitives plus user words, every lookup during compilation scans the full list.

### Solution

Maintain a `HashMap<String, (u32, WordId, bool)>` alongside the linked list. Update it on `create()` and `reveal()`. Lookup becomes O(1) average case. Keep the linked list for Forth-level traversal (`WORDS`, `FIND` at runtime).

This affects **compile time** (word lookup during parsing), not runtime (compiled code uses function table indices directly).

## 13. Startup Batching

**Status: Not started.** `compile_core_module()` stub exists in `codegen.rs`.

Currently, each of the 80+ primitives registered at boot creates a separate WASM module: `wasm-encoder` builds it, `wasmparser` validates it, Cranelift compiles it, and wasmtime instantiates it. This happens 80+ times sequentially.

### Solution

Batch all IR-based primitives into a single WASM module with multiple exported functions. One `Module::new()` + one `Instance::new()` replaces 80+ pairs. This is a subset of what Consolidation (section 8) achieves, but scoped to primitives only and simpler to implement.

## 14. Float and Double-Cell Stack

**Status: Not started.** `PushI64` and `PushF64` exist as IR ops but are stubs in codegen. Float stack operations are currently all host functions.

The float stack lives in its own memory region (0x2540--0x2D40). Float operations will have the same memory-based overhead as integer operations, but worse: `f64` values are 8 bytes, doubling the memory traffic per push/pop. Stack-to-local promotion (section 1) is even more impactful for floats because WASM has native `f64` locals and operand stack support.

## Suggested Implementation Order

Ordered by effort-to-impact ratio (cheapest wins first):

| Priority | Optimization                                       | Effort  | Unlocks                                         |
| -------- | -------------------------------------------------- | ------- | ----------------------------------------------- |
| 1        | Peephole optimization                              | Low     | Immediate code size reduction                   |
| 2        | Constant folding                                   | Low     | Composes with peephole                          |
| 3        | Tail call detection                                | Low     | Recursive word optimization                     |
| 4        | Dictionary hash index                              | Low     | Faster compilation                              |
| 5        | wasmtime config tuning                             | Trivial | Caching, interruption                           |
| 6        | Codegen improvements (global caching, loop locals) | Medium  | ~30% fewer instructions                         |
| 7        | Inlining                                           | Medium  | Unlocks cross-word folding and peephole         |
| 8        | Strength reduction                                 | Low     | Best after inlining exists                      |
| 9        | Dead code elimination                              | Low     | Best after constant folding exists              |
| 10       | Compound IR operations                             | Medium  | Cumulative gains                                |
| 11       | Stack-to-local promotion                           | High    | The single biggest speedup (~7x for arithmetic) |
| 12       | Startup batching                                   | Medium  | Faster boot                                     |
| 13       | Consolidation                                      | High    | Direct calls, cross-word optimization           |
| 14       | Float/double-cell                                  | Medium  | Depends on stack-to-local                       |

Stack-to-local promotion has the highest impact but also the highest implementation cost. The passes before it (peephole, folding, inlining) are simpler and their benefits multiply when stack-to-local promotion is eventually added. Consolidation is last because it requires storing IR bodies and restructuring the module generation -- it benefits most from having all other passes working first.
