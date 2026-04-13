# WAFER Trace-the-Compilation Exercises

For each exercise, manually trace the Forth code through the full pipeline:
1. **Outer interpreter** — tokenization, dictionary lookup, compile/interpret dispatch
2. **IR generation** — what Vec<IrOp> is produced
3. **Optimization** — which passes fire, what changes
4. **Codegen** — WASM instructions emitted (conceptual)
5. **Runtime** — how it executes

Answers are below each exercise (scroll down or cover with paper).

---

## Exercise 1: Simple Arithmetic
```forth
: SQUARE  DUP * ;
```

<details>
<summary>Answer</summary>

1. `:` → enter compile mode, next token "SQUARE" = word name, dictionary.create("SQUARE")
2. `DUP` → find in dictionary → IR primitive (WordId N) → append `Call(dup_id)`
3. `*` → find → IR primitive → append `Call(mul_id)`
4. `;` → raw IR: `[Call(dup_id), Call(mul_id)]`
5. **Optimize:**
   - Inline: DUP body=[Dup] (1 op ≤ 8), * body=[Mul] (1 op ≤ 8) → `[Dup, Mul]`
   - Peephole: no patterns match Dup,Mul
   - Constant fold: nothing to fold
   - Tail call: Mul is not a Call → skip
   - **Final IR: `[Dup, Mul]`**
6. **Codegen:**
   - Dup: `local.get $dsp; i32.load; local.set $tmp; dsp_dec; local.get $dsp; local.get $tmp; i32.store`
   - Mul: `pop; pop; i32.mul; push_via_local`
7. **Runtime:** WASM module instantiated, function registered at table[word_id]

</details>

---

## Exercise 2: Constant Folding
```forth
: TEN  5 5 + ;
```

<details>
<summary>Answer</summary>

1. `:` → compile mode, name="TEN"
2. `5` → not in dictionary → parse as number → append `PushI32(5)`
3. `5` → append `PushI32(5)`
4. `+` → find → IR primitive → append `Call(add_id)`
5. `;` → raw IR: `[PushI32(5), PushI32(5), Call(add_id)]`
6. **Optimize:**
   - Inline: + body=[Add] → `[PushI32(5), PushI32(5), Add]`
   - Constant fold: PushI32(5), PushI32(5), Add → `PushI32(10)`
   - **Final IR: `[PushI32(10)]`**
7. **Codegen:** Just `push_const(f, 10)` → `dsp_dec; local.get $dsp; i32.const 10; i32.store`

</details>

---

## Exercise 3: Peephole Elimination
```forth
: NOOP  DUP DROP ;
```

<details>
<summary>Answer</summary>

1. Raw IR after inlining: `[Dup, Drop]`
2. **Optimize:**
   - Peephole: Dup, Drop → removed (both eliminated)
   - **Final IR: `[]` (empty)**
3. **Codegen:** Empty function body — just DSP writeback at entry/exit

</details>

---

## Exercise 4: Strength Reduction
```forth
: DOUBLE  8 * ;
```

<details>
<summary>Answer</summary>

1. Raw IR after inlining: `[PushI32(8), Mul]`
2. **Optimize:**
   - Strength reduce: PushI32(8) is 2^3, so → `[PushI32(3), Lshift]`
   - 8 * x becomes x << 3
   - **Final IR: `[PushI32(3), Lshift]`**
3. **Codegen:** push_const(3), then pop two, i32.shl, push result

</details>

---

## Exercise 5: Tail Call Detection
```forth
: FOO  1 + BAR ;
```
(Assume BAR is a previously defined word)

<details>
<summary>Answer</summary>

1. Raw IR: `[PushI32(1), Call(add_id), Call(bar_id)]`
2. **Optimize:**
   - Inline + (1 op): `[PushI32(1), Add, Call(bar_id)]`
   - Tail call: last op is Call(bar_id), return stack balanced (no >R or R>) → `TailCall(bar_id)`
   - **Final IR: `[PushI32(1), Add, TailCall(bar_id)]`**
3. **Codegen:** TailCall emits `dsp_writeback; call_indirect bar_id; return`

</details>

---

## Exercise 6: Control Flow — IF/THEN
```forth
: ABS  DUP 0< IF NEGATE THEN ;
```

<details>
<summary>Answer</summary>

1. `DUP` → Call(dup_id), `0<` → Call(zerolt_id)
2. `IF` → push ControlEntry::If { then_body: [] }, start collecting
3. `NEGATE` → Call(negate_id) appended to then_body
4. `THEN` → pop ControlEntry::If, emit `If { then_body: [Call(negate_id)], else_body: None }`
5. Raw IR: `[Call(dup_id), Call(zerolt_id), If { then: [Call(negate_id)], else: None }]`
6. **Optimize:**
   - Inline all (each is 1 op): `[Dup, ZeroLt, If { then: [Negate], else: None }]`
   - Note: optimizer recurses into If bodies via apply_to_bodies
   - **Final IR: `[Dup, ZeroLt, If { then: [Negate], else: None }]`**
7. **Codegen:** pop flag → `if (block) ... end` WASM structure

</details>

---

## Exercise 7: DO LOOP
```forth
: STARS  0 DO 42 EMIT LOOP ;
```

<details>
<summary>Answer</summary>

1. `0` → PushI32(0)
2. `DO` → push ControlEntry::Do { body: [] }
3. `42` → PushI32(42) into body
4. `EMIT` → Call(emit_id) into body
5. `LOOP` → pop Do, emit `DoLoop { body: [PushI32(42), Call(emit_id)], is_plus_loop: false }`
6. Note: the 0 and the limit (already on stack from caller) are consumed by DoLoop
7. **Optimize:**
   - Inline EMIT (1 op): `DoLoop { body: [PushI32(42), Emit], is_plus_loop: false }`
   - **Final IR:** `[PushI32(0), DoLoop { body: [PushI32(42), Emit], is_plus_loop: false }]`
8. **Codegen:** Loop index+limit in WASM locals. WASM `loop { body; index++; br_if index<limit }`

</details>

---

## Exercise 8: BEGIN UNTIL
```forth
: COUNTDOWN  BEGIN DUP . 1 - DUP 0= UNTIL DROP ;
```

<details>
<summary>Answer</summary>

1. `BEGIN` → push ControlEntry::Begin { body: [] }
2. `DUP .` → Call(dup_id), Call(dot_id) into body
3. `1 -` → PushI32(1), Call(sub_id) into body
4. `DUP 0=` → Call(dup_id), Call(zeroeq_id) into body
5. `UNTIL` → pop Begin, emit `BeginUntil { body: [Call(dup), Call(dot), PushI32(1), Call(sub), Call(dup), Call(zeroeq)] }`
6. **Optimize:** Inline small primitives. `1 -` stays as `PushI32(1), Sub` (no further fold since operand unknown). `.` is a host function → NOT inlined.
7. `DROP` after loop.

</details>

---

## Exercise 9: Dead Code Elimination
```forth
: ALWAYS-TRUE  TRUE IF 42 ELSE 99 THEN ;
```

<details>
<summary>Answer</summary>

1. Raw IR after inlining TRUE (body=[PushI32(-1)]):
   `[PushI32(-1), If { then: [PushI32(42)], else: Some([PushI32(99)]) }]`
2. **DCE:** PushI32(-1) is nonzero → emit then_body only
   → `[PushI32(42)]`
3. Entire IF/ELSE/THEN eliminated. Just pushes 42.

</details>

---

## Exercise 10: Swap Peephole Patterns
```forth
: TEST  SWAP SWAP DROP DROP ;
```

<details>
<summary>Answer</summary>

1. After inlining: `[Swap, Swap, Drop, Drop]`
2. **Peephole pass 1:**
   - Swap, Swap → removed → `[Drop, Drop]`
   - Drop, Drop → TwoDrop → `[TwoDrop]`
3. **Final IR: `[TwoDrop]`**

</details>

---

## Exercise 11: Nested Control Flow
```forth
: CLASSIFY  DUP 0< IF DROP -1 ELSE 0> IF 1 ELSE 0 THEN THEN ;
```

<details>
<summary>Answer</summary>

1. IR structure (after inlining):
```
[Dup, ZeroLt, If {
  then: [Drop, PushI32(-1)],
  else: Some([Gt(implicit 0>), If {
    then: [PushI32(1)],
    else: Some([PushI32(0)])
  }])
}]
```
2. Optimizer recurses into both If bodies. No constant conditions → no DCE.
3. Codegen: nested WASM `if/else/end` blocks.

</details>

---

## Exercise 12: DOES> Defining Word
```forth
: CONSTANT  CREATE , DOES> @ ;
5 CONSTANT FIVE
FIVE .
```

<details>
<summary>Answer</summary>

1. `: CONSTANT` enters compile mode
2. `CREATE` — flagged as saw_create_in_def=true
3. `,` — compiled normally
4. `DOES>` — splits definition:
   - create_ir = everything before DOES> (the `,` call)
   - does_action = everything after DOES> (the `@` call) → compiled as separate word
   - Stores DoesDefinition { create_ir, does_action_id, has_create: true }
5. `5 CONSTANT FIVE`:
   - CONSTANT executes its defining behavior
   - CREATE makes dictionary entry "FIVE"
   - `,` stores 5 at FIVE's parameter field
   - DOES> patches FIVE to execute the does_action (which does `@`)
6. `FIVE .`:
   - FIVE executes: pushes its PFA, then calls does_action (`@`)
   - `@` fetches the 5 stored there
   - `.` prints "5 "

</details>

---

## Exercise 13: Consolidation
```forth
: A 1 ;
: B 2 ;
: C A B + ;
CONSOLIDATE
```

<details>
<summary>Answer</summary>

1. Before CONSOLIDATE: A, B, C are separate WASM modules. C calls A and B via `call_indirect` through the function table.
2. CONSOLIDATE:
   - Collects all IR bodies: A=[PushI32(1)], B=[PushI32(2)], C=[Call(a_id), Call(b_id), Add(inlined)]
   - Builds local_fn_map: A→1, B→2, C→3 (within consolidated module)
   - `compile_consolidated_module()`: all three become functions in one WASM module
   - C's Call(a_id) → direct `call 1` (not call_indirect)
   - Replaces all table entries with new functions
3. Result: C calling A and B is now a direct WASM `call` — much faster than table dispatch.

</details>

---

## Exercise 14: Host Function Execution
```forth
5 3 M*
```

<details>
<summary>Answer</summary>

1. `5` → push to data stack (dsp -= 4, mem[dsp] = 5)
2. `3` → push to data stack (dsp -= 4, mem[dsp] = 3)
3. `M*` → host function (Rust closure):
   - Read sp = dsp global value
   - Read n2 = mem[sp] = 3 (as i64)
   - Read n1 = mem[sp+4] = 5 (as i64)
   - result = 5i64 * 3i64 = 15i64
   - lo = 15 as i32 = 15
   - hi = (15 >> 32) as i32 = 0
   - Write mem[sp+4] = 15 (lo), mem[sp] = 0 (hi)
   - Stack unchanged (still 2 cells, now containing double-cell 15)
4. Note: M* is a host function because it needs 64-bit multiplication (WASM i32 only)

</details>

---

## Exercise 15: Float Operations
```forth
: HYPOTENUSE  FDUP F* FSWAP FDUP F* F+ FSQRT ;
```

<details>
<summary>Answer</summary>

1. After inlining: `[FDup, FMul, FSwap, FDup, FMul, FAdd, FSqrt]`
2. **Peephole:** No matching patterns (FDup+FMul not a known pair)
3. **Codegen:** All float ops use the float stack (FSP global):
   - FDup: `fpeek(f)` then `fpush_via_local`
   - FMul: `emit_float_binary` with `f64.mul`
   - FSqrt: `emit_float_unary` with `f64.sqrt`
4. Float stack lives at 0x2540-0x2D40 in linear memory

</details>

---

## Exercise 16: BEGIN WHILE REPEAT
```forth
: COUNTDOWN  BEGIN DUP WHILE DUP . 1 - REPEAT DROP ;
```

<details>
<summary>Answer</summary>

1. `BEGIN` → ControlEntry::Begin { body: [] }
2. `DUP` → Call(dup_id) into body
3. `WHILE` → pop Begin, create ControlEntry::BeginWhile { test: [Call(dup_id)], body: [] }
4. `DUP . 1 -` → into body
5. `REPEAT` → pop BeginWhile, emit `BeginWhileRepeat { test: [Dup], body: [Dup, Call(dot_id), PushI32(1), Sub] }`
6. Semantics: evaluate test; if false exit loop; execute body; jump to BEGIN

</details>

---

## Exercise 17: Batch Mode Compilation
```forth
( During ForthVM::new() )
```

<details>
<summary>Answer</summary>

1. `register_primitives()` sets `batch_mode = true`
2. Each `register_primitive("DUP", ...)`:
   - Creates dictionary entry (dictionary.create + reveal)
   - Stores IR body in ir_bodies
   - Pushes `(word_id, ir_body)` to `deferred_ir` (no WASM compilation yet)
3. After all ~40 IR primitives registered:
   - `compile_batch()` compiles ALL deferred IR into a single WASM module
   - One `rt.instantiate_and_install()` call — single module with ~40 functions
   - Each function registered in the table
4. Why batch? Amortizes runtime compilation overhead. One module instead of 40.
5. Host functions bypass batch_mode — registered via `rt.register_host_func()` with HostFn closures.

</details>

---

## Exercise 18: wafer build Pipeline
```forth
( file: hello.fth )
: MAIN  ." Hello, World!" CR ;
```
```bash
wafer build hello.fth -o hello.wasm
```

<details>
<summary>Answer</summary>

1. `cmd_build()`: create ForthVM, set recording=true, evaluate source
2. `evaluate()`: compiles MAIN normally (IR → optimize → codegen)
3. `recording_toplevel=true`: but MAIN is a definition, not top-level execution, so toplevel_ir stays empty
4. `export_module()`:
   - Collect IR words: MAIN + all boot.fth definitions
   - Entry point: no --entry flag, look for MAIN → found!
   - Build `local_fn_map`: all words get module-internal indices
   - `compile_exportable_module()`: single WASM module with all functions
   - Data section: snapshot of linear memory (dictionary, variables, etc.)
   - Metadata in "wafer" custom section: version, entry index, host functions, memory size, stack pointers
5. Output: hello.wasm file

</details>

---

## Exercise 19: Stack-to-Local Promotion
```forth
: ADD3  + + ;
```

<details>
<summary>Answer</summary>

1. After inlining: `[Add, Add]`
2. **Stack-to-local promotion** (codegen pass, not optimizer):
   - Analyzes stack flow: first Add pops 2, pushes 1; second Add pops 2 (including that 1), pushes 1
   - If stack depth is statically known at each point → can use WASM locals instead of memory stack
   - Result: operands stay in WASM locals/operand stack, no memory reads/writes
   - Much faster: avoids load/store through linear memory
3. Promotion only works for "straight-line" code (no calls that might modify the stack unpredictably)

</details>

---

## Exercise 20: MARKER and State Restore
```forth
MARKER CLEAN
: FOO 1 ;
: BAR 2 ;
CLEAN
FOO  \ Error: unknown word
```

<details>
<summary>Answer</summary>

1. `MARKER CLEAN`:
   - Creates a MarkerState snapshot: dictionary state, user_here, next_table_index, word_pfa_map, ir_bodies, does_definitions, host_word_names, two_value_words, fvalue_words
   - Registers CLEAN as a word that, when executed, restores this snapshot
2. `: FOO 1 ; : BAR 2 ;` — normal compilation, adds to dictionary
3. `CLEAN`:
   - Executes the marker word
   - Restores dictionary to state before FOO/BAR were defined
   - Resets user_here, ir_bodies, etc.
   - FOO and BAR are gone — dictionary.find("FOO") returns None
4. `FOO` → "unknown word: FOO"

Key: MARKER doesn't undo WASM table entries (they become unreachable but stay allocated). It restores the dictionary and Rust-side metadata.

</details>
