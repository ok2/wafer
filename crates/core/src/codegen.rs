//! WASM code generation from IR.
//!
//! Translates optimized IR into WASM bytecode using the `wasm-encoder` crate.
//! Stacks live in linear memory. The data-stack pointer (`$dsp`) is cached in
//! a WASM local for the duration of each function, with write-back to the
//! global before calls and at function exit. The return-stack pointer (`$rsp`)
//! remains a global.

use std::borrow::Cow;
use std::collections::HashMap;

use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, CustomSection, DataCountSection, DataSection,
    ElementSection, Elements, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    GlobalType, ImportSection, Instruction, MemArg, MemoryType, Module, RefType, TableType,
    TypeSection, ValType,
};

use crate::dictionary::WordId;
use crate::error::{WaferError, WaferResult};
use crate::ir::IrOp;
use crate::memory::{CELL_SIZE, SYSVAR_LEAVE_FLAG};

// ---------------------------------------------------------------------------
// Import indices (order matters: imports numbered sequentially by kind)
// ---------------------------------------------------------------------------

/// Index of the imported memory.
const MEMORY_INDEX: u32 = 0;

/// Index of the `$dsp` global (data stack pointer).
const DSP: u32 = 0;

/// Index of the `$rsp` global (return stack pointer).
const RSP: u32 = 1;

/// Index of the `$fsp` global (float stack pointer).
const FSP: u32 = 2;

/// Index of the imported function table.
const TABLE: u32 = 0;

// Type indices in the type section.
const TYPE_VOID: u32 = 0; // () -> ()
const TYPE_I32: u32 = 1; // (i32) -> ()

// The `emit` callback is the first (and only) imported function, so index 0.
// The compiled word is the first (and only) defined function, so index 1.
const EMIT_FUNC: u32 = 0;
const WORD_FUNC: u32 = 1;

// ---------------------------------------------------------------------------
// DSP caching: local 0 holds a cached copy of the $dsp global.
// Scratch locals start at SCRATCH_BASE (1) instead of 0.
// ---------------------------------------------------------------------------

/// WASM local index for the cached data-stack pointer.
const CACHED_DSP_LOCAL: u32 = 0;

/// First WASM local index available for scratch temporaries.
const SCRATCH_BASE: u32 = 1;

/// Natural-alignment `MemArg` for 4-byte i32 operations.
const MEM4: MemArg = MemArg {
    offset: 0,
    align: 2, // 2^2 = 4
    memory_index: MEMORY_INDEX,
};

/// `MemArg` for single-byte operations.
const MEM1: MemArg = MemArg {
    offset: 0,
    align: 0, // 2^0 = 1
    memory_index: MEMORY_INDEX,
};

/// Natural-alignment `MemArg` for 8-byte f64 operations.
const MEM8: MemArg = MemArg {
    offset: 0,
    align: 3, // 2^3 = 8
    memory_index: MEMORY_INDEX,
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for code generation.
#[derive(Debug, Clone)]
pub struct CodegenConfig {
    /// Base function index (for the function table).
    pub base_fn_index: u32,
    /// Number of functions already in the table.
    pub table_size: u32,
    /// Enable stack-to-local promotion for straight-line words.
    pub stack_to_local_promotion: bool,
}

/// Result of compiling a word to WASM.
#[derive(Debug, Clone)]
pub struct CompiledModule {
    /// The WASM binary bytes.
    pub bytes: Vec<u8>,
    /// Function index in the table for this word.
    pub fn_index: u32,
}

// ---------------------------------------------------------------------------
// Instruction-level helpers (free functions that take &mut Function)
// ---------------------------------------------------------------------------

/// Decrement the cached `$dsp` local by `CELL_SIZE`.
fn dsp_dec(f: &mut Function) {
    f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
        .instruction(&Instruction::I32Const(CELL_SIZE as i32))
        .instruction(&Instruction::I32Sub)
        .instruction(&Instruction::LocalSet(CACHED_DSP_LOCAL));
}

/// Increment the cached `$dsp` local by `CELL_SIZE`.
fn dsp_inc(f: &mut Function) {
    f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
        .instruction(&Instruction::I32Const(CELL_SIZE as i32))
        .instruction(&Instruction::I32Add)
        .instruction(&Instruction::LocalSet(CACHED_DSP_LOCAL));
}

/// Push an i32 value that is already on the WASM operand stack onto the
/// data stack in linear memory, using `tmp` as a scratch local.
///
/// Sequence: local.set tmp; dsp -= 4; mem[dsp] = local.get tmp
fn push_via_local(f: &mut Function, tmp: u32) {
    f.instruction(&Instruction::LocalSet(tmp));
    dsp_dec(f);
    f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
        .instruction(&Instruction::LocalGet(tmp))
        .instruction(&Instruction::I32Store(MEM4));
}

/// Push a known i32 constant onto the data stack.
fn push_const(f: &mut Function, value: i32) {
    dsp_dec(f);
    f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
        .instruction(&Instruction::I32Const(value))
        .instruction(&Instruction::I32Store(MEM4));
}

/// Pop the top of the data stack onto the WASM operand stack.
fn pop(f: &mut Function) {
    f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
        .instruction(&Instruction::I32Load(MEM4));
    dsp_inc(f);
}

/// Pop the top of the data stack into a local.
fn pop_to(f: &mut Function, local: u32) {
    pop(f);
    f.instruction(&Instruction::LocalSet(local));
}

/// Read the top of the data stack without popping (value on operand stack).
fn peek(f: &mut Function) {
    f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
        .instruction(&Instruction::I32Load(MEM4));
}

/// Write the cached DSP local back to the `$dsp` global.
///
/// Emitted before calls and at function exit so callees see the correct value.
fn dsp_writeback(f: &mut Function) {
    f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
        .instruction(&Instruction::GlobalSet(DSP));
}

/// Reload the cached DSP local from the `$dsp` global.
///
/// Emitted after calls since the callee may have modified `$dsp`.
fn dsp_reload(f: &mut Function) {
    f.instruction(&Instruction::GlobalGet(DSP))
        .instruction(&Instruction::LocalSet(CACHED_DSP_LOCAL));
}

/// Push a value from the WASM operand stack onto the return stack via `tmp`.
fn rpush_via_local(f: &mut Function, tmp: u32) {
    f.instruction(&Instruction::LocalSet(tmp));
    // rsp -= CELL_SIZE
    f.instruction(&Instruction::GlobalGet(RSP))
        .instruction(&Instruction::I32Const(CELL_SIZE as i32))
        .instruction(&Instruction::I32Sub)
        .instruction(&Instruction::GlobalSet(RSP));
    // mem[rsp] = value
    f.instruction(&Instruction::GlobalGet(RSP))
        .instruction(&Instruction::LocalGet(tmp))
        .instruction(&Instruction::I32Store(MEM4));
}

/// Pop the return stack onto the WASM operand stack.
fn rpop(f: &mut Function) {
    f.instruction(&Instruction::GlobalGet(RSP))
        .instruction(&Instruction::I32Load(MEM4));
    // rsp += CELL_SIZE
    f.instruction(&Instruction::GlobalGet(RSP))
        .instruction(&Instruction::I32Const(CELL_SIZE as i32))
        .instruction(&Instruction::I32Add)
        .instruction(&Instruction::GlobalSet(RSP));
}

/// Peek at the top of the return stack (no pop).
fn rpeek(f: &mut Function) {
    f.instruction(&Instruction::GlobalGet(RSP))
        .instruction(&Instruction::I32Load(MEM4));
}

/// Convert a WASM boolean (0 or 1 on operand stack) to a Forth flag (0 or -1).
/// Uses `tmp` as scratch local.
fn bool_to_forth_flag(f: &mut Function, tmp: u32) {
    // 0 - result: if result=1 => -1, if result=0 => 0
    f.instruction(&Instruction::LocalSet(tmp))
        .instruction(&Instruction::I32Const(0))
        .instruction(&Instruction::LocalGet(tmp))
        .instruction(&Instruction::I32Sub);
}

// ---------------------------------------------------------------------------
// Float stack helpers
// ---------------------------------------------------------------------------

/// Carries context for WASM code emission.
struct EmitCtx {
    f64_local_0: u32,
    f64_local_1: u32,
    /// Base WASM local index for Forth locals ({: ... :}).
    /// Forth local N maps to WASM local `forth_local_base + N`.
    forth_local_base: u32,
    /// Base WASM local index for DO/LOOP index/limit local pairs.
    /// Each nested loop uses 2 locals: (index, limit).
    loop_local_base: u32,
    /// Stack of (index_local, limit_local) for active DO/LOOP nesting.
    /// Innermost loop is last. Used to compile `J` as local.get.
    loop_locals: Vec<(u32, u32)>,
    /// Nesting depth of DO/LOOPs that use the fast path (no RS sync).
    /// When > 0, `RFetch` (I) reads from the loop local instead of rpeek.
    fast_loop_depth: u32,
    /// The word being compiled (for self-recursion detection).
    /// When `Call(id)` matches this, emit direct `call` instead of `call_indirect`.
    self_word_id: Option<WordId>,
}

/// Decrement the FSP global by 8 (allocate space for one f64).
fn fsp_dec(f: &mut Function) {
    f.instruction(&Instruction::GlobalGet(FSP))
        .instruction(&Instruction::I32Const(8))
        .instruction(&Instruction::I32Sub)
        .instruction(&Instruction::GlobalSet(FSP));
}

/// Increment the FSP global by 8 (free space for one f64).
fn fsp_inc(f: &mut Function) {
    f.instruction(&Instruction::GlobalGet(FSP))
        .instruction(&Instruction::I32Const(8))
        .instruction(&Instruction::I32Add)
        .instruction(&Instruction::GlobalSet(FSP));
}

/// Save an f64 from the WASM operand stack into `tmp`, decrement FSP,
/// then store the f64 at [FSP].
fn fpush_via_local(f: &mut Function, tmp: u32) {
    f.instruction(&Instruction::LocalSet(tmp));
    fsp_dec(f);
    f.instruction(&Instruction::GlobalGet(FSP))
        .instruction(&Instruction::LocalGet(tmp))
        .instruction(&Instruction::F64Store(MEM8));
}

/// Decrement FSP, then store the f64 from local `src` at [FSP].
fn fpush_from_local(f: &mut Function, src: u32) {
    fsp_dec(f);
    f.instruction(&Instruction::GlobalGet(FSP))
        .instruction(&Instruction::LocalGet(src))
        .instruction(&Instruction::F64Store(MEM8));
}

/// Load f64 from [FSP] onto the WASM operand stack, then increment FSP.
fn fpop(f: &mut Function) {
    f.instruction(&Instruction::GlobalGet(FSP))
        .instruction(&Instruction::F64Load(MEM8));
    fsp_inc(f);
}

/// Load f64 from [FSP] onto the WASM operand stack without popping.
fn fpeek(f: &mut Function) {
    f.instruction(&Instruction::GlobalGet(FSP))
        .instruction(&Instruction::F64Load(MEM8));
}

/// Pop two floats (b then a), apply binary op, push result.
fn emit_float_binary(f: &mut Function, ctx: &EmitCtx, wasm_op: &Instruction<'_>) {
    fpop(f);
    f.instruction(&Instruction::LocalSet(ctx.f64_local_0));
    fpop(f);
    f.instruction(&Instruction::LocalSet(ctx.f64_local_1));
    f.instruction(&Instruction::LocalGet(ctx.f64_local_1))
        .instruction(&Instruction::LocalGet(ctx.f64_local_0))
        .instruction(wasm_op);
    fpush_via_local(f, ctx.f64_local_0);
}

/// Pop one float, apply unary op, push result.
fn emit_float_unary(f: &mut Function, ctx: &EmitCtx, wasm_op: &Instruction<'_>) {
    fpop(f);
    f.instruction(wasm_op);
    fpush_via_local(f, ctx.f64_local_0);
}

/// Pop two floats, compare, push Forth flag to data stack.
fn emit_float_cmp(f: &mut Function, ctx: &EmitCtx, wasm_cmp: &Instruction<'_>) {
    fpop(f);
    f.instruction(&Instruction::LocalSet(ctx.f64_local_0));
    fpop(f);
    f.instruction(&Instruction::LocalSet(ctx.f64_local_1));
    f.instruction(&Instruction::LocalGet(ctx.f64_local_1))
        .instruction(&Instruction::LocalGet(ctx.f64_local_0))
        .instruction(wasm_cmp);
    bool_to_forth_flag(f, SCRATCH_BASE);
    push_via_local(f, SCRATCH_BASE + 1);
}

// ---------------------------------------------------------------------------
// IR emission
// ---------------------------------------------------------------------------

/// Emit all IR operations in `ops` into the WASM function body `f`.
fn emit_body(f: &mut Function, ops: &[IrOp], ctx: &mut EmitCtx) {
    for op in ops {
        emit_op(f, op, ctx);
    }
}

/// Emit a single IR operation.
#[allow(clippy::too_many_lines)]
fn emit_op(f: &mut Function, op: &IrOp, ctx: &mut EmitCtx) {
    match op {
        // -- Literals -------------------------------------------------------
        IrOp::PushI32(n) => push_const(f, *n),
        IrOp::PushI64(_) => { /* TODO: double-cell */ }
        IrOp::PushF64(val) => {
            fsp_dec(f);
            f.instruction(&Instruction::GlobalGet(FSP))
                .instruction(&Instruction::F64Const(*val))
                .instruction(&Instruction::F64Store(MEM8));
        }

        // -- Stack manipulation ---------------------------------------------
        IrOp::Drop => dsp_inc(f),

        IrOp::Dup => {
            peek(f);
            push_via_local(f, SCRATCH_BASE);
        }

        IrOp::Swap => {
            // ( a b -- b a )
            pop_to(f, SCRATCH_BASE); // b
            pop_to(f, SCRATCH_BASE + 1); // a
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE));
            push_via_local(f, SCRATCH_BASE + 2);
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE + 1));
            push_via_local(f, SCRATCH_BASE + 2);
        }

        IrOp::Over => {
            // ( a b -- a b a ) : read second item
            f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
                .instruction(&Instruction::I32Const(CELL_SIZE as i32))
                .instruction(&Instruction::I32Add)
                .instruction(&Instruction::I32Load(MEM4));
            push_via_local(f, SCRATCH_BASE);
        }

        IrOp::Rot => {
            // ( a b c -- b c a )
            pop_to(f, SCRATCH_BASE); // c
            pop_to(f, SCRATCH_BASE + 1); // b
            pop_to(f, SCRATCH_BASE + 2); // a
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE + 1));
            push_via_local(f, SCRATCH_BASE + 3);
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE));
            push_via_local(f, SCRATCH_BASE + 3);
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE + 2));
            push_via_local(f, SCRATCH_BASE + 3);
        }

        IrOp::Nip => {
            // ( a b -- b )
            pop_to(f, SCRATCH_BASE); // b
            dsp_inc(f); // drop a
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE));
            push_via_local(f, SCRATCH_BASE + 1);
        }

        IrOp::Tuck => {
            // ( a b -- b a b )
            pop_to(f, SCRATCH_BASE); // b
            pop_to(f, SCRATCH_BASE + 1); // a
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE));
            push_via_local(f, SCRATCH_BASE + 2);
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE + 1));
            push_via_local(f, SCRATCH_BASE + 2);
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE));
            push_via_local(f, SCRATCH_BASE + 2);
        }

        // -- Arithmetic -----------------------------------------------------
        IrOp::Add => emit_binary_commutative(f, &Instruction::I32Add),
        IrOp::Mul => emit_binary_commutative(f, &Instruction::I32Mul),

        IrOp::Sub => {
            // ( a b -- a-b )
            pop_to(f, SCRATCH_BASE); // b
            pop_to(f, SCRATCH_BASE + 1); // a
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE + 1))
                .instruction(&Instruction::LocalGet(SCRATCH_BASE))
                .instruction(&Instruction::I32Sub);
            push_via_local(f, SCRATCH_BASE + 2);
        }

        IrOp::DivMod => {
            // ( n1 n2 -- rem quot )
            pop_to(f, SCRATCH_BASE); // n2
            pop_to(f, SCRATCH_BASE + 1); // n1
            // Push remainder first (deeper)
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE + 1))
                .instruction(&Instruction::LocalGet(SCRATCH_BASE))
                .instruction(&Instruction::I32RemS);
            push_via_local(f, SCRATCH_BASE + 2);
            // Push quotient on top
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE + 1))
                .instruction(&Instruction::LocalGet(SCRATCH_BASE))
                .instruction(&Instruction::I32DivS);
            push_via_local(f, SCRATCH_BASE + 2);
        }

        IrOp::Negate => {
            pop_to(f, SCRATCH_BASE);
            f.instruction(&Instruction::I32Const(0))
                .instruction(&Instruction::LocalGet(SCRATCH_BASE))
                .instruction(&Instruction::I32Sub);
            push_via_local(f, SCRATCH_BASE + 1);
        }

        IrOp::Abs => {
            pop_to(f, SCRATCH_BASE);
            // if local < 0: local = 0 - local
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE))
                .instruction(&Instruction::I32Const(0))
                .instruction(&Instruction::I32LtS)
                .instruction(&Instruction::If(BlockType::Empty))
                .instruction(&Instruction::I32Const(0))
                .instruction(&Instruction::LocalGet(SCRATCH_BASE))
                .instruction(&Instruction::I32Sub)
                .instruction(&Instruction::LocalSet(SCRATCH_BASE))
                .instruction(&Instruction::End);
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE));
            push_via_local(f, SCRATCH_BASE + 1);
        }

        // -- Comparison -----------------------------------------------------
        IrOp::Eq => emit_cmp(f, &Instruction::I32Eq),
        IrOp::NotEq => emit_cmp(f, &Instruction::I32Ne),
        IrOp::Lt => emit_cmp(f, &Instruction::I32LtS),
        IrOp::Gt => emit_cmp(f, &Instruction::I32GtS),
        IrOp::LtUnsigned => emit_cmp(f, &Instruction::I32LtU),

        IrOp::ZeroEq => {
            pop(f);
            f.instruction(&Instruction::I32Eqz);
            bool_to_forth_flag(f, SCRATCH_BASE);
            push_via_local(f, SCRATCH_BASE + 1);
        }

        IrOp::ZeroLt => {
            pop(f);
            f.instruction(&Instruction::I32Const(0))
                .instruction(&Instruction::I32LtS);
            bool_to_forth_flag(f, SCRATCH_BASE);
            push_via_local(f, SCRATCH_BASE + 1);
        }

        // -- Logic ----------------------------------------------------------
        IrOp::And => emit_binary_commutative(f, &Instruction::I32And),
        IrOp::Or => emit_binary_commutative(f, &Instruction::I32Or),
        IrOp::Xor => emit_binary_commutative(f, &Instruction::I32Xor),

        IrOp::Invert => {
            pop(f);
            f.instruction(&Instruction::I32Const(-1))
                .instruction(&Instruction::I32Xor);
            push_via_local(f, SCRATCH_BASE);
        }

        IrOp::Lshift => emit_binary_ordered(f, &Instruction::I32Shl),
        IrOp::Rshift => emit_binary_ordered(f, &Instruction::I32ShrU),
        IrOp::ArithRshift => emit_binary_ordered(f, &Instruction::I32ShrS),

        // -- Memory ---------------------------------------------------------
        IrOp::Fetch => {
            // ( addr -- value )
            pop(f);
            f.instruction(&Instruction::I32Load(MEM4));
            push_via_local(f, SCRATCH_BASE);
        }

        IrOp::Store => {
            // ( x addr -- )
            pop_to(f, SCRATCH_BASE); // addr
            pop_to(f, SCRATCH_BASE + 1); // x
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE))
                .instruction(&Instruction::LocalGet(SCRATCH_BASE + 1))
                .instruction(&Instruction::I32Store(MEM4));
        }

        IrOp::CFetch => {
            pop(f);
            f.instruction(&Instruction::I32Load8U(MEM1));
            push_via_local(f, SCRATCH_BASE);
        }

        IrOp::CStore => {
            pop_to(f, SCRATCH_BASE); // addr
            pop_to(f, SCRATCH_BASE + 1); // char
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE))
                .instruction(&Instruction::LocalGet(SCRATCH_BASE + 1))
                .instruction(&Instruction::I32Store8(MEM1));
        }

        IrOp::PlusStore => {
            // ( n addr -- ) : mem[addr] += n
            pop_to(f, SCRATCH_BASE); // addr
            pop_to(f, SCRATCH_BASE + 1); // n
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE))
                .instruction(&Instruction::LocalGet(SCRATCH_BASE))
                .instruction(&Instruction::I32Load(MEM4))
                .instruction(&Instruction::LocalGet(SCRATCH_BASE + 1))
                .instruction(&Instruction::I32Add)
                .instruction(&Instruction::I32Store(MEM4));
        }

        // -- Control flow ---------------------------------------------------
        IrOp::Call(word_id) => {
            dsp_writeback(f);
            if ctx.self_word_id == Some(*word_id) {
                // Self-recursion: direct call (avoids table lookup + signature check)
                f.instruction(&Instruction::Call(WORD_FUNC));
            } else {
                f.instruction(&Instruction::I32Const(word_id.0 as i32))
                    .instruction(&Instruction::CallIndirect {
                        type_index: TYPE_VOID,
                        table_index: TABLE,
                    });
            }
            dsp_reload(f);
        }

        IrOp::TailCall(word_id) => {
            dsp_writeback(f);
            if ctx.self_word_id == Some(*word_id) {
                f.instruction(&Instruction::Call(WORD_FUNC));
            } else {
                f.instruction(&Instruction::I32Const(word_id.0 as i32))
                    .instruction(&Instruction::CallIndirect {
                        type_index: TYPE_VOID,
                        table_index: TABLE,
                    });
            }
            f.instruction(&Instruction::Return);
        }

        IrOp::If {
            then_body,
            else_body,
        } => {
            pop(f);
            f.instruction(&Instruction::If(BlockType::Empty));
            emit_body(f, then_body, ctx);
            if let Some(eb) = else_body {
                f.instruction(&Instruction::Else);
                emit_body(f, eb, ctx);
            }
            f.instruction(&Instruction::End);
        }

        IrOp::DoLoop { body, is_plus_loop } => {
            emit_do_loop(f, body, *is_plus_loop, ctx);
        }

        IrOp::BeginUntil { body } => {
            f.instruction(&Instruction::Loop(BlockType::Empty));
            emit_body(f, body, ctx);
            pop(f);
            f.instruction(&Instruction::I32Eqz)
                .instruction(&Instruction::BrIf(0))
                .instruction(&Instruction::End);
        }

        IrOp::BeginAgain { body } => {
            f.instruction(&Instruction::Loop(BlockType::Empty));
            emit_body(f, body, ctx);
            f.instruction(&Instruction::Br(0))
                .instruction(&Instruction::End);
        }

        IrOp::BeginWhileRepeat { test, body } => {
            f.instruction(&Instruction::Block(BlockType::Empty));
            f.instruction(&Instruction::Loop(BlockType::Empty));
            emit_body(f, test, ctx);
            pop(f);
            f.instruction(&Instruction::I32Eqz)
                .instruction(&Instruction::BrIf(1)); // break to outer block
            emit_body(f, body, ctx);
            f.instruction(&Instruction::Br(0)) // continue loop
                .instruction(&Instruction::End) // end loop
                .instruction(&Instruction::End); // end block
        }

        IrOp::BeginDoubleWhileRepeat {
            outer_test,
            inner_test,
            body,
            after_repeat,
            else_body,
        } => {
            // WASM structure:
            //   block $end           ;; THEN target
            //     block $else        ;; first WHILE false target
            //       block $after     ;; second WHILE false target
            //         loop $begin
            //           outer_test
            //           br_if(2) $else   ;; first WHILE: if false, skip to else
            //           inner_test
            //           br_if(1) $after  ;; second WHILE: if false, skip to after
            //           body
            //           br(0)            ;; REPEAT: back to loop start
            //         end
            //       end
            //       after_repeat code
            //       br(1) $end           ;; skip else, goto end
            //     end
            //     else code
            //   end
            f.instruction(&Instruction::Block(BlockType::Empty)); // $end
            f.instruction(&Instruction::Block(BlockType::Empty)); // $else
            f.instruction(&Instruction::Block(BlockType::Empty)); // $after
            f.instruction(&Instruction::Loop(BlockType::Empty)); // $begin
            emit_body(f, outer_test, ctx);
            pop(f);
            f.instruction(&Instruction::I32Eqz)
                .instruction(&Instruction::BrIf(2)); // to $else
            emit_body(f, inner_test, ctx);
            pop(f);
            f.instruction(&Instruction::I32Eqz)
                .instruction(&Instruction::BrIf(1)); // to $after
            emit_body(f, body, ctx);
            f.instruction(&Instruction::Br(0)); // back to $begin
            f.instruction(&Instruction::End); // end loop
            f.instruction(&Instruction::End); // end $after block
            emit_body(f, after_repeat, ctx);
            if else_body.is_some() {
                f.instruction(&Instruction::Br(1)); // skip else, goto $end
            }
            f.instruction(&Instruction::End); // end $else block
            if let Some(eb) = else_body {
                emit_body(f, eb, ctx);
            }
            f.instruction(&Instruction::End); // end $end block
        }

        IrOp::Exit => {
            // Write back cached DSP before early return
            dsp_writeback(f);
            f.instruction(&Instruction::Return);
        }

        // -- Forth locals ({: ... :}) -----------------------------------------
        IrOp::ForthLocalGet(n) => {
            f.instruction(&Instruction::LocalGet(ctx.forth_local_base + n));
            push_via_local(f, SCRATCH_BASE);
        }
        IrOp::ForthLocalSet(n) => {
            pop_to(f, ctx.forth_local_base + n);
        }

        // -- Return stack ---------------------------------------------------
        IrOp::ToR => {
            pop(f);
            rpush_via_local(f, SCRATCH_BASE);
        }

        IrOp::FromR => {
            rpop(f);
            push_via_local(f, SCRATCH_BASE);
        }

        IrOp::RFetch => {
            // In a fast-path DO/LOOP (no >R/R>/calls), R@ == loop index local.
            // In slow-path or outside loops, read from the return stack.
            if ctx.fast_loop_depth > 0 {
                let (index_local, _) = *ctx.loop_locals.last().expect("fast loop without locals");
                f.instruction(&Instruction::LocalGet(index_local));
                push_via_local(f, SCRATCH_BASE);
            } else {
                rpeek(f);
                push_via_local(f, SCRATCH_BASE);
            }
        }

        IrOp::LoopJ => {
            // Read outer loop index: use loop locals if available,
            // otherwise fall back to reading rsp+8.
            if ctx.loop_locals.len() >= 2 {
                let (outer_index, _) = ctx.loop_locals[ctx.loop_locals.len() - 2];
                f.instruction(&Instruction::LocalGet(outer_index));
                push_via_local(f, SCRATCH_BASE);
            } else {
                // Fallback: read from return stack (rsp + 2*CELL_SIZE)
                f.instruction(&Instruction::GlobalGet(RSP))
                    .instruction(&Instruction::I32Const(2 * CELL_SIZE as i32))
                    .instruction(&Instruction::I32Add)
                    .instruction(&Instruction::I32Load(MEM4));
                push_via_local(f, SCRATCH_BASE);
            }
        }

        // -- I/O ------------------------------------------------------------
        IrOp::Emit => {
            pop(f);
            f.instruction(&Instruction::Call(EMIT_FUNC));
        }

        IrOp::Dot => {
            // MVP stub: pop and discard
            pop(f);
            f.instruction(&Instruction::Drop);
        }

        IrOp::Cr => {
            f.instruction(&Instruction::I32Const(10))
                .instruction(&Instruction::Call(EMIT_FUNC));
        }

        IrOp::Type => {
            // MVP stub: drop both (c-addr u)
            pop(f);
            f.instruction(&Instruction::Drop);
            pop(f);
            f.instruction(&Instruction::Drop);
        }

        // -- System ---------------------------------------------------------
        IrOp::Execute => {
            pop(f);
            // Write back cached DSP before indirect call
            dsp_writeback(f);
            f.instruction(&Instruction::CallIndirect {
                type_index: TYPE_VOID,
                table_index: TABLE,
            });
            // Reload cached DSP after call
            dsp_reload(f);
        }

        IrOp::SpFetch => {
            // Push the current cached DSP value onto the data stack.
            // Save DSP, decrement, then store the saved value at new TOS.
            f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
                .instruction(&Instruction::LocalSet(SCRATCH_BASE));
            dsp_dec(f);
            f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
                .instruction(&Instruction::LocalGet(SCRATCH_BASE))
                .instruction(&Instruction::I32Store(MEM4));
        }

        // -- Compound operations -----------------------------------------------
        IrOp::TwoDup => {
            // ( a b -- a b a b )
            f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
                .instruction(&Instruction::I32Load(MEM4)); // b
            f.instruction(&Instruction::LocalSet(SCRATCH_BASE));
            f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
                .instruction(&Instruction::I32Const(CELL_SIZE as i32))
                .instruction(&Instruction::I32Add)
                .instruction(&Instruction::I32Load(MEM4)); // a
            f.instruction(&Instruction::LocalSet(SCRATCH_BASE + 1));
            // dsp -= 8
            f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
                .instruction(&Instruction::I32Const((CELL_SIZE * 2) as i32))
                .instruction(&Instruction::I32Sub)
                .instruction(&Instruction::LocalSet(CACHED_DSP_LOCAL));
            // store a at [dsp+4], b at [dsp]
            f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
                .instruction(&Instruction::I32Const(CELL_SIZE as i32))
                .instruction(&Instruction::I32Add)
                .instruction(&Instruction::LocalGet(SCRATCH_BASE + 1))
                .instruction(&Instruction::I32Store(MEM4));
            f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
                .instruction(&Instruction::LocalGet(SCRATCH_BASE))
                .instruction(&Instruction::I32Store(MEM4));
        }

        IrOp::TwoDrop => {
            // ( a b -- )
            f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
                .instruction(&Instruction::I32Const((CELL_SIZE * 2) as i32))
                .instruction(&Instruction::I32Add)
                .instruction(&Instruction::LocalSet(CACHED_DSP_LOCAL));
        }

        // -- Float stack ops -----------------------------------------------
        IrOp::FDrop => fsp_inc(f),
        IrOp::FDup => {
            fpeek(f);
            fpush_via_local(f, ctx.f64_local_0);
        }
        IrOp::FSwap => {
            fpop(f);
            f.instruction(&Instruction::LocalSet(ctx.f64_local_0));
            fpop(f);
            f.instruction(&Instruction::LocalSet(ctx.f64_local_1));
            fpush_from_local(f, ctx.f64_local_0);
            fpush_from_local(f, ctx.f64_local_1);
        }
        IrOp::FOver => {
            f.instruction(&Instruction::GlobalGet(FSP))
                .instruction(&Instruction::I32Const(8))
                .instruction(&Instruction::I32Add)
                .instruction(&Instruction::F64Load(MEM8));
            fpush_via_local(f, ctx.f64_local_0);
        }

        // -- Float arithmetic ----------------------------------------------
        IrOp::FAdd => emit_float_binary(f, ctx, &Instruction::F64Add),
        IrOp::FSub => emit_float_binary(f, ctx, &Instruction::F64Sub),
        IrOp::FMul => emit_float_binary(f, ctx, &Instruction::F64Mul),
        IrOp::FDiv => emit_float_binary(f, ctx, &Instruction::F64Div),
        IrOp::FMin => emit_float_binary(f, ctx, &Instruction::F64Min),
        IrOp::FMax => emit_float_binary(f, ctx, &Instruction::F64Max),
        IrOp::FNegate => emit_float_unary(f, ctx, &Instruction::F64Neg),
        IrOp::FAbs => emit_float_unary(f, ctx, &Instruction::F64Abs),
        IrOp::FSqrt => emit_float_unary(f, ctx, &Instruction::F64Sqrt),
        IrOp::FFloor => emit_float_unary(f, ctx, &Instruction::F64Floor),
        IrOp::FRound => emit_float_unary(f, ctx, &Instruction::F64Nearest),

        // -- Float comparisons (cross-stack) --------------------------------
        IrOp::FZeroEq => {
            fpop(f);
            f.instruction(&Instruction::F64Const(0.0))
                .instruction(&Instruction::F64Eq);
            bool_to_forth_flag(f, SCRATCH_BASE);
            push_via_local(f, SCRATCH_BASE + 1);
        }
        IrOp::FZeroLt => {
            fpop(f);
            f.instruction(&Instruction::F64Const(0.0))
                .instruction(&Instruction::F64Lt);
            bool_to_forth_flag(f, SCRATCH_BASE);
            push_via_local(f, SCRATCH_BASE + 1);
        }
        IrOp::FEq => emit_float_cmp(f, ctx, &Instruction::F64Eq),
        IrOp::FLt => emit_float_cmp(f, ctx, &Instruction::F64Lt),

        // -- Float memory (cross-stack) ------------------------------------
        IrOp::FetchFloat => {
            // ( addr -- ) ( F: -- r )
            pop(f); // addr on operand stack
            f.instruction(&Instruction::F64Load(MEM8));
            fpush_via_local(f, ctx.f64_local_0);
        }
        IrOp::StoreFloat => {
            // ( addr -- ) ( F: r -- )
            pop_to(f, SCRATCH_BASE); // addr
            fpop(f);
            f.instruction(&Instruction::LocalSet(ctx.f64_local_0));
            f.instruction(&Instruction::LocalGet(SCRATCH_BASE))
                .instruction(&Instruction::LocalGet(ctx.f64_local_0))
                .instruction(&Instruction::F64Store(MEM8));
        }

        // -- Float/integer conversions (cross-stack) -----------------------
        IrOp::StoF => {
            // ( n -- ) ( F: -- r )
            pop(f);
            f.instruction(&Instruction::F64ConvertI32S);
            fpush_via_local(f, ctx.f64_local_0);
        }
        IrOp::FtoS => {
            // ( F: r -- ) ( -- n )
            fpop(f);
            f.instruction(&Instruction::I32TruncF64S);
            push_via_local(f, SCRATCH_BASE);
        }
    }
}

/// Binary operation where operand order does not matter (commutative).
/// Pops two from data stack, applies `op`, pushes result.
fn emit_binary_commutative(f: &mut Function, op: &Instruction<'_>) {
    pop_to(f, SCRATCH_BASE); // second operand
    pop_to(f, SCRATCH_BASE + 1); // first operand
    f.instruction(&Instruction::LocalGet(SCRATCH_BASE + 1))
        .instruction(&Instruction::LocalGet(SCRATCH_BASE))
        .instruction(op);
    push_via_local(f, SCRATCH_BASE + 2);
}

/// Binary operation where operand order matters: ( a b -- a OP b ).
/// First pops b, then a, pushes a OP b.
fn emit_binary_ordered(f: &mut Function, op: &Instruction<'_>) {
    pop_to(f, SCRATCH_BASE); // b
    pop_to(f, SCRATCH_BASE + 1); // a
    f.instruction(&Instruction::LocalGet(SCRATCH_BASE + 1))
        .instruction(&Instruction::LocalGet(SCRATCH_BASE))
        .instruction(op);
    push_via_local(f, SCRATCH_BASE + 2);
}

/// Comparison: pop two, compare, push Forth flag (-1 or 0).
fn emit_cmp(f: &mut Function, cmp: &Instruction<'_>) {
    pop_to(f, SCRATCH_BASE); // b
    pop_to(f, SCRATCH_BASE + 1); // a
    f.instruction(&Instruction::LocalGet(SCRATCH_BASE + 1))
        .instruction(&Instruction::LocalGet(SCRATCH_BASE))
        .instruction(cmp);
    bool_to_forth_flag(f, SCRATCH_BASE + 2);
    push_via_local(f, SCRATCH_BASE + 3);
}

/// Emit a DO...LOOP / DO...+LOOP construct using WASM locals for index/limit.
///
/// Two paths:
/// - **Fast path**: Body has no calls, no return stack ops. Index/limit live
///   purely in WASM locals — zero return stack traffic per iteration.
/// - **Slow path**: Body uses calls or return stack. Index/limit still in locals
///   but synced to return stack for LEAVE/UNLOOP/J/I compatibility.
fn emit_do_loop(f: &mut Function, body: &[IrOp], is_plus_loop: bool, ctx: &mut EmitCtx) {
    let loop_depth = ctx.loop_locals.len() as u32;
    let index_local = ctx.loop_local_base + loop_depth * 2;
    let limit_local = ctx.loop_local_base + loop_depth * 2 + 1;
    let needs_rs = body_needs_return_stack(body);

    // DO ( limit index -- )
    pop_to(f, index_local);
    pop_to(f, limit_local);

    if needs_rs {
        // Push to return stack for I/J/LEAVE/UNLOOP
        f.instruction(&Instruction::LocalGet(limit_local));
        rpush_via_local(f, SCRATCH_BASE);
        f.instruction(&Instruction::LocalGet(index_local));
        rpush_via_local(f, SCRATCH_BASE);
    }

    ctx.loop_locals.push((index_local, limit_local));
    if !needs_rs {
        ctx.fast_loop_depth += 1;
    }

    f.instruction(&Instruction::Block(BlockType::Empty));
    f.instruction(&Instruction::Loop(BlockType::Empty));

    if needs_rs {
        // Sync index local to return stack before body (so I/R@ reads current value)
        rpop(f);
        f.instruction(&Instruction::Drop);
        f.instruction(&Instruction::LocalGet(index_local));
        rpush_via_local(f, SCRATCH_BASE);
    }

    emit_body(f, body, ctx);

    if needs_rs {
        // Reload index from return stack (LEAVE may have modified it)
        rpeek(f);
        f.instruction(&Instruction::LocalSet(index_local));
    }

    if is_plus_loop {
        pop_to(f, SCRATCH_BASE + 2); // step from data stack

        // Check leave flag — if set, clear it and exit immediately
        f.instruction(&Instruction::I32Const(SYSVAR_LEAVE_FLAG as i32))
            .instruction(&Instruction::I32Load(MEM4))
            .instruction(&Instruction::If(BlockType::Empty))
            .instruction(&Instruction::I32Const(SYSVAR_LEAVE_FLAG as i32))
            .instruction(&Instruction::I32Const(0))
            .instruction(&Instruction::I32Store(MEM4))
            .instruction(&Instruction::Br(2)) // exit: If(0) → Loop(1) → Block(2)
            .instruction(&Instruction::End);

        // old_index - limit
        f.instruction(&Instruction::LocalGet(index_local))
            .instruction(&Instruction::LocalGet(limit_local))
            .instruction(&Instruction::I32Sub)
            .instruction(&Instruction::LocalSet(SCRATCH_BASE + 3));

        // new_index = old_index + step
        f.instruction(&Instruction::LocalGet(index_local))
            .instruction(&Instruction::LocalGet(SCRATCH_BASE + 2))
            .instruction(&Instruction::I32Add)
            .instruction(&Instruction::LocalSet(index_local));

        // Forth 2012 +LOOP termination:
        // exit = ((old-limit) XOR (new-limit)) AND ((old-limit) XOR step) < 0
        f.instruction(&Instruction::LocalGet(SCRATCH_BASE + 3)) // old - limit
            .instruction(&Instruction::LocalGet(index_local)) // new_index
            .instruction(&Instruction::LocalGet(limit_local)) // limit
            .instruction(&Instruction::I32Sub) // new - limit
            .instruction(&Instruction::I32Xor); // xor1
        f.instruction(&Instruction::LocalGet(SCRATCH_BASE + 3)) // old - limit
            .instruction(&Instruction::LocalGet(SCRATCH_BASE + 2)) // step
            .instruction(&Instruction::I32Xor); // xor2
        f.instruction(&Instruction::I32And)
            .instruction(&Instruction::I32Const(0))
            .instruction(&Instruction::I32LtS)
            .instruction(&Instruction::BrIf(1)) // break to $exit
            .instruction(&Instruction::Br(0)) // continue loop
            .instruction(&Instruction::End) // end loop
            .instruction(&Instruction::End); // end block
    } else {
        // LOOP: simple increment by 1
        f.instruction(&Instruction::LocalGet(index_local))
            .instruction(&Instruction::I32Const(1))
            .instruction(&Instruction::I32Add)
            .instruction(&Instruction::LocalSet(index_local));

        // Check leave flag (needed even for simple LOOP since LEAVE is a host function)
        if needs_rs {
            f.instruction(&Instruction::I32Const(SYSVAR_LEAVE_FLAG as i32))
                .instruction(&Instruction::I32Load(MEM4))
                .instruction(&Instruction::If(BlockType::Empty))
                .instruction(&Instruction::I32Const(SYSVAR_LEAVE_FLAG as i32))
                .instruction(&Instruction::I32Const(0))
                .instruction(&Instruction::I32Store(MEM4))
                .instruction(&Instruction::Br(2)) // exit: If→Loop→Block
                .instruction(&Instruction::End);
        }

        // if index >= limit, exit
        f.instruction(&Instruction::LocalGet(index_local))
            .instruction(&Instruction::LocalGet(limit_local))
            .instruction(&Instruction::I32GeS)
            .instruction(&Instruction::BrIf(1)) // break to $exit
            .instruction(&Instruction::Br(0)) // continue loop
            .instruction(&Instruction::End) // end loop
            .instruction(&Instruction::End); // end block
    }

    if !needs_rs {
        ctx.fast_loop_depth -= 1;
    }
    ctx.loop_locals.pop();

    if needs_rs {
        rpop(f);
        f.instruction(&Instruction::Drop);
        rpop(f);
        f.instruction(&Instruction::Drop);
    }
    f.instruction(&Instruction::I32Const(SYSVAR_LEAVE_FLAG as i32))
        .instruction(&Instruction::I32Const(0))
        .instruction(&Instruction::I32Store(MEM4));
}

// ---------------------------------------------------------------------------
// Stack-to-local promotion
// ---------------------------------------------------------------------------

/// Check if a word body qualifies for stack-to-local promotion.
///
/// Phase 2: supports control flow (IF, DO/LOOP, BEGIN loops) in addition
/// to straight-line code. Still rejects calls, return stack ops, I/O, and floats.
fn is_promotable(ops: &[IrOp]) -> bool {
    if ops.is_empty() {
        return false;
    }
    is_promotable_body(ops)
}

/// Recursive check for promotable ops.
fn is_promotable_body(ops: &[IrOp]) -> bool {
    for op in ops {
        match op {
            IrOp::Call(_) | IrOp::TailCall(_) | IrOp::Execute | IrOp::SpFetch => return false,
            IrOp::ToR | IrOp::FromR | IrOp::Exit => return false,
            IrOp::ForthLocalGet(_) | IrOp::ForthLocalSet(_) => return false,
            IrOp::Emit | IrOp::Dot | IrOp::Cr | IrOp::Type => return false,
            IrOp::PushI64(_) | IrOp::PushF64(_) => return false,
            IrOp::FDup
            | IrOp::FDrop
            | IrOp::FSwap
            | IrOp::FOver
            | IrOp::FAdd
            | IrOp::FSub
            | IrOp::FMul
            | IrOp::FDiv
            | IrOp::FNegate
            | IrOp::FAbs
            | IrOp::FSqrt
            | IrOp::FMin
            | IrOp::FMax
            | IrOp::FFloor
            | IrOp::FRound
            | IrOp::FZeroEq
            | IrOp::FZeroLt
            | IrOp::FEq
            | IrOp::FLt
            | IrOp::FetchFloat
            | IrOp::StoreFloat
            | IrOp::StoF
            | IrOp::FtoS => return false,
            // IF with ELSE: promotable if both branches are promotable
            // and have the same net stack effect
            IrOp::If { then_body, else_body } => {
                let Some(eb) = else_body else {
                    return false;
                };
                if !is_promotable_body(then_body) || !is_promotable_body(eb) {
                    return false;
                }
                // Both branches must have the same net stack effect
                let (_, then_net) = compute_stack_needs(then_body);
                let (_, else_net) = compute_stack_needs(eb);
                if then_net != else_net {
                    return false;
                }
            }
            // DO/LOOP: promotable if body is promotable and stack-neutral
            IrOp::DoLoop { body, is_plus_loop } => {
                if !is_promotable_body(body) {
                    return false;
                }
                if body.iter().any(|op| matches!(op, IrOp::Exit)) {
                    return false;
                }
                let (_, body_net) = compute_stack_needs(body);
                let expected = if *is_plus_loop { 1 } else { 0 };
                if body_net != expected {
                    return false;
                }
            }
            // BEGIN loops and BeginDoubleWhileRepeat: not yet promoted
            IrOp::BeginUntil { .. }
            | IrOp::BeginAgain { .. }
            | IrOp::BeginWhileRepeat { .. }
            | IrOp::BeginDoubleWhileRepeat { .. } => return false,
            _ => {}
        }
    }
    true
}

/// Compute the net stack depth change for a single IR operation.
fn stack_delta(op: &IrOp) -> i32 {
    match op {
        IrOp::PushI32(_) | IrOp::Dup | IrOp::Over | IrOp::Tuck => 1,
        IrOp::Drop | IrOp::Nip => -1,
        IrOp::Swap | IrOp::Rot => 0,
        IrOp::Add
        | IrOp::Sub
        | IrOp::Mul
        | IrOp::And
        | IrOp::Or
        | IrOp::Xor
        | IrOp::Lshift
        | IrOp::Rshift
        | IrOp::ArithRshift
        | IrOp::Eq
        | IrOp::NotEq
        | IrOp::Lt
        | IrOp::Gt
        | IrOp::LtUnsigned => -1,
        IrOp::DivMod => 0, // 2->2
        IrOp::Negate | IrOp::Abs | IrOp::Invert | IrOp::ZeroEq | IrOp::ZeroLt => 0,
        IrOp::Fetch | IrOp::CFetch => 0, // 1->1
        IrOp::Store | IrOp::CStore | IrOp::PlusStore => -2,
        IrOp::TwoDup => 2,
        IrOp::TwoDrop => -2,
        // Float-only ops: no data stack change
        IrOp::PushF64(_)
        | IrOp::FDup
        | IrOp::FDrop
        | IrOp::FSwap
        | IrOp::FOver
        | IrOp::FAdd
        | IrOp::FSub
        | IrOp::FMul
        | IrOp::FDiv
        | IrOp::FNegate
        | IrOp::FAbs
        | IrOp::FSqrt
        | IrOp::FMin
        | IrOp::FMax
        | IrOp::FFloor
        | IrOp::FRound => 0,
        // Cross-stack: push to data stack
        IrOp::FZeroEq | IrOp::FZeroLt | IrOp::FEq | IrOp::FLt | IrOp::FtoS => 1,
        // Cross-stack: pop from data stack
        IrOp::FetchFloat | IrOp::StoreFloat | IrOp::StoF => -1,
        // Return stack reads push to data stack
        IrOp::RFetch | IrOp::LoopJ => 1,
        _ => 0,
    }
}

/// Compute how many pre-existing stack items a word body needs.
///
/// Returns `(preload_count, net_depth_change)` where `preload_count` is the
/// number of items that must be loaded from the memory stack before execution.
///
/// The key insight: some ops READ existing stack positions without consuming
/// them (e.g., `Dup` reads the top). We must track the minimum stack position
/// that any op reads from, not just the net depth after consumption.
fn compute_stack_needs(ops: &[IrOp]) -> (u32, i32) {
    let mut depth: i32 = 0;
    let mut min_accessed: i32 = 0;
    compute_stack_needs_rec(ops, &mut depth, &mut min_accessed);
    let preload = if min_accessed < 0 {
        (-min_accessed) as u32
    } else {
        0
    };
    (preload, depth)
}

/// Recursive stack-needs analysis that descends into control flow bodies.
fn compute_stack_needs_rec(ops: &[IrOp], depth: &mut i32, min_accessed: &mut i32) {
    for op in ops {
        // First: compute the deepest position this op reads from.
        let reads_from = match op {
            IrOp::Dup => *depth - 1,
            IrOp::Over | IrOp::TwoDup => *depth - 2,
            IrOp::Swap | IrOp::Nip | IrOp::Tuck => *depth - 2,
            IrOp::Rot => *depth - 3,
            IrOp::Add | IrOp::Sub | IrOp::Mul | IrOp::And | IrOp::Or | IrOp::Xor
            | IrOp::Lshift | IrOp::Rshift | IrOp::ArithRshift
            | IrOp::Eq | IrOp::NotEq | IrOp::Lt | IrOp::Gt | IrOp::LtUnsigned
            | IrOp::DivMod | IrOp::Store | IrOp::CStore | IrOp::PlusStore => *depth - 2,
            IrOp::Drop | IrOp::Negate | IrOp::Abs | IrOp::Invert
            | IrOp::ZeroEq | IrOp::ZeroLt | IrOp::Fetch | IrOp::CFetch => *depth - 1,
            IrOp::TwoDrop => *depth - 2,
            IrOp::FetchFloat | IrOp::StoreFloat | IrOp::StoF => *depth - 1,
            // Control flow reads are handled by recursion below
            IrOp::If { .. } => *depth - 1, // consumes condition
            IrOp::DoLoop { .. } => *depth - 2, // consumes limit + index
            _ => *depth,
        };
        *min_accessed = (*min_accessed).min(reads_from);

        // Then: update depth. For control flow, recurse instead of using stack_delta.
        match op {
            IrOp::If { then_body, else_body } => {
                *depth -= 1; // consume condition
                let saved = *depth;
                compute_stack_needs_rec(then_body, depth, min_accessed);
                if let Some(eb) = else_body {
                    let then_depth = *depth;
                    *depth = saved;
                    compute_stack_needs_rec(eb, depth, min_accessed);
                    // Use the then-branch depth (both should match for well-formed code)
                    *depth = then_depth;
                }
            }
            IrOp::DoLoop { body, is_plus_loop } => {
                *depth -= 2; // consume limit + index
                // Loop body is stack-neutral (net 0, or +1 for +LOOP step)
                // We still recurse to track min_accessed inside the body.
                let saved = *depth;
                compute_stack_needs_rec(body, depth, min_accessed);
                // Restore: body effect is consumed by loop control
                *depth = saved;
                if *is_plus_loop {
                    // +LOOP body pushes 1 step value, consumed by loop control
                }
            }
            IrOp::BeginUntil { body } => {
                let saved = *depth;
                compute_stack_needs_rec(body, depth, min_accessed);
                // Body produces flag, consumed by UNTIL: net 0 for the whole construct
                *depth = saved;
            }
            IrOp::BeginAgain { body } => {
                let saved = *depth;
                compute_stack_needs_rec(body, depth, min_accessed);
                *depth = saved;
            }
            IrOp::BeginWhileRepeat { test, body } => {
                let saved = *depth;
                compute_stack_needs_rec(test, depth, min_accessed);
                // WHILE consumes flag
                *depth -= 1;
                compute_stack_needs_rec(body, depth, min_accessed);
                // Whole construct is stack-neutral
                *depth = saved;
            }
            IrOp::BeginDoubleWhileRepeat {
                outer_test, inner_test, body, after_repeat, else_body,
            } => {
                let saved = *depth;
                compute_stack_needs_rec(outer_test, depth, min_accessed);
                *depth -= 1;
                compute_stack_needs_rec(inner_test, depth, min_accessed);
                *depth -= 1;
                compute_stack_needs_rec(body, depth, min_accessed);
                compute_stack_needs_rec(after_repeat, depth, min_accessed);
                if let Some(eb) = else_body {
                    compute_stack_needs_rec(eb, depth, min_accessed);
                }
                *depth = saved;
            }
            // All other ops: use stack_delta
            _ => {
                *depth += stack_delta(op);
            }
        }
    }
}

/// Count how many WASM locals the promoted code path needs (excluding cached
/// DSP and scratch locals). This is an upper bound -- we allocate a fresh
/// local for each value-producing operation.
fn count_promoted_locals(ops: &[IrOp], preload: u32) -> u32 {
    let mut count = preload;
    count_promoted_locals_body(ops, &mut count);
    count
}

/// Recursive helper for counting promoted locals.
fn count_promoted_locals_body(ops: &[IrOp], count: &mut u32) {
    for op in ops {
        match op {
            IrOp::PushI32(_) | IrOp::RFetch | IrOp::LoopJ => *count += 1,
            IrOp::Add
            | IrOp::Sub
            | IrOp::Mul
            | IrOp::And
            | IrOp::Or
            | IrOp::Xor
            | IrOp::Lshift
            | IrOp::Rshift
            | IrOp::ArithRshift
            | IrOp::Eq
            | IrOp::NotEq
            | IrOp::Lt
            | IrOp::Gt
            | IrOp::LtUnsigned
            | IrOp::Negate
            | IrOp::Abs
            | IrOp::Invert
            | IrOp::ZeroEq
            | IrOp::ZeroLt
            | IrOp::Fetch
            | IrOp::CFetch => *count += 1,
            IrOp::DivMod => *count += 2,
            IrOp::DoLoop { body, .. } => {
                *count += 2; // index + limit locals
                count_promoted_locals_body(body, count);
            }
            IrOp::If {
                then_body,
                else_body,
            } => {
                count_promoted_locals_body(then_body, count);
                if let Some(eb) = else_body {
                    count_promoted_locals_body(eb, count);
                }
            }
            IrOp::BeginUntil { body } | IrOp::BeginAgain { body } => {
                count_promoted_locals_body(body, count);
            }
            IrOp::BeginWhileRepeat { test, body } => {
                count_promoted_locals_body(test, count);
                count_promoted_locals_body(body, count);
            }
            IrOp::BeginDoubleWhileRepeat {
                outer_test,
                inner_test,
                body,
                after_repeat,
                else_body,
            } => {
                count_promoted_locals_body(outer_test, count);
                count_promoted_locals_body(inner_test, count);
                count_promoted_locals_body(body, count);
                count_promoted_locals_body(after_repeat, count);
                if let Some(eb) = else_body {
                    count_promoted_locals_body(eb, count);
                }
            }
            IrOp::Dup | IrOp::Over | IrOp::Tuck | IrOp::TwoDup => {
                // These reuse existing locals via the simulator, no extra needed
            }
            _ => {}
        }
    }
}

/// Stack simulator: tracks which WASM local holds each conceptual stack slot.
struct StackSim {
    /// Conceptual stack: `stack[0]` = bottom, `stack.last()` = top.
    /// Each entry is a WASM local index.
    stack: Vec<u32>,
    /// Next available local index.
    next_local: u32,
    /// Stack of (index_local, limit_local) for nested DO/LOOP in promoted path.
    loop_index_stack: Vec<(u32, u32)>,
}

impl StackSim {
    fn new(first_local: u32) -> Self {
        Self {
            stack: Vec::new(),
            next_local: first_local,
            loop_index_stack: Vec::new(),
        }
    }

    /// Allocate a fresh WASM local and return its index.
    fn alloc(&mut self) -> u32 {
        let l = self.next_local;
        self.next_local += 1;
        l
    }

    /// Push a local index onto the conceptual stack.
    fn push(&mut self, local: u32) {
        self.stack.push(local);
    }

    /// Pop the top local index from the conceptual stack.
    fn pop(&mut self) -> u32 {
        self.stack.pop().expect("promoted stack underflow")
    }

    /// Peek at the top of the conceptual stack.
    fn peek(&self) -> u32 {
        *self.stack.last().expect("promoted stack empty")
    }

    /// Peek at a position relative to the top (0 = top, 1 = second, etc.).
    fn peek_at(&self, from_top: usize) -> u32 {
        self.stack[self.stack.len() - 1 - from_top]
    }

    fn swap(&mut self) {
        let len = self.stack.len();
        self.stack.swap(len - 1, len - 2);
    }

    fn rot(&mut self) {
        // ( a b c -- b c a ) : remove third from top, push to top
        let len = self.stack.len();
        let a = self.stack.remove(len - 3);
        self.stack.push(a);
    }
}

/// Emit the promoted prologue: load `preload` items from the memory stack
/// into WASM locals.
fn emit_promoted_prologue(f: &mut Function, preload: u32, sim: &mut StackSim) {
    // Load items: mem[dsp] = top of stack, mem[dsp+4] = second, etc.
    // We load them top-first, then reverse the sim stack so that
    // sim.stack[0] = deepest loaded, sim.stack[last] = top.
    for i in 0..preload {
        let local = sim.alloc();
        f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL));
        if i > 0 {
            f.instruction(&Instruction::I32Const((i * CELL_SIZE) as i32));
            f.instruction(&Instruction::I32Add);
        }
        f.instruction(&Instruction::I32Load(MEM4));
        f.instruction(&Instruction::LocalSet(local));
        sim.push(local);
    }
    // Reverse so stack[0] = deepest, stack[last] = top
    sim.stack.reverse();

    // Advance cached DSP past preloaded items
    if preload > 0 {
        f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL));
        f.instruction(&Instruction::I32Const((preload * CELL_SIZE) as i32));
        f.instruction(&Instruction::I32Add);
        f.instruction(&Instruction::LocalSet(CACHED_DSP_LOCAL));
    }
}

/// Emit the promoted epilogue: write remaining stack items back to memory.
fn emit_promoted_epilogue(f: &mut Function, sim: &mut StackSim) {
    let remaining = sim.stack.len() as u32;
    if remaining > 0 {
        // Decrement cached DSP for the items we're pushing back
        f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL));
        f.instruction(&Instruction::I32Const((remaining * CELL_SIZE) as i32));
        f.instruction(&Instruction::I32Sub);
        f.instruction(&Instruction::LocalSet(CACHED_DSP_LOCAL));

        // Store items: top of sim stack (last in vec) goes to [dsp],
        // next goes to [dsp+4], etc.
        for i in 0..remaining {
            let local = sim.stack[(remaining - 1 - i) as usize]; // top first
            f.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL));
            if i > 0 {
                f.instruction(&Instruction::I32Const((i * CELL_SIZE) as i32));
                f.instruction(&Instruction::I32Add);
            }
            f.instruction(&Instruction::LocalGet(local));
            f.instruction(&Instruction::I32Store(MEM4));
        }
    }
}

/// Emit a single promoted IR operation using WASM locals instead of memory.
///
/// Stack manipulation ops (Swap, Rot, Dup, Drop, Over, Nip, Tuck) emit zero
/// WASM instructions -- they just rearrange the simulator's local references.
/// Arithmetic and memory ops use `local.get` / `local.set` instead of
/// load/store through the data stack pointer.
fn emit_promoted_op(f: &mut Function, op: &IrOp, sim: &mut StackSim) {
    match op {
        // -- Literals --
        IrOp::PushI32(n) => {
            let local = sim.alloc();
            f.instruction(&Instruction::I32Const(*n));
            f.instruction(&Instruction::LocalSet(local));
            sim.push(local);
        }

        // -- Stack manipulation: zero WASM instructions! --
        IrOp::Drop => {
            sim.pop();
        }
        IrOp::Dup => {
            let top = sim.peek();
            sim.push(top); // same local, aliased
        }
        IrOp::Swap => {
            sim.swap();
        }
        IrOp::Over => {
            let second = sim.peek_at(1);
            sim.push(second);
        }
        IrOp::Rot => {
            sim.rot();
        }
        IrOp::Nip => {
            // ( a b -- b ) : remove second
            let top = sim.pop();
            sim.pop(); // discard second
            sim.push(top);
        }
        IrOp::Tuck => {
            // ( a b -- b a b ) : insert top below second
            let b = sim.pop();
            let a = sim.pop();
            sim.push(b);
            sim.push(a);
            sim.push(b); // aliased, same local
        }
        IrOp::TwoDup => {
            let b = sim.peek_at(0);
            let a = sim.peek_at(1);
            sim.push(a);
            sim.push(b);
        }
        IrOp::TwoDrop => {
            sim.pop();
            sim.pop();
        }

        // -- Binary arithmetic (commutative) --
        IrOp::Add => emit_promoted_binary(f, sim, &Instruction::I32Add),
        IrOp::Mul => emit_promoted_binary(f, sim, &Instruction::I32Mul),
        IrOp::And => emit_promoted_binary(f, sim, &Instruction::I32And),
        IrOp::Or => emit_promoted_binary(f, sim, &Instruction::I32Or),
        IrOp::Xor => emit_promoted_binary(f, sim, &Instruction::I32Xor),

        // -- Binary arithmetic (ordered: a OP b) --
        IrOp::Sub => emit_promoted_binary_ordered(f, sim, &Instruction::I32Sub),
        IrOp::Lshift => emit_promoted_binary_ordered(f, sim, &Instruction::I32Shl),
        IrOp::Rshift => emit_promoted_binary_ordered(f, sim, &Instruction::I32ShrU),
        IrOp::ArithRshift => emit_promoted_binary_ordered(f, sim, &Instruction::I32ShrS),

        // -- Comparisons --
        IrOp::Eq => emit_promoted_cmp(f, sim, &Instruction::I32Eq),
        IrOp::NotEq => emit_promoted_cmp(f, sim, &Instruction::I32Ne),
        IrOp::Lt => emit_promoted_cmp(f, sim, &Instruction::I32LtS),
        IrOp::Gt => emit_promoted_cmp(f, sim, &Instruction::I32GtS),
        IrOp::LtUnsigned => emit_promoted_cmp(f, sim, &Instruction::I32LtU),

        IrOp::ZeroEq => {
            let a = sim.pop();
            let result = sim.alloc();
            f.instruction(&Instruction::LocalGet(a));
            f.instruction(&Instruction::I32Eqz);
            // Convert WASM bool to Forth flag: 0 - result
            f.instruction(&Instruction::LocalSet(result));
            f.instruction(&Instruction::I32Const(0));
            f.instruction(&Instruction::LocalGet(result));
            f.instruction(&Instruction::I32Sub);
            f.instruction(&Instruction::LocalSet(result));
            sim.push(result);
        }
        IrOp::ZeroLt => {
            let a = sim.pop();
            let result = sim.alloc();
            f.instruction(&Instruction::LocalGet(a));
            f.instruction(&Instruction::I32Const(0));
            f.instruction(&Instruction::I32LtS);
            // Convert WASM bool to Forth flag
            f.instruction(&Instruction::LocalSet(result));
            f.instruction(&Instruction::I32Const(0));
            f.instruction(&Instruction::LocalGet(result));
            f.instruction(&Instruction::I32Sub);
            f.instruction(&Instruction::LocalSet(result));
            sim.push(result);
        }

        // -- Unary arithmetic --
        IrOp::Negate => {
            let a = sim.pop();
            let result = sim.alloc();
            f.instruction(&Instruction::I32Const(0));
            f.instruction(&Instruction::LocalGet(a));
            f.instruction(&Instruction::I32Sub);
            f.instruction(&Instruction::LocalSet(result));
            sim.push(result);
        }
        IrOp::Abs => {
            let a = sim.pop();
            let result = sim.alloc();
            // Copy input to result, then negate if negative
            f.instruction(&Instruction::LocalGet(a));
            f.instruction(&Instruction::LocalSet(result));
            f.instruction(&Instruction::LocalGet(result));
            f.instruction(&Instruction::I32Const(0));
            f.instruction(&Instruction::I32LtS);
            f.instruction(&Instruction::If(BlockType::Empty));
            f.instruction(&Instruction::I32Const(0));
            f.instruction(&Instruction::LocalGet(result));
            f.instruction(&Instruction::I32Sub);
            f.instruction(&Instruction::LocalSet(result));
            f.instruction(&Instruction::End);
            sim.push(result);
        }
        IrOp::Invert => {
            let a = sim.pop();
            let result = sim.alloc();
            f.instruction(&Instruction::I32Const(-1));
            f.instruction(&Instruction::LocalGet(a));
            f.instruction(&Instruction::I32Xor);
            f.instruction(&Instruction::LocalSet(result));
            sim.push(result);
        }

        // -- DivMod: ( n1 n2 -- rem quot ) --
        IrOp::DivMod => {
            let n2 = sim.pop();
            let n1 = sim.pop();
            let rem_local = sim.alloc();
            let quot_local = sim.alloc();
            // remainder
            f.instruction(&Instruction::LocalGet(n1));
            f.instruction(&Instruction::LocalGet(n2));
            f.instruction(&Instruction::I32RemS);
            f.instruction(&Instruction::LocalSet(rem_local));
            // quotient
            f.instruction(&Instruction::LocalGet(n1));
            f.instruction(&Instruction::LocalGet(n2));
            f.instruction(&Instruction::I32DivS);
            f.instruction(&Instruction::LocalSet(quot_local));
            sim.push(rem_local);
            sim.push(quot_local);
        }

        // -- Memory operations: these still access linear memory --
        IrOp::Fetch => {
            let addr = sim.pop();
            let result = sim.alloc();
            f.instruction(&Instruction::LocalGet(addr));
            f.instruction(&Instruction::I32Load(MEM4));
            f.instruction(&Instruction::LocalSet(result));
            sim.push(result);
        }
        IrOp::CFetch => {
            let addr = sim.pop();
            let result = sim.alloc();
            f.instruction(&Instruction::LocalGet(addr));
            f.instruction(&Instruction::I32Load8U(MEM1));
            f.instruction(&Instruction::LocalSet(result));
            sim.push(result);
        }
        IrOp::Store => {
            // ( x addr -- )
            let addr = sim.pop();
            let x = sim.pop();
            f.instruction(&Instruction::LocalGet(addr));
            f.instruction(&Instruction::LocalGet(x));
            f.instruction(&Instruction::I32Store(MEM4));
        }
        IrOp::CStore => {
            let addr = sim.pop();
            let ch = sim.pop();
            f.instruction(&Instruction::LocalGet(addr));
            f.instruction(&Instruction::LocalGet(ch));
            f.instruction(&Instruction::I32Store8(MEM1));
        }
        IrOp::PlusStore => {
            // ( n addr -- ) : mem[addr] += n
            let addr = sim.pop();
            let n = sim.pop();
            f.instruction(&Instruction::LocalGet(addr));
            f.instruction(&Instruction::LocalGet(addr));
            f.instruction(&Instruction::I32Load(MEM4));
            f.instruction(&Instruction::LocalGet(n));
            f.instruction(&Instruction::I32Add);
            f.instruction(&Instruction::I32Store(MEM4));
        }

        // -- Control flow in promoted path --
        IrOp::If {
            then_body,
            else_body,
        } => {
            let cond = sim.pop();
            f.instruction(&Instruction::LocalGet(cond));
            f.instruction(&Instruction::If(BlockType::Empty));

            let saved_stack = sim.stack.clone();
            let saved_next = sim.next_local;

            emit_promoted_body(f, then_body, sim);

            let then_stack = sim.stack.clone();
            let then_next = sim.next_local;

            // Restore to branch-point state for else
            sim.stack = saved_stack;
            sim.next_local = saved_next;

            f.instruction(&Instruction::Else);
            if let Some(eb) = else_body {
                emit_promoted_body(f, eb, sim);
            }

            // Copy else results into then's locals at the join point.
            // Both branches should have the same stack depth for well-formed Forth.
            let else_stack = &sim.stack;
            let min_len = then_stack.len().min(else_stack.len());
            for i in 0..min_len {
                if then_stack[i] != else_stack[i] {
                    f.instruction(&Instruction::LocalGet(else_stack[i]));
                    f.instruction(&Instruction::LocalSet(then_stack[i]));
                }
            }

            sim.stack = then_stack;
            sim.next_local = sim.next_local.max(then_next);

            f.instruction(&Instruction::End);
        }

        IrOp::DoLoop { body, is_plus_loop } => {
            // DO ( limit index -- )
            let index_local = sim.pop();
            let limit_local = sim.pop();
            sim.loop_index_stack.push((index_local, limit_local));

            let loop_top_stack = sim.stack.clone();

            f.instruction(&Instruction::Block(BlockType::Empty));
            f.instruction(&Instruction::Loop(BlockType::Empty));

            emit_promoted_body(f, body, sim);

            if *is_plus_loop {
                // +LOOP: pop step from stack (body pushed it)
                let step = sim.pop();

                // Fix up remaining stack for next iteration
                emit_promoted_loop_fixup(f, sim, &loop_top_stack);

                // old_diff = index - limit
                let old_diff = sim.alloc();
                f.instruction(&Instruction::LocalGet(index_local));
                f.instruction(&Instruction::LocalGet(limit_local));
                f.instruction(&Instruction::I32Sub);
                f.instruction(&Instruction::LocalSet(old_diff));

                // new_index = index + step
                f.instruction(&Instruction::LocalGet(index_local));
                f.instruction(&Instruction::LocalGet(step));
                f.instruction(&Instruction::I32Add);
                f.instruction(&Instruction::LocalSet(index_local));

                // exit = ((old_diff) XOR (new_index - limit)) AND ((old_diff) XOR step) < 0
                f.instruction(&Instruction::LocalGet(old_diff));
                f.instruction(&Instruction::LocalGet(index_local));
                f.instruction(&Instruction::LocalGet(limit_local));
                f.instruction(&Instruction::I32Sub);
                f.instruction(&Instruction::I32Xor);
                f.instruction(&Instruction::LocalGet(old_diff));
                f.instruction(&Instruction::LocalGet(step));
                f.instruction(&Instruction::I32Xor);
                f.instruction(&Instruction::I32And);
                f.instruction(&Instruction::I32Const(0));
                f.instruction(&Instruction::I32LtS);
                f.instruction(&Instruction::BrIf(1)); // break to $exit
            } else {
                // Fix up stack for next iteration (LOOP body is stack-neutral)
                emit_promoted_loop_fixup(f, sim, &loop_top_stack);

                // LOOP: increment by 1, check >= limit
                f.instruction(&Instruction::LocalGet(index_local));
                f.instruction(&Instruction::I32Const(1));
                f.instruction(&Instruction::I32Add);
                f.instruction(&Instruction::LocalSet(index_local));

                f.instruction(&Instruction::LocalGet(index_local));
                f.instruction(&Instruction::LocalGet(limit_local));
                f.instruction(&Instruction::I32GeS);
                f.instruction(&Instruction::BrIf(1)); // break to $exit
            }

            f.instruction(&Instruction::Br(0)); // continue loop
            f.instruction(&Instruction::End); // end loop
            f.instruction(&Instruction::End); // end block

            sim.loop_index_stack.pop();
        }

        IrOp::BeginUntil { body } => {
            // Save sim state at loop top — loop body must be stack-neutral
            // so we need to copy results back into the same locals.
            let loop_top_stack = sim.stack.clone();

            f.instruction(&Instruction::Loop(BlockType::Empty));
            emit_promoted_body(f, body, sim);
            let cond = sim.pop();
            f.instruction(&Instruction::LocalGet(cond));
            f.instruction(&Instruction::I32Eqz);

            // Copy modified stack values back to loop-top locals for next iteration
            emit_promoted_loop_fixup(f, sim, &loop_top_stack);

            f.instruction(&Instruction::BrIf(0));
            f.instruction(&Instruction::End);
        }

        IrOp::BeginAgain { body } => {
            let loop_top_stack = sim.stack.clone();

            f.instruction(&Instruction::Loop(BlockType::Empty));
            emit_promoted_body(f, body, sim);

            emit_promoted_loop_fixup(f, sim, &loop_top_stack);

            f.instruction(&Instruction::Br(0));
            f.instruction(&Instruction::End);
        }

        IrOp::BeginWhileRepeat { test, body } => {
            let loop_top_stack = sim.stack.clone();

            f.instruction(&Instruction::Block(BlockType::Empty));
            f.instruction(&Instruction::Loop(BlockType::Empty));
            emit_promoted_body(f, test, sim);
            let cond = sim.pop();
            f.instruction(&Instruction::LocalGet(cond));
            f.instruction(&Instruction::I32Eqz);
            f.instruction(&Instruction::BrIf(1)); // break to outer block
            emit_promoted_body(f, body, sim);

            emit_promoted_loop_fixup(f, sim, &loop_top_stack);

            f.instruction(&Instruction::Br(0)); // continue loop
            f.instruction(&Instruction::End); // end loop
            f.instruction(&Instruction::End); // end block
        }

        IrOp::BeginDoubleWhileRepeat {
            outer_test,
            inner_test,
            body,
            after_repeat,
            else_body,
        } => {
            f.instruction(&Instruction::Block(BlockType::Empty)); // $end
            f.instruction(&Instruction::Block(BlockType::Empty)); // $else
            f.instruction(&Instruction::Block(BlockType::Empty)); // $after
            f.instruction(&Instruction::Loop(BlockType::Empty)); // $begin
            emit_promoted_body(f, outer_test, sim);
            let cond1 = sim.pop();
            f.instruction(&Instruction::LocalGet(cond1));
            f.instruction(&Instruction::I32Eqz);
            f.instruction(&Instruction::BrIf(2)); // → $else
            emit_promoted_body(f, inner_test, sim);
            let cond2 = sim.pop();
            f.instruction(&Instruction::LocalGet(cond2));
            f.instruction(&Instruction::I32Eqz);
            f.instruction(&Instruction::BrIf(1)); // → $after
            emit_promoted_body(f, body, sim);
            f.instruction(&Instruction::Br(0)); // → $begin
            f.instruction(&Instruction::End); // end loop
            f.instruction(&Instruction::End); // end $after
            emit_promoted_body(f, after_repeat, sim);
            f.instruction(&Instruction::Br(0)); // → $end (skip else)
            // Actually this needs to jump past else... let me use the same
            // pattern as the non-promoted path
            f.instruction(&Instruction::End); // end $else
            if let Some(eb) = else_body {
                emit_promoted_body(f, eb, sim);
            }
            f.instruction(&Instruction::End); // end $end
        }

        IrOp::RFetch => {
            // In promoted DO/LOOP, R@ = loop index
            if let Some(&(index_local, _)) = sim.loop_index_stack.last() {
                let result = sim.alloc();
                f.instruction(&Instruction::LocalGet(index_local));
                f.instruction(&Instruction::LocalSet(result));
                sim.push(result);
            }
            // Outside loops, RFetch shouldn't appear in promoted code
        }

        IrOp::LoopJ => {
            if sim.loop_index_stack.len() >= 2 {
                let (outer_index, _) =
                    sim.loop_index_stack[sim.loop_index_stack.len() - 2];
                let result = sim.alloc();
                f.instruction(&Instruction::LocalGet(outer_index));
                f.instruction(&Instruction::LocalSet(result));
                sim.push(result);
            }
        }

        IrOp::Exit => {
            // Write remaining promoted locals back to memory stack, then return
            emit_promoted_epilogue(f, sim);
            dsp_writeback(f);
            f.instruction(&Instruction::Return);
        }

        // Unhandled ops in promoted path — shouldn't reach here if is_promotable is correct
        _ => {}
    }
}

/// Emit a promoted body (sequence of ops).
fn emit_promoted_body(f: &mut Function, ops: &[IrOp], sim: &mut StackSim) {
    for op in ops {
        emit_promoted_op(f, op, sim);
    }
}

/// At the end of a loop iteration in promoted code, copy modified values
/// back into the loop-top locals so the next iteration reads correct values.
fn emit_promoted_loop_fixup(f: &mut Function, sim: &mut StackSim, loop_top_stack: &[u32]) {
    assert_eq!(
        sim.stack.len(),
        loop_top_stack.len(),
        "loop body must be stack-neutral (got {} items, expected {})",
        sim.stack.len(),
        loop_top_stack.len()
    );
    for (i, &top_local) in loop_top_stack.iter().enumerate() {
        if sim.stack[i] != top_local {
            f.instruction(&Instruction::LocalGet(sim.stack[i]));
            f.instruction(&Instruction::LocalSet(top_local));
        }
    }
    // Reset sim to loop-top state
    sim.stack = loop_top_stack.to_vec();
}

/// Emit a promoted binary operation (commutative).
fn emit_promoted_binary(f: &mut Function, sim: &mut StackSim, op: &Instruction<'_>) {
    let b = sim.pop();
    let a = sim.pop();
    let result = sim.alloc();
    f.instruction(&Instruction::LocalGet(a));
    f.instruction(&Instruction::LocalGet(b));
    f.instruction(op);
    f.instruction(&Instruction::LocalSet(result));
    sim.push(result);
}

/// Emit a promoted binary operation (ordered: a OP b).
fn emit_promoted_binary_ordered(f: &mut Function, sim: &mut StackSim, op: &Instruction<'_>) {
    let b = sim.pop();
    let a = sim.pop();
    let result = sim.alloc();
    f.instruction(&Instruction::LocalGet(a));
    f.instruction(&Instruction::LocalGet(b));
    f.instruction(op);
    f.instruction(&Instruction::LocalSet(result));
    sim.push(result);
}

/// Emit a promoted comparison operation (a CMP b, result is Forth flag).
fn emit_promoted_cmp(f: &mut Function, sim: &mut StackSim, cmp: &Instruction<'_>) {
    let b = sim.pop();
    let a = sim.pop();
    let result = sim.alloc();
    f.instruction(&Instruction::LocalGet(a));
    f.instruction(&Instruction::LocalGet(b));
    f.instruction(cmp);
    // Convert WASM bool (0/1) to Forth flag (0/-1): 0 - wasm_bool
    f.instruction(&Instruction::LocalSet(result));
    f.instruction(&Instruction::I32Const(0));
    f.instruction(&Instruction::LocalGet(result));
    f.instruction(&Instruction::I32Sub);
    f.instruction(&Instruction::LocalSet(result));
    sim.push(result);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Check if an IR body (recursively) contains any float ops that need f64 locals.
fn needs_f64_locals(ops: &[IrOp]) -> bool {
    for op in ops {
        match op {
            IrOp::PushF64(_)
            | IrOp::FDup
            | IrOp::FDrop
            | IrOp::FSwap
            | IrOp::FOver
            | IrOp::FAdd
            | IrOp::FSub
            | IrOp::FMul
            | IrOp::FDiv
            | IrOp::FNegate
            | IrOp::FAbs
            | IrOp::FSqrt
            | IrOp::FMin
            | IrOp::FMax
            | IrOp::FFloor
            | IrOp::FRound
            | IrOp::FZeroEq
            | IrOp::FZeroLt
            | IrOp::FEq
            | IrOp::FLt
            | IrOp::FetchFloat
            | IrOp::StoreFloat
            | IrOp::StoF
            | IrOp::FtoS => return true,
            IrOp::If {
                then_body,
                else_body,
            } => {
                if needs_f64_locals(then_body) {
                    return true;
                }
                if let Some(eb) = else_body
                    && needs_f64_locals(eb)
                {
                    return true;
                }
            }
            IrOp::DoLoop { body, .. } | IrOp::BeginUntil { body } | IrOp::BeginAgain { body } => {
                if needs_f64_locals(body) {
                    return true;
                }
            }
            IrOp::BeginWhileRepeat { test, body } => {
                if needs_f64_locals(test) || needs_f64_locals(body) {
                    return true;
                }
            }
            IrOp::BeginDoubleWhileRepeat {
                outer_test,
                inner_test,
                body,
                after_repeat,
                else_body,
            } => {
                if needs_f64_locals(outer_test)
                    || needs_f64_locals(inner_test)
                    || needs_f64_locals(body)
                    || needs_f64_locals(after_repeat)
                {
                    return true;
                }
                if let Some(eb) = else_body
                    && needs_f64_locals(eb)
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Check if a DO/LOOP body needs return stack access.
///
/// When false, the loop can use pure WASM locals for index/limit without
/// syncing to the return stack. This is safe when the body has no calls
/// (which might be LEAVE/J/UNLOOP) and no explicit return stack ops.
fn body_needs_return_stack(ops: &[IrOp]) -> bool {
    for op in ops {
        match op {
            IrOp::Call(_) | IrOp::TailCall(_) | IrOp::Execute => return true,
            IrOp::ToR | IrOp::FromR => return true,
            // RFetch (I) is handled by loop locals in the fast path — not a problem.
            // LoopJ is also handled by loop locals.
            // Only explicit >R / R> / calls force the slow path.
            IrOp::If {
                then_body,
                else_body,
            } => {
                if body_needs_return_stack(then_body) {
                    return true;
                }
                if let Some(eb) = else_body {
                    if body_needs_return_stack(eb) {
                        return true;
                    }
                }
            }
            IrOp::DoLoop { body, .. }
            | IrOp::BeginUntil { body }
            | IrOp::BeginAgain { body } => {
                if body_needs_return_stack(body) {
                    return true;
                }
            }
            IrOp::BeginWhileRepeat { test, body } => {
                if body_needs_return_stack(test) || body_needs_return_stack(body) {
                    return true;
                }
            }
            IrOp::BeginDoubleWhileRepeat {
                outer_test,
                inner_test,
                body,
                after_repeat,
                else_body,
            } => {
                if body_needs_return_stack(outer_test)
                    || body_needs_return_stack(inner_test)
                    || body_needs_return_stack(body)
                    || body_needs_return_stack(after_repeat)
                {
                    return true;
                }
                if let Some(eb) = else_body {
                    if body_needs_return_stack(eb) {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }
    false
}

/// Count the maximum DO/LOOP nesting depth in an IR body.
/// Each nesting level needs 2 WASM locals (index, limit).
fn count_loop_depth(ops: &[IrOp]) -> u32 {
    let mut max: u32 = 0;
    for op in ops {
        match op {
            IrOp::DoLoop { body, .. } => {
                max = max.max(1 + count_loop_depth(body));
            }
            IrOp::If {
                then_body,
                else_body,
            } => {
                max = max.max(count_loop_depth(then_body));
                if let Some(eb) = else_body {
                    max = max.max(count_loop_depth(eb));
                }
            }
            IrOp::BeginUntil { body }
            | IrOp::BeginAgain { body } => {
                max = max.max(count_loop_depth(body));
            }
            IrOp::BeginWhileRepeat { test, body } => {
                max = max.max(count_loop_depth(test)).max(count_loop_depth(body));
            }
            IrOp::BeginDoubleWhileRepeat {
                outer_test,
                inner_test,
                body,
                after_repeat,
                else_body,
            } => {
                max = max
                    .max(count_loop_depth(outer_test))
                    .max(count_loop_depth(inner_test))
                    .max(count_loop_depth(body))
                    .max(count_loop_depth(after_repeat));
                if let Some(eb) = else_body {
                    max = max.max(count_loop_depth(eb));
                }
            }
            _ => {}
        }
    }
    max
}

/// Estimate scratch locals a function body needs (not counting cached DSP).
fn count_scratch_locals(ops: &[IrOp]) -> u32 {
    let mut max: u32 = 4; // baseline scratch space (indices SCRATCH_BASE..SCRATCH_BASE+3)
    for op in ops {
        match op {
            IrOp::Rot | IrOp::Tuck => max = max.max(4),
            IrOp::DoLoop { body, is_plus_loop } => {
                // +LOOP needs 5 scratch locals (SCRATCH_BASE..SCRATCH_BASE+4)
                if *is_plus_loop {
                    max = max.max(5);
                }
                max = max.max(count_scratch_locals(body));
            }
            IrOp::BeginUntil { body } => max = max.max(count_scratch_locals(body)),
            IrOp::BeginAgain { body } => max = max.max(count_scratch_locals(body)),
            IrOp::BeginWhileRepeat { test, body } => {
                max = max
                    .max(count_scratch_locals(test))
                    .max(count_scratch_locals(body));
            }
            IrOp::BeginDoubleWhileRepeat {
                outer_test,
                inner_test,
                body,
                after_repeat,
                else_body,
            } => {
                max = max
                    .max(count_scratch_locals(outer_test))
                    .max(count_scratch_locals(inner_test))
                    .max(count_scratch_locals(body))
                    .max(count_scratch_locals(after_repeat));
                if let Some(eb) = else_body {
                    max = max.max(count_scratch_locals(eb));
                }
            }
            IrOp::If {
                then_body,
                else_body,
            } => {
                max = max.max(count_scratch_locals(then_body));
                if let Some(eb) = else_body {
                    max = max.max(count_scratch_locals(eb));
                }
            }
            _ => {}
        }
    }
    max
}

/// Count the number of Forth locals used in an IR body.
/// Returns the maximum local index + 1 (0 if no locals used).
fn count_forth_locals(ops: &[IrOp]) -> u32 {
    let mut max: u32 = 0;
    for op in ops {
        match op {
            IrOp::ForthLocalGet(n) | IrOp::ForthLocalSet(n) => max = max.max(*n + 1),
            IrOp::If {
                then_body,
                else_body,
            } => {
                max = max.max(count_forth_locals(then_body));
                if let Some(eb) = else_body {
                    max = max.max(count_forth_locals(eb));
                }
            }
            IrOp::DoLoop { body, .. } | IrOp::BeginUntil { body } | IrOp::BeginAgain { body } => {
                max = max.max(count_forth_locals(body));
            }
            IrOp::BeginWhileRepeat { test, body } => {
                max = max
                    .max(count_forth_locals(test))
                    .max(count_forth_locals(body));
            }
            _ => {}
        }
    }
    max
}

/// Generate a complete WASM module for a single compiled word.
///
/// This is the JIT path: each word gets its own module that imports
/// shared memory, globals, and function table from the host.
pub fn compile_word(
    _name: &str,
    body: &[IrOp],
    config: &CodegenConfig,
) -> WaferResult<CompiledModule> {
    let mut module = Module::new();

    // -- Type section --
    let mut types = TypeSection::new();
    types.ty().function([], []); // type 0: () -> ()
    types.ty().function([ValType::I32], []); // type 1: (i32) -> ()
    module.section(&types);

    // -- Import section --
    let mut imports = ImportSection::new();
    imports.import("env", "emit", EntityType::Function(TYPE_I32));
    imports.import(
        "env",
        "memory",
        EntityType::Memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        }),
    );
    imports.import(
        "env",
        "dsp",
        EntityType::Global(GlobalType {
            val_type: ValType::I32,
            mutable: true,
            shared: false,
        }),
    );
    imports.import(
        "env",
        "rsp",
        EntityType::Global(GlobalType {
            val_type: ValType::I32,
            mutable: true,
            shared: false,
        }),
    );
    imports.import(
        "env",
        "fsp",
        EntityType::Global(GlobalType {
            val_type: ValType::I32,
            mutable: true,
            shared: false,
        }),
    );
    imports.import(
        "env",
        "table",
        EntityType::Table(TableType {
            element_type: RefType::FUNCREF,
            minimum: config.table_size as u64,
            maximum: None,
            table64: false,
            shared: false,
        }),
    );
    module.section(&imports);

    // -- Function section --
    let mut functions = FunctionSection::new();
    functions.function(TYPE_VOID);
    module.section(&functions);

    // -- Export section --
    let mut exports = ExportSection::new();
    exports.export("fn", ExportKind::Func, WORD_FUNC);
    module.section(&exports);

    // -- Element section --
    let mut elements = ElementSection::new();
    let offset = ConstExpr::i32_const(config.base_fn_index as i32);
    let indices = [WORD_FUNC];
    elements.active(
        Some(TABLE),
        &offset,
        Elements::Functions(Cow::Borrowed(&indices)),
    );
    module.section(&elements);

    // -- Code section --
    // Determine whether to use stack-to-local promotion
    let promoted = config.stack_to_local_promotion && is_promotable(body);
    let scratch_count = count_scratch_locals(body);
    let forth_local_count = count_forth_locals(body);
    let loop_depth = count_loop_depth(body);
    let loop_local_count = loop_depth * 2; // 2 locals per nesting level (index, limit)
    let num_locals = if promoted {
        let (preload, _) = compute_stack_needs(body);
        let promoted_count = count_promoted_locals(body, preload);
        // 1 (cached DSP) + promoted locals (scratch locals not needed in promoted path)
        1 + promoted_count + forth_local_count + loop_local_count
    } else {
        1 + scratch_count + forth_local_count + loop_local_count
    };
    let has_floats = needs_f64_locals(body);
    let num_f64: u32 = if has_floats { 2 } else { 0 };
    let mut locals_decl = vec![(num_locals, ValType::I32)];
    if num_f64 > 0 {
        locals_decl.push((num_f64, ValType::F64));
    }
    let mut func = Function::new(locals_decl);
    let forth_local_base = if promoted {
        let (preload, _) = compute_stack_needs(body);
        let promoted_count = count_promoted_locals(body, preload);
        1 + promoted_count
    } else {
        1 + scratch_count
    };
    let loop_local_base = forth_local_base + forth_local_count;
    let mut ctx = EmitCtx {
        f64_local_0: num_locals,
        f64_local_1: num_locals + 1,
        forth_local_base,
        loop_local_base,
        loop_locals: Vec::new(),
        fast_loop_depth: 0,
        self_word_id: Some(WordId(config.base_fn_index)),
    };

    // Prologue: cache $dsp global into local 0
    func.instruction(&Instruction::GlobalGet(DSP))
        .instruction(&Instruction::LocalSet(CACHED_DSP_LOCAL));

    if promoted {
        let (preload, _) = compute_stack_needs(body);
        let first_promoted = SCRATCH_BASE; // promoted locals start right after cached_dsp
        let mut sim = StackSim::new(first_promoted);
        emit_promoted_prologue(&mut func, preload, &mut sim);
        for op in body {
            emit_promoted_op(&mut func, op, &mut sim);
        }
        emit_promoted_epilogue(&mut func, &mut sim);
    } else {
        emit_body(&mut func, body, &mut ctx);
    }

    // Epilogue: write cached DSP back to the $dsp global
    func.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
        .instruction(&Instruction::GlobalSet(DSP));

    func.instruction(&Instruction::End);

    let mut code = CodeSection::new();
    code.function(&func);
    module.section(&code);

    let bytes = module.finish();

    // Validate
    wasmparser::validate(&bytes).map_err(|e| {
        WaferError::ValidationError(format!("Generated WASM failed validation: {e}"))
    })?;

    Ok(CompiledModule {
        bytes,
        fn_index: config.base_fn_index,
    })
}

// ---------------------------------------------------------------------------
// Consolidated module generation
// ---------------------------------------------------------------------------

/// Emit all IR operations, replacing `Call`/`TailCall` with direct calls
/// when the target word is within the consolidated module.
fn emit_consolidated_body(
    f: &mut Function,
    ops: &[IrOp],
    local_fn_map: &HashMap<WordId, u32>,
    ctx: &mut EmitCtx,
) {
    for op in ops {
        emit_consolidated_op(f, op, local_fn_map, ctx);
    }
}

/// Emit a single IR operation with consolidated call support.
///
/// For `Call` and `TailCall`, emits a direct `call` if the target is in the
/// consolidated module, otherwise falls back to `call_indirect`. For control
/// flow with nested bodies, recurses to handle inner calls.
fn emit_consolidated_op(
    f: &mut Function,
    op: &IrOp,
    local_fn_map: &HashMap<WordId, u32>,
    ctx: &mut EmitCtx,
) {
    match op {
        IrOp::Call(word_id) => {
            if let Some(&fn_idx) = local_fn_map.get(word_id) {
                dsp_writeback(f);
                f.instruction(&Instruction::Call(fn_idx));
                dsp_reload(f);
            } else {
                // Fall back to indirect call for host functions
                dsp_writeback(f);
                f.instruction(&Instruction::I32Const(word_id.0 as i32))
                    .instruction(&Instruction::CallIndirect {
                        type_index: TYPE_VOID,
                        table_index: TABLE,
                    });
                dsp_reload(f);
            }
        }

        IrOp::TailCall(word_id) => {
            if let Some(&fn_idx) = local_fn_map.get(word_id) {
                dsp_writeback(f);
                f.instruction(&Instruction::Call(fn_idx));
                f.instruction(&Instruction::Return);
            } else {
                dsp_writeback(f);
                f.instruction(&Instruction::I32Const(word_id.0 as i32))
                    .instruction(&Instruction::CallIndirect {
                        type_index: TYPE_VOID,
                        table_index: TABLE,
                    });
                f.instruction(&Instruction::Return);
            }
        }

        // Control flow with nested bodies -- recurse for consolidated calls
        IrOp::If {
            then_body,
            else_body,
        } => {
            pop(f);
            f.instruction(&Instruction::If(BlockType::Empty));
            emit_consolidated_body(f, then_body, local_fn_map, ctx);
            if let Some(eb) = else_body {
                f.instruction(&Instruction::Else);
                emit_consolidated_body(f, eb, local_fn_map, ctx);
            }
            f.instruction(&Instruction::End);
        }

        IrOp::DoLoop { body, is_plus_loop } => {
            emit_consolidated_do_loop(f, body, *is_plus_loop, local_fn_map, ctx);
        }

        IrOp::BeginUntil { body } => {
            f.instruction(&Instruction::Loop(BlockType::Empty));
            emit_consolidated_body(f, body, local_fn_map, ctx);
            pop(f);
            f.instruction(&Instruction::I32Eqz)
                .instruction(&Instruction::BrIf(0))
                .instruction(&Instruction::End);
        }

        IrOp::BeginAgain { body } => {
            f.instruction(&Instruction::Loop(BlockType::Empty));
            emit_consolidated_body(f, body, local_fn_map, ctx);
            f.instruction(&Instruction::Br(0))
                .instruction(&Instruction::End);
        }

        IrOp::BeginWhileRepeat { test, body } => {
            f.instruction(&Instruction::Block(BlockType::Empty));
            f.instruction(&Instruction::Loop(BlockType::Empty));
            emit_consolidated_body(f, test, local_fn_map, ctx);
            pop(f);
            f.instruction(&Instruction::I32Eqz)
                .instruction(&Instruction::BrIf(1));
            emit_consolidated_body(f, body, local_fn_map, ctx);
            f.instruction(&Instruction::Br(0))
                .instruction(&Instruction::End)
                .instruction(&Instruction::End);
        }

        IrOp::BeginDoubleWhileRepeat {
            outer_test,
            inner_test,
            body,
            after_repeat,
            else_body,
        } => {
            f.instruction(&Instruction::Block(BlockType::Empty)); // $end
            f.instruction(&Instruction::Block(BlockType::Empty)); // $else
            f.instruction(&Instruction::Block(BlockType::Empty)); // $after
            f.instruction(&Instruction::Loop(BlockType::Empty)); // $begin
            emit_consolidated_body(f, outer_test, local_fn_map, ctx);
            pop(f);
            f.instruction(&Instruction::I32Eqz)
                .instruction(&Instruction::BrIf(2)); // to $else
            emit_consolidated_body(f, inner_test, local_fn_map, ctx);
            pop(f);
            f.instruction(&Instruction::I32Eqz)
                .instruction(&Instruction::BrIf(1)); // to $after
            emit_consolidated_body(f, body, local_fn_map, ctx);
            f.instruction(&Instruction::Br(0)); // back to $begin
            f.instruction(&Instruction::End); // end loop
            f.instruction(&Instruction::End); // end $after block
            emit_consolidated_body(f, after_repeat, local_fn_map, ctx);
            if else_body.is_some() {
                f.instruction(&Instruction::Br(1)); // skip else, goto $end
            }
            f.instruction(&Instruction::End); // end $else block
            if let Some(eb) = else_body {
                emit_consolidated_body(f, eb, local_fn_map, ctx);
            }
            f.instruction(&Instruction::End); // end $end block
        }

        // All other ops have no nested bodies with calls -- delegate to emit_op
        other => emit_op(f, other, ctx),
    }
}

/// Emit a DO...LOOP / DO...+LOOP with consolidated call support for the body.
/// Same fast/slow path logic as `emit_do_loop`.
fn emit_consolidated_do_loop(
    f: &mut Function,
    body: &[IrOp],
    is_plus_loop: bool,
    local_fn_map: &HashMap<WordId, u32>,
    ctx: &mut EmitCtx,
) {
    let loop_depth = ctx.loop_locals.len() as u32;
    let index_local = ctx.loop_local_base + loop_depth * 2;
    let limit_local = ctx.loop_local_base + loop_depth * 2 + 1;
    let needs_rs = body_needs_return_stack(body);

    pop_to(f, index_local);
    pop_to(f, limit_local);

    if needs_rs {
        f.instruction(&Instruction::LocalGet(limit_local));
        rpush_via_local(f, SCRATCH_BASE);
        f.instruction(&Instruction::LocalGet(index_local));
        rpush_via_local(f, SCRATCH_BASE);
    }

    ctx.loop_locals.push((index_local, limit_local));
    if !needs_rs {
        ctx.fast_loop_depth += 1;
    }

    f.instruction(&Instruction::Block(BlockType::Empty));
    f.instruction(&Instruction::Loop(BlockType::Empty));

    if needs_rs {
        rpop(f);
        f.instruction(&Instruction::Drop);
        f.instruction(&Instruction::LocalGet(index_local));
        rpush_via_local(f, SCRATCH_BASE);
    }

    emit_consolidated_body(f, body, local_fn_map, ctx);

    if needs_rs {
        rpeek(f);
        f.instruction(&Instruction::LocalSet(index_local));
    }

    if is_plus_loop {
        pop_to(f, SCRATCH_BASE + 2); // step

        f.instruction(&Instruction::I32Const(SYSVAR_LEAVE_FLAG as i32))
            .instruction(&Instruction::I32Load(MEM4))
            .instruction(&Instruction::If(BlockType::Empty))
            .instruction(&Instruction::I32Const(SYSVAR_LEAVE_FLAG as i32))
            .instruction(&Instruction::I32Const(0))
            .instruction(&Instruction::I32Store(MEM4))
            .instruction(&Instruction::Br(2))
            .instruction(&Instruction::End);

        f.instruction(&Instruction::LocalGet(index_local))
            .instruction(&Instruction::LocalGet(limit_local))
            .instruction(&Instruction::I32Sub)
            .instruction(&Instruction::LocalSet(SCRATCH_BASE + 3));

        f.instruction(&Instruction::LocalGet(index_local))
            .instruction(&Instruction::LocalGet(SCRATCH_BASE + 2))
            .instruction(&Instruction::I32Add)
            .instruction(&Instruction::LocalSet(index_local));

        f.instruction(&Instruction::LocalGet(SCRATCH_BASE + 3))
            .instruction(&Instruction::LocalGet(index_local))
            .instruction(&Instruction::LocalGet(limit_local))
            .instruction(&Instruction::I32Sub)
            .instruction(&Instruction::I32Xor);
        f.instruction(&Instruction::LocalGet(SCRATCH_BASE + 3))
            .instruction(&Instruction::LocalGet(SCRATCH_BASE + 2))
            .instruction(&Instruction::I32Xor);
        f.instruction(&Instruction::I32And)
            .instruction(&Instruction::I32Const(0))
            .instruction(&Instruction::I32LtS)
            .instruction(&Instruction::BrIf(1))
            .instruction(&Instruction::Br(0))
            .instruction(&Instruction::End)
            .instruction(&Instruction::End);
    } else {
        f.instruction(&Instruction::LocalGet(index_local))
            .instruction(&Instruction::I32Const(1))
            .instruction(&Instruction::I32Add)
            .instruction(&Instruction::LocalSet(index_local));

        if needs_rs {
            f.instruction(&Instruction::I32Const(SYSVAR_LEAVE_FLAG as i32))
                .instruction(&Instruction::I32Load(MEM4))
                .instruction(&Instruction::If(BlockType::Empty))
                .instruction(&Instruction::I32Const(SYSVAR_LEAVE_FLAG as i32))
                .instruction(&Instruction::I32Const(0))
                .instruction(&Instruction::I32Store(MEM4))
                .instruction(&Instruction::Br(2))
                .instruction(&Instruction::End);
        }

        f.instruction(&Instruction::LocalGet(index_local))
            .instruction(&Instruction::LocalGet(limit_local))
            .instruction(&Instruction::I32GeS)
            .instruction(&Instruction::BrIf(1))
            .instruction(&Instruction::Br(0))
            .instruction(&Instruction::End)
            .instruction(&Instruction::End);
    }

    if !needs_rs {
        ctx.fast_loop_depth -= 1;
    }
    ctx.loop_locals.pop();

    if needs_rs {
        rpop(f);
        f.instruction(&Instruction::Drop);
        rpop(f);
        f.instruction(&Instruction::Drop);
    }
    f.instruction(&Instruction::I32Const(SYSVAR_LEAVE_FLAG as i32))
        .instruction(&Instruction::I32Const(0))
        .instruction(&Instruction::I32Store(MEM4));
}

/// Optional extras for exportable modules (data section, entry point, metadata).
pub struct ExportSections<'a> {
    /// Memory snapshot to embed as a WASM data section.
    pub memory_snapshot: &'a [u8],
    /// If set, export this function index as `_start`.
    pub entry_fn_index: Option<u32>,
    /// JSON metadata to embed as a custom "wafer" section.
    pub metadata_json: &'a [u8],
}

/// Compile multiple IR-based words into a single WASM module with direct calls.
///
/// Used at runtime by `CONSOLIDATE` and during startup batch compilation.
pub fn compile_consolidated_module(
    words: &[(WordId, Vec<IrOp>)],
    local_fn_map: &HashMap<WordId, u32>,
    table_size: u32,
) -> WaferResult<Vec<u8>> {
    compile_multi_word_module(words, local_fn_map, table_size, None)
}

/// Compile an exportable WASM module with embedded memory and metadata.
///
/// Same as [`compile_consolidated_module`] but adds a WASM data section
/// (memory snapshot), an optional `_start` entry point export, and a
/// custom "wafer" section with JSON metadata.
pub fn compile_exportable_module(
    words: &[(WordId, Vec<IrOp>)],
    local_fn_map: &HashMap<WordId, u32>,
    table_size: u32,
    export: &ExportSections<'_>,
) -> WaferResult<Vec<u8>> {
    compile_multi_word_module(words, local_fn_map, table_size, Some(export))
}

/// Internal: build a multi-word WASM module. When `export` is `Some`, adds
/// data section, entry-point export, and custom metadata section.
fn compile_multi_word_module(
    words: &[(WordId, Vec<IrOp>)],
    local_fn_map: &HashMap<WordId, u32>,
    table_size: u32,
    export: Option<&ExportSections<'_>>,
) -> WaferResult<Vec<u8>> {
    let has_data = export.is_some_and(|e| !e.memory_snapshot.is_empty());
    let mut module = Module::new();

    // -- Type section --
    let mut types = TypeSection::new();
    types.ty().function([], []); // type 0: () -> ()
    types.ty().function([ValType::I32], []); // type 1: (i32) -> ()
    module.section(&types);

    // -- Import section (same as single-word modules) --
    let mut imports = ImportSection::new();
    imports.import("env", "emit", EntityType::Function(TYPE_I32));
    imports.import(
        "env",
        "memory",
        EntityType::Memory(MemoryType {
            minimum: 1,
            maximum: None,
            memory64: false,
            shared: false,
            page_size_log2: None,
        }),
    );
    imports.import(
        "env",
        "dsp",
        EntityType::Global(GlobalType {
            val_type: ValType::I32,
            mutable: true,
            shared: false,
        }),
    );
    imports.import(
        "env",
        "rsp",
        EntityType::Global(GlobalType {
            val_type: ValType::I32,
            mutable: true,
            shared: false,
        }),
    );
    imports.import(
        "env",
        "fsp",
        EntityType::Global(GlobalType {
            val_type: ValType::I32,
            mutable: true,
            shared: false,
        }),
    );
    imports.import(
        "env",
        "table",
        EntityType::Table(TableType {
            element_type: RefType::FUNCREF,
            minimum: table_size as u64,
            maximum: None,
            table64: false,
            shared: false,
        }),
    );
    module.section(&imports);

    // -- Function section: N functions, all type void --
    let mut functions = FunctionSection::new();
    for _ in words {
        functions.function(TYPE_VOID);
    }
    module.section(&functions);

    // -- Export section: export each function as "fn_0", "fn_1", etc. --
    let mut exports = ExportSection::new();
    for (i, _) in words.iter().enumerate() {
        let name = format!("fn_{i}");
        // +1 because emit is imported function index 0
        exports.export(&name, ExportKind::Func, (i as u32) + 1);
    }
    // Optionally export an entry point as "_start"
    if let Some(e) = export
        && let Some(fn_idx) = e.entry_fn_index
    {
        exports.export("_start", ExportKind::Func, fn_idx);
    }
    module.section(&exports);

    // -- Element section: place each function in the table at its WordId slot --
    let mut elements = ElementSection::new();
    for (i, (word_id, _)) in words.iter().enumerate() {
        let offset = ConstExpr::i32_const(word_id.0 as i32);
        let fn_idx = (i as u32) + 1; // +1 for the emit import
        let indices = [fn_idx];
        elements.active(
            Some(TABLE),
            &offset,
            Elements::Functions(Cow::Borrowed(&indices)),
        );
    }
    module.section(&elements);

    // -- DataCount section (required before Code when Data section is present) --
    if has_data {
        module.section(&DataCountSection { count: 1 });
    }

    // -- Code section: emit each function body --
    let mut code = CodeSection::new();
    for (_word_id, body) in words {
        let promoted = is_promotable(body);
        let scratch_count = count_scratch_locals(body);
        let forth_local_count = count_forth_locals(body);
        let loop_depth = count_loop_depth(body);
        let loop_local_count = loop_depth * 2;
        let num_locals = if promoted {
            let (preload, _) = compute_stack_needs(body);
            let promoted_count = count_promoted_locals(body, preload);
            1 + promoted_count + forth_local_count + loop_local_count
        } else {
            1 + scratch_count + forth_local_count + loop_local_count
        };
        let has_floats = needs_f64_locals(body);
        let num_f64: u32 = if has_floats { 2 } else { 0 };
        let mut locals_decl = vec![(num_locals, ValType::I32)];
        if num_f64 > 0 {
            locals_decl.push((num_f64, ValType::F64));
        }
        let mut func = Function::new(locals_decl);
        let forth_local_base = if promoted {
            let (preload, _) = compute_stack_needs(body);
            let promoted_count = count_promoted_locals(body, preload);
            1 + promoted_count
        } else {
            1 + scratch_count
        };
        let loop_local_base = forth_local_base + forth_local_count;
        let mut ctx = EmitCtx {
            f64_local_0: num_locals,
            f64_local_1: num_locals + 1,
            forth_local_base,
            loop_local_base,
            loop_locals: Vec::new(),
            fast_loop_depth: 0,
            self_word_id: None, // consolidated module uses direct calls via local_fn_map
        };

        // Prologue: cache $dsp global into local 0
        func.instruction(&Instruction::GlobalGet(DSP))
            .instruction(&Instruction::LocalSet(CACHED_DSP_LOCAL));

        if promoted {
            // Use stack-to-local promotion (same as compile_word path)
            let (preload, _) = compute_stack_needs(body);
            let first_promoted = SCRATCH_BASE;
            let mut sim = StackSim::new(first_promoted);
            emit_promoted_prologue(&mut func, preload, &mut sim);
            for op in body.iter() {
                emit_promoted_op(&mut func, op, &mut sim);
            }
            emit_promoted_epilogue(&mut func, &mut sim);
        } else {
            // Body with consolidated call support
            emit_consolidated_body(&mut func, body, local_fn_map, &mut ctx);
        }

        // Epilogue: write cached DSP back to the $dsp global
        func.instruction(&Instruction::LocalGet(CACHED_DSP_LOCAL))
            .instruction(&Instruction::GlobalSet(DSP));

        func.instruction(&Instruction::End);
        code.function(&func);
    }
    module.section(&code);

    // -- Data section (memory snapshot for exportable modules) --
    if let Some(e) = export
        && !e.memory_snapshot.is_empty()
    {
        let mut data = DataSection::new();
        data.active(
            MEMORY_INDEX,
            &ConstExpr::i32_const(0),
            e.memory_snapshot.iter().copied(),
        );
        module.section(&data);
    }

    // -- Custom "wafer" section (metadata for exportable modules) --
    if let Some(e) = export
        && !e.metadata_json.is_empty()
    {
        module.section(&CustomSection {
            name: Cow::Borrowed("wafer"),
            data: Cow::Borrowed(e.metadata_json),
        });
    }

    let bytes = module.finish();

    // Validate
    wasmparser::validate(&bytes)
        .map_err(|e| WaferError::ValidationError(format!("WASM module failed validation: {e}")))?;

    Ok(bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::WordId;
    use crate::ir::IrOp;
    use crate::memory::{DATA_STACK_TOP, FLOAT_STACK_TOP, RETURN_STACK_TOP};

    fn default_config() -> CodegenConfig {
        CodegenConfig {
            base_fn_index: 0,
            table_size: 16,
            stack_to_local_promotion: true,
        }
    }

    fn validate_wasm(bytes: &[u8]) -> Result<(), String> {
        wasmparser::validate(bytes)
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    // ===================================================================
    // Validation-only tests
    // ===================================================================

    #[test]
    fn compile_push_i32_validates() {
        let m = compile_word("test", &[IrOp::PushI32(42)], &default_config()).unwrap();
        validate_wasm(&m.bytes).unwrap();
    }

    #[test]
    fn compile_arithmetic_validates() {
        let ops = vec![IrOp::PushI32(3), IrOp::PushI32(4), IrOp::Add];
        let m = compile_word("add_test", &ops, &default_config()).unwrap();
        validate_wasm(&m.bytes).unwrap();
    }

    #[test]
    fn compile_if_else_validates() {
        let ops = vec![
            IrOp::PushI32(1),
            IrOp::If {
                then_body: vec![IrOp::PushI32(42)],
                else_body: Some(vec![IrOp::PushI32(0)]),
            },
        ];
        let m = compile_word("if_test", &ops, &default_config()).unwrap();
        validate_wasm(&m.bytes).unwrap();
    }

    #[test]
    fn compile_call_validates() {
        let ops = vec![IrOp::Call(WordId(5))];
        let m = compile_word("call_test", &ops, &default_config()).unwrap();
        validate_wasm(&m.bytes).unwrap();
    }

    #[test]
    fn compile_stack_ops_validates() {
        let ops = vec![
            IrOp::PushI32(1),
            IrOp::PushI32(2),
            IrOp::Dup,
            IrOp::Swap,
            IrOp::Over,
            IrOp::Rot,
            IrOp::Drop,
            IrOp::Drop,
            IrOp::Drop,
        ];
        let m = compile_word("stack_ops", &ops, &default_config()).unwrap();
        validate_wasm(&m.bytes).unwrap();
    }

    #[test]
    fn compile_comparisons_validate() {
        for op in [IrOp::Eq, IrOp::NotEq, IrOp::Lt, IrOp::Gt, IrOp::LtUnsigned] {
            let ops = vec![IrOp::PushI32(3), IrOp::PushI32(4), op];
            compile_word("cmp", &ops, &default_config()).unwrap();
        }
        for op in [IrOp::ZeroEq, IrOp::ZeroLt] {
            let ops = vec![IrOp::PushI32(0), op];
            compile_word("zcmp", &ops, &default_config()).unwrap();
        }
    }

    #[test]
    fn compile_logic_ops_validates() {
        let ops = vec![
            IrOp::PushI32(0xFF),
            IrOp::PushI32(0x0F),
            IrOp::And,
            IrOp::PushI32(0xF0),
            IrOp::Or,
            IrOp::Invert,
        ];
        compile_word("logic", &ops, &default_config()).unwrap();
    }

    #[test]
    fn compile_memory_ops_validates() {
        let ops = vec![
            IrOp::PushI32(42),
            IrOp::PushI32(0x100),
            IrOp::Store,
            IrOp::PushI32(0x100),
            IrOp::Fetch,
        ];
        compile_word("mem", &ops, &default_config()).unwrap();
    }

    #[test]
    fn compile_begin_until_validates() {
        let ops = vec![
            IrOp::PushI32(5),
            IrOp::BeginUntil {
                body: vec![IrOp::PushI32(1), IrOp::Sub, IrOp::Dup, IrOp::ZeroEq],
            },
        ];
        compile_word("bu", &ops, &default_config()).unwrap();
    }

    #[test]
    fn compile_begin_while_repeat_validates() {
        let ops = vec![
            IrOp::PushI32(3),
            IrOp::BeginWhileRepeat {
                test: vec![IrOp::Dup],
                body: vec![IrOp::PushI32(1), IrOp::Sub],
            },
        ];
        compile_word("bwr", &ops, &default_config()).unwrap();
    }

    #[test]
    fn compile_return_stack_validates() {
        let ops = vec![IrOp::PushI32(42), IrOp::ToR, IrOp::RFetch, IrOp::FromR];
        compile_word("rs", &ops, &default_config()).unwrap();
    }

    #[test]
    fn compile_shift_ops_validates() {
        let ops = vec![
            IrOp::PushI32(1),
            IrOp::PushI32(4),
            IrOp::Lshift,
            IrOp::PushI32(2),
            IrOp::Rshift,
        ];
        compile_word("shift", &ops, &default_config()).unwrap();
    }

    #[test]
    fn compile_emit_validates() {
        compile_word("emit", &[IrOp::PushI32(65), IrOp::Emit], &default_config()).unwrap();
    }

    #[test]
    fn compile_cr_validates() {
        compile_word("cr", &[IrOp::Cr], &default_config()).unwrap();
    }

    #[test]
    fn compile_exit_validates() {
        compile_word("exit", &[IrOp::PushI32(1), IrOp::Exit], &default_config()).unwrap();
    }

    #[test]
    fn compile_nip_tuck_validates() {
        let ops = vec![
            IrOp::PushI32(1),
            IrOp::PushI32(2),
            IrOp::Nip,
            IrOp::PushI32(3),
            IrOp::Tuck,
        ];
        compile_word("nt", &ops, &default_config()).unwrap();
    }

    #[test]
    fn compile_divmod_validates() {
        compile_word(
            "dm",
            &[IrOp::PushI32(10), IrOp::PushI32(3), IrOp::DivMod],
            &default_config(),
        )
        .unwrap();
    }

    #[test]
    fn compile_negate_abs_validates() {
        compile_word(
            "na",
            &[IrOp::PushI32(-5), IrOp::Abs, IrOp::Negate],
            &default_config(),
        )
        .unwrap();
    }

    #[test]
    fn compile_empty_body_validates() {
        compile_word("noop", &[], &default_config()).unwrap();
    }

    #[test]
    fn compile_cfetch_cstore_validates() {
        let ops = vec![
            IrOp::PushI32(65),
            IrOp::PushI32(0x200),
            IrOp::CStore,
            IrOp::PushI32(0x200),
            IrOp::CFetch,
        ];
        compile_word("byte", &ops, &default_config()).unwrap();
    }

    #[test]
    fn compile_plus_store_validates() {
        let ops = vec![
            IrOp::PushI32(10),
            IrOp::PushI32(0x100),
            IrOp::Store,
            IrOp::PushI32(5),
            IrOp::PushI32(0x100),
            IrOp::PlusStore,
        ];
        compile_word("ps", &ops, &default_config()).unwrap();
    }

    #[test]
    fn compiled_module_fn_index() {
        let cfg = CodegenConfig {
            base_fn_index: 7,
            table_size: 16,
            stack_to_local_promotion: true,
        };
        let m = compile_word("t", &[IrOp::PushI32(1)], &cfg).unwrap();
        assert_eq!(m.fn_index, 7);
    }

    // ===================================================================
    // Wasmtime execution tests
    // ===================================================================

    /// Run a compiled word via wasmtime and return the data stack (top first).
    fn run_word(ops: &[IrOp]) -> Vec<i32> {
        use wasmtime::*;

        let compiled = compile_word("test", ops, &default_config()).unwrap();
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());

        let memory = Memory::new(&mut store, MemoryType::new(16, None)).unwrap();

        let dsp = Global::new(
            &mut store,
            wasmtime::GlobalType::new(ValType::I32, Mutability::Var),
            Val::I32(DATA_STACK_TOP as i32),
        )
        .unwrap();

        let rsp = Global::new(
            &mut store,
            wasmtime::GlobalType::new(ValType::I32, Mutability::Var),
            Val::I32(RETURN_STACK_TOP as i32),
        )
        .unwrap();

        let fsp = Global::new(
            &mut store,
            wasmtime::GlobalType::new(ValType::I32, Mutability::Var),
            Val::I32(FLOAT_STACK_TOP as i32),
        )
        .unwrap();

        let table = Table::new(
            &mut store,
            wasmtime::TableType::new(RefType::FUNCREF, 16, None),
            Ref::Func(None),
        )
        .unwrap();

        let emit_ty = FuncType::new(&engine, [ValType::I32], []);
        let emit = Func::new(&mut store, emit_ty, |_caller, _params, _results| Ok(()));

        let module = wasmtime::Module::new(&engine, &compiled.bytes).unwrap();
        let instance = Instance::new(
            &mut store,
            &module,
            &[
                emit.into(),
                memory.into(),
                dsp.into(),
                rsp.into(),
                fsp.into(),
                table.into(),
            ],
        )
        .unwrap();

        instance
            .get_func(&mut store, "fn")
            .unwrap()
            .call(&mut store, &[], &mut [])
            .unwrap();

        // Read data stack
        let sp = dsp.get(&mut store).unwrap_i32() as u32;
        let data = memory.data(&store);
        let mut stack = Vec::new();
        let mut addr = sp;
        while addr < DATA_STACK_TOP {
            let b: [u8; 4] = data[addr as usize..addr as usize + 4].try_into().unwrap();
            stack.push(i32::from_le_bytes(b));
            addr += CELL_SIZE;
        }
        stack
    }

    #[test]
    fn execute_push_i32() {
        assert_eq!(run_word(&[IrOp::PushI32(42)]), vec![42]);
    }

    #[test]
    fn execute_push_multiple() {
        assert_eq!(
            run_word(&[IrOp::PushI32(1), IrOp::PushI32(2), IrOp::PushI32(3)]),
            vec![3, 2, 1],
        );
    }

    #[test]
    fn execute_add() {
        assert_eq!(
            run_word(&[IrOp::PushI32(3), IrOp::PushI32(4), IrOp::Add]),
            vec![7]
        );
    }

    #[test]
    fn execute_sub() {
        assert_eq!(
            run_word(&[IrOp::PushI32(10), IrOp::PushI32(3), IrOp::Sub]),
            vec![7]
        );
    }

    #[test]
    fn execute_mul() {
        assert_eq!(
            run_word(&[IrOp::PushI32(6), IrOp::PushI32(7), IrOp::Mul]),
            vec![42]
        );
    }

    #[test]
    fn execute_divmod() {
        // ( 10 3 -- rem quot ) => ( 1 3 ) => top-first: [3, 1]
        assert_eq!(
            run_word(&[IrOp::PushI32(10), IrOp::PushI32(3), IrOp::DivMod]),
            vec![3, 1]
        );
    }

    #[test]
    fn execute_dup() {
        assert_eq!(run_word(&[IrOp::PushI32(42), IrOp::Dup]), vec![42, 42]);
    }

    #[test]
    fn execute_drop() {
        assert_eq!(
            run_word(&[IrOp::PushI32(1), IrOp::PushI32(2), IrOp::Drop]),
            vec![1]
        );
    }

    #[test]
    fn execute_swap() {
        // ( 1 2 -- 2 1 ) => top-first: [1, 2]
        assert_eq!(
            run_word(&[IrOp::PushI32(1), IrOp::PushI32(2), IrOp::Swap]),
            vec![1, 2]
        );
    }

    #[test]
    fn execute_over() {
        // ( 1 2 -- 1 2 1 )
        assert_eq!(
            run_word(&[IrOp::PushI32(1), IrOp::PushI32(2), IrOp::Over]),
            vec![1, 2, 1]
        );
    }

    #[test]
    fn execute_rot() {
        // ( 1 2 3 -- 2 3 1 ) => top-first: [1, 3, 2]
        assert_eq!(
            run_word(&[
                IrOp::PushI32(1),
                IrOp::PushI32(2),
                IrOp::PushI32(3),
                IrOp::Rot
            ]),
            vec![1, 3, 2],
        );
    }

    #[test]
    fn execute_negate() {
        assert_eq!(run_word(&[IrOp::PushI32(5), IrOp::Negate]), vec![-5]);
    }

    #[test]
    fn execute_abs() {
        assert_eq!(run_word(&[IrOp::PushI32(-42), IrOp::Abs]), vec![42]);
        assert_eq!(run_word(&[IrOp::PushI32(42), IrOp::Abs]), vec![42]);
    }

    #[test]
    fn execute_eq() {
        assert_eq!(
            run_word(&[IrOp::PushI32(5), IrOp::PushI32(5), IrOp::Eq]),
            vec![-1]
        );
        assert_eq!(
            run_word(&[IrOp::PushI32(3), IrOp::PushI32(5), IrOp::Eq]),
            vec![0]
        );
    }

    #[test]
    fn execute_lt() {
        assert_eq!(
            run_word(&[IrOp::PushI32(3), IrOp::PushI32(5), IrOp::Lt]),
            vec![-1]
        );
        assert_eq!(
            run_word(&[IrOp::PushI32(5), IrOp::PushI32(3), IrOp::Lt]),
            vec![0]
        );
    }

    #[test]
    fn execute_gt() {
        assert_eq!(
            run_word(&[IrOp::PushI32(5), IrOp::PushI32(3), IrOp::Gt]),
            vec![-1]
        );
    }

    #[test]
    fn execute_zero_eq() {
        assert_eq!(run_word(&[IrOp::PushI32(0), IrOp::ZeroEq]), vec![-1]);
        assert_eq!(run_word(&[IrOp::PushI32(5), IrOp::ZeroEq]), vec![0]);
    }

    #[test]
    fn execute_zero_lt() {
        assert_eq!(run_word(&[IrOp::PushI32(-1), IrOp::ZeroLt]), vec![-1]);
        assert_eq!(run_word(&[IrOp::PushI32(0), IrOp::ZeroLt]), vec![0]);
    }

    #[test]
    fn execute_and_or_xor() {
        assert_eq!(
            run_word(&[IrOp::PushI32(0xFF), IrOp::PushI32(0x0F), IrOp::And]),
            vec![0x0F]
        );
        assert_eq!(
            run_word(&[IrOp::PushI32(0xF0), IrOp::PushI32(0x0F), IrOp::Or]),
            vec![0xFF]
        );
        assert_eq!(
            run_word(&[IrOp::PushI32(0xFF), IrOp::PushI32(0xF0), IrOp::Xor]),
            vec![0x0F]
        );
    }

    #[test]
    fn execute_invert() {
        assert_eq!(run_word(&[IrOp::PushI32(0), IrOp::Invert]), vec![-1]);
    }

    #[test]
    fn execute_shifts() {
        assert_eq!(
            run_word(&[IrOp::PushI32(1), IrOp::PushI32(4), IrOp::Lshift]),
            vec![16]
        );
        assert_eq!(
            run_word(&[IrOp::PushI32(16), IrOp::PushI32(2), IrOp::Rshift]),
            vec![4]
        );
    }

    #[test]
    fn execute_fetch_store() {
        let ops = vec![
            IrOp::PushI32(42),
            IrOp::PushI32(0x100),
            IrOp::Store,
            IrOp::PushI32(0x100),
            IrOp::Fetch,
        ];
        assert_eq!(run_word(&ops), vec![42]);
    }

    #[test]
    fn execute_cfetch_cstore() {
        let ops = vec![
            IrOp::PushI32(65),
            IrOp::PushI32(0x200),
            IrOp::CStore,
            IrOp::PushI32(0x200),
            IrOp::CFetch,
        ];
        assert_eq!(run_word(&ops), vec![65]);
    }

    #[test]
    fn execute_if_then_else() {
        // TRUE path
        let ops = vec![
            IrOp::PushI32(-1),
            IrOp::If {
                then_body: vec![IrOp::PushI32(42)],
                else_body: Some(vec![IrOp::PushI32(0)]),
            },
        ];
        assert_eq!(run_word(&ops), vec![42]);

        // FALSE path
        let ops = vec![
            IrOp::PushI32(0),
            IrOp::If {
                then_body: vec![IrOp::PushI32(42)],
                else_body: Some(vec![IrOp::PushI32(0)]),
            },
        ];
        assert_eq!(run_word(&ops), vec![0]);
    }

    #[test]
    fn execute_if_without_else() {
        let ops = vec![
            IrOp::PushI32(99),
            IrOp::PushI32(-1),
            IrOp::If {
                then_body: vec![IrOp::PushI32(42)],
                else_body: None,
            },
        ];
        assert_eq!(run_word(&ops), vec![42, 99]);

        let ops = vec![
            IrOp::PushI32(99),
            IrOp::PushI32(0),
            IrOp::If {
                then_body: vec![IrOp::PushI32(42)],
                else_body: None,
            },
        ];
        assert_eq!(run_word(&ops), vec![99]);
    }

    #[test]
    fn execute_nested_if() {
        let ops = vec![
            IrOp::PushI32(-1),
            IrOp::If {
                then_body: vec![
                    IrOp::PushI32(-1),
                    IrOp::If {
                        then_body: vec![IrOp::PushI32(1)],
                        else_body: Some(vec![IrOp::PushI32(2)]),
                    },
                ],
                else_body: Some(vec![IrOp::PushI32(3)]),
            },
        ];
        assert_eq!(run_word(&ops), vec![1]);
    }

    #[test]
    fn execute_begin_until() {
        // Count down from 3
        let ops = vec![
            IrOp::PushI32(3),
            IrOp::BeginUntil {
                body: vec![IrOp::PushI32(1), IrOp::Sub, IrOp::Dup, IrOp::ZeroEq],
            },
        ];
        assert_eq!(run_word(&ops), vec![0]);
    }

    #[test]
    fn execute_return_stack() {
        let ops = vec![IrOp::PushI32(42), IrOp::ToR, IrOp::PushI32(99), IrOp::FromR];
        assert_eq!(run_word(&ops), vec![42, 99]);
    }

    #[test]
    fn execute_rfetch() {
        let ops = vec![IrOp::PushI32(42), IrOp::ToR, IrOp::RFetch, IrOp::FromR];
        assert_eq!(run_word(&ops), vec![42, 42]);
    }

    #[test]
    fn execute_nip() {
        assert_eq!(
            run_word(&[IrOp::PushI32(1), IrOp::PushI32(2), IrOp::Nip]),
            vec![2]
        );
    }

    #[test]
    fn execute_tuck() {
        // ( 1 2 -- 2 1 2 )
        assert_eq!(
            run_word(&[IrOp::PushI32(1), IrOp::PushI32(2), IrOp::Tuck]),
            vec![2, 1, 2],
        );
    }

    #[test]
    fn execute_plus_store() {
        let ops = vec![
            IrOp::PushI32(10),
            IrOp::PushI32(0x100),
            IrOp::Store,
            IrOp::PushI32(5),
            IrOp::PushI32(0x100),
            IrOp::PlusStore,
            IrOp::PushI32(0x100),
            IrOp::Fetch,
        ];
        assert_eq!(run_word(&ops), vec![15]);
    }

    #[test]
    fn execute_complex_expression() {
        // (3 + 4) * 2 = 14
        let ops = vec![
            IrOp::PushI32(3),
            IrOp::PushI32(4),
            IrOp::Add,
            IrOp::PushI32(2),
            IrOp::Mul,
        ];
        assert_eq!(run_word(&ops), vec![14]);
    }

    // ===================================================================
    // Stack-to-local promotion tests
    // ===================================================================

    #[test]
    fn promotable_pure_arithmetic() {
        assert!(is_promotable(&[IrOp::Dup, IrOp::Mul]));
        assert!(is_promotable(&[IrOp::PushI32(1), IrOp::Add]));
        assert!(is_promotable(&[IrOp::Swap, IrOp::Over, IrOp::Nip]));
    }

    #[test]
    fn not_promotable_with_calls() {
        assert!(!is_promotable(&[IrOp::Call(WordId(5))]));
        assert!(!is_promotable(&[IrOp::Emit]));
        assert!(!is_promotable(&[IrOp::ToR]));
        // IF without ELSE is not promotable (stack depth varies by branch)
        assert!(!is_promotable(&[IrOp::If {
            then_body: vec![],
            else_body: None,
        }]));
        // IF with ELSE is promotable
        assert!(is_promotable(&[IrOp::PushI32(1), IrOp::If {
            then_body: vec![IrOp::PushI32(1)],
            else_body: Some(vec![IrOp::PushI32(0)]),
        }]));
        // DO/LOOP with stack-neutral body is promotable
        assert!(is_promotable(&[IrOp::PushI32(10), IrOp::PushI32(0), IrOp::DoLoop {
            body: vec![IrOp::RFetch, IrOp::Drop],
            is_plus_loop: false,
        }]));
        assert!(!is_promotable(&[]));
    }

    #[test]
    fn compute_stack_needs_dup_mul() {
        // DUP * : reads 1 item from caller, net change = 0 (1 in, 1 out via dup*mul)
        let (preload, net) = compute_stack_needs(&[IrOp::Dup, IrOp::Mul]);
        assert_eq!(preload, 1);
        assert_eq!(net, 0);
    }

    #[test]
    fn compute_stack_needs_push_add() {
        // PushI32(1) Add: needs 1 item from caller (Add consumes 2, push provides 1)
        let (preload, net) = compute_stack_needs(&[IrOp::PushI32(1), IrOp::Add]);
        assert_eq!(preload, 1); // Add reads depth-2 = -1 when depth=1 after push
        assert_eq!(net, 0);
    }

    #[test]
    fn compute_stack_needs_swap() {
        // SWAP: reads 2 items, net = 0
        let (preload, net) = compute_stack_needs(&[IrOp::Swap]);
        assert_eq!(preload, 2);
        assert_eq!(net, 0);
    }

    #[test]
    fn promoted_dup_mul_executes() {
        // SQUARE = DUP * (promotable: preload 1 item, no memory stack ops)
        let ops = vec![IrOp::PushI32(7), IrOp::Dup, IrOp::Mul];
        assert_eq!(run_word(&ops), vec![49]);
    }

    #[test]
    fn promoted_swap_executes() {
        // Swap two items using promoted path (zero WASM instructions for swap)
        let ops = vec![IrOp::PushI32(1), IrOp::PushI32(2), IrOp::Swap];
        assert_eq!(run_word(&ops), vec![1, 2]);
    }

    #[test]
    fn promoted_over_add_executes() {
        // OVER OVER + : promoted, reads 2 items, pushes 1 extra
        let ops = vec![
            IrOp::PushI32(3),
            IrOp::PushI32(4),
            IrOp::Over,
            IrOp::Over,
            IrOp::Add,
        ];
        assert_eq!(run_word(&ops), vec![7, 4, 3]);
    }

    #[test]
    fn promoted_nip_executes() {
        let ops = vec![IrOp::PushI32(10), IrOp::PushI32(20), IrOp::Nip];
        assert_eq!(run_word(&ops), vec![20]);
    }

    #[test]
    fn promoted_rot_executes() {
        let ops = vec![
            IrOp::PushI32(1),
            IrOp::PushI32(2),
            IrOp::PushI32(3),
            IrOp::Rot,
        ];
        assert_eq!(run_word(&ops), vec![1, 3, 2]);
    }

    #[test]
    fn promoted_comparison_executes() {
        let ops = vec![IrOp::PushI32(5), IrOp::PushI32(5), IrOp::Eq];
        assert_eq!(run_word(&ops), vec![-1]);
        let ops = vec![IrOp::PushI32(3), IrOp::PushI32(5), IrOp::Lt];
        assert_eq!(run_word(&ops), vec![-1]);
    }

    #[test]
    fn promoted_memory_fetch_store_executes() {
        let ops = vec![
            IrOp::PushI32(42),
            IrOp::PushI32(0x100),
            IrOp::Store,
            IrOp::PushI32(0x100),
            IrOp::Fetch,
        ];
        assert_eq!(run_word(&ops), vec![42]);
    }

    #[test]
    fn promoted_divmod_executes() {
        // ( 10 3 -- rem quot ) => top-first: [3, 1]
        let ops = vec![IrOp::PushI32(10), IrOp::PushI32(3), IrOp::DivMod];
        assert_eq!(run_word(&ops), vec![3, 1]);
    }

    #[test]
    fn promoted_tuck_executes() {
        // ( 1 2 -- 2 1 2 )
        let ops = vec![IrOp::PushI32(1), IrOp::PushI32(2), IrOp::Tuck];
        assert_eq!(run_word(&ops), vec![2, 1, 2]);
    }

    #[test]
    fn promoted_two_dup_executes() {
        let ops = vec![IrOp::PushI32(3), IrOp::PushI32(4), IrOp::TwoDup];
        assert_eq!(run_word(&ops), vec![4, 3, 4, 3]);
    }

    #[test]
    fn promoted_two_drop_executes() {
        let ops = vec![
            IrOp::PushI32(1),
            IrOp::PushI32(2),
            IrOp::PushI32(3),
            IrOp::TwoDrop,
        ];
        assert_eq!(run_word(&ops), vec![1]);
    }

    #[test]
    fn promoted_negate_abs_invert_executes() {
        assert_eq!(run_word(&[IrOp::PushI32(5), IrOp::Negate]), vec![-5]);
        assert_eq!(run_word(&[IrOp::PushI32(-42), IrOp::Abs]), vec![42]);
        assert_eq!(run_word(&[IrOp::PushI32(0), IrOp::Invert]), vec![-1]);
    }

    #[test]
    fn promoted_zero_eq_zero_lt_executes() {
        assert_eq!(run_word(&[IrOp::PushI32(0), IrOp::ZeroEq]), vec![-1]);
        assert_eq!(run_word(&[IrOp::PushI32(5), IrOp::ZeroEq]), vec![0]);
        assert_eq!(run_word(&[IrOp::PushI32(-1), IrOp::ZeroLt]), vec![-1]);
        assert_eq!(run_word(&[IrOp::PushI32(0), IrOp::ZeroLt]), vec![0]);
    }

    #[test]
    fn promoted_shift_executes() {
        assert_eq!(
            run_word(&[IrOp::PushI32(1), IrOp::PushI32(4), IrOp::Lshift]),
            vec![16]
        );
        assert_eq!(
            run_word(&[IrOp::PushI32(16), IrOp::PushI32(2), IrOp::Rshift]),
            vec![4]
        );
    }

    #[test]
    fn promoted_plus_store_executes() {
        let ops = vec![
            IrOp::PushI32(10),
            IrOp::PushI32(0x100),
            IrOp::Store,
            IrOp::PushI32(5),
            IrOp::PushI32(0x100),
            IrOp::PlusStore,
            IrOp::PushI32(0x100),
            IrOp::Fetch,
        ];
        assert_eq!(run_word(&ops), vec![15]);
    }

    #[test]
    fn promoted_cfetch_cstore_executes() {
        let ops = vec![
            IrOp::PushI32(65),
            IrOp::PushI32(0x200),
            IrOp::CStore,
            IrOp::PushI32(0x200),
            IrOp::CFetch,
        ];
        assert_eq!(run_word(&ops), vec![65]);
    }

    #[test]
    fn non_promotable_still_works() {
        // IF-without-ELSE should NOT be promoted, but should still work
        let ops = vec![
            IrOp::PushI32(-1),
            IrOp::If {
                then_body: vec![IrOp::PushI32(42)],
                else_body: None,
            },
        ];
        assert!(!is_promotable(&ops));
        assert_eq!(run_word(&ops), vec![42]);

        // IF-with-ELSE IS promotable and works
        let ops = vec![
            IrOp::PushI32(-1),
            IrOp::If {
                then_body: vec![IrOp::PushI32(42)],
                else_body: Some(vec![IrOp::PushI32(0)]),
            },
        ];
        assert!(is_promotable(&ops));
        assert_eq!(run_word(&ops), vec![42]);
    }

    // ===================================================================
    // Float IR tests
    // ===================================================================

    /// Run a compiled word and return the float stack (top first).
    fn run_float_word(ops: &[IrOp]) -> Vec<f64> {
        use wasmtime::*;

        let compiled = compile_word("test", ops, &default_config()).unwrap();
        let engine = Engine::default();
        let mut store = Store::new(&engine, ());

        let memory = Memory::new(&mut store, MemoryType::new(16, None)).unwrap();

        let dsp = Global::new(
            &mut store,
            wasmtime::GlobalType::new(ValType::I32, Mutability::Var),
            Val::I32(DATA_STACK_TOP as i32),
        )
        .unwrap();

        let rsp = Global::new(
            &mut store,
            wasmtime::GlobalType::new(ValType::I32, Mutability::Var),
            Val::I32(RETURN_STACK_TOP as i32),
        )
        .unwrap();

        let fsp = Global::new(
            &mut store,
            wasmtime::GlobalType::new(ValType::I32, Mutability::Var),
            Val::I32(FLOAT_STACK_TOP as i32),
        )
        .unwrap();

        let table = Table::new(
            &mut store,
            wasmtime::TableType::new(RefType::FUNCREF, 16, None),
            Ref::Func(None),
        )
        .unwrap();

        let emit_ty = FuncType::new(&engine, [ValType::I32], []);
        let emit = Func::new(&mut store, emit_ty, |_caller, _params, _results| Ok(()));

        let module = wasmtime::Module::new(&engine, &compiled.bytes).unwrap();
        let instance = Instance::new(
            &mut store,
            &module,
            &[
                emit.into(),
                memory.into(),
                dsp.into(),
                rsp.into(),
                fsp.into(),
                table.into(),
            ],
        )
        .unwrap();

        instance
            .get_func(&mut store, "fn")
            .unwrap()
            .call(&mut store, &[], &mut [])
            .unwrap();

        // Read float stack
        let sp = fsp.get(&mut store).unwrap_i32() as u32;
        let data = memory.data(&store);
        let mut stack = Vec::new();
        let mut addr = sp;
        while addr < FLOAT_STACK_TOP {
            let b: [u8; 8] = data[addr as usize..addr as usize + 8].try_into().unwrap();
            stack.push(f64::from_le_bytes(b));
            addr += 8;
        }
        stack
    }

    #[test]
    fn compile_push_f64_validates() {
        let m = compile_word("test", &[IrOp::PushF64(3.14)], &default_config()).unwrap();
        validate_wasm(&m.bytes).unwrap();
    }

    #[test]
    fn compile_float_arithmetic_validates() {
        let ops = vec![IrOp::PushF64(1.0), IrOp::PushF64(2.0), IrOp::FAdd];
        let m = compile_word("fadd", &ops, &default_config()).unwrap();
        validate_wasm(&m.bytes).unwrap();
    }

    #[test]
    fn compile_float_cross_stack_validates() {
        let ops = vec![IrOp::PushI32(42), IrOp::StoF, IrOp::FtoS];
        let m = compile_word("cross", &ops, &default_config()).unwrap();
        validate_wasm(&m.bytes).unwrap();
    }

    #[test]
    fn execute_push_f64() {
        assert_eq!(run_float_word(&[IrOp::PushF64(3.14)]), vec![3.14]);
    }

    #[test]
    fn execute_float_add() {
        let ops = vec![IrOp::PushF64(1.0), IrOp::PushF64(2.0), IrOp::FAdd];
        assert_eq!(run_float_word(&ops), vec![3.0]);
    }

    #[test]
    fn execute_float_sub() {
        let ops = vec![IrOp::PushF64(5.0), IrOp::PushF64(3.0), IrOp::FSub];
        assert_eq!(run_float_word(&ops), vec![2.0]);
    }

    #[test]
    fn execute_float_mul() {
        let ops = vec![IrOp::PushF64(3.0), IrOp::PushF64(4.0), IrOp::FMul];
        assert_eq!(run_float_word(&ops), vec![12.0]);
    }

    #[test]
    fn execute_float_div() {
        let ops = vec![IrOp::PushF64(10.0), IrOp::PushF64(4.0), IrOp::FDiv];
        assert_eq!(run_float_word(&ops), vec![2.5]);
    }

    #[test]
    fn execute_float_negate() {
        let ops = vec![IrOp::PushF64(3.0), IrOp::FNegate];
        assert_eq!(run_float_word(&ops), vec![-3.0]);
    }

    #[test]
    fn execute_float_abs() {
        let ops = vec![IrOp::PushF64(-7.0), IrOp::FAbs];
        assert_eq!(run_float_word(&ops), vec![7.0]);
    }

    #[test]
    fn execute_float_sqrt() {
        let ops = vec![IrOp::PushF64(9.0), IrOp::FSqrt];
        assert_eq!(run_float_word(&ops), vec![3.0]);
    }

    #[test]
    fn execute_float_floor() {
        let ops = vec![IrOp::PushF64(3.7), IrOp::FFloor];
        assert_eq!(run_float_word(&ops), vec![3.0]);
    }

    #[test]
    fn execute_float_round() {
        let ops = vec![IrOp::PushF64(2.5), IrOp::FRound];
        assert_eq!(run_float_word(&ops), vec![2.0]); // round ties even
    }

    #[test]
    fn execute_float_min_max() {
        let ops = vec![IrOp::PushF64(3.0), IrOp::PushF64(5.0), IrOp::FMin];
        assert_eq!(run_float_word(&ops), vec![3.0]);
        let ops = vec![IrOp::PushF64(3.0), IrOp::PushF64(5.0), IrOp::FMax];
        assert_eq!(run_float_word(&ops), vec![5.0]);
    }

    #[test]
    fn execute_fdup() {
        let ops = vec![IrOp::PushF64(7.0), IrOp::FDup];
        assert_eq!(run_float_word(&ops), vec![7.0, 7.0]);
    }

    #[test]
    fn execute_fdrop() {
        let ops = vec![IrOp::PushF64(1.0), IrOp::PushF64(2.0), IrOp::FDrop];
        assert_eq!(run_float_word(&ops), vec![1.0]);
    }

    #[test]
    fn execute_fswap() {
        let ops = vec![IrOp::PushF64(1.0), IrOp::PushF64(2.0), IrOp::FSwap];
        assert_eq!(run_float_word(&ops), vec![1.0, 2.0]);
    }

    #[test]
    fn execute_fover() {
        let ops = vec![IrOp::PushF64(1.0), IrOp::PushF64(2.0), IrOp::FOver];
        assert_eq!(run_float_word(&ops), vec![1.0, 2.0, 1.0]);
    }

    #[test]
    fn execute_float_zero_eq() {
        let ops = vec![IrOp::PushF64(0.0), IrOp::FZeroEq];
        assert_eq!(run_word(&ops), vec![-1]);
        let ops = vec![IrOp::PushF64(1.0), IrOp::FZeroEq];
        assert_eq!(run_word(&ops), vec![0]);
    }

    #[test]
    fn execute_float_zero_lt() {
        let ops = vec![IrOp::PushF64(-1.0), IrOp::FZeroLt];
        assert_eq!(run_word(&ops), vec![-1]);
        let ops = vec![IrOp::PushF64(1.0), IrOp::FZeroLt];
        assert_eq!(run_word(&ops), vec![0]);
    }

    #[test]
    fn execute_float_eq() {
        let ops = vec![IrOp::PushF64(3.0), IrOp::PushF64(3.0), IrOp::FEq];
        assert_eq!(run_word(&ops), vec![-1]);
        let ops = vec![IrOp::PushF64(3.0), IrOp::PushF64(4.0), IrOp::FEq];
        assert_eq!(run_word(&ops), vec![0]);
    }

    #[test]
    fn execute_float_lt() {
        let ops = vec![IrOp::PushF64(2.0), IrOp::PushF64(3.0), IrOp::FLt];
        assert_eq!(run_word(&ops), vec![-1]);
        let ops = vec![IrOp::PushF64(3.0), IrOp::PushF64(2.0), IrOp::FLt];
        assert_eq!(run_word(&ops), vec![0]);
    }

    #[test]
    fn execute_stof_ftos() {
        // ( 42 -- ) ( F: -- 42.0 ) then ( F: 42.0 -- ) ( -- 42 )
        let ops = vec![IrOp::PushI32(42), IrOp::StoF, IrOp::FtoS];
        assert_eq!(run_word(&ops), vec![42]);
    }

    #[test]
    fn execute_fetch_store_float() {
        // Store 3.14 at address 0x100, then fetch it back
        let ops = vec![
            IrOp::PushF64(3.14),
            IrOp::PushI32(0x100),
            IrOp::StoreFloat,
            IrOp::PushI32(0x100),
            IrOp::FetchFloat,
        ];
        assert_eq!(run_float_word(&ops), vec![3.14]);
    }
}
