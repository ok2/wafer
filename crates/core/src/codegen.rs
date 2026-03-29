//! WASM code generation from IR.
//!
//! Translates optimized IR into WASM bytecode using the `wasm-encoder` crate.
//! Currently implements **fallback mode**: all stacks live in linear memory
//! and are accessed via globals (`$dsp`, `$rsp`).

use std::borrow::Cow;

use wasm_encoder::{
    BlockType, CodeSection, ConstExpr, ElementSection, Elements, EntityType, ExportKind,
    ExportSection, Function, FunctionSection, GlobalType, ImportSection, Instruction, MemArg,
    MemoryType, Module, RefType, TableType, TypeSection, ValType,
};

use crate::error::{WaferError, WaferResult};
use crate::ir::IrOp;
use crate::memory::CELL_SIZE;

// ---------------------------------------------------------------------------
// Import indices (order matters: imports numbered sequentially by kind)
// ---------------------------------------------------------------------------

/// Index of the imported memory.
const MEMORY_INDEX: u32 = 0;

/// Index of the `$dsp` global (data stack pointer).
const DSP: u32 = 0;

/// Index of the `$rsp` global (return stack pointer).
const RSP: u32 = 1;

/// Index of the imported function table.
const TABLE: u32 = 0;

// Type indices in the type section.
const TYPE_VOID: u32 = 0; // () -> ()
const TYPE_I32: u32 = 1; // (i32) -> ()

// The `emit` callback is the first (and only) imported function, so index 0.
// The compiled word is the first (and only) defined function, so index 1.
const EMIT_FUNC: u32 = 0;
const WORD_FUNC: u32 = 1;

/// Natural-alignment MemArg for 4-byte i32 operations.
const MEM4: MemArg = MemArg {
    offset: 0,
    align: 2, // 2^2 = 4
    memory_index: MEMORY_INDEX,
};

/// MemArg for single-byte operations.
const MEM1: MemArg = MemArg {
    offset: 0,
    align: 0, // 2^0 = 1
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

/// Decrement `$dsp` by CELL_SIZE.
fn dsp_dec(f: &mut Function) {
    f.instruction(&Instruction::GlobalGet(DSP))
        .instruction(&Instruction::I32Const(CELL_SIZE as i32))
        .instruction(&Instruction::I32Sub)
        .instruction(&Instruction::GlobalSet(DSP));
}

/// Increment `$dsp` by CELL_SIZE.
fn dsp_inc(f: &mut Function) {
    f.instruction(&Instruction::GlobalGet(DSP))
        .instruction(&Instruction::I32Const(CELL_SIZE as i32))
        .instruction(&Instruction::I32Add)
        .instruction(&Instruction::GlobalSet(DSP));
}

/// Push an i32 value that is already on the WASM operand stack onto the
/// data stack in linear memory, using `tmp` as a scratch local.
///
/// Sequence: local.set tmp; dsp -= 4; mem[dsp] = local.get tmp
fn push_via_local(f: &mut Function, tmp: u32) {
    f.instruction(&Instruction::LocalSet(tmp));
    dsp_dec(f);
    f.instruction(&Instruction::GlobalGet(DSP))
        .instruction(&Instruction::LocalGet(tmp))
        .instruction(&Instruction::I32Store(MEM4));
}

/// Push a known i32 constant onto the data stack.
fn push_const(f: &mut Function, value: i32) {
    dsp_dec(f);
    f.instruction(&Instruction::GlobalGet(DSP))
        .instruction(&Instruction::I32Const(value))
        .instruction(&Instruction::I32Store(MEM4));
}

/// Pop the top of the data stack onto the WASM operand stack.
fn pop(f: &mut Function) {
    f.instruction(&Instruction::GlobalGet(DSP))
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
    f.instruction(&Instruction::GlobalGet(DSP))
        .instruction(&Instruction::I32Load(MEM4));
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
// IR emission
// ---------------------------------------------------------------------------

/// Emit all IR operations in `ops` into the WASM function body `f`.
fn emit_body(f: &mut Function, ops: &[IrOp]) {
    for op in ops {
        emit_op(f, op);
    }
}

/// Emit a single IR operation.
#[allow(clippy::too_many_lines)]
fn emit_op(f: &mut Function, op: &IrOp) {
    match op {
        // -- Literals -------------------------------------------------------
        IrOp::PushI32(n) => push_const(f, *n),
        IrOp::PushI64(_) | IrOp::PushF64(_) => { /* TODO: double / float stacks */ }

        // -- Stack manipulation ---------------------------------------------
        IrOp::Drop => dsp_inc(f),

        IrOp::Dup => {
            peek(f);
            push_via_local(f, 0);
        }

        IrOp::Swap => {
            // ( a b -- b a )
            pop_to(f, 0); // b
            pop_to(f, 1); // a
            f.instruction(&Instruction::LocalGet(0));
            push_via_local(f, 2);
            f.instruction(&Instruction::LocalGet(1));
            push_via_local(f, 2);
        }

        IrOp::Over => {
            // ( a b -- a b a ) : read second item
            f.instruction(&Instruction::GlobalGet(DSP))
                .instruction(&Instruction::I32Const(CELL_SIZE as i32))
                .instruction(&Instruction::I32Add)
                .instruction(&Instruction::I32Load(MEM4));
            push_via_local(f, 0);
        }

        IrOp::Rot => {
            // ( a b c -- b c a )
            pop_to(f, 0); // c
            pop_to(f, 1); // b
            pop_to(f, 2); // a
            f.instruction(&Instruction::LocalGet(1));
            push_via_local(f, 3);
            f.instruction(&Instruction::LocalGet(0));
            push_via_local(f, 3);
            f.instruction(&Instruction::LocalGet(2));
            push_via_local(f, 3);
        }

        IrOp::Nip => {
            // ( a b -- b )
            pop_to(f, 0); // b
            dsp_inc(f); // drop a
            f.instruction(&Instruction::LocalGet(0));
            push_via_local(f, 1);
        }

        IrOp::Tuck => {
            // ( a b -- b a b )
            pop_to(f, 0); // b
            pop_to(f, 1); // a
            f.instruction(&Instruction::LocalGet(0));
            push_via_local(f, 2);
            f.instruction(&Instruction::LocalGet(1));
            push_via_local(f, 2);
            f.instruction(&Instruction::LocalGet(0));
            push_via_local(f, 2);
        }

        // -- Arithmetic -----------------------------------------------------
        IrOp::Add => emit_binary_commutative(f, &Instruction::I32Add),
        IrOp::Mul => emit_binary_commutative(f, &Instruction::I32Mul),

        IrOp::Sub => {
            // ( a b -- a-b )
            pop_to(f, 0); // b
            pop_to(f, 1); // a
            f.instruction(&Instruction::LocalGet(1))
                .instruction(&Instruction::LocalGet(0))
                .instruction(&Instruction::I32Sub);
            push_via_local(f, 2);
        }

        IrOp::DivMod => {
            // ( n1 n2 -- rem quot )
            pop_to(f, 0); // n2
            pop_to(f, 1); // n1
            // Push remainder first (deeper)
            f.instruction(&Instruction::LocalGet(1))
                .instruction(&Instruction::LocalGet(0))
                .instruction(&Instruction::I32RemS);
            push_via_local(f, 2);
            // Push quotient on top
            f.instruction(&Instruction::LocalGet(1))
                .instruction(&Instruction::LocalGet(0))
                .instruction(&Instruction::I32DivS);
            push_via_local(f, 2);
        }

        IrOp::Negate => {
            pop_to(f, 0);
            f.instruction(&Instruction::I32Const(0))
                .instruction(&Instruction::LocalGet(0))
                .instruction(&Instruction::I32Sub);
            push_via_local(f, 1);
        }

        IrOp::Abs => {
            pop_to(f, 0);
            // if local0 < 0: local0 = 0 - local0
            f.instruction(&Instruction::LocalGet(0))
                .instruction(&Instruction::I32Const(0))
                .instruction(&Instruction::I32LtS)
                .instruction(&Instruction::If(BlockType::Empty))
                .instruction(&Instruction::I32Const(0))
                .instruction(&Instruction::LocalGet(0))
                .instruction(&Instruction::I32Sub)
                .instruction(&Instruction::LocalSet(0))
                .instruction(&Instruction::End);
            f.instruction(&Instruction::LocalGet(0));
            push_via_local(f, 1);
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
            bool_to_forth_flag(f, 0);
            push_via_local(f, 1);
        }

        IrOp::ZeroLt => {
            pop(f);
            f.instruction(&Instruction::I32Const(0))
                .instruction(&Instruction::I32LtS);
            bool_to_forth_flag(f, 0);
            push_via_local(f, 1);
        }

        // -- Logic ----------------------------------------------------------
        IrOp::And => emit_binary_commutative(f, &Instruction::I32And),
        IrOp::Or => emit_binary_commutative(f, &Instruction::I32Or),
        IrOp::Xor => emit_binary_commutative(f, &Instruction::I32Xor),

        IrOp::Invert => {
            pop(f);
            f.instruction(&Instruction::I32Const(-1))
                .instruction(&Instruction::I32Xor);
            push_via_local(f, 0);
        }

        IrOp::Lshift => emit_binary_ordered(f, &Instruction::I32Shl),
        IrOp::Rshift => emit_binary_ordered(f, &Instruction::I32ShrS),

        // -- Memory ---------------------------------------------------------
        IrOp::Fetch => {
            // ( addr -- value )
            pop(f);
            f.instruction(&Instruction::I32Load(MEM4));
            push_via_local(f, 0);
        }

        IrOp::Store => {
            // ( x addr -- )
            pop_to(f, 0); // addr
            pop_to(f, 1); // x
            f.instruction(&Instruction::LocalGet(0))
                .instruction(&Instruction::LocalGet(1))
                .instruction(&Instruction::I32Store(MEM4));
        }

        IrOp::CFetch => {
            pop(f);
            f.instruction(&Instruction::I32Load8U(MEM1));
            push_via_local(f, 0);
        }

        IrOp::CStore => {
            pop_to(f, 0); // addr
            pop_to(f, 1); // char
            f.instruction(&Instruction::LocalGet(0))
                .instruction(&Instruction::LocalGet(1))
                .instruction(&Instruction::I32Store8(MEM1));
        }

        IrOp::PlusStore => {
            // ( n addr -- ) : mem[addr] += n
            pop_to(f, 0); // addr
            pop_to(f, 1); // n
            f.instruction(&Instruction::LocalGet(0))
                .instruction(&Instruction::LocalGet(0))
                .instruction(&Instruction::I32Load(MEM4))
                .instruction(&Instruction::LocalGet(1))
                .instruction(&Instruction::I32Add)
                .instruction(&Instruction::I32Store(MEM4));
        }

        // -- Control flow ---------------------------------------------------
        IrOp::Call(word_id) => {
            f.instruction(&Instruction::I32Const(word_id.0 as i32))
                .instruction(&Instruction::CallIndirect {
                    type_index: TYPE_VOID,
                    table_index: TABLE,
                });
        }

        IrOp::TailCall(word_id) => {
            f.instruction(&Instruction::I32Const(word_id.0 as i32))
                .instruction(&Instruction::CallIndirect {
                    type_index: TYPE_VOID,
                    table_index: TABLE,
                })
                .instruction(&Instruction::Return);
        }

        IrOp::If {
            then_body,
            else_body,
        } => {
            pop(f);
            f.instruction(&Instruction::If(BlockType::Empty));
            emit_body(f, then_body);
            if let Some(eb) = else_body {
                f.instruction(&Instruction::Else);
                emit_body(f, eb);
            }
            f.instruction(&Instruction::End);
        }

        IrOp::DoLoop { body, is_plus_loop } => {
            emit_do_loop(f, body, *is_plus_loop);
        }

        IrOp::BeginUntil { body } => {
            f.instruction(&Instruction::Loop(BlockType::Empty));
            emit_body(f, body);
            pop(f);
            f.instruction(&Instruction::I32Eqz)
                .instruction(&Instruction::BrIf(0))
                .instruction(&Instruction::End);
        }

        IrOp::BeginWhileRepeat { test, body } => {
            f.instruction(&Instruction::Block(BlockType::Empty));
            f.instruction(&Instruction::Loop(BlockType::Empty));
            emit_body(f, test);
            pop(f);
            f.instruction(&Instruction::I32Eqz)
                .instruction(&Instruction::BrIf(1)); // break to outer block
            emit_body(f, body);
            f.instruction(&Instruction::Br(0)) // continue loop
                .instruction(&Instruction::End) // end loop
                .instruction(&Instruction::End); // end block
        }

        IrOp::Exit => {
            f.instruction(&Instruction::Return);
        }

        // -- Return stack ---------------------------------------------------
        IrOp::ToR => {
            pop(f);
            rpush_via_local(f, 0);
        }

        IrOp::FromR => {
            rpop(f);
            push_via_local(f, 0);
        }

        IrOp::RFetch => {
            rpeek(f);
            push_via_local(f, 0);
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
            f.instruction(&Instruction::CallIndirect {
                type_index: TYPE_VOID,
                table_index: TABLE,
            });
        }
    }
}

/// Binary operation where operand order does not matter (commutative).
/// Pops two from data stack, applies `op`, pushes result.
fn emit_binary_commutative(f: &mut Function, op: &Instruction<'_>) {
    pop_to(f, 0); // second operand
    pop_to(f, 1); // first operand
    f.instruction(&Instruction::LocalGet(1))
        .instruction(&Instruction::LocalGet(0))
        .instruction(op);
    push_via_local(f, 2);
}

/// Binary operation where operand order matters: ( a b -- a OP b ).
/// First pops b, then a, pushes a OP b.
fn emit_binary_ordered(f: &mut Function, op: &Instruction<'_>) {
    pop_to(f, 0); // b
    pop_to(f, 1); // a
    f.instruction(&Instruction::LocalGet(1))
        .instruction(&Instruction::LocalGet(0))
        .instruction(op);
    push_via_local(f, 2);
}

/// Comparison: pop two, compare, push Forth flag (-1 or 0).
fn emit_cmp(f: &mut Function, cmp: &Instruction<'_>) {
    pop_to(f, 0); // b
    pop_to(f, 1); // a
    f.instruction(&Instruction::LocalGet(1))
        .instruction(&Instruction::LocalGet(0))
        .instruction(cmp);
    bool_to_forth_flag(f, 2);
    push_via_local(f, 3);
}

/// Emit a DO...LOOP / DO...+LOOP construct.
fn emit_do_loop(f: &mut Function, body: &[IrOp], is_plus_loop: bool) {
    // DO ( limit index -- )
    pop_to(f, 0); // index
    pop_to(f, 1); // limit

    // Push limit then index to return stack
    f.instruction(&Instruction::LocalGet(1));
    rpush_via_local(f, 2);
    f.instruction(&Instruction::LocalGet(0));
    rpush_via_local(f, 2);

    // block $exit
    //   loop $continue
    //     <body>
    //     -- update index, check, branch
    //   end
    // end
    f.instruction(&Instruction::Block(BlockType::Empty));
    f.instruction(&Instruction::Loop(BlockType::Empty));

    emit_body(f, body);

    // Pop current index from return stack
    rpop(f);
    if is_plus_loop {
        f.instruction(&Instruction::LocalSet(0));
        pop_to(f, 2); // increment from data stack
        f.instruction(&Instruction::LocalGet(0))
            .instruction(&Instruction::LocalGet(2))
            .instruction(&Instruction::I32Add)
            .instruction(&Instruction::LocalSet(0));
    } else {
        f.instruction(&Instruction::I32Const(1))
            .instruction(&Instruction::I32Add)
            .instruction(&Instruction::LocalSet(0));
    }

    // Peek limit from return stack
    rpeek(f);
    f.instruction(&Instruction::LocalSet(1));

    // Push updated index back to return stack
    f.instruction(&Instruction::LocalGet(0));
    rpush_via_local(f, 2);

    // if index >= limit, exit
    f.instruction(&Instruction::LocalGet(0))
        .instruction(&Instruction::LocalGet(1))
        .instruction(&Instruction::I32GeS)
        .instruction(&Instruction::BrIf(1)) // break to $exit (block, depth 1)
        .instruction(&Instruction::Br(0)) // continue $continue (loop, depth 0)
        .instruction(&Instruction::End) // end loop
        .instruction(&Instruction::End); // end block

    // Clean up: pop index and limit from return stack
    rpop(f);
    f.instruction(&Instruction::Drop);
    rpop(f);
    f.instruction(&Instruction::Drop);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Estimate how many scratch locals a function body needs.
fn count_needed_locals(ops: &[IrOp]) -> u32 {
    let mut max: u32 = 4; // baseline scratch space
    for op in ops {
        match op {
            IrOp::Rot | IrOp::Tuck => max = max.max(4),
            IrOp::DoLoop { body, .. } => max = max.max(count_needed_locals(body)),
            IrOp::BeginUntil { body } => max = max.max(count_needed_locals(body)),
            IrOp::BeginWhileRepeat { test, body } => {
                max = max
                    .max(count_needed_locals(test))
                    .max(count_needed_locals(body));
            }
            IrOp::If {
                then_body,
                else_body,
            } => {
                max = max.max(count_needed_locals(then_body));
                if let Some(eb) = else_body {
                    max = max.max(count_needed_locals(eb));
                }
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
    let num_locals = count_needed_locals(body);
    let mut func = Function::new(vec![(num_locals, ValType::I32)]);
    emit_body(&mut func, body);
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

/// Generate the core/bootstrap WASM module.
///
/// Not yet implemented -- will be built in a future step.
pub fn compile_core_module(primitives: &[(String, Vec<IrOp>)]) -> WaferResult<Vec<u8>> {
    let _ = primitives;
    Err(WaferError::CodegenError(
        "compile_core_module not yet implemented".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::WordId;
    use crate::ir::IrOp;
    use crate::memory::{DATA_STACK_TOP, RETURN_STACK_TOP};

    fn default_config() -> CodegenConfig {
        CodegenConfig {
            base_fn_index: 0,
            table_size: 16,
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
}
