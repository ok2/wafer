//! Intermediate representation for WAFER's compilation pipeline.
//!
//! The IR sits between parsing/compilation and WASM codegen.
//! Optimization passes transform IR before it reaches codegen.

use crate::dictionary::WordId;

/// A single IR operation.
#[derive(Debug, Clone, PartialEq)]
pub enum IrOp {
    // -- Literals --
    /// Push a 32-bit integer constant.
    PushI32(i32),
    /// Push a 64-bit integer constant (double-cell).
    PushI64(i64),
    /// Push a 64-bit float constant.
    PushF64(f64),

    // -- Stack manipulation --
    Drop,
    Dup,
    Swap,
    Over,
    Rot,
    Nip,
    Tuck,
    /// Two-item duplication: ( a b -- a b a b )
    TwoDup,
    /// Two-item drop: ( a b -- )
    TwoDrop,

    // -- Arithmetic --
    Add,
    Sub,
    Mul,
    /// Combined division and modulus: ( n1 n2 -- rem quot )
    DivMod,
    Negate,
    Abs,

    // -- Comparison --
    Eq,
    NotEq,
    Lt,
    Gt,
    LtUnsigned,
    ZeroEq,
    ZeroLt,

    // -- Logic --
    And,
    Or,
    Xor,
    Invert,
    Lshift,
    Rshift,
    /// Arithmetic (signed) right shift -- used by 2/.
    ArithRshift,

    // -- Memory --
    /// Fetch cell from address: ( addr -- x )
    Fetch,
    /// Store cell to address: ( x addr -- )
    Store,
    /// Fetch byte: ( addr -- char )
    CFetch,
    /// Store byte: ( char addr -- )
    CStore,
    /// Add to cell at address: ( n addr -- )
    PlusStore,

    // -- Control flow --
    /// Call another word.
    Call(WordId),
    /// Tail-call optimization.
    TailCall(WordId),
    /// IF ... ELSE ... THEN
    If {
        then_body: Vec<IrOp>,
        else_body: Option<Vec<IrOp>>,
    },
    /// DO ... LOOP
    DoLoop {
        body: Vec<IrOp>,
        is_plus_loop: bool,
    },
    /// BEGIN ... UNTIL
    BeginUntil {
        body: Vec<IrOp>,
    },
    /// BEGIN ... AGAIN (infinite loop)
    BeginAgain {
        body: Vec<IrOp>,
    },
    /// BEGIN ... WHILE ... REPEAT
    BeginWhileRepeat {
        test: Vec<IrOp>,
        body: Vec<IrOp>,
    },
    /// BEGIN test1 WHILE test2 WHILE body REPEAT `after_repeat` ELSE `else_body` THEN
    ///
    /// Two nested WHILEs in a single BEGIN loop. When the first WHILE fails,
    /// control goes to `else_body`. When the second WHILE fails, control goes
    /// to `after_repeat`. REPEAT jumps back to BEGIN.
    BeginDoubleWhileRepeat {
        outer_test: Vec<IrOp>,
        inner_test: Vec<IrOp>,
        body: Vec<IrOp>,
        after_repeat: Vec<IrOp>,
        else_body: Option<Vec<IrOp>>,
    },
    /// Return from current word.
    Exit,
    /// Conditional restart of enclosing loop (used by CS-PICK'd BEGIN + UNTIL).
    /// Pops flag; if false, restart the loop. Desugared into nested `If` before codegen.
    LoopRestartIfFalse,

    // -- Flat forward branches (for CS-ROLL'd IF/THEN patterns) --
    /// Open a WASM `block`. `BranchIfFalse` can target this to skip to `EndBlock`.
    Block(u32),
    /// Pop flag; if false, branch to matching `EndBlock` with this label.
    BranchIfFalse(u32),
    /// Close the `block` with this label.
    EndBlock(u32),

    // -- Return stack --
    /// Move to return stack: ( x -- ) ( R: -- x )
    ToR,
    /// Move from return stack: ( -- x ) ( R: x -- )
    FromR,
    /// Copy from return stack: ( -- x ) ( R: x -- x )
    RFetch,
    /// Read outer DO/LOOP index (J): ( -- n )
    /// Compiled to local.get when loop locals are available.
    LoopJ,

    // -- Forth locals (from {: ... :} syntax) --
    /// Get Forth local variable N: ( -- x )
    ForthLocalGet(u32),
    /// Set Forth local variable N: ( x -- )
    ForthLocalSet(u32),
    /// Push float-typed Forth local N: ( F: -- r )
    ForthFLocalGet(u32),
    /// Set float-typed Forth local N: ( F: r -- )
    ForthFLocalSet(u32),

    // -- I/O --
    /// Output character: ( char -- )
    Emit,
    /// Print number: ( n -- )
    Dot,
    /// Output newline.
    Cr,
    /// Output string: ( c-addr u -- )
    Type,

    // -- System --
    /// Execute word by function table index: ( xt -- )
    Execute,
    /// Push the current data-stack pointer: ( -- addr )
    SpFetch,

    // -- Float stack manipulation --
    /// Float duplicate: ( F: r -- r r )
    FDup,
    /// Float drop: ( F: r -- )
    FDrop,
    /// Float swap: ( F: r1 r2 -- r2 r1 )
    FSwap,
    /// Float over: ( F: r1 r2 -- r1 r2 r1 )
    FOver,

    // -- Float arithmetic --
    /// Float add: ( F: r1 r2 -- r1+r2 )
    FAdd,
    /// Float subtract: ( F: r1 r2 -- r1-r2 )
    FSub,
    /// Float multiply: ( F: r1 r2 -- r1*r2 )
    FMul,
    /// Float divide: ( F: r1 r2 -- r1/r2 )
    FDiv,
    /// Float negate: ( F: r -- -r )
    FNegate,
    /// Float absolute value: ( F: r -- |r| )
    FAbs,
    /// Float square root: ( F: r -- sqrt(r) )
    FSqrt,
    /// Float minimum: ( F: r1 r2 -- min(r1,r2) )
    FMin,
    /// Float maximum: ( F: r1 r2 -- max(r1,r2) )
    FMax,
    /// Float floor: ( F: r -- floor(r) )
    FFloor,
    /// Float round to nearest even: ( F: r -- round(r) )
    FRound,

    // -- Float comparisons (cross-stack: pop float, push data) --
    /// Float zero equal: ( F: r -- ) ( -- flag )
    FZeroEq,
    /// Float zero less-than: ( F: r -- ) ( -- flag )
    FZeroLt,
    /// Float equal: ( F: r1 r2 -- ) ( -- flag )
    FEq,
    /// Float less-than: ( F: r1 r2 -- ) ( -- flag )
    FLt,

    // -- Float memory (cross-stack) --
    /// Float fetch: ( addr -- ) ( F: -- r )
    FetchFloat,
    /// Float store: ( addr -- ) ( F: r -- )
    StoreFloat,

    // -- Float/integer conversions (cross-stack) --
    /// Single to float: ( n -- ) ( F: -- r )
    StoF,
    /// Float to single: ( F: r -- ) ( -- n )
    FtoS,
}

/// A compiled word definition as IR.
#[derive(Debug, Clone)]
pub struct IrWord {
    /// Word name.
    pub name: String,
    /// The word's body as IR operations.
    pub body: Vec<IrOp>,
    /// Whether this word has the IMMEDIATE flag.
    pub is_immediate: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ir_word_construction() {
        let word = IrWord {
            name: "SQUARE".to_string(),
            body: vec![IrOp::Dup, IrOp::Mul],
            is_immediate: false,
        };
        assert_eq!(word.name, "SQUARE");
        assert_eq!(word.body.len(), 2);
    }

    #[test]
    fn ir_control_flow() {
        // : ABS DUP 0< IF NEGATE THEN ;
        let abs_word = IrWord {
            name: "ABS".to_string(),
            body: vec![
                IrOp::Dup,
                IrOp::ZeroLt,
                IrOp::If {
                    then_body: vec![IrOp::Negate],
                    else_body: None,
                },
            ],
            is_immediate: false,
        };
        assert_eq!(abs_word.body.len(), 3);
    }
}
