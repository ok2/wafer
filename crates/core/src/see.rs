//! IR pretty-printer for `SEE-IR` and the `SEE` fallback path.
//!
//! Renders a post-optimization IR body as indented, one-op-per-line text.
//! Simple ops print as short lowercase mnemonics (Forth glyphs where they
//! are universally recognizable: `@`, `!`, `0=`, `>r`, ...); structured ops
//! print as Forth control words with 2-space indented bodies. Calls resolve
//! `WordId`s to names through an optional resolver so the formatter itself
//! stays independent of the VM.

use crate::dictionary::WordId;
use crate::ir::IrOp;

/// Format an IR body as indented, one-op-per-line text.
pub fn format_ir(ops: &[IrOp]) -> String {
    format_ir_with(ops, &|_| None)
}

/// Like [`format_ir`], resolving `Call`/`TailCall`/`Execute` targets to word
/// names via `resolve`; unresolved ids print as `#N`.
pub fn format_ir_with(ops: &[IrOp], resolve: &dyn Fn(WordId) -> Option<String>) -> String {
    let mut out = String::new();
    write_ops(&mut out, ops, 0, resolve);
    out
}

fn line(out: &mut String, depth: usize, text: &str) {
    for _ in 0..depth {
        out.push_str("  ");
    }
    out.push_str(text);
    out.push('\n');
}

fn callee(id: WordId, resolve: &dyn Fn(WordId) -> Option<String>) -> String {
    resolve(id).unwrap_or_else(|| format!("#{}", id.0))
}

fn write_ops(
    out: &mut String,
    ops: &[IrOp],
    depth: usize,
    resolve: &dyn Fn(WordId) -> Option<String>,
) {
    for op in ops {
        write_op(out, op, depth, resolve);
    }
}

fn write_op(out: &mut String, op: &IrOp, depth: usize, resolve: &dyn Fn(WordId) -> Option<String>) {
    // Exhaustive on purpose: a new IrOp variant must show up here at
    // compile time, not silently render wrong.
    let simple: String = match op {
        // -- Literals --
        IrOp::PushI32(v) => format!("push {v}"),
        IrOp::PushI64(v) => format!("push64 {v}"),
        IrOp::PushF64(v) => format!("fpush {v}"),

        // -- Stack manipulation --
        IrOp::Drop => "drop".into(),
        IrOp::Dup => "dup".into(),
        IrOp::Swap => "swap".into(),
        IrOp::Over => "over".into(),
        IrOp::Rot => "rot".into(),
        IrOp::Nip => "nip".into(),
        IrOp::Tuck => "tuck".into(),
        IrOp::TwoDup => "2dup".into(),
        IrOp::TwoDrop => "2drop".into(),

        // -- Arithmetic --
        IrOp::Add => "add".into(),
        IrOp::Sub => "sub".into(),
        IrOp::Mul => "mul".into(),
        IrOp::DivMod => "divmod".into(),
        IrOp::Negate => "negate".into(),
        IrOp::Abs => "abs".into(),

        // -- Comparison --
        IrOp::Eq => "eq".into(),
        IrOp::NotEq => "ne".into(),
        IrOp::Lt => "lt".into(),
        IrOp::Gt => "gt".into(),
        IrOp::LtUnsigned => "u<".into(),
        IrOp::ZeroEq => "0=".into(),
        IrOp::ZeroLt => "0<".into(),

        // -- Logic --
        IrOp::And => "and".into(),
        IrOp::Or => "or".into(),
        IrOp::Xor => "xor".into(),
        IrOp::Invert => "invert".into(),
        IrOp::Lshift => "lshift".into(),
        IrOp::Rshift => "rshift".into(),
        IrOp::ArithRshift => "arshift".into(),

        // -- Memory --
        IrOp::Fetch => "@".into(),
        IrOp::Store => "!".into(),
        IrOp::CFetch => "c@".into(),
        IrOp::CStore => "c!".into(),
        IrOp::PlusStore => "+!".into(),

        // -- Calls --
        IrOp::Call(id) => format!("call {}", callee(*id, resolve)),
        IrOp::TailCall(id) => format!("tail-call {}", callee(*id, resolve)),

        // -- Structured control flow (multi-line) --
        IrOp::If {
            then_body,
            else_body,
        } => {
            line(out, depth, "if");
            write_ops(out, then_body, depth + 1, resolve);
            if let Some(eb) = else_body {
                line(out, depth, "else");
                write_ops(out, eb, depth + 1, resolve);
            }
            line(out, depth, "then");
            return;
        }
        IrOp::DoLoop { body, is_plus_loop } => {
            line(out, depth, "do");
            write_ops(out, body, depth + 1, resolve);
            line(out, depth, if *is_plus_loop { "+loop" } else { "loop" });
            return;
        }
        IrOp::BeginUntil { body } => {
            line(out, depth, "begin");
            write_ops(out, body, depth + 1, resolve);
            line(out, depth, "until");
            return;
        }
        IrOp::BeginAgain { body } => {
            line(out, depth, "begin");
            write_ops(out, body, depth + 1, resolve);
            line(out, depth, "again");
            return;
        }
        IrOp::BeginWhileRepeat { test, body } => {
            line(out, depth, "begin");
            write_ops(out, test, depth + 1, resolve);
            line(out, depth, "while");
            write_ops(out, body, depth + 1, resolve);
            line(out, depth, "repeat");
            return;
        }
        IrOp::BeginDoubleWhileRepeat {
            outer_test,
            inner_test,
            body,
            after_repeat,
            else_body,
        } => {
            line(out, depth, "begin");
            write_ops(out, outer_test, depth + 1, resolve);
            line(out, depth, "while");
            write_ops(out, inner_test, depth + 1, resolve);
            line(out, depth, "while");
            write_ops(out, body, depth + 1, resolve);
            line(out, depth, "repeat");
            write_ops(out, after_repeat, depth + 1, resolve);
            if let Some(eb) = else_body {
                line(out, depth, "else");
                write_ops(out, eb, depth + 1, resolve);
            }
            line(out, depth, "then");
            return;
        }
        IrOp::Exit => "exit".into(),
        IrOp::LoopRestartIfFalse => "loop-restart-if-false".into(),

        // -- Flat forward branches --
        IrOp::Block(l) => format!("block L{l}"),
        IrOp::BranchIfFalse(l) => format!("branch-if-false L{l}"),
        IrOp::EndBlock(l) => format!("end-block L{l}"),

        // -- Return stack --
        IrOp::ToR => ">r".into(),
        IrOp::FromR => "r>".into(),
        IrOp::RFetch => "r@".into(),
        IrOp::LoopJ => "j".into(),

        // -- Forth locals --
        IrOp::ForthLocalGet(n) => format!("local@ {n}"),
        IrOp::ForthLocalSet(n) => format!("local! {n}"),
        IrOp::ForthFLocalGet(n) => format!("flocal@ {n}"),
        IrOp::ForthFLocalSet(n) => format!("flocal! {n}"),

        // -- I/O --
        IrOp::Emit => "emit".into(),
        IrOp::Dot => ".".into(),
        IrOp::Cr => "cr".into(),
        IrOp::Type => "type".into(),

        // -- System --
        IrOp::Execute => "execute".into(),
        IrOp::SpFetch => "sp@".into(),
        IrOp::RpFetch => "rp@".into(),

        // -- Float stack --
        IrOp::FDup => "fdup".into(),
        IrOp::FDrop => "fdrop".into(),
        IrOp::FSwap => "fswap".into(),
        IrOp::FOver => "fover".into(),

        // -- Float arithmetic --
        IrOp::FAdd => "fadd".into(),
        IrOp::FSub => "fsub".into(),
        IrOp::FMul => "fmul".into(),
        IrOp::FDiv => "fdiv".into(),
        IrOp::FNegate => "fnegate".into(),
        IrOp::FAbs => "fabs".into(),
        IrOp::FSqrt => "fsqrt".into(),
        IrOp::FMin => "fmin".into(),
        IrOp::FMax => "fmax".into(),
        IrOp::FFloor => "ffloor".into(),
        IrOp::FRound => "fround".into(),

        // -- Float comparisons --
        IrOp::FZeroEq => "f0=".into(),
        IrOp::FZeroLt => "f0<".into(),
        IrOp::FEq => "f=".into(),
        IrOp::FLt => "f<".into(),

        // -- Float memory --
        IrOp::FetchFloat => "f@".into(),
        IrOp::StoreFloat => "f!".into(),

        // -- Conversions --
        IrOp::StoF => "s>f".into(),
        IrOp::FtoS => "f>s".into(),
    };
    line(out, depth, &simple);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_ops_one_per_line() {
        let out = format_ir(&[IrOp::Dup, IrOp::Mul, IrOp::PushI32(7)]);
        assert_eq!(out, "dup\nmul\npush 7\n");
    }

    #[test]
    fn call_resolves_via_resolver() {
        let ops = [IrOp::Call(WordId(12)), IrOp::TailCall(WordId(13))];
        assert_eq!(format_ir(&ops), "call #12\ntail-call #13\n");
        let named = format_ir_with(&ops, &|id| (id.0 == 12).then(|| "SQ".to_string()));
        assert_eq!(named, "call SQ\ntail-call #13\n");
    }

    #[test]
    fn nested_if_inside_do_loop_indents() {
        let ops = [IrOp::DoLoop {
            body: vec![
                IrOp::Dup,
                IrOp::If {
                    then_body: vec![IrOp::Dup, IrOp::Mul],
                    else_body: Some(vec![IrOp::Drop]),
                },
            ],
            is_plus_loop: false,
        }];
        let expected = "do\n  dup\n  if\n    dup\n    mul\n  else\n    drop\n  then\nloop\n";
        assert_eq!(format_ir(&ops), expected);
    }

    #[test]
    fn while_loops_and_flat_branches() {
        let ops = [
            IrOp::BeginWhileRepeat {
                test: vec![IrOp::Dup],
                body: vec![IrOp::PushI32(1), IrOp::Sub],
            },
            IrOp::Block(3),
            IrOp::BranchIfFalse(3),
            IrOp::EndBlock(3),
        ];
        let expected = "begin\n  dup\nwhile\n  push 1\n  sub\nrepeat\nblock L3\nbranch-if-false L3\nend-block L3\n";
        assert_eq!(format_ir(&ops), expected);
    }

    #[test]
    fn every_simple_variant_renders() {
        // One of each non-structured op; count of output lines must match.
        let ops = vec![
            IrOp::PushI32(1),
            IrOp::PushI64(2),
            IrOp::PushF64(1.5),
            IrOp::Drop,
            IrOp::Dup,
            IrOp::Swap,
            IrOp::Over,
            IrOp::Rot,
            IrOp::Nip,
            IrOp::Tuck,
            IrOp::TwoDup,
            IrOp::TwoDrop,
            IrOp::Add,
            IrOp::Sub,
            IrOp::Mul,
            IrOp::DivMod,
            IrOp::Negate,
            IrOp::Abs,
            IrOp::Eq,
            IrOp::NotEq,
            IrOp::Lt,
            IrOp::Gt,
            IrOp::LtUnsigned,
            IrOp::ZeroEq,
            IrOp::ZeroLt,
            IrOp::And,
            IrOp::Or,
            IrOp::Xor,
            IrOp::Invert,
            IrOp::Lshift,
            IrOp::Rshift,
            IrOp::ArithRshift,
            IrOp::Fetch,
            IrOp::Store,
            IrOp::CFetch,
            IrOp::CStore,
            IrOp::PlusStore,
            IrOp::Call(WordId(1)),
            IrOp::TailCall(WordId(2)),
            IrOp::Exit,
            IrOp::LoopRestartIfFalse,
            IrOp::Block(1),
            IrOp::BranchIfFalse(1),
            IrOp::EndBlock(1),
            IrOp::ToR,
            IrOp::FromR,
            IrOp::RFetch,
            IrOp::LoopJ,
            IrOp::ForthLocalGet(0),
            IrOp::ForthLocalSet(0),
            IrOp::ForthFLocalGet(0),
            IrOp::ForthFLocalSet(0),
            IrOp::Emit,
            IrOp::Dot,
            IrOp::Cr,
            IrOp::Type,
            IrOp::Execute,
            IrOp::SpFetch,
            IrOp::RpFetch,
            IrOp::FDup,
            IrOp::FDrop,
            IrOp::FSwap,
            IrOp::FOver,
            IrOp::FAdd,
            IrOp::FSub,
            IrOp::FMul,
            IrOp::FDiv,
            IrOp::FNegate,
            IrOp::FAbs,
            IrOp::FSqrt,
            IrOp::FMin,
            IrOp::FMax,
            IrOp::FFloor,
            IrOp::FRound,
            IrOp::FZeroEq,
            IrOp::FZeroLt,
            IrOp::FEq,
            IrOp::FLt,
            IrOp::FetchFloat,
            IrOp::StoreFloat,
            IrOp::StoF,
            IrOp::FtoS,
        ];
        let out = format_ir(&ops);
        assert_eq!(out.lines().count(), ops.len());
        // Every line non-empty, no accidental blank rendering.
        assert!(out.lines().all(|l| !l.trim().is_empty()));
    }
}
