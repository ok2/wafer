//! Optimization passes for WAFER's IR.
//!
//! Each pass is a function `Vec<IrOp> -> Vec<IrOp>`, composable in sequence:
//! 1. Peephole optimization
//! 2. Constant folding
//! 3. Strength reduction
//! 4. Dead code elimination
//! 5. Tail call detection

use std::collections::HashMap;

use crate::dictionary::WordId;
use crate::ir::IrOp;

/// Configuration for the optimization pipeline.
#[derive(Debug, Clone, Default)]
pub struct OptConfig {
    /// Enable peephole optimization patterns.
    pub peephole: bool,
    /// Enable constant folding.
    pub constant_fold: bool,
    /// Enable tail call detection.
    pub tail_call: bool,
    /// Enable strength reduction (e.g., multiply by power of 2 -> shift).
    pub strength_reduce: bool,
    /// Enable dead code elimination.
    pub dce: bool,
    /// Enable inlining of small word bodies.
    pub inline: bool,
    /// Expand a recursive word's base-case guard into its own call sites, so
    /// the leaves of the recursion cost a test instead of a call.
    pub self_guard: bool,
}

/// Run all enabled optimization passes.
pub fn optimize(
    ops: Vec<IrOp>,
    config: &OptConfig,
    bodies: &HashMap<WordId, Vec<IrOp>>,
    self_id: Option<WordId>,
) -> Vec<IrOp> {
    let mut ir = ops;

    // Phase 1: simplify
    if config.peephole {
        ir = peephole(ir);
    }
    if config.constant_fold {
        ir = constant_fold(ir);
    }
    if config.strength_reduce {
        ir = strength_reduce(ir);
    }
    if config.peephole {
        ir = peephole(ir);
    }

    // Phase 2: inline then simplify again
    if config.inline {
        // A caller that can never leave the memory data stack would drag an
        // inlined loop down with it, so leave those callees where they are:
        // as their own word the loop keeps its registers, and one call is far
        // cheaper than a loop's worth of memory traffic.
        //
        // This takes two passes, because before the primitives are substituted
        // the caller is nothing but `Call`s -- `DROP` and `CR` included -- and
        // the promotability check deliberately looks through calls. Asked too
        // early it says "promotable" about almost anything, which is how this
        // guard managed to be a no-op. Inline the loop-free callees first, then
        // ask, then let the loop-bearing ones in if the answer was yes.
        ir = inline(ir, bodies, 8, true);
        let keep_loops_out = !crate::codegen::promotable_modulo_calls(&ir);
        ir = inline(ir, bodies, 8, keep_loops_out);
    }
    if config.self_guard
        && let Some(id) = self_id
    {
        ir = expand_self_guard(ir, id);
    }
    if config.peephole {
        ir = peephole(ir);
    }
    if config.constant_fold {
        ir = constant_fold(ir);
    }
    if config.strength_reduce {
        ir = strength_reduce(ir);
    }
    if config.peephole {
        ir = peephole(ir);
    }

    // Phase 3: eliminate dead code
    if config.dce {
        ir = dce(ir);
    }
    if config.peephole {
        ir = peephole(ir);
    }

    // Phase 4: tail calls (must be last)
    if config.tail_call {
        ir = tail_call_detect(ir);
    }

    ir
}

// ---------------------------------------------------------------------------
// Helper: recurse into control-flow bodies
// ---------------------------------------------------------------------------

/// Apply a pass function to all nested bodies within a control-flow IR op.
fn apply_to_bodies<F: Fn(Vec<IrOp>) -> Vec<IrOp>>(op: IrOp, pass: &F) -> IrOp {
    match op {
        IrOp::If {
            then_body,
            else_body,
        } => IrOp::If {
            then_body: pass(then_body),
            else_body: else_body.map(pass),
        },
        IrOp::DoLoop { body, is_plus_loop } => IrOp::DoLoop {
            body: pass(body),
            is_plus_loop,
        },
        IrOp::BeginUntil { body } => IrOp::BeginUntil { body: pass(body) },
        IrOp::BeginAgain { body } => IrOp::BeginAgain { body: pass(body) },
        IrOp::BeginWhileRepeat { test, body } => IrOp::BeginWhileRepeat {
            test: pass(test),
            body: pass(body),
        },
        IrOp::BeginDoubleWhileRepeat {
            outer_test,
            inner_test,
            body,
            after_repeat,
            else_body,
        } => IrOp::BeginDoubleWhileRepeat {
            outer_test: pass(outer_test),
            inner_test: pass(inner_test),
            body: pass(body),
            after_repeat: pass(after_repeat),
            else_body: else_body.map(pass),
        },
        other => other,
    }
}

// ---------------------------------------------------------------------------
// Pass 1: Peephole optimization
// ---------------------------------------------------------------------------

/// Peephole optimizer: pattern-match adjacent ops and simplify.
fn peephole(ops: Vec<IrOp>) -> Vec<IrOp> {
    let mut ir = ops;
    loop {
        let before_len = ir.len();
        ir = peephole_one_pass(ir);
        if ir.len() == before_len {
            break;
        }
    }
    ir
}

/// Single peephole pass (one sweep through the IR).
fn peephole_one_pass(ops: Vec<IrOp>) -> Vec<IrOp> {
    let mut out: Vec<IrOp> = Vec::with_capacity(ops.len());

    for op in ops {
        // Recurse into control-flow bodies first
        let op = apply_to_bodies(op, &peephole);

        // Try to match the new op against the last item in output
        if let Some(prev) = out.last() {
            match (&prev, &op) {
                // PushI32(n), Drop => remove both
                (IrOp::PushI32(_), IrOp::Drop) => {
                    out.pop();
                    continue;
                }
                // Dup, Drop => remove both
                (IrOp::Dup, IrOp::Drop) => {
                    out.pop();
                    continue;
                }
                // Swap, Swap => remove both
                (IrOp::Swap, IrOp::Swap) => {
                    out.pop();
                    continue;
                }
                // Swap, Drop => Nip
                (IrOp::Swap, IrOp::Drop) => {
                    out.pop();
                    out.push(IrOp::Nip);
                    continue;
                }
                // PushI32(0), Add => identity, remove both
                (IrOp::PushI32(0), IrOp::Add) => {
                    out.pop();
                    continue;
                }
                // PushI32(0), Or => identity, remove both
                (IrOp::PushI32(0), IrOp::Or) => {
                    out.pop();
                    continue;
                }
                // PushI32(-1), And => identity, remove both
                (IrOp::PushI32(-1), IrOp::And) => {
                    out.pop();
                    continue;
                }
                // PushI32(1), Mul => identity, remove both
                (IrOp::PushI32(1), IrOp::Mul) => {
                    out.pop();
                    continue;
                }
                // PushF64, FDrop => remove both
                (IrOp::PushF64(_), IrOp::FDrop) => {
                    out.pop();
                    continue;
                }
                // FDup, FDrop => remove both
                (IrOp::FDup, IrOp::FDrop) => {
                    out.pop();
                    continue;
                }
                // FSwap, FSwap => remove both
                (IrOp::FSwap, IrOp::FSwap) => {
                    out.pop();
                    continue;
                }
                // FNegate, FNegate => remove both
                (IrOp::FNegate, IrOp::FNegate) => {
                    out.pop();
                    continue;
                }
                // Over, Over => TwoDup
                (IrOp::Over, IrOp::Over) => {
                    out.pop();
                    out.push(IrOp::TwoDup);
                    continue;
                }
                // Drop, Drop => TwoDrop
                (IrOp::Drop, IrOp::Drop) => {
                    out.pop();
                    out.push(IrOp::TwoDrop);
                    continue;
                }
                _ => {}
            }
        }
        out.push(op);
    }
    out
}

// ---------------------------------------------------------------------------
// Pass 2: Constant folding
// ---------------------------------------------------------------------------

/// Constant folder: evaluate operations on known constants at compile time.
fn constant_fold(ops: Vec<IrOp>) -> Vec<IrOp> {
    let mut out: Vec<IrOp> = Vec::with_capacity(ops.len());

    for op in ops {
        // Recurse into control-flow bodies
        let op = apply_to_bodies(op, &constant_fold);

        // Try binary fold: last two outputs are PushI32, current op is foldable
        if out.len() >= 2
            && let Some(result) = try_binary_fold(&out[out.len() - 2], &out[out.len() - 1], &op)
        {
            out.pop();
            out.pop();
            out.push(IrOp::PushI32(result));
            continue;
        }

        // Try float binary fold: last two outputs are PushF64
        if out.len() >= 2
            && let Some(result) =
                try_float_binary_fold(&out[out.len() - 2], &out[out.len() - 1], &op)
        {
            out.pop();
            out.pop();
            out.push(IrOp::PushF64(result));
            continue;
        }

        // Try unary fold: last output is PushI32, current op is foldable
        if !out.is_empty()
            && let Some(result) = try_unary_fold(&out[out.len() - 1], &op)
        {
            out.pop();
            out.push(IrOp::PushI32(result));
            continue;
        }

        // Try float unary fold: last output is PushF64
        if !out.is_empty()
            && let Some(result) = try_float_unary_fold(&out[out.len() - 1], &op)
        {
            out.pop();
            out.push(IrOp::PushF64(result));
            continue;
        }

        out.push(op);
    }
    out
}

/// Try to fold a binary operation on two constants.
fn try_binary_fold(a_op: &IrOp, b_op: &IrOp, op: &IrOp) -> Option<i32> {
    let (a, b) = match (a_op, b_op) {
        (IrOp::PushI32(a), IrOp::PushI32(b)) => (*a, *b),
        _ => return None,
    };

    match op {
        IrOp::Add => Some(a.wrapping_add(b)),
        IrOp::Sub => Some(a.wrapping_sub(b)),
        IrOp::Mul => Some(a.wrapping_mul(b)),
        IrOp::And => Some(a & b),
        IrOp::Or => Some(a | b),
        IrOp::Xor => Some(a ^ b),
        IrOp::Lshift => {
            if (0..32).contains(&b) {
                Some(a.wrapping_shl(b as u32))
            } else {
                None
            }
        }
        IrOp::Rshift => {
            if (0..32).contains(&b) {
                Some((a as u32).wrapping_shr(b as u32) as i32)
            } else {
                None
            }
        }
        IrOp::ArithRshift => {
            if (0..32).contains(&b) {
                Some(a.wrapping_shr(b as u32))
            } else {
                None
            }
        }
        IrOp::Eq => Some(if a == b { -1 } else { 0 }),
        IrOp::NotEq => Some(if a != b { -1 } else { 0 }),
        IrOp::Lt => Some(if a < b { -1 } else { 0 }),
        IrOp::Gt => Some(if a > b { -1 } else { 0 }),
        IrOp::LtUnsigned => Some(if (a as u32) < (b as u32) { -1 } else { 0 }),
        _ => None,
    }
}

/// Try to fold a unary operation on a constant.
fn try_unary_fold(n_op: &IrOp, op: &IrOp) -> Option<i32> {
    let n = match n_op {
        IrOp::PushI32(n) => *n,
        _ => return None,
    };

    match op {
        IrOp::Negate => Some(n.wrapping_neg()),
        IrOp::Abs => {
            if n == i32::MIN {
                Some(i32::MIN)
            } else {
                Some(n.abs())
            }
        }
        IrOp::Invert => Some(!n),
        IrOp::ZeroEq => Some(if n == 0 { -1 } else { 0 }),
        IrOp::ZeroLt => Some(if n < 0 { -1 } else { 0 }),
        _ => None,
    }
}

/// Try to fold a binary float operation on two constants.
fn try_float_binary_fold(a_op: &IrOp, b_op: &IrOp, op: &IrOp) -> Option<f64> {
    let (a, b) = match (a_op, b_op) {
        (IrOp::PushF64(a), IrOp::PushF64(b)) => (*a, *b),
        _ => return None,
    };

    match op {
        IrOp::FAdd => Some(a + b),
        IrOp::FSub => Some(a - b),
        IrOp::FMul => Some(a * b),
        IrOp::FDiv => {
            if b != 0.0 {
                Some(a / b)
            } else {
                None
            }
        }
        IrOp::FMin => Some(a.min(b)),
        IrOp::FMax => Some(a.max(b)),
        _ => None,
    }
}

/// Try to fold a unary float operation on a constant.
fn try_float_unary_fold(n_op: &IrOp, op: &IrOp) -> Option<f64> {
    let n = match n_op {
        IrOp::PushF64(n) => *n,
        _ => return None,
    };

    match op {
        IrOp::FNegate => Some(-n),
        IrOp::FAbs => Some(n.abs()),
        IrOp::FSqrt => {
            if n >= 0.0 {
                Some(n.sqrt())
            } else {
                None
            }
        }
        IrOp::FFloor => Some(n.floor()),
        IrOp::FRound => Some(n.round_ties_even()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Pass 3: Strength reduction
// ---------------------------------------------------------------------------

/// Strength reduction: replace expensive ops with cheaper equivalents.
fn strength_reduce(ops: Vec<IrOp>) -> Vec<IrOp> {
    let mut out: Vec<IrOp> = Vec::with_capacity(ops.len());

    for op in ops {
        // Recurse into control-flow bodies
        let op = apply_to_bodies(op, &strength_reduce);

        if let Some(prev) = out.last() {
            match (prev, &op) {
                // PushI32(n) * where n is power of 2 => shift left
                (IrOp::PushI32(n), IrOp::Mul) if *n > 0 && (*n as u32).is_power_of_two() => {
                    let shift = (*n as u32).trailing_zeros() as i32;
                    out.pop();
                    out.push(IrOp::PushI32(shift));
                    out.push(IrOp::Lshift);
                    continue;
                }
                // PushI32(0) = => ZeroEq
                (IrOp::PushI32(0), IrOp::Eq) => {
                    out.pop();
                    out.push(IrOp::ZeroEq);
                    continue;
                }
                // PushI32(0) < => ZeroLt
                (IrOp::PushI32(0), IrOp::Lt) => {
                    out.pop();
                    out.push(IrOp::ZeroLt);
                    continue;
                }
                _ => {}
            }
        }
        out.push(op);
    }
    out
}

// ---------------------------------------------------------------------------
// Pass 4: Dead code elimination
// ---------------------------------------------------------------------------

/// Dead code elimination: remove unreachable code.
fn dce(ops: Vec<IrOp>) -> Vec<IrOp> {
    let mut out: Vec<IrOp> = Vec::with_capacity(ops.len());

    for op in ops {
        // Recurse into control-flow bodies
        let op = apply_to_bodies(op, &dce);

        // Constant conditional: if last output is PushI32 and current is If
        if let IrOp::If {
            then_body,
            else_body,
        } = &op
            && let Some(IrOp::PushI32(n)) = out.last()
        {
            let n = *n;
            out.pop();
            if n == 0 {
                // False: emit else_body only
                if let Some(eb) = else_body {
                    out.extend(eb.iter().cloned());
                }
            } else {
                // True: emit then_body only
                out.extend(then_body.iter().cloned());
            }
            continue;
        }

        // Truncate after Exit in linear sequence
        if matches!(op, IrOp::Exit) {
            out.push(op);
            break;
        }

        out.push(op);
    }
    out
}

// ---------------------------------------------------------------------------
// Pass 6: Inlining
// ---------------------------------------------------------------------------

/// Inline small word bodies: replaces `Call(id)` with the word's IR body
/// if the body is small enough and not recursive.
fn inline(
    ops: Vec<IrOp>,
    bodies: &HashMap<WordId, Vec<IrOp>>,
    max_size: usize,
    keep_loops_out: bool,
) -> Vec<IrOp> {
    let mut out = Vec::new();
    for op in ops {
        match &op {
            IrOp::Call(id) => {
                if let Some(body) = bodies.get(id)
                    && body.len() <= max_size
                    && !contains_call_to(body, *id)
                    && !contains_exit(body)
                    && !(keep_loops_out && crate::codegen::contains_loop(body))
                {
                    // Inline the body, recursively converting TailCall back to Call
                    // (tail position in the callee is not tail position in the caller).
                    for inlined_op in body {
                        out.push(detailcall(inlined_op.clone()));
                    }
                    continue;
                }
                out.push(op);
            }
            _ => {
                out.push(apply_to_bodies(op, &|inner| {
                    inline(inner, bodies, max_size, keep_loops_out)
                }));
            }
        }
    }
    out
}

/// Recursively convert all `TailCall` ops to `Call` in an IR tree.
///
/// When inlining a callee, its tail-call positions are no longer tail positions
/// in the caller. The `TailCall` codegen emits `Return` after the call, which
/// would prematurely exit the caller's function. This must recurse into
/// control-flow bodies (If, loops) where `convert_tail_call` may have placed
/// `TailCall` ops.
fn detailcall(op: IrOp) -> IrOp {
    match op {
        IrOp::TailCall(id) => IrOp::Call(id),
        IrOp::If {
            then_body,
            else_body,
        } => IrOp::If {
            then_body: then_body.into_iter().map(detailcall).collect(),
            else_body: else_body.map(|eb| eb.into_iter().map(detailcall).collect()),
        },
        IrOp::DoLoop { body, is_plus_loop } => IrOp::DoLoop {
            body: body.into_iter().map(detailcall).collect(),
            is_plus_loop,
        },
        IrOp::BeginUntil { body } => IrOp::BeginUntil {
            body: body.into_iter().map(detailcall).collect(),
        },
        IrOp::BeginAgain { body } => IrOp::BeginAgain {
            body: body.into_iter().map(detailcall).collect(),
        },
        IrOp::BeginWhileRepeat { test, body } => IrOp::BeginWhileRepeat {
            test: test.into_iter().map(detailcall).collect(),
            body: body.into_iter().map(detailcall).collect(),
        },
        IrOp::BeginDoubleWhileRepeat {
            outer_test,
            inner_test,
            body,
            after_repeat,
            else_body,
        } => IrOp::BeginDoubleWhileRepeat {
            outer_test: outer_test.into_iter().map(detailcall).collect(),
            inner_test: inner_test.into_iter().map(detailcall).collect(),
            body: body.into_iter().map(detailcall).collect(),
            after_repeat: after_repeat.into_iter().map(detailcall).collect(),
            else_body: else_body.map(|eb| eb.into_iter().map(detailcall).collect()),
        },
        other => other,
    }
}

/// Check if an IR body contains a direct call to the given word (recursion guard).
/// Largest guard the expander is willing to run twice, in IR operations.
const MAX_GUARD_OPS: usize = 6;
/// Most self-call sites worth expanding, to bound the code growth.
const MAX_GUARD_SITES: usize = 4;

/// Expand a recursive word's base-case guard into its own call sites.
///
/// A recursive Forth word almost always opens with a guard that returns early
/// -- `: FIB DUP 2 < IF EXIT THEN ... RECURSE ... ;` -- so every leaf of the
/// recursion costs a call whose whole body is that test. Testing at the call
/// site instead removes the call for the leaves, which in fib's tree is half
/// of all nodes.
///
/// `Call(self)` becomes `<guard> IF <what the guard returns> ELSE Call(self)
/// THEN`, which computes the same thing: the callee would have run the guard,
/// taken the branch and returned. The price is that the guard runs twice along
/// the recursive path, which is why it has to be small and free of effects.
fn expand_self_guard(ops: Vec<IrOp>, self_id: WordId) -> Vec<IrOp> {
    let Some((cond, base)) = split_guard(&ops) else {
        return ops;
    };
    if count_self_calls(&ops, self_id) > MAX_GUARD_SITES {
        return ops;
    }
    let (cond, base) = (cond.to_vec(), base.to_vec());
    replace_self_calls(ops, self_id, &cond, &base)
}

/// Split a body into the condition of its leading base-case guard and what
/// that guard leaves behind, or `None` if it does not open with one.
fn split_guard(ops: &[IrOp]) -> Option<(&[IrOp], &[IrOp])> {
    let at = ops.iter().position(|op| matches!(op, IrOp::If { .. }))?;
    let cond = &ops[..at];
    if at > MAX_GUARD_OPS || !cond.iter().all(is_duplicable) {
        return None;
    }
    let IrOp::If {
        then_body,
        else_body: None,
    } = &ops[at]
    else {
        return None;
    };
    // The guard is only a guard if it returns; what precedes the `EXIT` is
    // the value it returns, and has to be as harmless as the condition.
    let (IrOp::Exit, base) = then_body.split_last()? else {
        return None;
    };
    if base.len() > MAX_GUARD_OPS || !base.iter().all(is_duplicable) {
        return None;
    }
    Some((cond, base))
}

/// Can this operation be duplicated at every call site -- cheap, effect-free,
/// and not itself a call or a branch?
fn is_duplicable(op: &IrOp) -> bool {
    matches!(
        op,
        IrOp::PushI32(_)
            | IrOp::Drop
            | IrOp::Dup
            | IrOp::Swap
            | IrOp::Over
            | IrOp::Rot
            | IrOp::Nip
            | IrOp::Tuck
            | IrOp::TwoDup
            | IrOp::TwoDrop
            | IrOp::Add
            | IrOp::Sub
            | IrOp::Mul
            | IrOp::Negate
            | IrOp::Abs
            | IrOp::Eq
            | IrOp::NotEq
            | IrOp::Lt
            | IrOp::Gt
            | IrOp::LtUnsigned
            | IrOp::ZeroEq
            | IrOp::ZeroLt
            | IrOp::And
            | IrOp::Or
            | IrOp::Xor
            | IrOp::Invert
            | IrOp::Lshift
            | IrOp::Rshift
            | IrOp::ArithRshift
    )
}

fn count_self_calls(ops: &[IrOp], self_id: WordId) -> usize {
    ops.iter()
        .map(|op| match op {
            IrOp::Call(id) if *id == self_id => 1,
            IrOp::If {
                then_body,
                else_body,
            } => {
                count_self_calls(then_body, self_id)
                    + else_body
                        .as_deref()
                        .map_or(0, |eb| count_self_calls(eb, self_id))
            }
            _ => 0,
        })
        .sum()
}

/// Wrap every `Call(self_id)` in the guard. Only plain calls: a `TailCall` is
/// followed by a return, and leaving those alone keeps tail-call detection and
/// this pass from having to agree about what tail position means.
fn replace_self_calls(ops: Vec<IrOp>, self_id: WordId, cond: &[IrOp], base: &[IrOp]) -> Vec<IrOp> {
    let mut out = Vec::with_capacity(ops.len());
    for op in ops {
        match op {
            IrOp::Call(id) if id == self_id => {
                out.extend_from_slice(cond);
                out.push(IrOp::If {
                    then_body: base.to_vec(),
                    else_body: Some(vec![IrOp::Call(id)]),
                });
            }
            IrOp::If {
                then_body,
                else_body,
            } => out.push(IrOp::If {
                then_body: replace_self_calls(then_body, self_id, cond, base),
                else_body: else_body.map(|eb| replace_self_calls(eb, self_id, cond, base)),
            }),
            other => out.push(other),
        }
    }
    out
}

fn contains_call_to(ops: &[IrOp], target: WordId) -> bool {
    for op in ops {
        match op {
            IrOp::Call(id) | IrOp::TailCall(id) if *id == target => return true,
            IrOp::If {
                then_body,
                else_body,
            } => {
                if contains_call_to(then_body, target) {
                    return true;
                }
                if let Some(eb) = else_body
                    && contains_call_to(eb, target)
                {
                    return true;
                }
            }
            IrOp::DoLoop { body, .. } | IrOp::BeginUntil { body } | IrOp::BeginAgain { body }
                if contains_call_to(body, target) =>
            {
                return true;
            }
            IrOp::BeginWhileRepeat { test, body }
                if contains_call_to(test, target) || contains_call_to(body, target) =>
            {
                return true;
            }
            IrOp::BeginDoubleWhileRepeat {
                outer_test,
                inner_test,
                body,
                after_repeat,
                else_body,
            } => {
                if contains_call_to(outer_test, target)
                    || contains_call_to(inner_test, target)
                    || contains_call_to(body, target)
                    || contains_call_to(after_repeat, target)
                {
                    return true;
                }
                if let Some(eb) = else_body
                    && contains_call_to(eb, target)
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Check if an IR body contains ops that prevent safe inlining.
/// - `Exit`: WASM `return` would exit the caller's function
/// - `ForthLocalGet/Set`: would collide with the caller's WASM locals
fn contains_exit(ops: &[IrOp]) -> bool {
    for op in ops {
        match op {
            IrOp::Exit
            | IrOp::ForthLocalGet(_)
            | IrOp::ForthLocalSet(_)
            | IrOp::ForthFLocalGet(_)
            | IrOp::ForthFLocalSet(_) => return true,
            IrOp::If {
                then_body,
                else_body,
            } => {
                if contains_exit(then_body) {
                    return true;
                }
                if let Some(eb) = else_body
                    && contains_exit(eb)
                {
                    return true;
                }
            }
            IrOp::DoLoop { body, .. } | IrOp::BeginUntil { body } | IrOp::BeginAgain { body }
                if contains_exit(body) =>
            {
                return true;
            }
            IrOp::BeginWhileRepeat { test, body } if contains_exit(test) || contains_exit(body) => {
                return true;
            }
            _ => {}
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Pass 7: Tail call detection
// ---------------------------------------------------------------------------

/// Tail call detection: replace the last `Call` with `TailCall` when safe.
fn tail_call_detect(ops: Vec<IrOp>) -> Vec<IrOp> {
    if ops.is_empty() || !is_return_stack_balanced(&ops) {
        return ops;
    }

    let mut ir = ops;
    let last_idx = ir.len() - 1;
    ir[last_idx] = convert_tail_call(ir[last_idx].clone());
    ir
}

/// Check if return stack usage is balanced (equal number of `ToR` and `FromR`).
fn is_return_stack_balanced(ops: &[IrOp]) -> bool {
    let mut depth: i32 = 0;
    for op in ops {
        match op {
            IrOp::ToR => depth += 1,
            IrOp::FromR => depth -= 1,
            _ => {}
        }
    }
    depth == 0
}

/// Convert a `Call` at tail position to `TailCall`, recursing into `If` branches.
fn convert_tail_call(op: IrOp) -> IrOp {
    match op {
        IrOp::Call(id) => IrOp::TailCall(id),
        IrOp::If {
            mut then_body,
            else_body,
        } => {
            // Recursively check then_body tail
            if let Some(last) = then_body.pop() {
                then_body.push(convert_tail_call(last));
            }
            // Recursively check else_body tail
            let else_body = else_body.map(|mut eb| {
                if let Some(last) = eb.pop() {
                    eb.push(convert_tail_call(last));
                }
                eb
            });
            IrOp::If {
                then_body,
                else_body,
            }
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dictionary::WordId;

    fn opt(ops: Vec<IrOp>) -> Vec<IrOp> {
        let config = OptConfig {
            peephole: true,
            constant_fold: true,
            tail_call: true,
            strength_reduce: true,
            dce: true,
            inline: false,
            self_guard: false,
        };
        optimize(ops, &config, &HashMap::new(), None)
    }

    /// A body shaped like a recursive Forth word: a base-case guard, then the
    /// recursive step. `SELF` is the word being compiled.
    const SELF: WordId = WordId(9);

    fn guarded_body(step: Vec<IrOp>) -> Vec<IrOp> {
        let mut ops = vec![
            IrOp::Dup,
            IrOp::PushI32(2),
            IrOp::Lt,
            IrOp::If {
                then_body: vec![IrOp::Exit],
                else_body: None,
            },
        ];
        ops.extend(step);
        ops
    }

    #[test]
    fn self_guard_moves_the_base_case_to_the_call_site() {
        let out = expand_self_guard(guarded_body(vec![IrOp::Call(SELF)]), SELF);
        assert_eq!(
            out,
            guarded_body(vec![
                IrOp::Dup,
                IrOp::PushI32(2),
                IrOp::Lt,
                IrOp::If {
                    then_body: vec![],
                    else_body: Some(vec![IrOp::Call(SELF)]),
                },
            ])
        );
    }

    #[test]
    fn self_guard_carries_the_value_the_guard_returns() {
        // `: F DUP 2 < IF DROP 0 EXIT THEN RECURSE ;` -- the base case is not
        // "leave the argument", it is "replace it with 0".
        let body = vec![
            IrOp::Dup,
            IrOp::PushI32(2),
            IrOp::Lt,
            IrOp::If {
                then_body: vec![IrOp::Drop, IrOp::PushI32(0), IrOp::Exit],
                else_body: None,
            },
            IrOp::Call(SELF),
        ];
        let out = expand_self_guard(body, SELF);
        let IrOp::If { then_body, .. } = &out[7] else {
            panic!("expected the expanded guard at index 7, got {:?}", out);
        };
        assert_eq!(then_body, &vec![IrOp::Drop, IrOp::PushI32(0)]);
    }

    #[test]
    fn self_guard_leaves_a_body_without_a_guard_alone() {
        // An `IF` with an `ELSE` is a branch, not an early return.
        let body = vec![
            IrOp::Dup,
            IrOp::If {
                then_body: vec![IrOp::Drop],
                else_body: Some(vec![IrOp::Call(SELF)]),
            },
        ];
        assert_eq!(expand_self_guard(body.clone(), SELF), body);

        // No `EXIT` in the then-branch: also not a guard.
        let body = guarded_body(vec![IrOp::Call(SELF)])
            .into_iter()
            .map(|op| match op {
                IrOp::If { .. } => IrOp::If {
                    then_body: vec![IrOp::Drop],
                    else_body: None,
                },
                other => other,
            })
            .collect::<Vec<_>>();
        assert_eq!(expand_self_guard(body.clone(), SELF), body);
    }

    #[test]
    fn self_guard_refuses_a_condition_it_cannot_run_twice() {
        // A guard reached through a call or a memory write would be evaluated
        // once at the call site and again inside the callee.
        let body = vec![
            IrOp::Call(WordId(3)),
            IrOp::If {
                then_body: vec![IrOp::Exit],
                else_body: None,
            },
            IrOp::Call(SELF),
        ];
        assert_eq!(expand_self_guard(body.clone(), SELF), body);

        let body = vec![
            IrOp::Dup,
            IrOp::Fetch,
            IrOp::If {
                then_body: vec![IrOp::Exit],
                else_body: None,
            },
            IrOp::Call(SELF),
        ];
        assert_eq!(expand_self_guard(body.clone(), SELF), body);
    }

    #[test]
    fn self_guard_stops_at_the_call_site_budget() {
        let step = std::iter::repeat_n(IrOp::Call(SELF), MAX_GUARD_SITES + 1).collect();
        let body = guarded_body(step);
        assert_eq!(expand_self_guard(body.clone(), SELF), body);
    }

    #[test]
    fn self_guard_leaves_tail_calls_alone() {
        let body = guarded_body(vec![IrOp::TailCall(SELF)]);
        assert_eq!(expand_self_guard(body.clone(), SELF), body);
    }

    #[test]
    fn a_loop_stays_out_of_a_caller_that_is_only_unpromotable_through_a_call() {
        // The shape the benchmark harness uses, and the one that made this
        // guard a no-op for its whole life: at the moment the guard runs, the
        // caller's `CR` is still `Call(cr_word)`, not `IrOp::Cr`. A test built
        // from `IrOp::Cr` directly passes even with the bug.
        let cross = WordId(7);
        let cr = WordId(9);
        let mut bodies = HashMap::new();
        bodies.insert(cr, vec![IrOp::Cr]);
        bodies.insert(
            cross,
            vec![
                IrOp::PushI32(0),
                IrOp::Swap,
                IrOp::PushI32(0),
                IrOp::DoLoop {
                    body: vec![IrOp::RFetch, IrOp::Call(WordId(8)), IrOp::Xor],
                    is_plus_loop: false,
                },
            ],
        );
        let out = opt_with_inline(
            vec![
                IrOp::PushI32(300000),
                IrOp::Call(cross),
                IrOp::Drop,
                IrOp::Call(cr),
            ],
            &bodies,
        );
        assert!(
            out.iter()
                .any(|op| matches!(op, IrOp::Call(id) if *id == cross)),
            "a loop-bearing callee must not be inlined into a caller that cannot \
             be promoted -- it would lose its registers: {out:?}"
        );
        assert!(
            out.iter().any(|op| matches!(op, IrOp::Cr)),
            "the loop-free callee should still have been inlined: {out:?}"
        );
    }

    fn opt_with_inline(ops: Vec<IrOp>, bodies: &HashMap<WordId, Vec<IrOp>>) -> Vec<IrOp> {
        let config = OptConfig {
            peephole: true,
            constant_fold: true,
            tail_call: true,
            strength_reduce: true,
            dce: true,
            inline: true,
            self_guard: false,
        };
        optimize(ops, &config, bodies, None)
    }

    // Peephole tests
    #[test]
    fn push_drop_removed() {
        assert_eq!(opt(vec![IrOp::PushI32(5), IrOp::Drop]), vec![]);
    }

    #[test]
    fn dup_drop_removed() {
        assert_eq!(
            opt(vec![IrOp::PushI32(1), IrOp::Dup, IrOp::Drop]),
            vec![IrOp::PushI32(1)]
        );
    }

    #[test]
    fn swap_swap_removed() {
        assert_eq!(opt(vec![IrOp::Swap, IrOp::Swap]), vec![]);
    }

    #[test]
    fn swap_drop_to_nip() {
        assert_eq!(opt(vec![IrOp::Swap, IrOp::Drop]), vec![IrOp::Nip]);
    }

    #[test]
    fn add_zero_identity() {
        assert_eq!(opt(vec![IrOp::PushI32(0), IrOp::Add]), vec![]);
    }

    // Constant folding tests
    #[test]
    fn fold_add() {
        assert_eq!(
            opt(vec![IrOp::PushI32(5), IrOp::PushI32(3), IrOp::Add]),
            vec![IrOp::PushI32(8)]
        );
    }

    #[test]
    fn fold_negate() {
        assert_eq!(
            opt(vec![IrOp::PushI32(7), IrOp::Negate]),
            vec![IrOp::PushI32(-7)]
        );
    }

    #[test]
    fn fold_chain() {
        // 2 3 + 4 * => 5 4 * => 20
        assert_eq!(
            opt(vec![
                IrOp::PushI32(2),
                IrOp::PushI32(3),
                IrOp::Add,
                IrOp::PushI32(4),
                IrOp::Mul,
            ]),
            vec![IrOp::PushI32(20)]
        );
    }

    #[test]
    fn fold_comparison() {
        assert_eq!(
            opt(vec![IrOp::PushI32(4), IrOp::PushI32(3), IrOp::Lt]),
            vec![IrOp::PushI32(0)]
        );
    }

    // Strength reduction tests
    #[test]
    fn power_of_2_mul_to_shift() {
        assert_eq!(
            opt(vec![IrOp::PushI32(4), IrOp::Mul]),
            vec![IrOp::PushI32(2), IrOp::Lshift]
        );
    }

    #[test]
    fn non_power_of_2_unchanged() {
        assert_eq!(
            opt(vec![IrOp::PushI32(3), IrOp::Mul]),
            vec![IrOp::PushI32(3), IrOp::Mul]
        );
    }

    // Tail call tests
    #[test]
    fn tail_call_simple() {
        assert_eq!(
            opt(vec![IrOp::PushI32(5), IrOp::Call(WordId(3))]),
            vec![IrOp::PushI32(5), IrOp::TailCall(WordId(3))]
        );
    }

    #[test]
    fn no_tail_call_with_unbalanced_rstack() {
        assert_eq!(
            opt(vec![IrOp::ToR, IrOp::Call(WordId(3))]),
            vec![IrOp::ToR, IrOp::Call(WordId(3))]
        );
    }

    // DCE tests
    #[test]
    fn remove_after_exit() {
        assert_eq!(
            opt(vec![IrOp::PushI32(1), IrOp::Exit, IrOp::PushI32(2)]),
            vec![IrOp::PushI32(1), IrOp::Exit]
        );
    }

    #[test]
    fn constant_true_if() {
        assert_eq!(
            opt(vec![
                IrOp::PushI32(1),
                IrOp::If {
                    then_body: vec![IrOp::PushI32(10)],
                    else_body: Some(vec![IrOp::PushI32(20)]),
                }
            ]),
            vec![IrOp::PushI32(10)]
        );
    }

    #[test]
    fn constant_false_if() {
        assert_eq!(
            opt(vec![
                IrOp::PushI32(0),
                IrOp::If {
                    then_body: vec![IrOp::PushI32(10)],
                    else_body: Some(vec![IrOp::PushI32(20)]),
                }
            ]),
            vec![IrOp::PushI32(20)]
        );
    }

    // Compound ops tests
    #[test]
    fn over_over_to_twdup() {
        assert_eq!(opt(vec![IrOp::Over, IrOp::Over]), vec![IrOp::TwoDup]);
    }

    #[test]
    fn drop_drop_to_twodrop() {
        assert_eq!(opt(vec![IrOp::Drop, IrOp::Drop]), vec![IrOp::TwoDrop]);
    }

    // Nested optimization
    #[test]
    fn nested_if_optimized() {
        assert_eq!(
            opt(vec![IrOp::If {
                then_body: vec![IrOp::PushI32(5), IrOp::Drop],
                else_body: None,
            }]),
            vec![IrOp::If {
                then_body: vec![],
                else_body: None
            }]
        );
    }

    // Inlining tests
    #[test]
    fn inline_simple() {
        let mut bodies = HashMap::new();
        // SQUARE = DUP *
        bodies.insert(WordId(5), vec![IrOp::Dup, IrOp::Mul]);
        let result = opt_with_inline(vec![IrOp::PushI32(7), IrOp::Call(WordId(5))], &bodies);
        // After inlining: 7 DUP * (Dup isn't folded by constant folder)
        assert_eq!(result, vec![IrOp::PushI32(7), IrOp::Dup, IrOp::Mul]);
    }

    #[test]
    fn inline_folds_constants() {
        let mut bodies = HashMap::new();
        // ADD3 = 3 +
        bodies.insert(WordId(5), vec![IrOp::PushI32(3), IrOp::Add]);
        let result = opt_with_inline(vec![IrOp::PushI32(5), IrOp::Call(WordId(5))], &bodies);
        // After inlining: PushI32(5) PushI32(3) Add => folded to PushI32(8)
        assert_eq!(result, vec![IrOp::PushI32(8)]);
    }

    #[test]
    fn no_inline_recursive() {
        let mut bodies = HashMap::new();
        bodies.insert(WordId(5), vec![IrOp::Dup, IrOp::Call(WordId(5))]);
        let result = opt_with_inline(vec![IrOp::Call(WordId(5))], &bodies);
        // Should NOT inline (recursive), but tail call detect may convert
        assert!(matches!(
            result.last(),
            Some(IrOp::Call(WordId(5)) | IrOp::TailCall(WordId(5)))
        ));
    }

    // Float peephole tests
    #[test]
    fn float_push_fdrop_removed() {
        assert_eq!(opt(vec![IrOp::PushF64(1.0), IrOp::FDrop]), vec![]);
    }

    #[test]
    fn float_fdup_fdrop_removed() {
        assert_eq!(opt(vec![IrOp::FDup, IrOp::FDrop]), vec![]);
    }

    #[test]
    fn float_fswap_fswap_removed() {
        assert_eq!(opt(vec![IrOp::FSwap, IrOp::FSwap]), vec![]);
    }

    #[test]
    fn float_fnegate_fnegate_removed() {
        assert_eq!(opt(vec![IrOp::FNegate, IrOp::FNegate]), vec![]);
    }

    // Float constant folding tests
    #[test]
    fn float_constant_fold_add() {
        assert_eq!(
            opt(vec![IrOp::PushF64(1.5), IrOp::PushF64(2.5), IrOp::FAdd]),
            vec![IrOp::PushF64(4.0)]
        );
    }

    #[test]
    fn float_constant_fold_negate() {
        assert_eq!(
            opt(vec![IrOp::PushF64(3.0), IrOp::FNegate]),
            vec![IrOp::PushF64(-3.0)]
        );
    }

    #[test]
    fn float_constant_fold_sqrt() {
        assert_eq!(
            opt(vec![IrOp::PushF64(9.0), IrOp::FSqrt]),
            vec![IrOp::PushF64(3.0)]
        );
    }

    #[test]
    fn no_inline_large() {
        let mut bodies = HashMap::new();
        // Body with 9 ops (> max_size of 8)
        bodies.insert(WordId(5), vec![IrOp::Dup; 9]);
        let config = OptConfig {
            peephole: false,
            constant_fold: false,
            tail_call: false,
            strength_reduce: false,
            dce: false,
            inline: true,
            self_guard: false,
        };
        let result = optimize(vec![IrOp::Call(WordId(5))], &config, &bodies, None);
        assert_eq!(result, vec![IrOp::Call(WordId(5))]);
    }

    #[test]
    fn keeps_a_loop_out_of_a_caller_stuck_on_the_memory_stack() {
        // The caller has a `.`, so it can never leave the memory data stack.
        // Inlining the loop would drag it down too; as its own word the loop
        // keeps its registers and the caller just pays one call.
        let mut bodies = HashMap::new();
        bodies.insert(
            WordId(5),
            vec![IrOp::DoLoop {
                body: vec![IrOp::PushI32(1), IrOp::Add],
                is_plus_loop: false,
            }],
        );
        let result = opt_with_inline(vec![IrOp::Call(WordId(5)), IrOp::Dot], &bodies);
        assert!(
            matches!(result.first(), Some(IrOp::Call(WordId(5)))),
            "loop should not have been inlined, got {result:?}"
        );
    }

    #[test]
    fn still_inlines_a_loop_into_a_caller_that_can_be_promoted() {
        let mut bodies = HashMap::new();
        bodies.insert(
            WordId(5),
            vec![IrOp::DoLoop {
                body: vec![IrOp::PushI32(1), IrOp::Add],
                is_plus_loop: false,
            }],
        );
        let result = opt_with_inline(vec![IrOp::Call(WordId(5)), IrOp::Dup], &bodies);
        assert!(
            !result.iter().any(|op| matches!(op, IrOp::Call(_))),
            "loop should have been inlined, got {result:?}"
        );
    }

    #[test]
    fn still_inlines_straight_line_words_anywhere() {
        // Only loops are held back; a small straight-line word is still
        // better off inlined even into an unpromotable caller.
        let mut bodies = HashMap::new();
        bodies.insert(WordId(5), vec![IrOp::Dup, IrOp::Mul]);
        let result = opt_with_inline(vec![IrOp::Call(WordId(5)), IrOp::Dot], &bodies);
        assert!(
            !result.iter().any(|op| matches!(op, IrOp::Call(_))),
            "straight-line word should still inline, got {result:?}"
        );
    }
}
