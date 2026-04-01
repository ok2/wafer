//! Optimization passes for WAFER's IR.
//!
//! Each pass is a function `Vec<IrOp> -> Vec<IrOp>`, composable in sequence:
//! 1. Peephole optimization
//! 2. Constant folding
//! 3. Strength reduction
//! 4. Dead code elimination
//! 5. Tail call detection

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
}

/// Run all enabled optimization passes.
pub fn optimize(ops: Vec<IrOp>, config: &OptConfig) -> Vec<IrOp> {
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

    // Phase 2: eliminate dead code
    if config.dce {
        ir = dce(ir);
    }
    if config.peephole {
        ir = peephole(ir);
    }

    // Phase 3: tail calls (must be last)
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

        // Try unary fold: last output is PushI32, current op is foldable
        if !out.is_empty()
            && let Some(result) = try_unary_fold(&out[out.len() - 1], &op)
        {
            out.pop();
            out.push(IrOp::PushI32(result));
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
// Pass 5: Tail call detection
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
        };
        optimize(ops, &config)
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
}
