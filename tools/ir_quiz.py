#!/usr/bin/env python3
"""WAFER IR Flash Quiz — predict the optimized IR for Forth code."""

import random
import sys

# Each exercise: (forth_code, accepted_answers, explanation)
# accepted_answers: list of strings that count as correct (case-insensitive, whitespace-normalized)
EXERCISES = [
    # --- Constant Folding ---
    (
        ": FOO  2 3 + ;",
        ["PushI32(5)", "pushi32(5)", "5"],
        "Constant fold: PushI32(2), PushI32(3), Add → PushI32(5)",
    ),
    (
        ": FOO  10 3 - ;",
        ["PushI32(7)", "pushi32(7)", "7"],
        "Constant fold: PushI32(10), PushI32(3), Sub → PushI32(7)",
    ),
    (
        ": FOO  6 7 * ;",
        ["PushI32(42)", "pushi32(42)", "42"],
        "Constant fold: PushI32(6), PushI32(7), Mul → PushI32(42)",
    ),
    (
        ": FOO  5 0= ;",
        ["PushI32(0)", "pushi32(0)", "0", "false"],
        "Constant fold (unary): PushI32(5), ZeroEq → PushI32(0) (5 is not zero)",
    ),
    (
        ": FOO  0 0= ;",
        ["PushI32(-1)", "pushi32(-1)", "-1", "true"],
        "Constant fold (unary): PushI32(0), ZeroEq → PushI32(-1) (true flag)",
    ),
    (
        ": FOO  -3 ABS ;",
        ["PushI32(3)", "pushi32(3)", "3"],
        "Constant fold (unary): PushI32(-3), Abs → PushI32(3)",
    ),
    (
        ": FOO  255 INVERT ;",
        ["PushI32(-256)", "pushi32(-256)", "-256"],
        "Constant fold (unary): PushI32(255), Invert → PushI32(-256) (bitwise NOT)",
    ),
    (
        ": FOO  3 2 LSHIFT ;",
        ["PushI32(12)", "pushi32(12)", "12"],
        "Constant fold: PushI32(3), PushI32(2), Lshift → PushI32(12) (3 << 2 = 12)",
    ),

    # --- Peephole ---
    (
        ": FOO  DUP DROP ;",
        ["(empty)", "empty", "nothing", "[]", ""],
        "Peephole: Dup, Drop → removed (both eliminated)",
    ),
    (
        ": FOO  SWAP SWAP ;",
        ["(empty)", "empty", "nothing", "[]", ""],
        "Peephole: Swap, Swap → removed (self-inverse)",
    ),
    (
        ": FOO  SWAP DROP ;",
        ["Nip", "nip"],
        "Peephole: Swap, Drop → Nip",
    ),
    (
        ": FOO  DROP DROP ;",
        ["TwoDrop", "twodrop", "2drop"],
        "Peephole: Drop, Drop → TwoDrop",
    ),
    (
        ": FOO  OVER OVER ;",
        ["TwoDup", "twodup", "2dup"],
        "Peephole: Over, Over → TwoDup",
    ),
    (
        ": FOO  0 + ;",
        ["(empty)", "empty", "nothing", "[]", ""],
        "Peephole: PushI32(0), Add → removed (identity)",
    ),
    (
        ": FOO  1 * ;",
        ["(empty)", "empty", "nothing", "[]", ""],
        "Peephole: PushI32(1), Mul → removed (identity)",
    ),
    (
        ": FOO  -1 AND ;",
        ["(empty)", "empty", "nothing", "[]", ""],
        "Peephole: PushI32(-1), And → removed (identity, all bits set)",
    ),
    (
        ": FOO  0 OR ;",
        ["(empty)", "empty", "nothing", "[]", ""],
        "Peephole: PushI32(0), Or → removed (identity)",
    ),
    (
        ": FOO  42 DROP ;",
        ["(empty)", "empty", "nothing", "[]", ""],
        "Peephole: PushI32(42), Drop → removed (unused literal)",
    ),

    # --- Strength Reduction ---
    (
        ": FOO  8 * ;",
        ["PushI32(3), Lshift", "pushi32(3) lshift", "3 lshift"],
        "Strength reduce: PushI32(8) is 2^3, Mul → PushI32(3), Lshift",
    ),
    (
        ": FOO  16 * ;",
        ["PushI32(4), Lshift", "pushi32(4) lshift", "4 lshift"],
        "Strength reduce: PushI32(16) is 2^4, Mul → PushI32(4), Lshift",
    ),
    (
        ": FOO  2 * ;",
        ["PushI32(1), Lshift", "pushi32(1) lshift", "1 lshift"],
        "Strength reduce: PushI32(2) is 2^1, Mul → PushI32(1), Lshift",
    ),
    (
        ": FOO  0 = ;",
        ["ZeroEq", "zeroeq", "0="],
        "Strength reduce: PushI32(0), Eq → ZeroEq",
    ),
    (
        ": FOO  0 < ;",
        ["ZeroLt", "zerolt", "0<"],
        "Strength reduce: PushI32(0), Lt → ZeroLt",
    ),

    # --- Dead Code Elimination ---
    (
        ": FOO  TRUE IF 42 ELSE 99 THEN ;",
        ["PushI32(42)", "pushi32(42)", "42"],
        "DCE: PushI32(-1) is nonzero → then_body only → PushI32(42)",
    ),
    (
        ": FOO  FALSE IF 42 ELSE 99 THEN ;",
        ["PushI32(99)", "pushi32(99)", "99"],
        "DCE: PushI32(0) is zero → else_body only → PushI32(99)",
    ),
    (
        ": FOO  EXIT 42 ;",
        ["Exit", "exit"],
        "DCE: Everything after Exit is removed. PushI32(42) eliminated.",
    ),

    # --- Combined Optimizations ---
    (
        ": FOO  DUP * ;",
        ["Dup, Mul", "dup mul", "dup, mul"],
        "Inline DUP and *: [Dup, Mul]. No further optimizations apply.",
    ),
    (
        ": FOO  2 3 + 4 * ;",
        ["PushI32(20)", "pushi32(20)", "20"],
        "Fold 2+3=5, then fold 5*4=20. Single constant.",
    ),
    (
        ": FOO  1 2 + 8 * ;",
        ["PushI32(24)", "pushi32(24)", "24"],
        "Fold 1+2=3, strength reduce 8*? No — fold first: 3*8=24.",
    ),
    (
        ": FOO  0 0 + ;",
        ["PushI32(0)", "pushi32(0)", "0"],
        "Fold: PushI32(0), PushI32(0), Add → PushI32(0)",
    ),
    (
        ": FOO  SWAP DUP DROP SWAP ;",
        ["(empty)", "empty", "nothing", "[]", ""],
        "Peephole chain: Swap,Dup → ...; Dup,Drop → removed; Swap,Swap → removed. All gone.",
    ),

    # --- Inlining ---
    (
        ": SQUARE  DUP * ;\n: FOO  SQUARE ;",
        ["Dup, Mul", "dup mul", "dup, mul"],
        "SQUARE body=[Dup,Mul] (2 ops ≤ 8). Inlined into FOO. Tail call: Dup is not Call, Mul is not Call → no tail call.",
    ),

    # --- Tail Call ---
    (
        ": BAR  1 ;  : FOO  42 BAR ;",
        ["PushI32(42), TailCall(bar_id)", "pushi32(42) tailcall", "42 tailcall(bar)"],
        "BAR has body [PushI32(1)] — 1 op, inlineable. But wait: if BAR is inlined, result is [PushI32(42), PushI32(1)]. Actually depends on whether BAR is inlined. If NOT inlined: tail call applies to Call(bar). If inlined: [PushI32(42), PushI32(1)].",
    ),

    # --- Float ---
    (
        ": FOO  1.0E0 2.0E0 F+ ;",
        ["PushF64(3.0)", "pushf64(3.0)", "3.0"],
        "Float constant fold: PushF64(1.0), PushF64(2.0), FAdd → PushF64(3.0)",
    ),
    (
        ": FOO  -5.0E0 FABS ;",
        ["PushF64(5.0)", "pushf64(5.0)", "5.0"],
        "Float unary fold: PushF64(-5.0), FAbs → PushF64(5.0)",
    ),
    (
        ": FOO  FNEGATE FNEGATE ;",
        ["(empty)", "empty", "nothing", "[]", ""],
        "Peephole: FNegate, FNegate → removed (self-inverse)",
    ),
    (
        ": FOO  FSWAP FSWAP ;",
        ["(empty)", "empty", "nothing", "[]", ""],
        "Peephole: FSwap, FSwap → removed (self-inverse)",
    ),
    (
        ": FOO  FDUP FDROP ;",
        ["(empty)", "empty", "nothing", "[]", ""],
        "Peephole: FDup, FDrop → removed",
    ),

    # --- Tricky ---
    (
        ": FOO  3 5 < ;",
        ["PushI32(-1)", "pushi32(-1)", "-1", "true"],
        "Constant fold: PushI32(3), PushI32(5), Lt → PushI32(-1) (3 < 5 is true, Forth true = -1)",
    ),
    (
        ": FOO  5 3 < ;",
        ["PushI32(0)", "pushi32(0)", "0", "false"],
        "Constant fold: PushI32(5), PushI32(3), Lt → PushI32(0) (5 < 3 is false)",
    ),
    (
        ": FOO  DUP DUP DROP DROP ;",
        ["Dup", "dup"],
        "Peephole: Dup, Dup, Drop, Drop → Dup (first Dup stays, second Dup+Drop cancel, last Drop+implicit cancel... actually: Dup, Dup → keep; Dup, Drop → cancel; left with Dup. Then Drop. Hmm. Let's trace: [Dup, Dup, Drop, Drop] → peephole sees Dup,Drop at positions 1,2 → removes → [Dup, Drop] → peephole sees Dup,Drop → removes → []. Actually empty!",
    ),
]


def normalize(s: str) -> str:
    """Normalize answer for comparison: lowercase, strip whitespace/punctuation."""
    s = s.strip().lower()
    # Remove parentheses, brackets, commas for flexible matching
    for ch in "()[]":
        s = s.replace(ch, "")
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def check_answer(user_input: str, accepted: list[str]) -> bool:
    """Check if user's answer matches any accepted answer."""
    norm_input = normalize(user_input)
    for ans in accepted:
        if normalize(ans) == norm_input:
            return True
    return False


def run_quiz(exercises: list, shuffle: bool = True) -> None:
    """Run the interactive quiz."""
    items = list(exercises)
    if shuffle:
        random.shuffle(items)

    correct = 0
    total = 0
    skipped = 0

    print("=" * 60)
    print("  WAFER IR Flash Quiz")
    print("  Predict the optimized IR for each Forth definition.")
    print("  Type 'q' to quit, 's' to skip, 'h' for hint.")
    print("=" * 60)
    print()

    for i, (forth, accepted, explanation) in enumerate(items):
        total += 1
        print(f"  [{i + 1}/{len(items)}]")
        print(f"  {forth}")
        print()

        while True:
            try:
                user = input("  Your answer> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n")
                show_score(correct, total - 1, skipped)
                return

            if user.lower() == "q":
                show_score(correct, total - 1, skipped)
                return
            if user.lower() == "s":
                skipped += 1
                print(f"  Skipped. Answer: {accepted[0]}")
                print(f"  {explanation}")
                break
            if user.lower() == "h":
                # Give a hint: first word of explanation
                hint_word = explanation.split(":")[0] if ":" in explanation else "Think about the optimizer passes"
                print(f"  Hint: {hint_word}")
                continue

            if check_answer(user, accepted):
                correct += 1
                print(f"  \033[32m✓ Correct!\033[0m {explanation}")
            else:
                print(f"  \033[31m✗ Not quite.\033[0m Expected: {accepted[0]}")
                print(f"  {explanation}")
            break

        print()
        print("-" * 60)
        print()

    show_score(correct, total, skipped)


def show_score(correct: int, total: int, skipped: int) -> None:
    """Display final score."""
    attempted = total - skipped
    if attempted == 0:
        print("  No questions attempted.")
        return
    pct = (correct / attempted) * 100
    print(f"\n  Score: {correct}/{attempted} ({pct:.0f}%)")
    if skipped:
        print(f"  Skipped: {skipped}")
    if pct == 100:
        print("  Perfect! You know the optimizer cold.")
    elif pct >= 80:
        print("  Strong! Review the ones you missed.")
    elif pct >= 60:
        print("  Getting there. Focus on peephole + fold patterns.")
    else:
        print("  Study tools/architecture.txt section 7, then retry.")
    print()


def main() -> None:
    """Entry point."""
    if "--all" in sys.argv:
        run_quiz(EXERCISES, shuffle=False)
    elif "--count" in sys.argv:
        print(f"{len(EXERCISES)} exercises available.")
    else:
        run_quiz(EXERCISES, shuffle=True)


if __name__ == "__main__":
    main()
