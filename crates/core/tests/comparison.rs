#![allow(dead_code)]
//! Cross-engine comparison tests: WAFER vs gforth (and `SwiftForth` for perf).
//!
//! Validates that WAFER produces identical output to gforth for standard
//! Forth programs, and benchmarks performance of the engines. `SwiftForth`
//! (`sf64`, native-code commercial compiler) joins the performance report
//! as an upper-bound reference when installed.
//!
//! WAFER-only correctness:  `cargo test -p wafer-core --test comparison`
//! Full comparison + perf:  `cargo test -p wafer-core --test comparison -- --nocapture --ignored`

use std::process::Command;
use std::sync::OnceLock;

use wafer_core::config::WaferConfig;
use wafer_core::outer::ForthVM;
use wafer_core::runtime_native::NativeRuntime;

// -----------------------------------------------------------------------
// Gforth discovery (cached)
// -----------------------------------------------------------------------

static GFORTH_PATH: OnceLock<Option<String>> = OnceLock::new();
static GFORTH_FAST_PATH: OnceLock<Option<String>> = OnceLock::new();

fn probe_gforth(candidate: &str) -> bool {
    Command::new(candidate)
        .arg("-e")
        .arg("bye")
        .output()
        .is_ok_and(|o| o.status.success())
}

fn find_gforth() -> Option<&'static str> {
    GFORTH_PATH
        .get_or_init(|| {
            for candidate in &[
                "/opt/homebrew/bin/gforth",
                "/usr/local/bin/gforth",
                "gforth",
            ] {
                if probe_gforth(candidate) {
                    return Some(candidate.to_string());
                }
            }
            None
        })
        .as_deref()
}

fn find_gforth_fast() -> Option<&'static str> {
    GFORTH_FAST_PATH
        .get_or_init(|| {
            for candidate in &[
                "/opt/homebrew/bin/gforth-fast",
                "/usr/local/bin/gforth-fast",
                "gforth-fast",
            ] {
                if probe_gforth(candidate) {
                    return Some(candidate.to_string());
                }
            }
            None
        })
        .as_deref()
}

// -----------------------------------------------------------------------
// SwiftForth (sf64) discovery (cached)
// -----------------------------------------------------------------------

static SF64_PATH: OnceLock<Option<String>> = OnceLock::new();

/// Probe sf64 by piping `bye` via stdin — sf64 has no `-e` flag; it takes
/// Forth source from stdin or as bare command-line arguments.
fn probe_sf64(candidate: &str) -> bool {
    run_via_stdin(candidate, "bye\n").is_some_and(|o| o.status.success())
}

fn find_sf64() -> Option<&'static str> {
    SF64_PATH
        .get_or_init(|| {
            for candidate in &["/Applications/ForthInc/SwiftForth/bin/macos/sf64", "sf64"] {
                if probe_sf64(candidate) {
                    return Some(candidate.to_string());
                }
            }
            None
        })
        .as_deref()
}

/// Spawn `binary`, write `input` to its stdin, and collect the output.
fn run_via_stdin(binary: &str, input: &str) -> Option<std::process::Output> {
    Command::new(binary)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take().unwrap().write_all(input.as_bytes())?;
            child.wait_with_output()
        })
        .ok()
}

// -----------------------------------------------------------------------
// Engine runners
// -----------------------------------------------------------------------

struct EngineResult {
    output: String,
    success: bool,
}

/// Run Forth code through WAFER (in-process via `ForthVM`).
fn run_wafer(code: &str) -> EngineResult {
    let mut vm = ForthVM::<NativeRuntime>::new().expect("Failed to create ForthVM");
    let mut output = String::new();
    for line in code.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match vm.evaluate(trimmed) {
            Ok(()) => output.push_str(&vm.take_output()),
            Err(_) => {
                return EngineResult {
                    output,
                    success: false,
                };
            }
        }
    }
    EngineResult {
        output,
        success: true,
    }
}

/// Run Forth code through WAFER with all optimizations enabled.
fn run_wafer_optimized(code: &str) -> EngineResult {
    let mut vm = ForthVM::<NativeRuntime>::new_with_config(WaferConfig::all())
        .expect("Failed to create ForthVM");
    let mut output = String::new();
    for line in code.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match vm.evaluate(trimmed) {
            Ok(()) => output.push_str(&vm.take_output()),
            Err(_) => {
                return EngineResult {
                    output,
                    success: false,
                };
            }
        }
    }
    EngineResult {
        output,
        success: true,
    }
}

/// Run Forth code through gforth. Returns `None` if gforth is unavailable.
fn run_gforth_engine(gforth: &str, code: &str) -> Option<EngineResult> {
    // Flatten to single line and append bye
    let flat = code
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let with_bye = if flat.ends_with("bye") || flat.ends_with("BYE") {
        flat
    } else {
        format!("{flat} bye")
    };
    let output = Command::new(gforth)
        .arg("-e")
        .arg(&with_bye)
        .output()
        .ok()?;
    Some(EngineResult {
        output: String::from_utf8_lossy(&output.stdout).into_owned(),
        success: output.status.success(),
    })
}

fn run_gforth(code: &str) -> Option<EngineResult> {
    run_gforth_engine(find_gforth()?, code)
}

fn run_gforth_fast(code: &str) -> Option<EngineResult> {
    run_gforth_engine(find_gforth_fast()?, code)
}

// -----------------------------------------------------------------------
// Output normalization
// -----------------------------------------------------------------------

/// Normalize Forth output for comparison: trim trailing whitespace per line,
/// collapse to single trailing newline.
fn normalize(s: &str) -> String {
    let trimmed: Vec<&str> = s.lines().map(str::trim_end).collect();
    let mut result = trimmed.join("\n");
    // Ensure exactly one trailing newline (or empty if no content)
    let end = result.trim_end_matches('\n');
    if !end.is_empty() {
        result = format!("{end}\n");
    } else {
        result.clear();
    }
    result
}

// -----------------------------------------------------------------------
// Assertion helpers
// -----------------------------------------------------------------------

/// Assert that WAFER produces the expected output for a program.
fn assert_wafer_output(name: &str, code: &str, expected: &str) {
    let result = run_wafer(code);
    assert!(result.success, "{name}: WAFER execution failed");
    assert_eq!(
        normalize(&result.output),
        normalize(expected),
        "{name}: WAFER output mismatch\n  got:      {:?}\n  expected: {:?}",
        result.output,
        expected
    );
}

/// Assert that WAFER and gforth produce identical output.
/// Skips gracefully if gforth is unavailable.
fn assert_same_output(name: &str, code: &str) {
    let wafer = run_wafer(code);
    assert!(wafer.success, "{name}: WAFER execution failed");

    let Some(gforth) = run_gforth(code) else {
        eprintln!("  SKIP {name}: gforth not available");
        return;
    };
    assert!(gforth.success, "{name}: gforth execution failed");
    assert_eq!(
        normalize(&wafer.output),
        normalize(&gforth.output),
        "{name}: output differs\n  WAFER:  {:?}\n  gforth: {:?}",
        wafer.output,
        gforth.output
    );
}

// -----------------------------------------------------------------------
// Test program catalog
// -----------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Category {
    Arithmetic,
    StackOps,
    ControlFlow,
    Loops,
    Definitions,
    Strings,
    Recursion,
    Memory,
}

struct Program {
    name: &'static str,
    code: &'static str,
    expected: &'static str,
    category: Category,
}

fn programs() -> Vec<Program> {
    vec![
        // -- Arithmetic --
        Program {
            name: "add",
            code: "2 3 + . CR",
            expected: "5 \n",
            category: Category::Arithmetic,
        },
        Program {
            name: "subtract",
            code: "10 3 - . CR",
            expected: "7 \n",
            category: Category::Arithmetic,
        },
        Program {
            name: "multiply",
            code: "6 7 * . CR",
            expected: "42 \n",
            category: Category::Arithmetic,
        },
        Program {
            name: "divide",
            code: "100 7 / . CR",
            expected: "14 \n",
            category: Category::Arithmetic,
        },
        Program {
            name: "mod",
            code: "100 7 MOD . CR",
            expected: "2 \n",
            category: Category::Arithmetic,
        },
        Program {
            name: "negate",
            code: "7 NEGATE . CR",
            expected: "-7 \n",
            category: Category::Arithmetic,
        },
        Program {
            name: "abs",
            code: "5 ABS . CR -5 ABS . CR",
            expected: "5 \n5 \n",
            category: Category::Arithmetic,
        },
        Program {
            name: "min-max",
            code: "3 7 MIN . CR 3 7 MAX . CR",
            expected: "3 \n7 \n",
            category: Category::Arithmetic,
        },
        Program {
            name: "divmod",
            code: "100 7 /MOD . . CR",
            expected: "14 2 \n",
            category: Category::Arithmetic,
        },
        // -- Stack operations --
        Program {
            name: "swap",
            code: "1 2 SWAP . . CR",
            expected: "1 2 \n",
            category: Category::StackOps,
        },
        Program {
            name: "dup",
            code: "5 DUP . . CR",
            expected: "5 5 \n",
            category: Category::StackOps,
        },
        Program {
            name: "over",
            code: "1 2 OVER . . . CR",
            expected: "1 2 1 \n",
            category: Category::StackOps,
        },
        Program {
            name: "rot",
            code: "1 2 3 ROT . . . CR",
            expected: "1 3 2 \n",
            category: Category::StackOps,
        },
        Program {
            name: "2dup",
            code: "1 2 2DUP . . . . CR",
            expected: "2 1 2 1 \n",
            category: Category::StackOps,
        },
        Program {
            name: "depth",
            code: "1 2 3 DEPTH . DROP DROP DROP CR",
            expected: "3 \n",
            category: Category::StackOps,
        },
        // -- Control flow --
        Program {
            name: "if-else",
            code: ": SGN DUP 0> IF DROP 1 ELSE DUP 0< IF DROP -1 ELSE DROP 0 THEN THEN ;\n\
                   5 SGN . CR -3 SGN . CR 0 SGN . CR",
            expected: "1 \n-1 \n0 \n",
            category: Category::ControlFlow,
        },
        Program {
            name: "max-word",
            code: ": MAX2 2DUP < IF SWAP THEN DROP ;\n\
                   3 7 MAX2 . CR 9 2 MAX2 . CR",
            expected: "7 \n9 \n",
            category: Category::ControlFlow,
        },
        Program {
            name: "abs-word",
            code: ": MYABS DUP 0< IF NEGATE THEN ;\n\
                   -5 MYABS . CR 3 MYABS . CR 0 MYABS . CR",
            expected: "5 \n3 \n0 \n",
            category: Category::ControlFlow,
        },
        // -- Loops --
        Program {
            name: "do-loop",
            code: ": SUM10 0 10 0 DO I + LOOP ; SUM10 . CR",
            expected: "45 \n",
            category: Category::Loops,
        },
        Program {
            name: "do-loop-emit",
            code: ": COUNTDOWN 5 0 DO I . LOOP CR ; COUNTDOWN",
            expected: "0 1 2 3 4 \n",
            category: Category::Loops,
        },
        Program {
            name: "plus-loop",
            code: ": SUM-EVEN 0 10 0 DO I + 2 +LOOP ; SUM-EVEN . CR",
            expected: "20 \n",
            category: Category::Loops,
        },
        Program {
            name: "begin-until",
            code: ": COUNT-DOWN 5 BEGIN DUP . 1- DUP 0= UNTIL DROP CR ; COUNT-DOWN",
            expected: "5 4 3 2 1 \n",
            category: Category::Loops,
        },
        Program {
            name: "begin-while-repeat",
            code: ": COUNT-UP 0 BEGIN DUP 5 < WHILE DUP . 1+ REPEAT DROP CR ; COUNT-UP",
            expected: "0 1 2 3 4 \n",
            category: Category::Loops,
        },
        // -- Definitions --
        Program {
            name: "variable",
            code: "VARIABLE X 42 X ! X @ . CR",
            expected: "42 \n",
            category: Category::Definitions,
        },
        Program {
            name: "constant",
            code: "7 CONSTANT SEVEN SEVEN . CR",
            expected: "7 \n",
            category: Category::Definitions,
        },
        Program {
            name: "colon-def",
            code: ": SQUARE DUP * ; 6 SQUARE . CR 11 SQUARE . CR",
            expected: "36 \n121 \n",
            category: Category::Definitions,
        },
        Program {
            name: "create-does",
            code: ": CONST CREATE , DOES> @ ;\n\
                   99 CONST NINETY-NINE\n\
                   NINETY-NINE . CR",
            expected: "99 \n",
            category: Category::Definitions,
        },
        // -- Strings --
        Program {
            name: "s-quote-type",
            code: "S\" hello\" TYPE CR",
            expected: "hello\n",
            category: Category::Strings,
        },
        Program {
            name: "dot-quote",
            code: ".\" world\" CR",
            expected: "world\n",
            category: Category::Strings,
        },
        Program {
            name: "char-emit",
            code: ": EMIT-AB [CHAR] A EMIT [CHAR] B EMIT ; EMIT-AB CR",
            expected: "AB\n",
            category: Category::Strings,
        },
        // -- Recursion --
        Program {
            name: "fibonacci",
            code: ": FIB DUP 2 < IF EXIT THEN DUP 1- RECURSE SWAP 2 - RECURSE + ;\n\
                   25 FIB . CR",
            expected: "75025 \n",
            category: Category::Recursion,
        },
        Program {
            name: "factorial",
            code: ": FACT 1 SWAP 1+ 1 ?DO I * LOOP ; 12 FACT . CR",
            expected: "479001600 \n",
            category: Category::Recursion,
        },
        Program {
            name: "gcd",
            code: ": GCD BEGIN DUP WHILE TUCK MOD REPEAT DROP ; 48 36 GCD . CR",
            expected: "12 \n",
            category: Category::Recursion,
        },
        // -- Memory --
        Program {
            name: "create-allot",
            code: "CREATE ARR 5 CELLS ALLOT\n\
                   99 ARR 3 CELLS + !\n\
                   ARR 3 CELLS + @ . CR",
            expected: "99 \n",
            category: Category::Memory,
        },
        Program {
            name: "fill-sum",
            code: "CREATE BUF 10 CELLS ALLOT\n\
                   : FILL-BUF 10 0 DO I I * BUF I CELLS + ! LOOP ;\n\
                   : SUM-BUF 0 10 0 DO BUF I CELLS + @ + LOOP ;\n\
                   FILL-BUF SUM-BUF . CR",
            expected: "285 \n",
            category: Category::Memory,
        },
    ]
}

// -----------------------------------------------------------------------
// WAFER-only correctness tests (always run in CI)
// -----------------------------------------------------------------------

fn run_category(cat: Category) {
    for prog in programs().iter().filter(|p| p.category == cat) {
        assert_wafer_output(prog.name, prog.code, prog.expected);
    }
}

#[test]
fn wafer_arithmetic() {
    run_category(Category::Arithmetic);
}

#[test]
fn wafer_stack_ops() {
    run_category(Category::StackOps);
}

#[test]
fn wafer_control_flow() {
    run_category(Category::ControlFlow);
}

#[test]
fn wafer_loops() {
    run_category(Category::Loops);
}

#[test]
fn wafer_definitions() {
    run_category(Category::Definitions);
}

#[test]
fn wafer_strings() {
    run_category(Category::Strings);
}

#[test]
fn wafer_recursion() {
    run_category(Category::Recursion);
}

#[test]
fn wafer_memory() {
    run_category(Category::Memory);
}

/// Verify that all optimizations produce the same output as unoptimized.
#[test]
fn wafer_optimized_matches_unoptimized() {
    for prog in programs() {
        let base = run_wafer(prog.code);
        let opt = run_wafer_optimized(prog.code);
        assert!(base.success, "{}: unoptimized failed", prog.name);
        assert!(opt.success, "{}: optimized failed", prog.name);
        assert_eq!(
            normalize(&base.output),
            normalize(&opt.output),
            "{}: optimized output differs from unoptimized",
            prog.name
        );
    }
}

// -----------------------------------------------------------------------
// Cross-engine behavioral comparison (requires gforth)
// -----------------------------------------------------------------------

#[test]
#[ignore = "requires gforth installation"]
fn compare_all_programs() {
    if find_gforth().is_none() {
        eprintln!("SKIP: gforth not found in PATH");
        return;
    }
    let progs = programs();
    let mut passed = 0;
    let mut skipped = 0;
    for prog in &progs {
        let wafer = run_wafer(prog.code);
        if !wafer.success {
            panic!("{}: WAFER execution failed", prog.name);
        }
        let Some(gforth) = run_gforth(prog.code) else {
            skipped += 1;
            continue;
        };
        if !gforth.success {
            eprintln!("  WARN {}: gforth execution failed, skipping", prog.name);
            skipped += 1;
            continue;
        }
        assert_eq!(
            normalize(&wafer.output),
            normalize(&gforth.output),
            "{}: output differs\n  WAFER:  {:?}\n  gforth: {:?}",
            prog.name,
            wafer.output,
            gforth.output
        );
        passed += 1;
    }
    eprintln!(
        "\nBehavioral comparison: {passed} passed, {skipped} skipped (of {})",
        progs.len()
    );
}

// -----------------------------------------------------------------------
// Performance comparison (requires gforth)
// -----------------------------------------------------------------------

struct PerfBenchmark {
    name: &'static str,
    define: &'static str,
    /// The workload to time — should include its own iteration loop for
    /// fast operations so that total execution time is measurable.
    run_code: &'static str,
    verify: &'static str,
    expected: i32,
    samples: u32, // Number of runs for WAFER median
    /// Maximum acceptable WAFER/gforth ratio (< 1.0 = WAFER faster).
    /// Test fails if ratio exceeds this. Set ~40-50% above measured baseline.
    max_ratio: f64,
}

fn perf_benchmarks() -> Vec<PerfBenchmark> {
    vec![
        PerfBenchmark {
            name: "Fibonacci(25)",
            define: ": FIB DUP 2 < IF EXIT THEN DUP 1- RECURSE SWAP 2 - RECURSE + ;",
            run_code: "25 FIB DROP",
            verify: "25 FIB",
            expected: 75025,
            samples: 5,
            max_ratio: 0.65,
        },
        PerfBenchmark {
            name: "Factorial(12)x10K",
            define: ": FACT 1 SWAP 1+ 1 ?DO I * LOOP ; \
                     : FACT-BENCH 10000 0 DO 12 FACT DROP LOOP ;",
            run_code: "FACT-BENCH",
            verify: "12 FACT",
            expected: 479001600,
            samples: 5,
            max_ratio: 0.75,
        },
        PerfBenchmark {
            name: "GCD-bench(500)",
            define: ": GCD BEGIN DUP WHILE TUCK MOD REPEAT DROP ; \
                     : GCD-BENCH 0 DO 10000 I 1+ GCD DROP LOOP ;",
            run_code: "500 GCD-BENCH",
            verify: "48 36 GCD",
            expected: 12,
            samples: 5,
            max_ratio: 0.70,
        },
        PerfBenchmark {
            name: "NestedLoops(50)",
            define: ": NESTED 0 SWAP 0 DO I 0 ?DO I J + DROP LOOP LOOP ; \
                     : NESTED-BENCH 100 0 DO 50 NESTED DROP LOOP ;",
            run_code: "NESTED-BENCH",
            verify: "5 NESTED",
            expected: 0,
            samples: 3,
            max_ratio: 0.20,
        },
        PerfBenchmark {
            name: "Collatz(2K)",
            define: ": COLLATZ 0 SWAP BEGIN DUP 1 > WHILE \
                     DUP 1 AND IF 3 * 1+ ELSE 2 / THEN \
                     SWAP 1+ SWAP REPEAT DROP ; \
                     : COLLATZ-BENCH 0 DO I 1+ COLLATZ DROP LOOP ;",
            run_code: "2000 COLLATZ-BENCH",
            verify: "27 COLLATZ",
            expected: 111,
            samples: 3,
            max_ratio: 0.45,
        },
    ]
}

/// Build the WAFER release binary and return its path.
/// Returns None if the build fails.
fn build_wafer_release() -> Option<String> {
    // Find workspace root (two levels up from crates/core)
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest_dir).parent()?.parent()?;
    let output = Command::new("cargo")
        .args(["build", "--release", "-p", "wafer"])
        .current_dir(workspace_root)
        .output()
        .ok()?;
    if !output.status.success() {
        eprintln!(
            "WARN: cargo build --release failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return None;
    }
    let target_dir = workspace_root
        .join(std::env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".to_string()));
    let binary = target_dir.join("release/wafer");
    if binary.exists() {
        Some(binary.to_string_lossy().into_owned())
    } else {
        None
    }
}

static WAFER_RELEASE: OnceLock<Option<String>> = OnceLock::new();

fn find_wafer_release() -> Option<&'static str> {
    WAFER_RELEASE.get_or_init(build_wafer_release).as_deref()
}

/// Measure WAFER execution time using a release-mode binary with UTIME.
/// Same approach as gforth: Forth-level timing excludes startup.
fn measure_wafer_release(wafer: &str, bench: &PerfBenchmark) -> Option<u64> {
    let code = format!(
        "{define} {run} \
         : TIMED-BENCH UTIME {run} UTIME 2SWAP D- DROP . CR ; \
         TIMED-BENCH TIMED-BENCH TIMED-BENCH",
        define = bench.define,
        run = bench.run_code,
    );
    let output = run_via_stdin(wafer, &code)?;
    if !output.status.success() {
        return None;
    }
    median_printed_time(&output.stdout)
}

/// Measure WAFER execution time after CONSOLIDATE (direct calls between all words).
fn measure_wafer_consolidated(wafer: &str, bench: &PerfBenchmark) -> Option<u64> {
    let code = format!(
        "{define} CONSOLIDATE {run} \
         : TIMED-BENCH UTIME {run} UTIME 2SWAP D- DROP . CR ; \
         TIMED-BENCH TIMED-BENCH TIMED-BENCH",
        define = bench.define,
        run = bench.run_code,
    );
    let output = run_via_stdin(wafer, &code)?;
    if !output.status.success() {
        return None;
    }
    median_printed_time(&output.stdout)
}

/// Parse the microsecond values printed by TIMED-BENCH (one per line) and
/// return the median.
fn median_printed_time(stdout: &[u8]) -> Option<u64> {
    let stdout = String::from_utf8_lossy(stdout);
    let mut times: Vec<u64> = stdout
        .trim()
        .lines()
        .filter_map(|l| l.trim().parse::<u64>().ok())
        .collect();
    times.sort();
    if times.is_empty() {
        return None;
    }
    Some(times[times.len() / 2])
}

/// Measure gforth execution time using Forth-level `utime` (excludes startup).
/// Both engines run the exact same `run_code`, so the comparison is apples-to-apples.
/// Returns microseconds, or None if gforth is unavailable.
fn measure_gforth(gforth: &str, bench: &PerfBenchmark) -> Option<u64> {
    // The timing wrapper must be inside a word (DO/LOOP is compile-only in gforth).
    // We take the median of 3 runs.
    let code = format!(
        "{define} {run} \
         : TIMED-BENCH utime {run} utime 2swap d- drop . CR ; \
         TIMED-BENCH TIMED-BENCH TIMED-BENCH bye",
        define = bench.define,
        run = bench.run_code,
    );
    let output = Command::new(gforth).arg("-e").arg(&code).output().ok()?;
    if !output.status.success() {
        return None;
    }
    median_printed_time(&output.stdout)
}

/// Measure `SwiftForth` (`sf64`) execution time using Forth-level `ucounter`
/// (double-cell microsecond counter; `2swap d- drop` yields elapsed us —
/// the same wrapper shape as gforth's `utime`). Timing excludes startup.
/// sf64 has no `-e` flag, so the program is piped via stdin — one statement
/// per line, because sf64 truncates input lines at ~256 chars.
/// Returns microseconds, or None if sf64 is unavailable or fails.
fn measure_sf64(sf64: &str, bench: &PerfBenchmark) -> Option<u64> {
    let code = format!(
        "{define}\n{run}\n\
         : TIMED-BENCH ucounter {run} ucounter 2swap d- drop . cr ;\n\
         TIMED-BENCH\nTIMED-BENCH\nTIMED-BENCH\nbye\n",
        define = bench.define,
        run = bench.run_code,
    );
    let output = run_via_stdin(sf64, &code)?;
    if !output.status.success() {
        return None;
    }
    median_printed_time(&output.stdout)
}

#[test]
#[ignore = "requires gforth installation"]
fn performance_report() {
    let gforth = find_gforth();
    let gforth_fast = find_gforth_fast();
    let wafer_release = find_wafer_release();
    if gforth.is_none() {
        eprintln!("SKIP: gforth not found");
        return;
    }
    if wafer_release.is_none() {
        eprintln!("WARN: could not build WAFER release binary, using in-process (debug) timing");
    }

    let benchmarks = perf_benchmarks();

    // Verify correctness first
    for bench in &benchmarks {
        let mut vm = ForthVM::<NativeRuntime>::new().expect("VM creation failed");
        for line in bench.define.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                let _ = vm.evaluate(trimmed);
            }
        }
        vm.take_output();
        vm.evaluate(bench.verify)
            .unwrap_or_else(|e| panic!("{}: verify failed: {e}", bench.name));
        vm.take_output();
        let stack = vm.data_stack();
        assert_eq!(
            stack.first().copied().unwrap_or(-1),
            bench.expected,
            "{}: wrong result",
            bench.name
        );
    }

    let sf64 = find_sf64();
    if sf64.is_none() {
        eprintln!("NOTE: sf64 (SwiftForth) not found — column skipped");
    }

    let sep = "=".repeat(100);
    let thin = "-".repeat(100);
    println!("\n{sep}");
    println!("  WAFER vs Gforth vs SwiftForth Performance Comparison (release mode)");
    println!("{sep}\n");
    println!(
        "{:<22} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "Benchmark",
        "WAFER",
        "CONSOL",
        "gforth",
        "gf-fast",
        "sf64",
        "WAFER/gf",
        "WAFER/sf",
        "limit"
    );
    println!(
        "{:<22} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "", "(us)", "(us)", "(us)", "(us)", "(us)", "", "", ""
    );
    println!("{thin}");

    let mut regressions: Vec<String> = Vec::new();

    for bench in &benchmarks {
        let wafer = wafer_release
            .and_then(|w| measure_wafer_release(w, bench))
            .unwrap_or(0);
        let consol = wafer_release
            .and_then(|w| measure_wafer_consolidated(w, bench))
            .unwrap_or(0);
        let gf = gforth.and_then(|g| measure_gforth(g, bench));
        let gf_fast = gforth_fast.and_then(|g| measure_gforth(g, bench));
        let sf = sf64.and_then(|s| measure_sf64(s, bench));

        let gf_str = gf.map_or_else(|| "-".to_string(), |v| format!("{v}"));
        let gf_fast_str = gf_fast.map_or_else(|| "-".to_string(), |v| format!("{v}"));
        let sf_str = sf.map_or_else(|| "-".to_string(), |v| format!("{v}"));
        let best_wafer = if consol > 0 && consol < wafer {
            consol
        } else {
            wafer
        };
        let ratio_val = gf.and_then(|g| {
            if g > 0 {
                Some(best_wafer as f64 / g as f64)
            } else {
                None
            }
        });
        let ratio = ratio_val.map_or_else(|| "-".to_string(), |r| format!("{r:.2}x"));
        let sf_ratio = sf.filter(|&s| s > 0).map_or_else(
            || "-".to_string(),
            |s| format!("{:.2}x", best_wafer as f64 / s as f64),
        );
        let limit_str = format!("{:.2}x", bench.max_ratio);

        println!(
            "{:<22} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
            bench.name, wafer, consol, gf_str, gf_fast_str, sf_str, ratio, sf_ratio, limit_str
        );

        // Check regression limits
        if let Some(r) = ratio_val {
            if r > bench.max_ratio {
                regressions.push(format!(
                    "  {} ratio {:.2}x exceeds limit {:.2}x (REGRESSION)",
                    bench.name, r, bench.max_ratio
                ));
            }
            if r < 0.02 {
                regressions.push(format!(
                    "  {} ratio {:.4}x suspiciously low (measurement error?)",
                    bench.name, r
                ));
            }
        }
    }

    println!("{thin}");
    println!("  WAFER = all optimizations, CONSOL = after CONSOLIDATE");
    println!("  WAFER/gf = best(WAFER,CONSOL) vs gforth, < 1.0 means WAFER faster");
    println!(
        "  WAFER/sf = best(WAFER,CONSOL) vs SwiftForth sf64 (native code; informational, no limit)"
    );
    println!("{sep}\n");

    if !regressions.is_empty() {
        let msg = regressions.join("\n");
        panic!("Performance regression detected:\n{msg}");
    }
}
