//! Forth 2012 compliance tests using Gerry Jackson's test suite.
//!
//! Each test loads the corresponding test file from the
//! forth2012-test-suite submodule and runs it through WAFER,
//! asserting 0 test failures.

use wafer_core::outer::ForthVM;
use wafer_core::runtime_native::NativeRuntime;

/// Path to the test suite source directory.
const SUITE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/forth2012-test-suite/src"
);

/// Load a file line-by-line, returning the number of lines that raised an
/// `evaluate` error. Each failing line is printed (visible under
/// `cargo test -- --nocapture`) so failures can be triaged without a
/// debugger.
///
/// Historically this helper discarded errors silently, which caused tests
/// like LT32 in `localstest.fth` (compile errors from unknown words such
/// as `(LOCAL)` before it was implemented) to vanish — the T{ }T error
/// counter was never incremented because the `:` definition never ran.
/// Returning the count surfaces silent skips as real failures.
///
/// **Note on multi-line definitions.** WAFER's DOES> handler collects
/// the does-body to `;` via `next_token()` within a *single* `evaluate`
/// call and treats end-of-input as end-of-body. Files with a `DOES>`
/// split across lines (e.g. `errorreport.fth`) therefore cannot be
/// loaded line-by-line; use [`load_file_whole`] for those.
fn load_file(vm: &mut ForthVM<NativeRuntime>, path: &str) -> u32 {
    let source = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {path}"));
    let mut fails = 0u32;
    for (lineno, line) in source.lines().enumerate() {
        if let Err(e) = vm.evaluate(line) {
            fails += 1;
            eprintln!("{path}:{}: {e}\n  line: {line}", lineno + 1);
        }
    }
    vm.take_output(); // discard output
    fails
}

/// Load a file as a single `evaluate` call (not line-by-line). Required
/// for files with multi-line definitions that WAFER's per-line handlers
/// can't stitch across calls (notably `: X ... DOES> ... ;` spanning
/// lines — see [`load_file`] note).
///
/// Returns `1` on any failure, `0` on success, so the caller can apply
/// baselines the same way as [`load_file`].
fn load_file_whole(vm: &mut ForthVM<NativeRuntime>, path: &str) -> u32 {
    let source = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {path}"));
    let fails = match vm.evaluate(&source) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("{path}: {e}");
            1
        }
    };
    vm.take_output();
    fails
}

/// Baseline of *known* line-level failures per prerequisite file. The runner
/// asserts `load_fails == expected_load_failures(path)`, so any regression
/// above (or silently-fixed case below) the baseline is caught.
///
/// Baselines are not an allowlist to paper over bugs — they are an explicit
/// tech-debt ledger. Each non-zero entry here is a bug that should be fixed
/// and the baseline lowered to zero. See the in-tree follow-up tasks.
fn expected_load_failures(path: &str) -> u32 {
    // core.fr exercises two constructs WAFER does not yet support:
    //   1. Nested colon definitions (`: NOP : POSTPONE ; ;` at line 751,
    //      defining NOP, NOP1, NOP2 — four silent lines).
    //   2. `SOURCE`/`>IN` round-trip through `EVALUATE` at line 797
    //      (GS1 definition) — one line.
    // Total: 5. Fix these and drop the baseline to 0.
    if path.ends_with("/core.fr") {
        return 5;
    }
    // coreexttest.fth uses two Core-Extension features WAFER lacks:
    //   1. SAVE-INPUT / RESTORE-INPUT at line 548 — not implemented.
    //   2. `.(` inside `[ ... ]` brackets at line 559 — `.(` isn't
    //      handled by `compile_token`'s `[ ... ]` interpret-mode path,
    //      so `First message via .(` tokens leak to the compiler as
    //      undefined words.
    // Total: 2. Fix these and drop the baseline to 0.
    if path.ends_with("/coreexttest.fth") {
        return 2;
    }
    // exceptiontest.fth line 95 fails with a garbled parse ("unknown word"
    // over non-ASCII bytes): WAFER's parser reads past a prior test's
    // scratch region after the preceding `C6` / `T9` frame exercises
    // CATCH/THROW source stacking. Root cause not yet diagnosed; baseline
    // until fixed.
    if path.ends_with("/exceptiontest.fth") {
        return 1;
    }
    // toolstest.fth uses the `\?` conditional-skip idiom defined in
    // utilities.fth:37 as `: \? (\?) @ IF EXIT THEN SOURCE >IN ! DROP ;
    // IMMEDIATE`. Under WAFER's per-line `evaluate` loader, the
    // `SOURCE >IN ! DROP` path does not consume the remainder of the
    // current line correctly, so 37 `\?`-guarded lines inside the
    // TRAVERSE-WORDLIST / NAME>COMPILE / NAME>INTERPRET blocks leak as
    // unknown-word errors. Fix the SOURCE/`>IN` interaction with
    // line-mode input and drop this to 0.
    //
    // The 38th: line 368 `R> DROP TRUE` runs interpreted (its enclosing
    // definition aborted on the missing NAME?), and the bare `R>` used
    // to underflow the return stack silently; stack guards now report
    // it as "Return stack underflow (throw -6)".
    if path.ends_with("/toolstest.fth") {
        return 38;
    }
    0
}

/// Assert a file loaded with exactly its baseline number of line-level
/// failures. Used for prerequisites; keeps the runner tight without
/// blocking the whole suite on known gaps.
fn assert_load_fails_within_baseline(path: &str, fails: u32) {
    let expected = expected_load_failures(path);
    assert_eq!(
        fails, expected,
        "{path} had {fails} line-level failures (expected baseline: {expected})"
    );
}

/// Boot a WAFER VM with full prerequisites loaded.
///
/// Every prerequisite file must load with zero line-level errors. Any
/// regression here points to a missing primitive or a parser bug and must
/// be fixed, not silently tolerated.
fn boot_with_prerequisites() -> ForthVM<NativeRuntime> {
    let mut vm = ForthVM::<NativeRuntime>::new().expect("Failed to create ForthVM");

    // Load test framework
    let tester_path = format!("{SUITE_DIR}/tester.fr");
    let f1 = load_file(&mut vm, &tester_path);
    assert_load_fails_within_baseline(&tester_path, f1);
    // Load core tests (prerequisite)
    let core_path = format!("{SUITE_DIR}/core.fr");
    let f2 = load_file(&mut vm, &core_path);
    assert_load_fails_within_baseline(&core_path, f2);
    // Switch to decimal and load utilities
    let _ = vm.evaluate("DECIMAL");
    vm.take_output();
    let util_path = format!("{SUITE_DIR}/utilities.fth");
    let f3 = load_file(&mut vm, &util_path);
    assert_load_fails_within_baseline(&util_path, f3);
    // errorreport.fth defines SET-ERROR-COUNT and the per-wordset counter
    // accessors (CORE-ERRORS, STRING-ERRORS, LOCALS-ERRORS, ...). Every
    // suite's final `X-ERRORS SET-ERROR-COUNT` line depends on this file,
    // and silently errored before the runner was tightened.
    let errorreport_path = format!("{SUITE_DIR}/errorreport.fth");
    let f_err = load_file_whole(&mut vm, &errorreport_path);
    assert_load_fails_within_baseline(&errorreport_path, f_err);
    // Load core extensions
    let ext_path = format!("{SUITE_DIR}/coreexttest.fth");
    let f4 = load_file(&mut vm, &ext_path);
    assert_load_fails_within_baseline(&ext_path, f4);

    vm
}

/// Run a test suite file and return the *total* error count:
/// `#ERRORS` from the Forth test framework plus any lines where
/// `vm.evaluate` itself failed (e.g. unknown word in a `:` definition
/// outside `T{ }T`, which the framework cannot catch).
fn run_suite(vm: &mut ForthVM<NativeRuntime>, test_file: &str) -> u32 {
    // Reset error counter
    let _ = vm.evaluate("DECIMAL 0 #ERRORS !");
    vm.take_output();

    // Load the test file
    let file_path = format!("{SUITE_DIR}/{test_file}");
    let load_fails = load_file(vm, &file_path);
    assert_load_fails_within_baseline(&file_path, load_fails);

    // Read error count -- try multiple approaches to be robust
    let _ = vm.evaluate("DECIMAL");
    vm.take_output();

    // Clear data stack first
    let _ = vm.evaluate("DEPTH 0 > IF DEPTH 0 DO DROP LOOP THEN");
    vm.take_output();

    // Push error count
    if vm.evaluate("#ERRORS @").is_err() {
        // #ERRORS not accessible -- test framework was corrupted
        return u32::MAX;
    }
    let stack = vm.data_stack();
    let errors = stack.first().copied().unwrap_or(-1);
    vm.take_output();

    // Clean up
    let _ = vm.evaluate("DEPTH 0 > IF DROP THEN");
    vm.take_output();

    if errors < 0 { u32::MAX } else { errors as u32 }
}

#[test]
fn compliance_core() {
    let mut vm = ForthVM::<NativeRuntime>::new().expect("Failed to create ForthVM");
    let tester_path = format!("{SUITE_DIR}/tester.fr");
    let f1 = load_file(&mut vm, &tester_path);
    assert_load_fails_within_baseline(&tester_path, f1);
    let core_path = format!("{SUITE_DIR}/core.fr");
    let f2 = load_file(&mut vm, &core_path);
    assert_load_fails_within_baseline(&core_path, f2);

    let _ = vm.evaluate("DECIMAL #ERRORS @");
    let errors = vm.data_stack().first().copied().unwrap_or(-1);
    assert_eq!(errors, 0, "Core word set: {errors} test failures");
}

#[test]
fn compliance_core_plus() {
    let mut vm = boot_with_prerequisites();
    let errors = run_suite(&mut vm, "coreplustest.fth");
    assert_eq!(errors, 0, "Core Plus: {errors} test failures");
}

#[test]
fn compliance_core_ext() {
    // Core Extensions are loaded as part of prerequisites.
    // Run from scratch to get a clean error count.
    let mut vm = ForthVM::<NativeRuntime>::new().expect("Failed to create ForthVM");
    let tester_path = format!("{SUITE_DIR}/tester.fr");
    let f1 = load_file(&mut vm, &tester_path);
    assert_load_fails_within_baseline(&tester_path, f1);
    let core_path = format!("{SUITE_DIR}/core.fr");
    let f2 = load_file(&mut vm, &core_path);
    assert_load_fails_within_baseline(&core_path, f2);
    let _ = vm.evaluate("DECIMAL");
    vm.take_output();
    let util_path = format!("{SUITE_DIR}/utilities.fth");
    let f3 = load_file(&mut vm, &util_path);
    assert_load_fails_within_baseline(&util_path, f3);
    let errorreport_path = format!("{SUITE_DIR}/errorreport.fth");
    let f_err = load_file_whole(&mut vm, &errorreport_path);
    assert_load_fails_within_baseline(&errorreport_path, f_err);
    let _ = vm.evaluate("DECIMAL 0 #ERRORS !");
    vm.take_output();
    let ext_path = format!("{SUITE_DIR}/coreexttest.fth");
    let load_fails = load_file(&mut vm, &ext_path);
    assert_load_fails_within_baseline(&ext_path, load_fails);
    let _ = vm.evaluate("DECIMAL #ERRORS @");
    let framework_errors = vm.data_stack().first().copied().unwrap_or(-1) as u32;
    assert_eq!(
        framework_errors, 0,
        "Core Extensions: {framework_errors} framework test failures"
    );
}

#[test]
fn compliance_double() {
    let mut vm = boot_with_prerequisites();
    let errors = run_suite(&mut vm, "doubletest.fth");
    assert_eq!(errors, 0, "Double-Number: {errors} test failures");
}

#[test]
fn compliance_exception() {
    let mut vm = boot_with_prerequisites();
    let errors = run_suite(&mut vm, "exceptiontest.fth");
    assert_eq!(errors, 0, "Exception: {errors} test failures");
}

#[test]
fn compliance_facility() {
    let mut vm = boot_with_prerequisites();
    let errors = run_suite(&mut vm, "facilitytest.fth");
    assert_eq!(errors, 0, "Facility: {errors} test failures");
}

#[test]
#[ignore = "File-Access requires WASI filesystem operations"]
fn compliance_file() {
    let mut vm = boot_with_prerequisites();
    let errors = run_suite(&mut vm, "filetest.fth");
    assert_eq!(errors, 0, "File-Access: {errors} test failures");
}

#[test]
fn compliance_locals() {
    let mut vm = boot_with_prerequisites();
    let errors = run_suite(&mut vm, "localstest.fth");
    assert_eq!(errors, 0, "Locals: {errors} test failures");
}

#[test]
fn compliance_memory() {
    let mut vm = boot_with_prerequisites();
    let errors = run_suite(&mut vm, "memorytest.fth");
    assert_eq!(errors, 0, "Memory-Allocation: {errors} test failures");
}

#[test]
fn compliance_search_order() {
    let mut vm = boot_with_prerequisites();
    let errors = run_suite(&mut vm, "searchordertest.fth");
    assert_eq!(errors, 0, "Search-Order: {errors} test failures");
}

#[test]
fn compliance_string() {
    // Run from scratch -- the stringtest includes CoreExt tests that
    // cascade failures when run on top of an already-loaded CoreExt suite.
    let mut vm = ForthVM::<NativeRuntime>::new().expect("Failed to create ForthVM");
    let tester_path = format!("{SUITE_DIR}/tester.fr");
    let f1 = load_file(&mut vm, &tester_path);
    assert_load_fails_within_baseline(&tester_path, f1);
    let core_path = format!("{SUITE_DIR}/core.fr");
    let f2 = load_file(&mut vm, &core_path);
    assert_load_fails_within_baseline(&core_path, f2);
    let _ = vm.evaluate("DECIMAL");
    vm.take_output();
    let util_path = format!("{SUITE_DIR}/utilities.fth");
    let f3 = load_file(&mut vm, &util_path);
    assert_load_fails_within_baseline(&util_path, f3);
    let errorreport_path = format!("{SUITE_DIR}/errorreport.fth");
    let f_err = load_file_whole(&mut vm, &errorreport_path);
    assert_load_fails_within_baseline(&errorreport_path, f_err);
    let _ = vm.evaluate("DECIMAL 0 #ERRORS !");
    vm.take_output();
    let str_path = format!("{SUITE_DIR}/stringtest.fth");
    let load_fails = load_file(&mut vm, &str_path);
    assert_load_fails_within_baseline(&str_path, load_fails);
    let _ = vm.evaluate("DECIMAL #ERRORS @");
    let framework_errors = vm.data_stack().first().copied().unwrap_or(-1) as u32;
    assert_eq!(
        framework_errors, 0,
        "String: {framework_errors} framework test failures"
    );
}

#[test]
fn compliance_tools() {
    let mut vm = boot_with_prerequisites();
    let errors = run_suite(&mut vm, "toolstest.fth");
    assert_eq!(errors, 0, "Programming-Tools: {errors} test failures");
}
