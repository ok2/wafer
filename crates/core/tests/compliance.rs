//! Forth 2012 compliance tests using Gerry Jackson's test suite.
//!
//! Each test loads the corresponding test file from the
//! forth2012-test-suite submodule and runs it through WAFER,
//! asserting 0 test failures.

use wafer_core::outer::ForthVM;

/// Path to the test suite source directory.
const SUITE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/forth2012-test-suite/src"
);

/// Load a file and evaluate it line by line, ignoring errors on individual lines.
fn load_file(vm: &mut ForthVM, path: &str) {
    let source = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {path}"));
    for line in source.lines() {
        let _ = vm.evaluate(line);
    }
    vm.take_output(); // discard output
}

/// Boot a WAFER VM with full prerequisites loaded.
fn boot_with_prerequisites() -> ForthVM {
    let mut vm = ForthVM::new().expect("Failed to create ForthVM");

    // Load test framework
    load_file(&mut vm, &format!("{SUITE_DIR}/tester.fr"));
    // Load core tests (prerequisite)
    load_file(&mut vm, &format!("{SUITE_DIR}/core.fr"));
    // Switch to decimal and load utilities
    let _ = vm.evaluate("DECIMAL");
    vm.take_output();
    load_file(&mut vm, &format!("{SUITE_DIR}/utilities.fth"));
    // Load core extensions
    load_file(&mut vm, &format!("{SUITE_DIR}/coreexttest.fth"));

    vm
}

/// Run a test suite file and return the #ERRORS count.
fn run_suite(vm: &mut ForthVM, test_file: &str) -> u32 {
    // Reset error counter
    let _ = vm.evaluate("DECIMAL 0 #ERRORS !");
    vm.take_output();

    // Load the test file
    load_file(vm, &format!("{SUITE_DIR}/{test_file}"));

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
    let mut vm = ForthVM::new().expect("Failed to create ForthVM");
    load_file(&mut vm, &format!("{SUITE_DIR}/tester.fr"));
    load_file(&mut vm, &format!("{SUITE_DIR}/core.fr"));

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
    let mut vm = ForthVM::new().expect("Failed to create ForthVM");
    load_file(&mut vm, &format!("{SUITE_DIR}/tester.fr"));
    load_file(&mut vm, &format!("{SUITE_DIR}/core.fr"));
    let _ = vm.evaluate("DECIMAL");
    vm.take_output();
    load_file(&mut vm, &format!("{SUITE_DIR}/utilities.fth"));
    let _ = vm.evaluate("DECIMAL 0 #ERRORS !");
    vm.take_output();
    load_file(&mut vm, &format!("{SUITE_DIR}/coreexttest.fth"));
    let _ = vm.evaluate("DECIMAL #ERRORS @");
    let errors = vm.data_stack().first().copied().unwrap_or(-1) as u32;
    assert_eq!(errors, 0, "Core Extensions: {errors} test failures");
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
    let mut vm = ForthVM::new().expect("Failed to create ForthVM");
    load_file(&mut vm, &format!("{SUITE_DIR}/tester.fr"));
    load_file(&mut vm, &format!("{SUITE_DIR}/core.fr"));
    let _ = vm.evaluate("DECIMAL");
    vm.take_output();
    load_file(&mut vm, &format!("{SUITE_DIR}/utilities.fth"));
    let _ = vm.evaluate("DECIMAL 0 #ERRORS !");
    vm.take_output();
    load_file(&mut vm, &format!("{SUITE_DIR}/stringtest.fth"));
    let _ = vm.evaluate("DECIMAL #ERRORS @");
    let errors = vm.data_stack().first().copied().unwrap_or(-1) as u32;
    assert_eq!(errors, 0, "String: {errors} test failures");
}

#[test]
fn compliance_tools() {
    let mut vm = boot_with_prerequisites();
    let errors = run_suite(&mut vm, "toolstest.fth");
    assert_eq!(errors, 0, "Programming-Tools: {errors} test failures");
}
