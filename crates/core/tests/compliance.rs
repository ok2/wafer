//! Forth 2012 compliance tests using Gerry Jackson's test suite.
//!
//! Each test function loads the corresponding test file from the
//! forth2012-test-suite submodule and runs it through WAFER.
//! Tests are initially `#[ignore]` and enabled as word sets are implemented.

/// Path to the test suite source directory.
/// The submodule lives at the workspace root: tests/forth2012-test-suite/
const _TEST_SUITE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/forth2012-test-suite/src"
);

// TODO: Test harness that boots WAFER, loads tester.fr, and runs test files.
// For now, these are placeholder tests that document the compliance targets.

#[test]
#[ignore = "Step 10: Core word set not yet implemented"]
fn compliance_core() {
    // Will load: tester.fr, then core.fr
    // Must pass with 0 failures
    todo!("Boot WAFER, load tester.fr + core.fr, assert 0 failures");
}

#[test]
#[ignore = "Step 10: Core word set not yet implemented"]
fn compliance_core_plus() {
    // Will load: tester.fr, then coreplustest.fth
    todo!("Boot WAFER, load tester.fr + coreplustest.fth");
}

#[test]
#[ignore = "Step 13: Core extensions not yet implemented"]
fn compliance_core_ext() {
    // Will load: tester.fr, utilities.fth, then coreexttest.fth
    todo!("Boot WAFER, load coreexttest.fth");
}

#[test]
#[ignore = "Step 13: Double-number word set not yet implemented"]
fn compliance_double() {
    todo!("Boot WAFER, load doubletest.fth");
}

#[test]
#[ignore = "Step 13: Exception word set not yet implemented"]
fn compliance_exception() {
    todo!("Boot WAFER, load exceptiontest.fth");
}

#[test]
#[ignore = "Step 13: Facility word set not yet implemented"]
fn compliance_facility() {
    todo!("Boot WAFER, load facilitytest.fth");
}

#[test]
#[ignore = "Step 13: File-access word set not yet implemented"]
fn compliance_file() {
    todo!("Boot WAFER, load filetest.fth");
}

#[test]
#[ignore = "Step 13: Locals word set not yet implemented"]
fn compliance_locals() {
    todo!("Boot WAFER, load localstest.fth");
}

#[test]
#[ignore = "Step 13: Memory-allocation word set not yet implemented"]
fn compliance_memory() {
    todo!("Boot WAFER, load memorytest.fth");
}

#[test]
#[ignore = "Step 13: Search-order word set not yet implemented"]
fn compliance_search_order() {
    todo!("Boot WAFER, load searchordertest.fth");
}

#[test]
#[ignore = "Step 13: String word set not yet implemented"]
fn compliance_string() {
    todo!("Boot WAFER, load stringtest.fth");
}

#[test]
#[ignore = "Step 13: Programming-tools word set not yet implemented"]
fn compliance_tools() {
    todo!("Boot WAFER, load toolstest.fth");
}
