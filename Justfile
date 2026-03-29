default:
    just --list

# Build everything
build:
    cargo build --workspace

# Run all tests
test:
    cargo test --workspace

# Run Forth 2012 compliance tests
compliance:
    cargo test --test compliance -- --nocapture

# Run clippy lints
clippy:
    cargo clippy --workspace -- -D warnings

# Check formatting
fmt:
    cargo fmt --all --check

# Format code
fmt-fix:
    cargo fmt --all

# Run the REPL
repl:
    cargo run -p wafer

# Run a Forth file
run file:
    cargo run -p wafer -- {{file}}

# Run benchmarks
bench:
    cargo bench --workspace

# Full CI check (what CI runs)
ci: fmt clippy test

# Check compilation without running
check:
    cargo check --workspace
