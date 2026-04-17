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

# Check formatting (Rust + Markdown)
fmt:
    cargo fmt --all --check
    dprint check

# Format code (Rust + Markdown)
fmt-fix:
    cargo fmt --all
    dprint fmt

# Run the REPL
repl:
    cargo run -p wafer

# Run a Forth file
run file:
    cargo run -p wafer -- {{file}}

# Run benchmarks
bench:
    cargo bench --workspace

# Run optimization benchmark report
bench-opts:
    cargo test -p wafer-core --test benchmark_report -- --nocapture --ignored

# Check dependency licenses and advisories
deny:
    cargo deny check

# Detect unused dependencies
machete:
    cargo machete --skip-target-dir

# Full CI check (what CI runs)
ci: fmt clippy deny test

# Check compilation without running
check:
    cargo check --workspace

# Install bat syntax highlighting for WAFER / Forth
install-syntax:
    mkdir -p ~/.config/bat/syntaxes
    cp tools/editor-support/bat/WAFER.sublime-syntax ~/.config/bat/syntaxes/
    bat cache --build
