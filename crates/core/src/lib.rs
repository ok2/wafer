//! WAFER Core: WebAssembly Forth Engine in Rust
//!
//! This crate provides the core compiler and runtime for WAFER,
//! an optimizing Forth 2012 compiler targeting WebAssembly.
//!
//! # Architecture
//!
//! ```text
//! Forth Source -> Outer Interpreter -> IR -> Optimize -> WASM Codegen
//! ```
//!
//! The compilation pipeline:
//! 1. **Outer interpreter** tokenizes input and dispatches to interpret/compile mode
//! 2. **Optimizer** applies transformation passes (constant folding, inlining, etc.)
//! 3. **Codegen** translates optimized IR to WASM bytecode via `wasm-encoder`

pub mod codegen;
pub mod config;
#[cfg(feature = "crypto")]
pub mod crypto;
pub mod dictionary;
pub mod error;
pub mod ir;
pub mod memory;
pub mod optimizer;
pub mod runtime;
pub mod see;
pub mod wordhelp;

// Outer interpreter: runtime-agnostic, works with any Runtime impl
#[allow(trivial_numeric_casts, clippy::unnecessary_cast)]
pub mod outer;

// Modules requiring the native wasmtime runtime
#[cfg(feature = "native")]
pub mod consolidate;
#[cfg(feature = "native")]
pub mod export;
#[cfg(feature = "native")]
pub mod js_loader;
#[cfg(feature = "native")]
pub mod runner;
#[cfg(feature = "native")]
pub mod runtime_native;
