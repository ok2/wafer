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
//! 2. **Compiler** builds an intermediate representation (IR) for each word definition
//! 3. **Type inference** annotates the IR with stack types
//! 4. **Optimizer** applies transformation passes (constant folding, inlining, etc.)
//! 5. **Codegen** translates optimized IR to WASM bytecode via `wasm-encoder`

pub mod codegen;
pub mod compiler;
pub mod config;
pub mod consolidate;
pub mod dictionary;
pub mod error;
pub mod ir;
pub mod memory;
pub mod optimizer;
pub mod outer;
pub mod primitives;
pub mod types;
pub mod words;
