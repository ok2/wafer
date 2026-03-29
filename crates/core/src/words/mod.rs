//! Forth 2012 word set implementations.
//!
//! Each submodule implements one word set from the Forth 2012 standard.
//! Words are implemented in Rust only when they require direct WASM instructions;
//! most words are defined in Forth source files under `forth/`.

// Word set modules will be added as each set is implemented:
// pub mod core;
// pub mod core_ext;
// pub mod double;
// pub mod exception;
// pub mod floating;
// pub mod locals;
// pub mod string;
// pub mod tools;
// pub mod memory_alloc;
// pub mod search_order;
// pub mod file;
// pub mod facility;
