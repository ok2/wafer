//! Built-in primitive words for WAFER.
//!
//! Primitives are the ~35 words that must be implemented in Rust because
//! they require direct WASM instructions or host interaction.
//! Everything else is defined in Forth (loaded from .fth files).

// TODO: Step 6 - Primitive word implementations
// Each primitive provides:
// - Its StackEffect (type signature)
// - Its IR representation (for inlining by the optimizer)
// - Direct WASM instruction generation

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {
        // Primitive tests will be added in Step 6
    }
}
