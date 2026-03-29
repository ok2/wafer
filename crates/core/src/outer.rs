//! Outer interpreter: tokenizer, number parser, and interpret/compile dispatch.
//!
//! The outer interpreter is the main loop of Forth:
//! 1. Read a token (whitespace-delimited word)
//! 2. Look it up in the dictionary
//! 3. If found: execute (interpret mode) or compile (compile mode)
//! 4. If not found: try to parse as a number
//! 5. If number: push (interpret) or compile as literal (compile mode)
//! 6. If neither: error

// TODO: Step 8 - Outer interpreter implementation
// - Tokenizer (whitespace splitting, string literals)
// - Number parsing (decimal, #decimal, $hex, %binary per Forth 2012)
// - Main interpret/compile dispatch loop
// - STATE management
// - EVALUATE support (nested interpretation)

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {
        // Outer interpreter tests will be added in Step 8
    }
}
