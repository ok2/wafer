//! Forth compile mode: builds IR from word definitions.
//!
//! When the outer interpreter encounters `:`, it switches to compile mode.
//! The compiler collects tokens and builds an IR representation until `;`.
//! IMMEDIATE words are executed during compilation (e.g., IF, ELSE, THEN).

// TODO: Step 7 - Compiler implementation
// - : (colon) starts compilation, ; (semicolon) ends it
// - Build Vec<IrOp> for the word body
// - Handle IMMEDIATE words
// - Handle control structures (IF/ELSE/THEN, DO/LOOP, BEGIN/UNTIL)
// - LITERAL, POSTPONE, ['], [CHAR]
// - Defining words: VARIABLE, CONSTANT, CREATE, DOES>

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {
        // Compiler tests will be added in Step 7
    }
}
