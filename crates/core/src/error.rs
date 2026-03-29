//! Error types for the WAFER compiler and runtime.

use thiserror::Error;

/// Errors that can occur during WAFER compilation and execution.
#[derive(Debug, Error)]
pub enum WaferError {
    #[error("stack underflow")]
    StackUnderflow,

    #[error("stack overflow")]
    StackOverflow,

    #[error("return stack underflow")]
    ReturnStackUnderflow,

    #[error("return stack overflow")]
    ReturnStackOverflow,

    #[error("float stack underflow")]
    FloatStackUnderflow,

    #[error("float stack overflow")]
    FloatStackOverflow,

    #[error("unknown word: {0}")]
    UnknownWord(String),

    #[error("division by zero")]
    DivisionByZero,

    #[error("invalid memory address: {0:#x}")]
    InvalidAddress(u32),

    #[error("dictionary overflow")]
    DictionaryOverflow,

    #[error("compilation error: {0}")]
    CompileError(String),

    #[error("invalid number: {0}")]
    InvalidNumber(String),

    #[error("word name too long: {0}")]
    NameTooLong(String),

    #[error("control structure mismatch: {0}")]
    ControlMismatch(String),

    #[error("WASM codegen error: {0}")]
    CodegenError(String),

    #[error("WASM validation error: {0}")]
    ValidationError(String),

    #[error("I/O error: {0}")]
    IoError(String),

    #[error("THROW code {0}")]
    Throw(i32),

    #[error("{0}")]
    Abort(String),
}

/// Result type alias for WAFER operations.
pub type WaferResult<T> = Result<T, WaferError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = WaferError::UnknownWord("FOO".to_string());
        assert_eq!(err.to_string(), "unknown word: FOO");
    }

    #[test]
    fn error_throw_code() {
        let err = WaferError::Throw(-1);
        assert_eq!(err.to_string(), "THROW code -1");
    }
}
