//! Type inference engine for WAFER's multi-typed stack.
//!
//! WAFER uses type inference to determine when values on the stack have
//! statically known types. When types are known, codegen uses WASM's native
//! typed operand stack and locals instead of simulating stacks in linear memory.

/// Types that can appear on WAFER's stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StackType {
    /// 32-bit integer (default Forth cell).
    I32,
    /// 64-bit integer (double-cell).
    I64,
    /// 32-bit float.
    F32,
    /// 64-bit float (Forth floating-point).
    F64,
    /// Boolean (result of comparisons). Represented as i32 at WASM level.
    Bool,
    /// Memory address. Represented as i32 at WASM level.
    Addr,
    /// Type is unknown or cannot be determined statically.
    Unknown,
}

impl StackType {
    /// Returns the WASM value type for this stack type.
    pub fn wasm_type(self) -> wasm_encoder::ValType {
        match self {
            StackType::I32 | StackType::Bool | StackType::Addr => wasm_encoder::ValType::I32,
            StackType::I64 => wasm_encoder::ValType::I64,
            StackType::F32 => wasm_encoder::ValType::F32,
            StackType::F64 => wasm_encoder::ValType::F64,
            StackType::Unknown => wasm_encoder::ValType::I32, // default to i32
        }
    }

    /// Returns true if this type's WASM representation is i32.
    pub fn is_i32_compatible(self) -> bool {
        matches!(
            self,
            StackType::I32 | StackType::Bool | StackType::Addr | StackType::Unknown
        )
    }
}

/// Describes the stack effect of a Forth word.
///
/// For example, `+` has effect `( I32 I32 -- I32 )`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackEffect {
    /// Types consumed from the stack (bottom to top).
    pub inputs: Vec<StackType>,
    /// Types produced on the stack (bottom to top).
    pub outputs: Vec<StackType>,
}

impl StackEffect {
    /// Create a new stack effect.
    pub fn new(inputs: Vec<StackType>, outputs: Vec<StackType>) -> Self {
        Self { inputs, outputs }
    }

    /// Number of items consumed.
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Number of items produced.
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Net stack depth change.
    pub fn depth_change(&self) -> i32 {
        self.outputs.len() as i32 - self.inputs.len() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_type_wasm_mapping() {
        assert_eq!(StackType::I32.wasm_type(), wasm_encoder::ValType::I32);
        assert_eq!(StackType::F64.wasm_type(), wasm_encoder::ValType::F64);
        assert_eq!(StackType::Bool.wasm_type(), wasm_encoder::ValType::I32);
        assert_eq!(StackType::Addr.wasm_type(), wasm_encoder::ValType::I32);
    }

    #[test]
    fn stack_effect_depth() {
        // DUP ( x -- x x )
        let dup = StackEffect::new(vec![StackType::I32], vec![StackType::I32, StackType::I32]);
        assert_eq!(dup.depth_change(), 1);

        // + ( x y -- z )
        let add = StackEffect::new(vec![StackType::I32, StackType::I32], vec![StackType::I32]);
        assert_eq!(add.depth_change(), -1);

        // DROP ( x -- )
        let drop_e = StackEffect::new(vec![StackType::I32], vec![]);
        assert_eq!(drop_e.depth_change(), -1);
    }
}
