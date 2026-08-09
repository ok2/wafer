//! Unified configuration for all WAFER optimizations.

use crate::optimizer::OptConfig;

/// Codegen-level optimization flags.
#[derive(Debug, Clone)]
pub struct CodegenOpts {
    /// Enable stack-to-local promotion for straight-line words.
    pub stack_to_local_promotion: bool,
    /// Emit stack under/overflow guards in compiled words. Faults throw
    /// standard codes (-3/-4/-5/-6/-44/-45) instead of silently
    /// corrupting stack pointers. On by default; benchmarks and
    /// exported production modules turn it off.
    pub stack_guards: bool,
    /// Compile words with a statically known stack effect to a typed entry
    /// point that carries stack items in WASM values, so a call keeps them
    /// in registers instead of round-tripping through the memory stack.
    /// On by default; `WAFER_TYPED_CALLS=0` turns it off.
    pub typed_calls: bool,
}

/// Master configuration for all WAFER optimizations.
#[derive(Debug, Clone)]
pub struct WaferConfig {
    /// IR-level optimization passes.
    pub opt: OptConfig,
    /// Codegen-level optimizations.
    pub codegen: CodegenOpts,
}

impl WaferConfig {
    /// All optimizations enabled.
    pub fn all() -> Self {
        Self {
            opt: OptConfig {
                peephole: true,
                constant_fold: true,
                tail_call: true,
                strength_reduce: true,
                dce: true,
                inline: true,
                self_guard: true,
            },
            codegen: CodegenOpts {
                stack_to_local_promotion: true,
                stack_guards: true,
                typed_calls: true,
            },
        }
    }

    /// All optimizations disabled.
    pub fn none() -> Self {
        Self {
            opt: OptConfig {
                peephole: false,
                constant_fold: false,
                tail_call: false,
                strength_reduce: false,
                dce: false,
                inline: false,
                self_guard: false,
            },
            codegen: CodegenOpts {
                stack_to_local_promotion: false,
                stack_guards: false,
                typed_calls: false,
            },
        }
    }
}

impl Default for WaferConfig {
    fn default() -> Self {
        Self::all()
    }
}
