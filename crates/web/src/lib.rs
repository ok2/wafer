//! WAFER Web REPL — browser-based Forth REPL using WebAssembly.

mod runtime_web;

use wasm_bindgen::prelude::*;

use wafer_core::outer::ForthVM;

use crate::runtime_web::WebRuntime;

/// Browser REPL for WAFER Forth.
#[wasm_bindgen]
pub struct WaferRepl {
    vm: ForthVM<WebRuntime>,
}

#[wasm_bindgen]
impl WaferRepl {
    /// Create a new WAFER REPL instance with all built-in words.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WaferRepl, JsError> {
        let vm = ForthVM::<WebRuntime>::new().map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WaferRepl { vm })
    }

    /// Evaluate a line of Forth input. Returns output text.
    pub fn evaluate(&mut self, input: &str) -> Result<String, JsError> {
        self.vm
            .evaluate(input)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(self.vm.take_output())
    }

    /// Get the current data stack as an array (top-first).
    pub fn data_stack(&mut self) -> Vec<i32> {
        self.vm.data_stack()
    }

    /// Check if the VM is currently in compile mode.
    pub fn is_compiling(&self) -> bool {
        self.vm.is_compiling()
    }

    /// Get the current number base (10 = decimal, 16 = hex).
    pub fn base(&mut self) -> u32 {
        // BASE is stored at SYSVAR_BASE_VAR in WASM memory
        self.vm.take_output(); // no-op side effect; just return base
        10 // TODO: read from memory once we have a getter
    }

    /// Reset the VM to initial state.
    pub fn reset(&mut self) -> Result<(), JsError> {
        self.vm = ForthVM::<WebRuntime>::new().map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }
}
