//! WAFER Web REPL — browser-based Forth REPL using WebAssembly.

mod runtime_web;

use send_wrapper::SendWrapper;
use wasm_bindgen::prelude::*;

use wafer_core::config::WaferConfig;
use wafer_core::memory::{CELL_SIZE, PAD_BASE, PAD_SIZE, SYSVAR_BASE_VAR};
use wafer_core::outer::ForthVM;
use wafer_core::runtime::Runtime;
use wafer_core::runtime::{HostAccess, HostFn};

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
        // Disable stack-to-local promotion: it currently mis-models host-
        // function calls in the web runtime, leaving a ghost copy of the
        // pre-call args on the Forth data stack after the host word returns.
        let mut cfg = WaferConfig::all();
        cfg.codegen.stack_to_local_promotion = false;
        let vm = ForthVM::<WebRuntime>::new_with_config(cfg)
            .map_err(|e| JsError::new(&e.to_string()))?;
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
        self.vm.runtime_mut().mem_read_i32(SYSVAR_BASE_VAR) as u32
    }

    /// Names of all user-facing words (visible, non-internal), newest first.
    pub fn words(&self) -> Vec<String> {
        self.vm.word_names()
    }

    /// Reset the VM to initial state.
    pub fn reset(&mut self) -> Result<(), JsError> {
        let mut cfg = WaferConfig::all();
        cfg.codegen.stack_to_local_promotion = false;
        self.vm = ForthVM::<WebRuntime>::new_with_config(cfg)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }

    /// Register a JavaScript function as a Forth word with stack effect
    /// `( prompt-a prompt-u -- pw-a pw-u )`.
    ///
    /// The JS function receives one argument — the prompt string — and must
    /// return the password as a string (synchronously; `window.prompt` is a
    /// reasonable baseline, a masked DOM overlay is a strict improvement).
    /// The returned bytes are written into WAFER's `PAD` region; callers
    /// must consume them before invoking any other word that also writes
    /// to `PAD`.
    ///
    /// Registering under a dedicated name (e.g. `"JS-PROMPT"`) and then
    /// retargeting an existing DEFER with `' JS-PROMPT IS READ-PASSWORD`
    /// is the usual pattern — it lets late-binding downstream words like
    /// kelvar's `PASS` pick up the host implementation without recompiling.
    pub fn set_prompter(&mut self, name: &str, js_fn: js_sys::Function) -> Result<(), JsError> {
        let holder = SendWrapper::new(js_fn);
        let max = (PAD_SIZE - 1) as usize;

        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // Pop ( prompt-a prompt-u ): advance dsp by 2 cells.
            let mut sp = ctx.get_dsp();
            let u = ctx.mem_read_i32(sp) as u32;
            sp += CELL_SIZE;
            let a = ctx.mem_read_i32(sp) as u32;
            sp += CELL_SIZE;
            ctx.set_dsp(sp);

            let prompt_bytes = ctx.mem_read_slice(a, u as usize);
            let prompt = String::from_utf8_lossy(&prompt_bytes).to_string();

            let result = holder
                .call1(&JsValue::NULL, &JsValue::from_str(&prompt))
                .map_err(|e| {
                    anyhow::anyhow!(
                        "prompter threw: {}",
                        e.as_string().unwrap_or_else(|| "<non-string>".into())
                    )
                })?;
            let pw = result.as_string().unwrap_or_default();

            let bytes = pw.as_bytes();
            if bytes.len() > max {
                anyhow::bail!(
                    "READ-PASSWORD: master too long ({} > {} bytes)",
                    bytes.len(),
                    max
                );
            }

            ctx.mem_write_slice(PAD_BASE, bytes);
            // Push ( PAD bytes.len() )
            sp -= CELL_SIZE;
            ctx.mem_write_i32(sp, PAD_BASE as i32);
            sp -= CELL_SIZE;
            ctx.mem_write_i32(sp, bytes.len() as i32);
            ctx.set_dsp(sp);
            Ok(())
        });

        self.vm
            .register_host_primitive(name, false, func)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(())
    }
}
