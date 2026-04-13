//! Runtime abstraction for WASM execution.
//!
//! The [`Runtime`] trait decouples the Forth VM from any specific WASM engine.
//! Two implementations exist:
//! - `NativeRuntime` (wasmtime) — for CLI and native tests
//! - `WebRuntime` (`js_sys`) — for browser REPL
//!
//! The [`HostAccess`] trait provides memory and global access to host function
//! callbacks, abstracting over wasmtime's `Caller` and browser's `js_sys` APIs.

use std::sync::{Arc, Mutex};

/// Access to WASM memory and globals from within a host function callback.
///
/// Both wasmtime (via `Caller`) and browser (via `js_sys`) implement this trait,
/// allowing host function logic to be shared across runtimes.
pub trait HostAccess {
    /// Read a 32-bit integer from linear memory at `addr` (little-endian).
    fn mem_read_i32(&mut self, addr: u32) -> i32;

    /// Write a 32-bit integer to linear memory at `addr` (little-endian).
    fn mem_write_i32(&mut self, addr: u32, val: i32);

    /// Read a single byte from linear memory.
    fn mem_read_u8(&mut self, addr: u32) -> u8;

    /// Write a single byte to linear memory.
    fn mem_write_u8(&mut self, addr: u32, val: u8);

    /// Read a slice of bytes from linear memory.
    fn mem_read_slice(&mut self, addr: u32, len: usize) -> Vec<u8>;

    /// Write a slice of bytes to linear memory.
    fn mem_write_slice(&mut self, addr: u32, data: &[u8]);

    /// Total size of linear memory in bytes.
    fn mem_len(&mut self) -> usize;

    /// Read the data stack pointer global.
    fn get_dsp(&mut self) -> u32;

    /// Write the data stack pointer global.
    fn set_dsp(&mut self, val: u32);

    /// Read the return stack pointer global.
    fn get_rsp(&mut self) -> u32;

    /// Write the return stack pointer global.
    fn set_rsp(&mut self, val: u32);

    /// Read the float stack pointer global.
    fn get_fsp(&mut self) -> u32;

    /// Write the float stack pointer global.
    fn set_fsp(&mut self, val: u32);

    /// Call a function in the shared table by index.
    /// Needed by CATCH to invoke the xt it receives.
    fn call_func(&mut self, fn_index: u32) -> anyhow::Result<()>;
}

/// Host function callback type.
///
/// A boxed closure that receives mutable [`HostAccess`] for memory/global ops.
/// Captures shared state (e.g. output buffer) via `Arc<Mutex<...>>`.
pub type HostFn = Box<dyn Fn(&mut dyn HostAccess) -> anyhow::Result<()> + Send + Sync>;

/// Abstraction over a WASM execution runtime.
///
/// Provides memory access, global management, module instantiation,
/// function execution, and host function registration.
pub trait Runtime: Sized {
    /// Create a new runtime with shared linear memory, function table,
    /// stack pointer globals, and an `emit` host function wired to `output`.
    fn new(
        memory_pages: u32,
        table_size: u32,
        dsp_init: u32,
        rsp_init: u32,
        fsp_init: u32,
        output: Arc<Mutex<String>>,
    ) -> anyhow::Result<Self>;

    // -- Linear memory access --

    /// Read a 32-bit integer from linear memory at `addr` (little-endian).
    fn mem_read_i32(&mut self, addr: u32) -> i32;

    /// Write a 32-bit integer to linear memory at `addr` (little-endian).
    fn mem_write_i32(&mut self, addr: u32, val: i32);

    /// Read a single byte from linear memory.
    fn mem_read_u8(&mut self, addr: u32) -> u8;

    /// Write a single byte to linear memory.
    fn mem_write_u8(&mut self, addr: u32, val: u8);

    /// Read a slice of bytes from linear memory.
    fn mem_read_slice(&mut self, addr: u32, len: usize) -> Vec<u8>;

    /// Write a slice of bytes to linear memory.
    fn mem_write_slice(&mut self, addr: u32, data: &[u8]);

    /// Total size of linear memory in bytes.
    fn mem_len(&mut self) -> usize;

    // -- Globals --

    /// Read the data stack pointer global.
    fn get_dsp(&mut self) -> u32;

    /// Write the data stack pointer global.
    fn set_dsp(&mut self, val: u32);

    /// Read the return stack pointer global.
    fn get_rsp(&mut self) -> u32;

    /// Write the return stack pointer global.
    fn set_rsp(&mut self, val: u32);

    /// Read the float stack pointer global.
    fn get_fsp(&mut self) -> u32;

    /// Write the float stack pointer global.
    fn set_fsp(&mut self, val: u32);

    // -- Function table --

    /// Current number of entries in the function table.
    fn table_size(&mut self) -> u32;

    /// Grow the table if needed so that index `needed` is valid.
    fn ensure_table_size(&mut self, needed: u32) -> anyhow::Result<()>;

    // -- Compilation and execution --

    /// Compile WASM bytes into a module, instantiate it with shared imports
    /// (memory, table, globals, emit), and install the exported function
    /// at `fn_index` in the shared table.
    fn instantiate_and_install(&mut self, wasm_bytes: &[u8], fn_index: u32) -> anyhow::Result<()>;

    /// Call the function at `fn_index` in the shared table.
    fn call_func(&mut self, fn_index: u32) -> anyhow::Result<()>;

    // -- Host functions --

    /// Register a void→void host function at `fn_index` in the shared table.
    ///
    /// The callback receives a [`HostAccess`] for memory and global operations.
    /// It may also capture shared state via `Arc<Mutex<...>>`.
    fn register_host_func(&mut self, fn_index: u32, f: HostFn) -> anyhow::Result<()>;
}
