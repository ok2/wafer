//! WASM runner: execute a pre-compiled `.wasm` module produced by `wafer build`.
//!
//! Provides the six imports the module expects (emit, memory, dsp, rsp, fsp,
//! table) and registers host-function stubs for known Forth words.

use std::sync::{Arc, Mutex};

use wasmtime::{
    Engine, Func, FuncType, Global, GlobalType, Memory, MemoryType, Module, Ref, Store, Table,
    TableType, Val, ValType,
};

use crate::export::{ExportMetadata, deserialize_metadata};
use crate::memory::{CELL_SIZE, DATA_STACK_TOP, SYSVAR_BASE_VAR};

/// Host state for the runner (currently unused by wasmtime `Store` but
/// required as the generic parameter).
struct RunnerHost {}

/// Create a wasmtime engine with the standard WAFER configuration.
fn make_engine() -> anyhow::Result<Engine> {
    let mut config = wasmtime::Config::new();
    config.cranelift_nan_canonicalization(false);
    Ok(Engine::new(&config)?)
}

/// Execute a pre-compiled `.wasm` module from a file path.
pub fn run_wasm_file(path: &str) -> anyhow::Result<String> {
    let wasm_bytes = std::fs::read(path)?;
    run_wasm_bytes(&wasm_bytes)
}

/// Execute WASM bytes directly (used by tests and the CLI).
pub fn run_wasm_bytes(wasm_bytes: &[u8]) -> anyhow::Result<String> {
    let metadata_json = extract_custom_section(wasm_bytes, "wafer")?;
    let metadata = deserialize_metadata(&metadata_json)?;
    let engine = make_engine()?;
    let module = Module::new(&engine, wasm_bytes)?;
    run_module(&engine, module, &metadata)
}

/// Execute an AOT-precompiled module with separate metadata.
///
/// The `precompiled` bytes must have been produced by
/// `Engine::precompile_module` with a compatible wasmtime version
/// and platform.
#[allow(unsafe_code)]
pub fn run_precompiled_bytes(precompiled: &[u8], metadata_json: &str) -> anyhow::Result<String> {
    let metadata = deserialize_metadata(metadata_json)?;
    let engine = make_engine()?;
    // SAFETY: precompiled bytes are produced by wafer build --native using
    // the same wasmtime version. The caller guarantees compatibility.
    let module = unsafe { Module::deserialize(&engine, precompiled)? };
    run_module(&engine, module, &metadata)
}

/// Shared runner logic: given a compiled module and metadata, set up the
/// six WASM imports, register host functions, call `_start`, return output.
fn run_module(
    engine: &Engine,
    module: Module,
    metadata: &ExportMetadata,
) -> anyhow::Result<String> {
    let output = Arc::new(Mutex::new(String::new()));
    let mut store = Store::new(engine, RunnerHost {});

    // Create the 6 imports the module expects.
    let memory_pages = metadata.memory_size.div_ceil(65536).max(16);
    let memory = Memory::new(&mut store, MemoryType::new(memory_pages, None))?;

    let dsp = Global::new(
        &mut store,
        GlobalType::new(ValType::I32, wasmtime::Mutability::Var),
        Val::I32(metadata.dsp_init as i32),
    )?;
    let rsp = Global::new(
        &mut store,
        GlobalType::new(ValType::I32, wasmtime::Mutability::Var),
        Val::I32(metadata.rsp_init as i32),
    )?;
    let fsp = Global::new(
        &mut store,
        GlobalType::new(ValType::I32, wasmtime::Mutability::Var),
        Val::I32(metadata.fsp_init as i32),
    )?;

    // Determine table size from the module's imports.
    let table_min: u32 = module
        .imports()
        .find(|i| i.name() == "table")
        .and_then(|i| i.ty().table().map(|t| t.minimum() as u32))
        .unwrap_or(256);
    let table = Table::new(
        &mut store,
        TableType::new(wasmtime::RefType::FUNCREF, table_min, None),
        Ref::Func(None),
    )?;

    // Create the emit function.
    let out_ref = Arc::clone(&output);
    let emit_func = Func::new(
        &mut store,
        FuncType::new(engine, [ValType::I32], []),
        move |_caller, params, _results| {
            let code = params[0].unwrap_i32();
            if let Some(ch) = char::from_u32(code as u32) {
                out_ref.lock().unwrap().push(ch);
            }
            Ok(())
        },
    );

    // Instantiate the module.
    let instance = wasmtime::Instance::new(
        &mut store,
        &module,
        &[
            emit_func.into(),
            memory.into(),
            dsp.into(),
            rsp.into(),
            fsp.into(),
            table.into(),
        ],
    )?;

    // Register host functions in the table at the metadata-specified indices.
    for (idx, name) in &metadata.host_functions {
        let func = create_host_func(&mut store, engine, memory, dsp, &output, name);
        table.set(&mut store, *idx as u64, Ref::Func(Some(func)))?;
    }

    // Call _start if it exists.
    if let Some(start) = instance.get_func(&mut store, "_start") {
        start.call(&mut store, &[], &mut [])?;
    }

    let result = output.lock().unwrap().clone();
    Ok(result)
}

/// Create a host function implementation for a known Forth word.
fn create_host_func(
    store: &mut Store<RunnerHost>,
    engine: &Engine,
    memory: Memory,
    dsp: Global,
    output: &Arc<Mutex<String>>,
    name: &str,
) -> Func {
    let void_type = FuncType::new(engine, [], []);

    match name {
        "." => {
            // ( n -- ) print number followed by space
            let out = Arc::clone(output);
            Func::new(store, void_type, move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                // Read all values from memory before mutable borrow.
                let (n, base) = {
                    let data = memory.data(&caller);
                    let n =
                        i32::from_le_bytes(data[sp as usize..sp as usize + 4].try_into().unwrap());
                    let base = u32::from_le_bytes(
                        data[SYSVAR_BASE_VAR as usize..SYSVAR_BASE_VAR as usize + 4]
                            .try_into()
                            .unwrap(),
                    );
                    (n, base)
                };
                dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))?;
                let s = if base == 16 {
                    format!("{n:X} ")
                } else {
                    format!("{n} ")
                };
                out.lock().unwrap().push_str(&s);
                Ok(())
            })
        }

        "U." => {
            // ( u -- ) print unsigned number followed by space
            let out = Arc::clone(output);
            Func::new(store, void_type, move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let (n, base) = {
                    let data = memory.data(&caller);
                    let n =
                        u32::from_le_bytes(data[sp as usize..sp as usize + 4].try_into().unwrap());
                    let base = u32::from_le_bytes(
                        data[SYSVAR_BASE_VAR as usize..SYSVAR_BASE_VAR as usize + 4]
                            .try_into()
                            .unwrap(),
                    );
                    (n, base)
                };
                dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))?;
                let s = if base == 16 {
                    format!("{n:X} ")
                } else {
                    format!("{n} ")
                };
                out.lock().unwrap().push_str(&s);
                Ok(())
            })
        }

        "TYPE" => {
            // ( c-addr u -- ) output u characters from memory at c-addr
            let out = Arc::clone(output);
            Func::new(store, void_type, move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let text = {
                    let data = memory.data(&caller);
                    let len =
                        i32::from_le_bytes(data[sp as usize..sp as usize + 4].try_into().unwrap())
                            as u32;
                    let addr = i32::from_le_bytes(
                        data[sp as usize + 4..sp as usize + 8].try_into().unwrap(),
                    ) as u32;
                    let end = (addr + len) as usize;
                    String::from_utf8_lossy(&data[addr as usize..end]).to_string()
                };
                dsp.set(&mut caller, Val::I32((sp + 2 * CELL_SIZE) as i32))?;
                out.lock().unwrap().push_str(&text);
                Ok(())
            })
        }

        "SPACES" => {
            // ( n -- ) output n spaces
            let out = Arc::clone(output);
            Func::new(store, void_type, move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let n = {
                    let data = memory.data(&caller);
                    i32::from_le_bytes(data[sp as usize..sp as usize + 4].try_into().unwrap())
                };
                dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))?;
                if n > 0 {
                    let spaces: String = std::iter::repeat_n(' ', n as usize).collect();
                    out.lock().unwrap().push_str(&spaces);
                }
                Ok(())
            })
        }

        ".S" => {
            // ( -- ) print stack contents non-destructively (no mutable borrow needed)
            let out = Arc::clone(output);
            Func::new(store, void_type, move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let depth = (DATA_STACK_TOP - sp) / CELL_SIZE;
                let mut buf = format!("<{depth}> ");
                let mut addr = DATA_STACK_TOP - CELL_SIZE;
                while addr >= sp {
                    let n = i32::from_le_bytes(
                        data[addr as usize..addr as usize + 4].try_into().unwrap(),
                    );
                    buf.push_str(&format!("{n} "));
                    if addr < CELL_SIZE {
                        break;
                    }
                    addr -= CELL_SIZE;
                }
                out.lock().unwrap().push_str(&buf);
                Ok(())
            })
        }

        "M*" => {
            // ( n1 n2 -- d )  signed multiply producing double-cell result
            Func::new(store, void_type, move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let (n1, n2) = {
                    let data = memory.data(&caller);
                    let n2 =
                        i32::from_le_bytes(data[sp as usize..sp as usize + 4].try_into().unwrap())
                            as i64;
                    let n1 = i32::from_le_bytes(
                        data[sp as usize + 4..sp as usize + 8].try_into().unwrap(),
                    ) as i64;
                    (n1, n2)
                };
                let result = n1 * n2;
                let lo = result as i32;
                let hi = (result >> 32) as i32;
                let data = memory.data_mut(&mut caller);
                data[sp as usize + 4..sp as usize + 8].copy_from_slice(&lo.to_le_bytes());
                data[sp as usize..sp as usize + 4].copy_from_slice(&hi.to_le_bytes());
                Ok(())
            })
        }

        "UM*" => {
            // ( u1 u2 -- ud )  unsigned multiply producing double-cell result
            Func::new(store, void_type, move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let (u1, u2) = {
                    let data = memory.data(&caller);
                    let u2 =
                        u32::from_le_bytes(data[sp as usize..sp as usize + 4].try_into().unwrap())
                            as u64;
                    let u1 = u32::from_le_bytes(
                        data[sp as usize + 4..sp as usize + 8].try_into().unwrap(),
                    ) as u64;
                    (u1, u2)
                };
                let result = u1 * u2;
                let lo = result as u32;
                let hi = (result >> 32) as u32;
                let data = memory.data_mut(&mut caller);
                data[sp as usize + 4..sp as usize + 8].copy_from_slice(&lo.to_le_bytes());
                data[sp as usize..sp as usize + 4].copy_from_slice(&hi.to_le_bytes());
                Ok(())
            })
        }

        "UM/MOD" => {
            // ( ud u -- rem quot )  unsigned double-cell divide
            Func::new(store, void_type, move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let (dividend, divisor) = {
                    let data = memory.data(&caller);
                    let divisor =
                        u32::from_le_bytes(data[sp as usize..sp as usize + 4].try_into().unwrap())
                            as u64;
                    let hi = u32::from_le_bytes(
                        data[sp as usize + 4..sp as usize + 8].try_into().unwrap(),
                    ) as u64;
                    let lo = u32::from_le_bytes(
                        data[sp as usize + 8..sp as usize + 12].try_into().unwrap(),
                    ) as u64;
                    ((hi << 32) | lo, divisor)
                };
                if divisor == 0 {
                    wasmtime::bail!("division by zero");
                }
                let quot = (dividend / divisor) as u32;
                let rem = (dividend % divisor) as u32;
                let new_sp = sp + CELL_SIZE;
                let data = memory.data_mut(&mut caller);
                data[new_sp as usize + 4..new_sp as usize + 8]
                    .copy_from_slice(&(rem as i32).to_le_bytes());
                data[new_sp as usize..new_sp as usize + 4]
                    .copy_from_slice(&(quot as i32).to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            })
        }

        "DEPTH" => {
            // ( -- n ) push current stack depth
            Func::new(store, void_type, move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let depth = (DATA_STACK_TOP - sp) / CELL_SIZE;
                let new_sp = sp - CELL_SIZE;
                memory.data_mut(&mut caller)[new_sp as usize..new_sp as usize + 4]
                    .copy_from_slice(&(depth as i32).to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            })
        }

        _ => {
            // Unimplemented host function: trap with a clear message.
            let name_owned = name.to_string();
            Func::new(store, void_type, move |_caller, _params, _results| {
                wasmtime::bail!("host function '{name_owned}' is not available in standalone mode")
            })
        }
    }
}

/// Extract a named custom section from raw WASM bytes.
fn extract_custom_section(wasm_bytes: &[u8], section_name: &str) -> anyhow::Result<String> {
    let parsed = wasmparser::Parser::new(0).parse_all(wasm_bytes);
    for payload in parsed {
        if let wasmparser::Payload::CustomSection(reader) = payload?
            && reader.name() == section_name
        {
            return Ok(String::from_utf8_lossy(reader.data()).to_string());
        }
    }
    anyhow::bail!("no '{section_name}' custom section found in WASM module")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_missing_section_is_error() {
        // Minimal valid WASM module (magic + version only won't validate,
        // but we can test with a trivial module).
        let empty_module = wasm_encoder::Module::new().finish();
        let result = extract_custom_section(&empty_module, "wafer");
        assert!(result.is_err());
    }
}
