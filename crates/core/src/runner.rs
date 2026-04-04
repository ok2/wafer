//! WASM runner: execute a pre-compiled `.wasm` module produced by `wafer build`.
//!
//! Provides the six imports the module expects (emit, memory, dsp, rsp, fsp,
//! table) and registers host-function stubs for known Forth words.

use std::sync::{Arc, Mutex};

use wasmtime::{
    Engine, Func, FuncType, Global, GlobalType, Memory, MemoryType, Module, Ref, Store, Table,
    TableType, Val, ValType,
};

use crate::export::deserialize_metadata;
use crate::memory::{CELL_SIZE, DATA_STACK_TOP, SYSVAR_BASE_VAR};

/// Host state for the runner (currently unused by wasmtime `Store` but
/// required as the generic parameter).
struct RunnerHost {}

/// Execute a pre-compiled `.wasm` module and return its output.
pub fn run_wasm_file(path: &str) -> anyhow::Result<String> {
    let wasm_bytes = std::fs::read(path)?;
    run_wasm_bytes(&wasm_bytes)
}

/// Execute WASM bytes directly (used by tests and the CLI).
pub fn run_wasm_bytes(wasm_bytes: &[u8]) -> anyhow::Result<String> {
    // Parse the "wafer" custom section for metadata.
    let metadata_json = extract_custom_section(wasm_bytes, "wafer")?;
    let metadata = deserialize_metadata(&metadata_json)?;

    // Set up wasmtime runtime.
    let mut config = wasmtime::Config::new();
    config.cranelift_nan_canonicalization(false);
    let engine = Engine::new(&config)?;

    let output = Arc::new(Mutex::new(String::new()));
    let mut store = Store::new(&engine, RunnerHost {});

    // Create the 6 imports the module expects.
    let memory_pages = metadata.memory_size.div_ceil(65536).max(16); // at least 16 pages like the VM
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

    // Determine table size from the module's import.
    let parsed = wasmparser::Parser::new(0).parse_all(wasm_bytes);
    let mut table_min: u64 = 256;
    for payload in parsed {
        if let wasmparser::Payload::ImportSection(reader) = payload? {
            for import in reader {
                let import = import?;
                if import.name == "table"
                    && let wasmparser::TypeRef::Table(t) = import.ty
                {
                    table_min = t.initial;
                }
            }
        }
    }
    let table = Table::new(
        &mut store,
        TableType::new(wasmtime::RefType::FUNCREF, table_min as u32, None),
        Ref::Func(None),
    )?;

    // Create the emit function.
    let out_ref = Arc::clone(&output);
    let emit_func = Func::new(
        &mut store,
        FuncType::new(&engine, [ValType::I32], []),
        move |_caller, params, _results| {
            let code = params[0].unwrap_i32();
            if let Some(ch) = char::from_u32(code as u32) {
                out_ref.lock().unwrap().push(ch);
            }
            Ok(())
        },
    );

    // Instantiate the module.
    let module = Module::new(&engine, wasm_bytes)?;
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
        let func = create_host_func(&mut store, &engine, memory, dsp, &output, name);
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
                anyhow::bail!("host function '{name_owned}' is not available in standalone mode")
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
