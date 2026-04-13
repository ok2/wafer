//! Native runtime implementation using wasmtime.

use std::sync::{Arc, Mutex};

use wasmtime::{
    Engine, Func, FuncType, Global, Instance, Memory, Module, Mutability, Ref, RefType, Store,
    Table, Val, ValType,
};

use crate::runtime::{HostAccess, HostFn, Runtime};

/// Host-side state accessible from WASM callbacks.
struct NativeVmHost {
    #[allow(dead_code)]
    output: Arc<Mutex<String>>,
}

/// [`HostAccess`] implementation for wasmtime, wrapping a `Caller`.
struct CallerHostAccess<'a, 'b> {
    caller: &'a mut wasmtime::Caller<'b, NativeVmHost>,
    memory: Memory,
    table: Table,
    dsp: Global,
    rsp: Global,
    fsp: Global,
}

impl HostAccess for CallerHostAccess<'_, '_> {
    fn mem_read_i32(&mut self, addr: u32) -> i32 {
        let data = self.memory.data(&self.caller);
        let a = addr as usize;
        i32::from_le_bytes(data[a..a + 4].try_into().unwrap())
    }

    fn mem_write_i32(&mut self, addr: u32, val: i32) {
        let a = addr as usize;
        let bytes = val.to_le_bytes();
        self.memory.data_mut(&mut *self.caller)[a..a + 4].copy_from_slice(&bytes);
    }

    fn mem_read_u8(&mut self, addr: u32) -> u8 {
        self.memory.data(&self.caller)[addr as usize]
    }

    fn mem_write_u8(&mut self, addr: u32, val: u8) {
        self.memory.data_mut(&mut *self.caller)[addr as usize] = val;
    }

    fn mem_read_slice(&mut self, addr: u32, len: usize) -> Vec<u8> {
        let a = addr as usize;
        self.memory.data(&self.caller)[a..a + len].to_vec()
    }

    fn mem_write_slice(&mut self, addr: u32, data: &[u8]) {
        let a = addr as usize;
        self.memory.data_mut(&mut *self.caller)[a..a + data.len()].copy_from_slice(data);
    }

    fn mem_len(&mut self) -> usize {
        self.memory.data(&self.caller).len()
    }

    fn get_dsp(&mut self) -> u32 {
        self.dsp.get(&mut *self.caller).unwrap_i32() as u32
    }

    fn set_dsp(&mut self, val: u32) {
        self.dsp
            .set(&mut *self.caller, Val::I32(val as i32))
            .unwrap();
    }

    fn get_rsp(&mut self) -> u32 {
        self.rsp.get(&mut *self.caller).unwrap_i32() as u32
    }

    fn set_rsp(&mut self, val: u32) {
        self.rsp
            .set(&mut *self.caller, Val::I32(val as i32))
            .unwrap();
    }

    fn get_fsp(&mut self) -> u32 {
        self.fsp.get(&mut *self.caller).unwrap_i32() as u32
    }

    fn set_fsp(&mut self, val: u32) {
        self.fsp
            .set(&mut *self.caller, Val::I32(val as i32))
            .unwrap();
    }

    fn call_func(&mut self, fn_index: u32) -> anyhow::Result<()> {
        let func_ref = self
            .table
            .get(&mut *self.caller, fn_index as u64)
            .ok_or_else(|| anyhow::anyhow!("call_func: invalid index {fn_index}"))?;
        let func = *func_ref
            .unwrap_func()
            .ok_or_else(|| anyhow::anyhow!("call_func: null funcref {fn_index}"))?;
        func.call(&mut *self.caller, &[], &mut [])?;
        Ok(())
    }
}

/// Wasmtime-based native runtime.
pub struct NativeRuntime {
    engine: Engine,
    store: Store<NativeVmHost>,
    memory: Memory,
    table: Table,
    dsp: Global,
    rsp: Global,
    fsp: Global,
    emit_func: Func,
}

impl Runtime for NativeRuntime {
    fn new(
        memory_pages: u32,
        table_size: u32,
        dsp_init: u32,
        rsp_init: u32,
        fsp_init: u32,
        output: Arc<Mutex<String>>,
    ) -> anyhow::Result<Self> {
        let mut config = wasmtime::Config::new();
        config.cranelift_nan_canonicalization(false);
        let engine = Engine::new(&config)?;

        let host = NativeVmHost {
            output: Arc::clone(&output),
        };
        let mut store = Store::new(&engine, host);

        let memory = Memory::new(&mut store, wasmtime::MemoryType::new(memory_pages, None))?;

        let dsp = Global::new(
            &mut store,
            wasmtime::GlobalType::new(ValType::I32, Mutability::Var),
            Val::I32(dsp_init as i32),
        )?;
        let rsp = Global::new(
            &mut store,
            wasmtime::GlobalType::new(ValType::I32, Mutability::Var),
            Val::I32(rsp_init as i32),
        )?;
        let fsp = Global::new(
            &mut store,
            wasmtime::GlobalType::new(ValType::I32, Mutability::Var),
            Val::I32(fsp_init as i32),
        )?;

        let table = Table::new(
            &mut store,
            wasmtime::TableType::new(RefType::FUNCREF, table_size, None),
            Ref::Func(None),
        )?;

        let out_ref = Arc::clone(&output);
        let emit_func = Func::new(
            &mut store,
            FuncType::new(&engine, [ValType::I32], []),
            move |_caller, params, _results| {
                let ch = params[0].unwrap_i32() as u8 as char;
                out_ref.lock().unwrap().push(ch);
                Ok(())
            },
        );

        Ok(NativeRuntime {
            engine,
            store,
            memory,
            table,
            dsp,
            rsp,
            fsp,
            emit_func,
        })
    }

    // -- Memory --

    fn mem_read_i32(&mut self, addr: u32) -> i32 {
        let a = addr as usize;
        let data = self.memory.data(&self.store);
        i32::from_le_bytes(data[a..a + 4].try_into().unwrap())
    }

    fn mem_write_i32(&mut self, addr: u32, val: i32) {
        let a = addr as usize;
        let bytes = val.to_le_bytes();
        self.memory.data_mut(&mut self.store)[a..a + 4].copy_from_slice(&bytes);
    }

    fn mem_read_u8(&mut self, addr: u32) -> u8 {
        self.memory.data(&self.store)[addr as usize]
    }

    fn mem_write_u8(&mut self, addr: u32, val: u8) {
        self.memory.data_mut(&mut self.store)[addr as usize] = val;
    }

    fn mem_read_slice(&mut self, addr: u32, len: usize) -> Vec<u8> {
        let a = addr as usize;
        self.memory.data(&self.store)[a..a + len].to_vec()
    }

    fn mem_write_slice(&mut self, addr: u32, data: &[u8]) {
        let a = addr as usize;
        self.memory.data_mut(&mut self.store)[a..a + data.len()].copy_from_slice(data);
    }

    fn mem_len(&mut self) -> usize {
        self.memory.data(&self.store).len()
    }

    // -- Globals --

    fn get_dsp(&mut self) -> u32 {
        self.dsp.get(&mut self.store).unwrap_i32() as u32
    }

    fn set_dsp(&mut self, val: u32) {
        self.dsp.set(&mut self.store, Val::I32(val as i32)).unwrap();
    }

    fn get_rsp(&mut self) -> u32 {
        self.rsp.get(&mut self.store).unwrap_i32() as u32
    }

    fn set_rsp(&mut self, val: u32) {
        self.rsp.set(&mut self.store, Val::I32(val as i32)).unwrap();
    }

    fn get_fsp(&mut self) -> u32 {
        self.fsp.get(&mut self.store).unwrap_i32() as u32
    }

    fn set_fsp(&mut self, val: u32) {
        self.fsp.set(&mut self.store, Val::I32(val as i32)).unwrap();
    }

    // -- Table --

    fn table_size(&mut self) -> u32 {
        self.table.size(&self.store) as u32
    }

    fn ensure_table_size(&mut self, needed: u32) -> anyhow::Result<()> {
        let current = self.table.size(&self.store) as u32;
        if needed >= current {
            let grow = (needed - current + 64) as u64;
            self.table.grow(&mut self.store, grow, Ref::Func(None))?;
        }
        Ok(())
    }

    // -- Compilation and execution --

    fn instantiate_and_install(&mut self, wasm_bytes: &[u8], fn_index: u32) -> anyhow::Result<()> {
        self.ensure_table_size(fn_index)?;
        let module = Module::new(&self.engine, wasm_bytes)?;
        let instance = Instance::new(
            &mut self.store,
            &module,
            &[
                self.emit_func.into(),
                self.memory.into(),
                self.dsp.into(),
                self.rsp.into(),
                self.fsp.into(),
                self.table.into(),
            ],
        )?;

        // Single-word modules export "fn"; multi-word (consolidated/batch)
        // modules use the element section to place functions in the table.
        if let Some(func) = instance.get_func(&mut self.store, "fn") {
            self.table
                .set(&mut self.store, fn_index as u64, Ref::Func(Some(func)))?;
        }

        Ok(())
    }

    fn call_func(&mut self, fn_index: u32) -> anyhow::Result<()> {
        let r = self
            .table
            .get(&mut self.store, fn_index as u64)
            .ok_or_else(|| anyhow::anyhow!("word {fn_index} not in function table"))?;
        let func = *r
            .unwrap_func()
            .ok_or_else(|| anyhow::anyhow!("word {fn_index} is null funcref"))?;
        func.call(&mut self.store, &[], &mut [])?;
        Ok(())
    }

    // -- Host functions --

    fn register_host_func(&mut self, fn_index: u32, f: HostFn) -> anyhow::Result<()> {
        let mem = self.memory;
        let tbl = self.table;
        let dsp = self.dsp;
        let rsp = self.rsp;
        let fsp = self.fsp;
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let mut ctx = CallerHostAccess {
                    caller: &mut caller,
                    memory: mem,
                    table: tbl,
                    dsp,
                    rsp,
                    fsp,
                };
                f(&mut ctx).map_err(|e| wasmtime::Error::msg(e.to_string()))
            },
        );
        self.ensure_table_size(fn_index)?;
        self.table
            .set(&mut self.store, fn_index as u64, Ref::Func(Some(func)))?;
        Ok(())
    }
}
