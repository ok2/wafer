//! Browser runtime implementation using js-sys WebAssembly API.

use std::sync::{Arc, Mutex};

use js_sys::{Function, Object, Reflect, Uint8Array, WebAssembly};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

use wafer_core::runtime::{HostAccess, HostFn, Runtime};

/// Browser-based WASM runtime using the WebAssembly JS API.
pub(crate) struct WebRuntime {
    memory: JsValue,
    table: JsValue,
    dsp_global: JsValue,
    rsp_global: JsValue,
    fsp_global: JsValue,
    emit_func: JsValue,
    #[allow(dead_code)]
    output: Arc<Mutex<String>>,
    /// Keep closures and wrapper-module Instances alive to prevent GC.
    _closures: Vec<JsValue>,
}

/// [`HostAccess`] for browser — wraps `js_sys` Memory/Globals.
struct WebHostAccess {
    memory: JsValue,
    table: JsValue,
    dsp_global: JsValue,
    rsp_global: JsValue,
    fsp_global: JsValue,
}

impl WebHostAccess {
    fn buffer(&self) -> js_sys::ArrayBuffer {
        let buf = Reflect::get(&self.memory, &"buffer".into()).unwrap();
        buf.unchecked_into()
    }
}

impl HostAccess for WebHostAccess {
    fn mem_read_i32(&mut self, addr: u32) -> i32 {
        let view = js_sys::Int32Array::new(&self.buffer());
        view.get_index(addr / 4)
    }

    fn mem_write_i32(&mut self, addr: u32, val: i32) {
        let view = js_sys::Int32Array::new(&self.buffer());
        view.set_index(addr / 4, val);
    }

    fn mem_read_u8(&mut self, addr: u32) -> u8 {
        let view = Uint8Array::new(&self.buffer());
        view.get_index(addr)
    }

    fn mem_write_u8(&mut self, addr: u32, val: u8) {
        let view = Uint8Array::new(&self.buffer());
        view.set_index(addr, val);
    }

    fn mem_read_slice(&mut self, addr: u32, len: usize) -> Vec<u8> {
        let view = Uint8Array::new(&self.buffer());
        let sub = view.subarray(addr, addr + len as u32);
        sub.to_vec()
    }

    fn mem_write_slice(&mut self, addr: u32, data: &[u8]) {
        let view = Uint8Array::new(&self.buffer());
        let src = Uint8Array::from(data);
        view.set(&src, addr);
    }

    fn mem_len(&mut self) -> usize {
        let buf = self.buffer();
        buf.byte_length() as usize
    }

    fn get_dsp(&mut self) -> u32 {
        Reflect::get(&self.dsp_global, &"value".into())
            .unwrap()
            .as_f64()
            .unwrap() as u32
    }

    fn set_dsp(&mut self, val: u32) {
        Reflect::set(
            &self.dsp_global,
            &"value".into(),
            &JsValue::from(val as i32),
        )
        .unwrap();
    }

    fn get_rsp(&mut self) -> u32 {
        Reflect::get(&self.rsp_global, &"value".into())
            .unwrap()
            .as_f64()
            .unwrap() as u32
    }

    fn set_rsp(&mut self, val: u32) {
        Reflect::set(
            &self.rsp_global,
            &"value".into(),
            &JsValue::from(val as i32),
        )
        .unwrap();
    }

    fn get_fsp(&mut self) -> u32 {
        Reflect::get(&self.fsp_global, &"value".into())
            .unwrap()
            .as_f64()
            .unwrap() as u32
    }

    fn set_fsp(&mut self, val: u32) {
        Reflect::set(
            &self.fsp_global,
            &"value".into(),
            &JsValue::from(val as i32),
        )
        .unwrap();
    }

    fn call_func(&mut self, fn_index: u32) -> anyhow::Result<()> {
        let get_fn = Reflect::get(&self.table, &"get".into()).unwrap();
        let get_fn: Function = get_fn.unchecked_into();
        let func = get_fn
            .call1(&self.table, &JsValue::from(fn_index))
            .map_err(|e| anyhow::anyhow!("table.get({fn_index}) failed: {e:?}"))?;
        let func: Function = func
            .dyn_into()
            .map_err(|_| anyhow::anyhow!("table entry {fn_index} is not a function"))?;
        func.call0(&JsValue::NULL)
            .map_err(|e| anyhow::anyhow!("call_func({fn_index}) failed: {e:?}"))?;
        Ok(())
    }
}

/// Helper: create a WebAssembly.Global with mutable i32.
fn make_global(init: u32) -> JsValue {
    let desc = Object::new();
    Reflect::set(&desc, &"value".into(), &"i32".into()).unwrap();
    Reflect::set(&desc, &"mutable".into(), &JsValue::TRUE).unwrap();
    let ctor = Reflect::get(&js_sys::global(), &"WebAssembly".into())
        .and_then(|wa| Reflect::get(&wa, &"Global".into()))
        .unwrap();
    let args = js_sys::Array::new();
    args.push(&desc);
    args.push(&JsValue::from(init as i32));
    Reflect::construct(&ctor.unchecked_into::<Function>(), &args).unwrap()
}

/// Helper: build the import object for instantiating compiled Forth modules.
fn build_imports(
    emit: &JsValue,
    memory: &JsValue,
    dsp: &JsValue,
    rsp: &JsValue,
    fsp: &JsValue,
    table: &JsValue,
) -> Object {
    let env = Object::new();
    Reflect::set(&env, &"emit".into(), emit).unwrap();
    Reflect::set(&env, &"memory".into(), memory).unwrap();
    Reflect::set(&env, &"dsp".into(), dsp).unwrap();
    Reflect::set(&env, &"rsp".into(), rsp).unwrap();
    Reflect::set(&env, &"fsp".into(), fsp).unwrap();
    Reflect::set(&env, &"table".into(), table).unwrap();
    let imports = Object::new();
    Reflect::set(&imports, &"env".into(), &env).unwrap();
    imports
}

impl Runtime for WebRuntime {
    fn new(
        memory_pages: u32,
        table_size: u32,
        dsp_init: u32,
        rsp_init: u32,
        fsp_init: u32,
        output: Arc<Mutex<String>>,
    ) -> anyhow::Result<Self> {
        // WebAssembly.Memory({initial: pages})
        let mem_desc = Object::new();
        Reflect::set(&mem_desc, &"initial".into(), &JsValue::from(memory_pages)).unwrap();
        let memory = WebAssembly::Memory::new(&mem_desc)
            .map_err(|e| anyhow::anyhow!("Memory::new failed: {e:?}"))?;
        let memory: JsValue = memory.into();

        // WebAssembly.Table({element: 'anyfunc', initial: size})
        let tbl_desc = Object::new();
        Reflect::set(&tbl_desc, &"element".into(), &"anyfunc".into()).unwrap();
        Reflect::set(&tbl_desc, &"initial".into(), &JsValue::from(table_size)).unwrap();
        let table = WebAssembly::Table::new(&tbl_desc)
            .map_err(|e| anyhow::anyhow!("Table::new failed: {e:?}"))?;
        let table: JsValue = table.into();

        let dsp_global = make_global(dsp_init);
        let rsp_global = make_global(rsp_init);
        let fsp_global = make_global(fsp_init);

        // Create emit function: WebAssembly.Function({parameters:['i32'],results:[]}, closure)
        let out_ref = Arc::clone(&output);
        let emit_closure = Closure::wrap(Box::new(move |code: i32| {
            let ch = code as u8 as char;
            out_ref.lock().unwrap().push(ch);
        }) as Box<dyn FnMut(i32)>);

        // Use WebAssembly.Function if available, else wrap in a tiny module.
        // The wrapper-module path also returns the Instance so we can keep it
        // alive for the lifetime of the runtime (browsers can otherwise GC
        // the instance and orphan the funcref's body).
        let (emit_func, emit_keep) = make_wasm_function_i32(&emit_closure.as_ref().into());
        let mut closures = vec![emit_closure.into_js_value(), emit_keep];
        let _ = &mut closures; // keep alive

        Ok(WebRuntime {
            memory,
            table,
            dsp_global,
            rsp_global,
            fsp_global,
            emit_func,
            output,
            _closures: closures,
        })
    }

    // -- Memory --

    fn mem_read_i32(&mut self, addr: u32) -> i32 {
        let buf: js_sys::ArrayBuffer = Reflect::get(&self.memory, &"buffer".into())
            .unwrap()
            .unchecked_into();
        let view = js_sys::Int32Array::new(&buf);
        view.get_index(addr / 4)
    }

    fn mem_write_i32(&mut self, addr: u32, val: i32) {
        let buf: js_sys::ArrayBuffer = Reflect::get(&self.memory, &"buffer".into())
            .unwrap()
            .unchecked_into();
        let view = js_sys::Int32Array::new(&buf);
        view.set_index(addr / 4, val);
    }

    fn mem_read_u8(&mut self, addr: u32) -> u8 {
        let buf: js_sys::ArrayBuffer = Reflect::get(&self.memory, &"buffer".into())
            .unwrap()
            .unchecked_into();
        let view = Uint8Array::new(&buf);
        view.get_index(addr)
    }

    fn mem_write_u8(&mut self, addr: u32, val: u8) {
        let buf: js_sys::ArrayBuffer = Reflect::get(&self.memory, &"buffer".into())
            .unwrap()
            .unchecked_into();
        let view = Uint8Array::new(&buf);
        view.set_index(addr, val);
    }

    fn mem_read_slice(&mut self, addr: u32, len: usize) -> Vec<u8> {
        let buf: js_sys::ArrayBuffer = Reflect::get(&self.memory, &"buffer".into())
            .unwrap()
            .unchecked_into();
        let view = Uint8Array::new(&buf);
        view.subarray(addr, addr + len as u32).to_vec()
    }

    fn mem_write_slice(&mut self, addr: u32, data: &[u8]) {
        let buf: js_sys::ArrayBuffer = Reflect::get(&self.memory, &"buffer".into())
            .unwrap()
            .unchecked_into();
        let view = Uint8Array::new(&buf);
        let src = Uint8Array::from(data);
        view.set(&src, addr);
    }

    fn mem_len(&mut self) -> usize {
        let buf: js_sys::ArrayBuffer = Reflect::get(&self.memory, &"buffer".into())
            .unwrap()
            .unchecked_into();
        buf.byte_length() as usize
    }

    // -- Globals --

    fn get_dsp(&mut self) -> u32 {
        Reflect::get(&self.dsp_global, &"value".into())
            .unwrap()
            .as_f64()
            .unwrap() as u32
    }

    fn set_dsp(&mut self, val: u32) {
        Reflect::set(
            &self.dsp_global,
            &"value".into(),
            &JsValue::from(val as i32),
        )
        .unwrap();
    }

    fn get_rsp(&mut self) -> u32 {
        Reflect::get(&self.rsp_global, &"value".into())
            .unwrap()
            .as_f64()
            .unwrap() as u32
    }

    fn set_rsp(&mut self, val: u32) {
        Reflect::set(
            &self.rsp_global,
            &"value".into(),
            &JsValue::from(val as i32),
        )
        .unwrap();
    }

    fn get_fsp(&mut self) -> u32 {
        Reflect::get(&self.fsp_global, &"value".into())
            .unwrap()
            .as_f64()
            .unwrap() as u32
    }

    fn set_fsp(&mut self, val: u32) {
        Reflect::set(
            &self.fsp_global,
            &"value".into(),
            &JsValue::from(val as i32),
        )
        .unwrap();
    }

    // -- Table --

    fn table_size(&mut self) -> u32 {
        let len = Reflect::get(&self.table, &"length".into()).unwrap();
        len.as_f64().unwrap() as u32
    }

    fn ensure_table_size(&mut self, needed: u32) -> anyhow::Result<()> {
        let current = self.table_size();
        if needed >= current {
            let grow = needed - current + 64;
            let grow_fn: Function = Reflect::get(&self.table, &"grow".into())
                .unwrap()
                .unchecked_into();
            grow_fn
                .call1(&self.table, &JsValue::from(grow))
                .map_err(|e| anyhow::anyhow!("table.grow failed: {e:?}"))?;
        }
        Ok(())
    }

    // -- Compilation and execution --

    fn instantiate_and_install(&mut self, wasm_bytes: &[u8], fn_index: u32) -> anyhow::Result<()> {
        self.ensure_table_size(fn_index)?;

        let bytes = Uint8Array::from(wasm_bytes);
        let module = WebAssembly::Module::new(&bytes.into())
            .map_err(|e| anyhow::anyhow!("Module::new failed: {e:?}"))?;

        let imports = build_imports(
            &self.emit_func,
            &self.memory,
            &self.dsp_global,
            &self.rsp_global,
            &self.fsp_global,
            &self.table,
        );

        let instance = WebAssembly::Instance::new(&module, &imports)
            .map_err(|e| anyhow::anyhow!("Instance::new failed: {e:?}"))?;

        // Single-word modules export "fn"; multi-word modules use element section.
        let exports = Reflect::get(&instance, &"exports".into()).unwrap();
        if let Ok(func) = Reflect::get(&exports, &"fn".into())
            && func.is_function()
        {
            let set_fn: Function = Reflect::get(&self.table, &"set".into())
                .unwrap()
                .unchecked_into();
            set_fn
                .call2(&self.table, &JsValue::from(fn_index), &func)
                .map_err(|e| anyhow::anyhow!("table.set failed: {e:?}"))?;
        }

        Ok(())
    }

    fn call_func(&mut self, fn_index: u32) -> anyhow::Result<()> {
        let get_fn: Function = Reflect::get(&self.table, &"get".into())
            .unwrap()
            .unchecked_into();
        let func = get_fn
            .call1(&self.table, &JsValue::from(fn_index))
            .map_err(|e| anyhow::anyhow!("table.get({fn_index}) failed: {e:?}"))?;
        let func: Function = func
            .dyn_into()
            .map_err(|_| anyhow::anyhow!("table entry {fn_index} is not callable"))?;
        func.call0(&JsValue::NULL)
            .map_err(|e| anyhow::anyhow!("call_func({fn_index}) failed: {e:?}"))?;
        Ok(())
    }

    // -- Host functions --

    fn register_host_func(&mut self, fn_index: u32, f: HostFn) -> anyhow::Result<()> {
        self.ensure_table_size(fn_index)?;

        let memory = self.memory.clone();
        let table = self.table.clone();
        let dsp = self.dsp_global.clone();
        let rsp = self.rsp_global.clone();
        let fsp = self.fsp_global.clone();

        let closure = Closure::wrap(Box::new(move || {
            let mut ctx = WebHostAccess {
                memory: memory.clone(),
                table: table.clone(),
                dsp_global: dsp.clone(),
                rsp_global: rsp.clone(),
                fsp_global: fsp.clone(),
            };
            if let Err(e) = f(&mut ctx) {
                // Throw a JS error to propagate the Forth error (e.g. ABORT, THROW)
                wasm_bindgen::throw_str(&e.to_string());
            }
        }) as Box<dyn FnMut()>);

        let (wasm_func, keep_alive) = make_wasm_function_void(&closure.as_ref().into());

        let set_fn: Function = Reflect::get(&self.table, &"set".into())
            .unwrap()
            .unchecked_into();
        set_fn
            .call2(&self.table, &JsValue::from(fn_index), &wasm_func)
            .map_err(|e| anyhow::anyhow!("table.set({fn_index}) failed: {e:?}"))?;

        // Stash both the closure AND the wrapper-module instance so neither
        // is garbage-collected while the funcref is still in WAFER's table.
        // Without keeping the Instance alive, browsers can orphan the funcref's
        // body and cross-module `call_indirect` silently fails to dispatch.
        self._closures.push(closure.into_js_value());
        self._closures.push(keep_alive);
        Ok(())
    }
}

/// Create a wasm-callable funcref of type `(i32) -> ()` from a JS callback.
///
/// Returns `(funcref, keep_alive)`. Callers MUST stash `keep_alive` somewhere
/// the GC can see — for the wrapper-module path it's the underlying
/// `WebAssembly.Instance`, and the funcref is only valid as long as the
/// instance lives. The `WebAssembly.Function` constructor path returns
/// `JsValue::NULL` for `keep_alive`.
fn make_wasm_function_i32(js_fn: &JsValue) -> (JsValue, JsValue) {
    if let Ok(wasm_func_ctor) = get_wasm_function_ctor() {
        let desc = Object::new();
        let params = js_sys::Array::new();
        params.push(&"i32".into());
        Reflect::set(&desc, &"parameters".into(), &params).unwrap();
        Reflect::set(&desc, &"results".into(), &js_sys::Array::new()).unwrap();
        let args = js_sys::Array::new();
        args.push(&desc);
        args.push(js_fn);
        let f = Reflect::construct(&wasm_func_ctor.unchecked_into::<Function>(), &args).unwrap();
        (f, JsValue::NULL)
    } else {
        make_wrapper_module_i32(js_fn)
    }
}

/// Create a wasm-callable funcref of type `() -> ()`. See [`make_wasm_function_i32`].
fn make_wasm_function_void(js_fn: &JsValue) -> (JsValue, JsValue) {
    if let Ok(wasm_func_ctor) = get_wasm_function_ctor() {
        let desc = Object::new();
        Reflect::set(&desc, &"parameters".into(), &js_sys::Array::new()).unwrap();
        Reflect::set(&desc, &"results".into(), &js_sys::Array::new()).unwrap();
        let args = js_sys::Array::new();
        args.push(&desc);
        args.push(js_fn);
        let f = Reflect::construct(&wasm_func_ctor.unchecked_into::<Function>(), &args).unwrap();
        (f, JsValue::NULL)
    } else {
        make_wrapper_module_void(js_fn)
    }
}

/// Try to get the WebAssembly.Function constructor (Chrome 78+, Firefox 78+).
fn get_wasm_function_ctor() -> Result<Function, ()> {
    let wa = Reflect::get(&js_sys::global(), &"WebAssembly".into()).map_err(|_| ())?;
    let ctor = Reflect::get(&wa, &"Function".into()).map_err(|_| ())?;
    if ctor.is_function() {
        Ok(ctor.unchecked_into())
    } else {
        Err(())
    }
}

/// Fallback: create a minimal WASM module that imports a JS function and
/// exports a **local** trampoline which calls it.
///
/// Exporting the imported funcref directly (the previous approach) produces a
/// funcref that works when invoked via JS `.call()`, but v8 and other engines
/// do not always dispatch to the underlying JS body when that funcref is
/// stored in a different module's table and invoked via `call_indirect`. A
/// local trampoline (a real WASM function that `call`s the import) gives a
/// stable, spec-correct funcref usable from any caller.
fn make_wrapper_module_void(js_fn: &JsValue) -> (JsValue, JsValue) {
    // (module
    //   (import "e" "f" (func $imp))     ;; func 0
    //   (func (export "f") (call $imp))  ;; func 1
    // )
    #[rustfmt::skip]
    let bytes: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,       // magic + version
        // type: 1 type, () -> ()
        0x01, 0x04, 0x01, 0x60, 0x00, 0x00,
        // import: "e"."f" func type 0  -> imported func index 0
        0x02, 0x07, 0x01, 0x01, 0x65, 0x01, 0x66, 0x00, 0x00,
        // function: 1 local func of type 0 -> local func index 1
        0x03, 0x02, 0x01, 0x00,
        // export: "f" -> func index 1 (the local trampoline)
        0x07, 0x05, 0x01, 0x01, 0x66, 0x00, 0x01,
        // code: 1 body, 4 bytes, 0 locals, `call 0`, `end`
        0x0a, 0x06, 0x01, 0x04, 0x00, 0x10, 0x00, 0x0b,
    ];
    instantiate_wrapper(bytes, js_fn)
}

/// Fallback: same idea as [`make_wrapper_module_void`] but for `(i32) -> ()`.
/// The trampoline forwards its one parameter to the import.
fn make_wrapper_module_i32(js_fn: &JsValue) -> (JsValue, JsValue) {
    // (module
    //   (import "e" "f" (func $imp (param i32)))
    //   (func (export "f") (param i32) (local.get 0) (call $imp))
    // )
    #[rustfmt::skip]
    let bytes: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        // type: 1 type, (i32) -> ()
        0x01, 0x05, 0x01, 0x60, 0x01, 0x7f, 0x00,
        // import: "e"."f" func type 0
        0x02, 0x07, 0x01, 0x01, 0x65, 0x01, 0x66, 0x00, 0x00,
        // function: 1 local func of type 0
        0x03, 0x02, 0x01, 0x00,
        // export: "f" -> func index 1
        0x07, 0x05, 0x01, 0x01, 0x66, 0x00, 0x01,
        // code: 1 body, 6 bytes, 0 locals, `local.get 0`, `call 0`, `end`
        0x0a, 0x08, 0x01, 0x06, 0x00, 0x20, 0x00, 0x10, 0x00, 0x0b,
    ];
    instantiate_wrapper(bytes, js_fn)
}

/// Shared: compile the wrapper module, instantiate with `js_fn` bound to
/// import `"e"."f"`, and return both the exported local trampoline and
/// the underlying `Instance` so callers can keep it alive (browsers can
/// otherwise GC the instance and orphan the funcref's body).
fn instantiate_wrapper(bytes: &[u8], js_fn: &JsValue) -> (JsValue, JsValue) {
    let u8arr = Uint8Array::from(bytes);
    let module = WebAssembly::Module::new(&u8arr.into()).unwrap();
    let env = Object::new();
    Reflect::set(&env, &"f".into(), js_fn).unwrap();
    let imports = Object::new();
    Reflect::set(&imports, &"e".into(), &env).unwrap();
    let instance = WebAssembly::Instance::new(&module, &imports).unwrap();
    let instance_jv: JsValue = instance.clone().into();
    let exports = Reflect::get(&instance, &"exports".into()).unwrap();
    let func = Reflect::get(&exports, &"f".into()).unwrap();
    (func, instance_jv)
}
