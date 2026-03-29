//! Outer interpreter: tokenizer, number parser, and interpret/compile dispatch.
//!
//! The outer interpreter is the main loop of Forth:
//! 1. Read a token (whitespace-delimited word)
//! 2. Look it up in the dictionary
//! 3. If found: execute (interpret mode) or compile (compile mode)
//! 4. If not found: try to parse as a number
//! 5. If number: push (interpret) or compile as literal (compile mode)
//! 6. If neither: error

use std::sync::{Arc, Mutex};

use wasmtime::{
    Engine, Func, FuncType, Global, Instance, Memory, Module, Mutability, Ref, RefType, Store,
    Table, Val, ValType,
};

use crate::codegen::{CodegenConfig, CompiledModule, compile_word};
use crate::dictionary::{Dictionary, WordId};
use crate::ir::IrOp;
use crate::memory::{CELL_SIZE, DATA_STACK_TOP, RETURN_STACK_TOP};

// ---------------------------------------------------------------------------
// Control-flow compilation state
// ---------------------------------------------------------------------------

/// Control-flow entry on the compile-time control stack.
#[derive(Debug)]
enum ControlEntry {
    If {
        then_body: Vec<IrOp>,
    },
    IfElse {
        then_body: Vec<IrOp>,
        else_body: Vec<IrOp>,
    },
    Do {
        body: Vec<IrOp>,
    },
    Begin {
        body: Vec<IrOp>,
    },
    BeginWhile {
        test: Vec<IrOp>,
        body: Vec<IrOp>,
    },
}

// ---------------------------------------------------------------------------
// VM state stored in the wasmtime Store
// ---------------------------------------------------------------------------

/// Host-side state accessible from WASM callbacks.
struct VmHost {
    #[allow(dead_code)]
    output: Arc<Mutex<String>>,
}

// ---------------------------------------------------------------------------
// ForthVM
// ---------------------------------------------------------------------------

/// The complete Forth virtual machine -- owns dictionary, WASM runtime, and state.
pub struct ForthVM {
    dictionary: Dictionary,
    engine: Engine,
    store: Store<VmHost>,
    memory: Memory,
    table: Table,
    dsp: Global,
    rsp: Global,
    /// 0 = interpreting, -1 = compiling
    state: i32,
    /// Number base (default 10)
    base: u32,
    input_buffer: String,
    input_pos: usize,
    // Compilation state
    compiling_name: Option<String>,
    compiling_ir: Vec<IrOp>,
    control_stack: Vec<ControlEntry>,
    compiling_word_id: Option<WordId>,
    // Output buffer
    output: Arc<Mutex<String>>,
    // Next table index (mirrors dictionary.next_fn_index conceptually,
    // but we track what's actually in the wasmtime table)
    next_table_index: u32,
    // The emit function (shared across all instantiated modules)
    emit_func: Func,
    // Dot (print number) function -- kept for potential future use
    #[allow(dead_code)]
    dot_func: Func,
    // Shared HERE value for host functions (synced with user_here)
    here_cell: Option<Arc<Mutex<u32>>>,
    // User data allocation pointer in WASM linear memory.
    // Variables and user data are allocated here (not in dictionary internal memory).
    user_here: u32,
    // Shared BASE value for host functions
    base_cell: Arc<Mutex<u32>>,
}

impl ForthVM {
    /// Boot a new Forth VM with all primitives registered.
    pub fn new() -> anyhow::Result<Self> {
        let engine = Engine::default();
        let output = Arc::new(Mutex::new(String::new()));

        let host = VmHost {
            output: Arc::clone(&output),
        };
        let mut store = Store::new(&engine, host);

        // Shared linear memory (16 pages = 1 MiB)
        let memory = Memory::new(&mut store, wasmtime::MemoryType::new(16, None))?;

        // Data stack pointer global
        let dsp = Global::new(
            &mut store,
            wasmtime::GlobalType::new(ValType::I32, Mutability::Var),
            Val::I32(DATA_STACK_TOP as i32),
        )?;

        // Return stack pointer global
        let rsp = Global::new(
            &mut store,
            wasmtime::GlobalType::new(ValType::I32, Mutability::Var),
            Val::I32(RETURN_STACK_TOP as i32),
        )?;

        // Function table (initial 256 entries)
        let table = Table::new(
            &mut store,
            wasmtime::TableType::new(RefType::FUNCREF, 256, None),
            Ref::Func(None),
        )?;

        // Create emit host function: (i32) -> ()
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

        // Create dot host function: (i32) -> ()
        // This is used to implement `.` -- it pops TOS and prints it.
        // We create a host function that takes i32, converts to string, appends to output.
        let out_ref2 = Arc::clone(&output);
        let dot_func = Func::new(
            &mut store,
            FuncType::new(&engine, [ValType::I32], []),
            move |_caller, params, _results| {
                let n = params[0].unwrap_i32();
                let s = format!("{n} ");
                out_ref2.lock().unwrap().push_str(&s);
                Ok(())
            },
        );

        let dictionary = Dictionary::new();

        let mut vm = ForthVM {
            dictionary,
            engine,
            store,
            memory,
            table,
            dsp,
            rsp,
            state: 0,
            base: 10,
            input_buffer: String::new(),
            input_pos: 0,
            compiling_name: None,
            compiling_ir: Vec::new(),
            control_stack: Vec::new(),
            compiling_word_id: None,
            output,
            next_table_index: 0,
            emit_func,
            dot_func,
            here_cell: None,
            // User data starts at 64K in WASM memory, well clear of all system regions
            user_here: 0x10000,
            base_cell: Arc::new(Mutex::new(10)),
        };

        vm.register_primitives()?;

        Ok(vm)
    }

    /// Evaluate a line of Forth input.
    pub fn evaluate(&mut self, input: &str) -> anyhow::Result<()> {
        self.input_buffer = input.to_string();
        self.input_pos = 0;

        while let Some(token) = self.next_token() {
            self.interpret_token(&token)?;
        }

        Ok(())
    }

    /// Check if the VM is currently in compile mode.
    pub fn is_compiling(&self) -> bool {
        self.state != 0
    }

    /// Get and clear the output buffer.
    pub fn take_output(&mut self) -> String {
        let mut out = self.output.lock().unwrap();
        let s = out.clone();
        out.clear();
        s
    }

    /// Read the current data stack contents (top-first).
    pub fn data_stack(&mut self) -> Vec<i32> {
        let sp = self.dsp.get(&mut self.store).unwrap_i32() as u32;
        let data = self.memory.data(&self.store);
        let mut stack = Vec::new();
        let mut addr = sp;
        while addr < DATA_STACK_TOP {
            let b: [u8; 4] = data[addr as usize..addr as usize + 4].try_into().unwrap();
            stack.push(i32::from_le_bytes(b));
            addr += CELL_SIZE;
        }
        stack
    }

    // -----------------------------------------------------------------------
    // Internal: tokenizer
    // -----------------------------------------------------------------------

    /// Read the next whitespace-delimited token from the input buffer.
    fn next_token(&mut self) -> Option<String> {
        let bytes = self.input_buffer.as_bytes();
        // Skip whitespace
        while self.input_pos < bytes.len() && bytes[self.input_pos].is_ascii_whitespace() {
            self.input_pos += 1;
        }
        if self.input_pos >= bytes.len() {
            return None;
        }
        let start = self.input_pos;
        while self.input_pos < bytes.len() && !bytes[self.input_pos].is_ascii_whitespace() {
            self.input_pos += 1;
        }
        Some(String::from_utf8_lossy(&bytes[start..self.input_pos]).to_string())
    }

    /// Read from the input buffer until the given delimiter character.
    /// Returns the collected string (not including the delimiter).
    fn parse_until(&mut self, delim: char) -> Option<String> {
        let bytes = self.input_buffer.as_bytes();
        // Skip one leading space if present
        if self.input_pos < bytes.len() && bytes[self.input_pos] == b' ' {
            self.input_pos += 1;
        }
        let start = self.input_pos;
        while self.input_pos < bytes.len() && bytes[self.input_pos] != delim as u8 {
            self.input_pos += 1;
        }
        if self.input_pos > start || self.input_pos < bytes.len() {
            let result = String::from_utf8_lossy(&bytes[start..self.input_pos]).to_string();
            // Skip past the delimiter
            if self.input_pos < bytes.len() {
                self.input_pos += 1;
            }
            Some(result)
        } else {
            None
        }
    }

    // -----------------------------------------------------------------------
    // Internal: interpret/compile dispatch
    // -----------------------------------------------------------------------

    /// Process a single token in the current mode (interpret or compile).
    fn interpret_token(&mut self, token: &str) -> anyhow::Result<()> {
        let token_upper = token.to_ascii_uppercase();

        // Handle colon definition start
        if token_upper == ":" {
            return self.start_colon_def();
        }

        // Handle semicolon
        if token_upper == ";" {
            if self.state == 0 {
                anyhow::bail!("unexpected ;");
            }
            return self.finish_colon_def();
        }

        // Words that must be handled in the outer interpreter because they
        // modify Rust-side VM state that host functions cannot access.
        match token_upper.as_str() {
            "IMMEDIATE" => {
                self.dictionary
                    .toggle_immediate()
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                return Ok(());
            }
            "DECIMAL" => {
                self.base = 10;
                *self.base_cell.lock().unwrap() = 10;
                return Ok(());
            }
            "HEX" => {
                self.base = 16;
                *self.base_cell.lock().unwrap() = 16;
                return Ok(());
            }
            _ => {}
        }

        if self.state != 0 {
            // Compile mode
            self.compile_token(token)?;
        } else {
            // Interpret mode
            self.interpret_token_immediate(token)?;
        }

        Ok(())
    }

    /// Interpret a token in immediate (interpret) mode.
    fn interpret_token_immediate(&mut self, token: &str) -> anyhow::Result<()> {
        // Special handling for string literals in interpret mode
        let token_upper = token.to_ascii_uppercase();
        if token_upper == ".\"" {
            // Parse until closing quote and print
            if let Some(s) = self.parse_until('"') {
                self.output.lock().unwrap().push_str(&s);
            }
            return Ok(());
        }
        if token_upper == "(" {
            // Comment -- skip until )
            self.parse_until(')');
            return Ok(());
        }
        if token_upper == "\\" {
            // Line comment -- skip rest of input
            self.input_pos = self.input_buffer.len();
            return Ok(());
        }

        // -- Defining words (special tokens handled in interpret mode) --
        match token_upper.as_str() {
            "VARIABLE" => return self.define_variable(),
            "CONSTANT" => return self.define_constant(),
            "CREATE" => return self.define_create(),
            "DOES>" => anyhow::bail!("DOES> not yet implemented"),
            "'" => return self.interpret_tick(),
            "[CHAR]" => {
                // In interpret mode, CHAR is the standard word
                return self.interpret_char();
            }
            "CHAR" => return self.interpret_char(),
            _ => {}
        }

        // Look up in dictionary
        if let Some((_addr, word_id, _is_immediate)) = self.dictionary.find(token) {
            self.execute_word(word_id)?;
            return Ok(());
        }

        // Try to parse as number
        if let Some(n) = self.parse_number(token) {
            self.push_data_stack(n)?;
            return Ok(());
        }

        anyhow::bail!("unknown word: {}", token);
    }

    /// Compile a token in compile mode.
    fn compile_token(&mut self, token: &str) -> anyhow::Result<()> {
        let token_upper = token.to_ascii_uppercase();

        // Handle string literals in compile mode
        if token_upper == ".\"" {
            // Parse until closing quote, emit characters as EMIT calls
            if let Some(s) = self.parse_until('"') {
                for ch in s.chars() {
                    self.push_ir(IrOp::PushI32(ch as i32));
                    self.push_ir(IrOp::Emit);
                }
            }
            return Ok(());
        }
        if token_upper == "S\"" {
            // TODO: string literal on stack
            self.parse_until('"');
            return Ok(());
        }
        if token_upper == "(" {
            self.parse_until(')');
            return Ok(());
        }
        if token_upper == "\\" {
            self.input_pos = self.input_buffer.len();
            return Ok(());
        }

        // Check control flow words (these are handled structurally)
        match token_upper.as_str() {
            "IF" => return self.compile_if(),
            "ELSE" => return self.compile_else(),
            "THEN" => return self.compile_then(),
            "DO" => return self.compile_do(),
            "LOOP" => return self.compile_loop(false),
            "+LOOP" => return self.compile_loop(true),
            "BEGIN" => return self.compile_begin(),
            "UNTIL" => return self.compile_until(),
            "WHILE" => return self.compile_while(),
            "REPEAT" => return self.compile_repeat(),
            "RECURSE" => {
                if let Some(word_id) = self.compiling_word_id {
                    self.push_ir(IrOp::Call(word_id));
                }
                return Ok(());
            }
            "EXIT" => {
                self.push_ir(IrOp::Exit);
                return Ok(());
            }
            "[" => {
                self.state = 0;
                return Ok(());
            }
            "]" => {
                self.state = -1;
                return Ok(());
            }
            "LITERAL" => {
                // compile-time: pop from data stack, compile as literal
                let stack = self.data_stack();
                if let Some(&n) = stack.first() {
                    self.pop_data_stack()?;
                    self.push_ir(IrOp::PushI32(n));
                }
                return Ok(());
            }
            "POSTPONE" => {
                // Read next token, compile a call to it
                if let Some(next) = self.next_token() {
                    if let Some((_addr, word_id, _imm)) = self.dictionary.find(&next) {
                        self.push_ir(IrOp::Call(word_id));
                    } else {
                        anyhow::bail!("POSTPONE: unknown word: {}", next);
                    }
                }
                return Ok(());
            }
            "[CHAR]" => {
                // compile-time: read next token, push first char as literal
                if let Some(next) = self.next_token()
                    && let Some(ch) = next.chars().next()
                {
                    self.push_ir(IrOp::PushI32(ch as i32));
                }
                return Ok(());
            }
            "CHAR" => {
                // In compile mode, CHAR reads next word and compiles its first char
                if let Some(next) = self.next_token()
                    && let Some(ch) = next.chars().next()
                {
                    self.push_ir(IrOp::PushI32(ch as i32));
                }
                return Ok(());
            }
            "[']" => {
                // compile-time: read next token, look up, compile as literal
                if let Some(next) = self.next_token() {
                    if let Some((_addr, word_id, _imm)) = self.dictionary.find(&next) {
                        self.push_ir(IrOp::PushI32(word_id.0 as i32));
                    } else {
                        anyhow::bail!("['] unknown word: {}", next);
                    }
                }
                return Ok(());
            }
            _ => {}
        }

        // Look up in dictionary
        if let Some((_addr, word_id, is_immediate)) = self.dictionary.find(token) {
            if is_immediate {
                // Execute immediately even in compile mode
                self.execute_word(word_id)?;
            } else {
                self.push_ir(IrOp::Call(word_id));
            }
            return Ok(());
        }

        // Try to parse as number
        if let Some(n) = self.parse_number(token) {
            self.push_ir(IrOp::PushI32(n));
            return Ok(());
        }

        anyhow::bail!("unknown word: {}", token);
    }

    // -----------------------------------------------------------------------
    // Control flow compilation
    // -----------------------------------------------------------------------

    fn compile_if(&mut self) -> anyhow::Result<()> {
        // Save current IR and start collecting then_body
        let saved = std::mem::take(&mut self.compiling_ir);
        self.control_stack.push(ControlEntry::If {
            then_body: Vec::new(),
        });
        // The saved IR goes back as the "outer" compiling_ir -- but we need a
        // different approach. Let's store the prefix in the control entry and
        // make compiling_ir the then_body.
        // Actually, the right pattern: we push a frame, and the current IR
        // becomes the prefix. When THEN is reached, we pop the frame, build
        // the IrOp::If, and append it to the prefix.

        // Put the prefix aside in the control entry itself.
        // We'll repurpose: then_body starts empty (will be compiling_ir from now on).
        // The prefix (current compiling_ir) is stashed.
        // On THEN, we pop the control entry, take compiling_ir as then_body,
        // restore the prefix, and append If{then_body, else_body}.

        // Let me restructure: use a separate prefix stack.
        // Actually the simplest approach: stash the current compiling_ir into
        // the control entry, and start fresh for the then_body.
        self.control_stack.pop(); // remove the one we just pushed
        self.control_stack.push(ControlEntry::If {
            then_body: saved, // this is actually the prefix
        });
        // compiling_ir is now empty and will collect the then_body
        Ok(())
    }

    fn compile_else(&mut self) -> anyhow::Result<()> {
        match self.control_stack.pop() {
            Some(ControlEntry::If { then_body: prefix }) => {
                // compiling_ir has the then_body ops
                let then_body = std::mem::take(&mut self.compiling_ir);
                self.control_stack.push(ControlEntry::IfElse {
                    then_body,
                    else_body: prefix, // stash prefix as else_body temporarily
                });
                // compiling_ir is now empty and will collect the else_body
            }
            _ => anyhow::bail!("ELSE without matching IF"),
        }
        Ok(())
    }

    fn compile_then(&mut self) -> anyhow::Result<()> {
        match self.control_stack.pop() {
            Some(ControlEntry::If { then_body: prefix }) => {
                // compiling_ir has the then_body ops
                let then_body = std::mem::take(&mut self.compiling_ir);
                // Restore prefix and append the If node
                self.compiling_ir = prefix;
                self.compiling_ir.push(IrOp::If {
                    then_body,
                    else_body: None,
                });
            }
            Some(ControlEntry::IfElse {
                then_body,
                else_body: prefix,
            }) => {
                // compiling_ir has the else_body ops
                let else_body = std::mem::take(&mut self.compiling_ir);
                self.compiling_ir = prefix;
                self.compiling_ir.push(IrOp::If {
                    then_body,
                    else_body: Some(else_body),
                });
            }
            _ => anyhow::bail!("THEN without matching IF"),
        }
        Ok(())
    }

    fn compile_do(&mut self) -> anyhow::Result<()> {
        let prefix = std::mem::take(&mut self.compiling_ir);
        self.control_stack.push(ControlEntry::Do { body: prefix });
        Ok(())
    }

    fn compile_loop(&mut self, is_plus_loop: bool) -> anyhow::Result<()> {
        match self.control_stack.pop() {
            Some(ControlEntry::Do { body: prefix }) => {
                let body = std::mem::take(&mut self.compiling_ir);
                self.compiling_ir = prefix;
                self.compiling_ir.push(IrOp::DoLoop { body, is_plus_loop });
            }
            _ => anyhow::bail!("LOOP without matching DO"),
        }
        Ok(())
    }

    fn compile_begin(&mut self) -> anyhow::Result<()> {
        let prefix = std::mem::take(&mut self.compiling_ir);
        self.control_stack
            .push(ControlEntry::Begin { body: prefix });
        Ok(())
    }

    fn compile_until(&mut self) -> anyhow::Result<()> {
        match self.control_stack.pop() {
            Some(ControlEntry::Begin { body: prefix }) => {
                let body = std::mem::take(&mut self.compiling_ir);
                self.compiling_ir = prefix;
                self.compiling_ir.push(IrOp::BeginUntil { body });
            }
            _ => anyhow::bail!("UNTIL without matching BEGIN"),
        }
        Ok(())
    }

    fn compile_while(&mut self) -> anyhow::Result<()> {
        match self.control_stack.pop() {
            Some(ControlEntry::Begin { body: prefix }) => {
                let test = std::mem::take(&mut self.compiling_ir);
                self.control_stack.push(ControlEntry::BeginWhile {
                    test,
                    body: prefix, // stash prefix
                });
                // compiling_ir now empty, collects the body
            }
            _ => anyhow::bail!("WHILE without matching BEGIN"),
        }
        Ok(())
    }

    fn compile_repeat(&mut self) -> anyhow::Result<()> {
        match self.control_stack.pop() {
            Some(ControlEntry::BeginWhile { test, body: prefix }) => {
                let body = std::mem::take(&mut self.compiling_ir);
                self.compiling_ir = prefix;
                self.compiling_ir
                    .push(IrOp::BeginWhileRepeat { test, body });
            }
            _ => anyhow::bail!("REPEAT without matching BEGIN...WHILE"),
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Colon definition
    // -----------------------------------------------------------------------

    fn start_colon_def(&mut self) -> anyhow::Result<()> {
        if self.state != 0 {
            anyhow::bail!("nested colon definitions not allowed");
        }
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("expected word name after :"))?;

        // Create the dictionary entry (hidden until ; reveals it)
        let word_id = self
            .dictionary
            .create(&name, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        self.compiling_name = Some(name);
        self.compiling_word_id = Some(word_id);
        self.compiling_ir.clear();
        self.control_stack.clear();
        self.state = -1;
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);

        Ok(())
    }

    fn finish_colon_def(&mut self) -> anyhow::Result<()> {
        if self.state == 0 {
            anyhow::bail!("not in compile mode");
        }
        if !self.control_stack.is_empty() {
            anyhow::bail!("unresolved control structure");
        }

        let name = self
            .compiling_name
            .take()
            .ok_or_else(|| anyhow::anyhow!("no word being compiled"))?;
        let word_id = self
            .compiling_word_id
            .take()
            .ok_or_else(|| anyhow::anyhow!("no word being compiled"))?;
        let ir = std::mem::take(&mut self.compiling_ir);

        // Compile to WASM
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
        };
        let compiled = compile_word(&name, &ir, &config)
            .map_err(|e| anyhow::anyhow!("codegen error: {}", e))?;

        // Instantiate and install in the table
        self.instantiate_and_install(&compiled, word_id)?;

        // Reveal the word
        self.dictionary.reveal();
        self.state = 0;
        self.sync_here_cell();

        Ok(())
    }

    // -----------------------------------------------------------------------
    // WASM instantiation
    // -----------------------------------------------------------------------

    /// Get the current table size.
    fn table_size(&self) -> u32 {
        self.table.size(&self.store) as u32
    }

    /// Ensure the table is large enough for the given index.
    fn ensure_table_size(&mut self, needed: u32) -> anyhow::Result<()> {
        let current = self.table.size(&self.store);
        let needed64 = needed as u64;
        if needed64 >= current {
            let grow_by = needed64 - current + 1;
            self.table.grow(&mut self.store, grow_by, Ref::Func(None))?;
        }
        Ok(())
    }

    /// Instantiate a compiled WASM module and install its function in the table.
    fn instantiate_and_install(
        &mut self,
        compiled: &CompiledModule,
        word_id: WordId,
    ) -> anyhow::Result<()> {
        self.ensure_table_size(word_id.0)?;

        let module = Module::new(&self.engine, &compiled.bytes)?;
        let instance = Instance::new(
            &mut self.store,
            &module,
            &[
                self.emit_func.into(),
                self.memory.into(),
                self.dsp.into(),
                self.rsp.into(),
                self.table.into(),
            ],
        )?;

        // Get the exported function and install it in our shared table
        let func = instance
            .get_func(&mut self.store, "fn")
            .ok_or_else(|| anyhow::anyhow!("compiled module missing 'fn' export"))?;

        self.table
            .set(&mut self.store, word_id.0 as u64, Ref::Func(Some(func)))?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Word execution
    // -----------------------------------------------------------------------

    /// Execute a word by its WordId (calls through the function table).
    fn execute_word(&mut self, word_id: WordId) -> anyhow::Result<()> {
        let r = self
            .table
            .get(&mut self.store, word_id.0 as u64)
            .ok_or_else(|| anyhow::anyhow!("word {} not in function table", word_id.0))?;
        let func = *r
            .unwrap_func()
            .ok_or_else(|| anyhow::anyhow!("word {} is null funcref", word_id.0))?;

        func.call(&mut self.store, &[], &mut [])?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Data stack operations
    // -----------------------------------------------------------------------

    /// Push a value onto the data stack.
    fn push_data_stack(&mut self, value: i32) -> anyhow::Result<()> {
        let sp = self.dsp.get(&mut self.store).unwrap_i32() as u32;
        let new_sp = sp - CELL_SIZE;
        let data = self.memory.data_mut(&mut self.store);
        let bytes = value.to_le_bytes();
        data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&bytes);
        self.dsp.set(&mut self.store, Val::I32(new_sp as i32))?;
        Ok(())
    }

    /// Pop a value from the data stack.
    fn pop_data_stack(&mut self) -> anyhow::Result<i32> {
        let sp = self.dsp.get(&mut self.store).unwrap_i32() as u32;
        if sp >= DATA_STACK_TOP {
            anyhow::bail!("stack underflow");
        }
        let data = self.memory.data(&self.store);
        let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
        let value = i32::from_le_bytes(b);
        self.dsp
            .set(&mut self.store, Val::I32((sp + CELL_SIZE) as i32))?;
        Ok(value)
    }

    // -----------------------------------------------------------------------
    // Number parsing
    // -----------------------------------------------------------------------

    /// Try to parse a token as a number.
    fn parse_number(&self, token: &str) -> Option<i32> {
        let token = token.trim();
        if token.is_empty() {
            return None;
        }

        // Check for negative prefix
        let (negative, rest) = if let Some(stripped) = token.strip_prefix('-') {
            (true, stripped)
        } else {
            (false, token)
        };

        if rest.is_empty() {
            return None;
        }

        // Parse based on prefix
        let result = if let Some(hex) = rest.strip_prefix('$') {
            i64::from_str_radix(hex, 16).ok()
        } else if let Some(dec) = rest.strip_prefix('#') {
            dec.parse::<i64>().ok()
        } else if let Some(bin) = rest.strip_prefix('%') {
            i64::from_str_radix(bin, 2).ok()
        } else {
            i64::from_str_radix(rest, self.base).ok()
        };

        result.map(|n| if negative { -(n as i32) } else { n as i32 })
    }

    // -----------------------------------------------------------------------
    // Push IR to the active body
    // -----------------------------------------------------------------------

    /// Push an IR op into the current compilation target.
    fn push_ir(&mut self, op: IrOp) {
        self.compiling_ir.push(op);
    }

    // -----------------------------------------------------------------------
    // Primitive registration
    // -----------------------------------------------------------------------

    /// Register a primitive word by compiling its IR body and installing it.
    fn register_primitive(
        &mut self,
        name: &str,
        immediate: bool,
        ir_body: Vec<IrOp>,
    ) -> anyhow::Result<WordId> {
        let word_id = self
            .dictionary
            .create(name, immediate)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
        };
        let compiled = compile_word(name, &ir_body, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for {}: {}", name, e))?;

        self.instantiate_and_install(&compiled, word_id)?;
        self.dictionary.reveal();
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);

        Ok(word_id)
    }

    /// Register a primitive whose implementation is a host function (not IR-compiled).
    fn register_host_primitive(
        &mut self,
        name: &str,
        immediate: bool,
        func: Func,
    ) -> anyhow::Result<WordId> {
        let word_id = self
            .dictionary
            .create(name, immediate)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        self.ensure_table_size(word_id.0)?;
        self.table
            .set(&mut self.store, word_id.0 as u64, Ref::Func(Some(func)))?;
        self.dictionary.reveal();
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);

        Ok(word_id)
    }

    /// Register all built-in primitive words.
    fn register_primitives(&mut self) -> anyhow::Result<()> {
        // -- Stack manipulation --
        self.register_primitive("DUP", false, vec![IrOp::Dup])?;
        self.register_primitive("DROP", false, vec![IrOp::Drop])?;
        self.register_primitive("SWAP", false, vec![IrOp::Swap])?;
        self.register_primitive("OVER", false, vec![IrOp::Over])?;
        self.register_primitive("ROT", false, vec![IrOp::Rot])?;
        self.register_primitive("NIP", false, vec![IrOp::Nip])?;
        self.register_primitive("TUCK", false, vec![IrOp::Tuck])?;

        // -- Arithmetic --
        self.register_primitive("+", false, vec![IrOp::Add])?;
        self.register_primitive("-", false, vec![IrOp::Sub])?;
        self.register_primitive("*", false, vec![IrOp::Mul])?;
        self.register_primitive("/MOD", false, vec![IrOp::DivMod])?;
        self.register_primitive("NEGATE", false, vec![IrOp::Negate])?;
        self.register_primitive("ABS", false, vec![IrOp::Abs])?;
        // / and MOD in terms of /MOD
        self.register_primitive("/", false, vec![IrOp::DivMod, IrOp::Swap, IrOp::Drop])?;
        self.register_primitive("MOD", false, vec![IrOp::DivMod, IrOp::Drop])?;

        // -- Comparison --
        self.register_primitive("=", false, vec![IrOp::Eq])?;
        self.register_primitive("<>", false, vec![IrOp::NotEq])?;
        self.register_primitive("<", false, vec![IrOp::Lt])?;
        self.register_primitive(">", false, vec![IrOp::Gt])?;
        self.register_primitive("U<", false, vec![IrOp::LtUnsigned])?;
        self.register_primitive("0=", false, vec![IrOp::ZeroEq])?;
        self.register_primitive("0<", false, vec![IrOp::ZeroLt])?;

        // -- Logic --
        self.register_primitive("AND", false, vec![IrOp::And])?;
        self.register_primitive("OR", false, vec![IrOp::Or])?;
        self.register_primitive("XOR", false, vec![IrOp::Xor])?;
        self.register_primitive("INVERT", false, vec![IrOp::Invert])?;
        self.register_primitive("LSHIFT", false, vec![IrOp::Lshift])?;
        self.register_primitive("RSHIFT", false, vec![IrOp::Rshift])?;

        // -- Memory --
        self.register_primitive("@", false, vec![IrOp::Fetch])?;
        self.register_primitive("!", false, vec![IrOp::Store])?;
        self.register_primitive("C@", false, vec![IrOp::CFetch])?;
        self.register_primitive("C!", false, vec![IrOp::CStore])?;
        self.register_primitive("+!", false, vec![IrOp::PlusStore])?;

        // -- Return stack --
        self.register_primitive(">R", false, vec![IrOp::ToR])?;
        self.register_primitive("R>", false, vec![IrOp::FromR])?;
        self.register_primitive("R@", false, vec![IrOp::RFetch])?;

        // -- I/O --
        self.register_primitive("EMIT", false, vec![IrOp::Emit])?;
        self.register_primitive("CR", false, vec![IrOp::Cr])?;

        // -- Constants --
        self.register_primitive("TRUE", false, vec![IrOp::PushI32(-1)])?;
        self.register_primitive("FALSE", false, vec![IrOp::PushI32(0)])?;
        self.register_primitive("BL", false, vec![IrOp::PushI32(32)])?;
        self.register_primitive("SPACE", false, vec![IrOp::PushI32(32), IrOp::Emit])?;

        // -- 1+ 1- 2* 2/ --
        self.register_primitive("1+", false, vec![IrOp::PushI32(1), IrOp::Add])?;
        self.register_primitive("1-", false, vec![IrOp::PushI32(1), IrOp::Sub])?;
        self.register_primitive("2*", false, vec![IrOp::PushI32(1), IrOp::Lshift])?;
        self.register_primitive("2/", false, vec![IrOp::PushI32(1), IrOp::Rshift])?;

        // -- Priority 1: Loop support --
        // I -- push loop index (top of return stack)
        self.register_primitive("I", false, vec![IrOp::RFetch])?;
        // J -- outer loop counter (third item on return stack)
        self.register_j()?;
        // UNLOOP -- remove loop parameters from return stack
        self.register_primitive(
            "UNLOOP",
            false,
            vec![IrOp::FromR, IrOp::Drop, IrOp::FromR, IrOp::Drop],
        )?;
        // LEAVE -- set index to limit so loop exits
        self.register_leave()?;

        // -- Priority 2: Defining words handled in interpret_token --
        // (VARIABLE, CONSTANT, CREATE are special tokens)

        // -- Priority 3: Memory/system words --
        self.register_here()?;
        self.register_allot()?;
        self.register_comma()?;
        self.register_c_comma()?;
        self.register_primitive("CELLS", false, vec![IrOp::PushI32(4), IrOp::Mul])?;
        self.register_primitive("CELL+", false, vec![IrOp::PushI32(4), IrOp::Add])?;
        // CHARS is a no-op (byte addressed)
        self.register_primitive("CHARS", false, vec![])?;
        self.register_primitive("CHAR+", false, vec![IrOp::PushI32(1), IrOp::Add])?;
        self.register_align()?;
        self.register_aligned()?;
        self.register_move()?;
        self.register_fill()?;

        // -- Priority 4: Stack/arithmetic --
        self.register_primitive("2DUP", false, vec![IrOp::Over, IrOp::Over])?;
        self.register_primitive("2DROP", false, vec![IrOp::Drop, IrOp::Drop])?;
        self.register_primitive(
            "2SWAP",
            false,
            vec![
                IrOp::Rot, IrOp::ToR, IrOp::Rot, IrOp::FromR,
            ],
        )?;
        self.register_2over()?;
        self.register_qdup()?;
        self.register_pick()?;
        self.register_min()?;
        self.register_max()?;
        self.register_within()?;

        // -- Priority 5: Comparison --
        self.register_primitive("0<>", false, vec![IrOp::ZeroEq, IrOp::ZeroEq])?;
        self.register_primitive("0>", false, vec![IrOp::PushI32(0), IrOp::Gt])?;

        // -- Priority 6: System/compiler --
        self.register_primitive("EXECUTE", false, vec![IrOp::Execute])?;
        self.register_immediate_word()?;
        self.register_decimal()?;
        self.register_hex()?;
        self.register_type_word()?;
        self.register_spaces()?;
        self.register_tick()?;
        self.register_to_body()?;
        self.register_environment_q()?;
        self.register_source()?;
        self.register_abort()?;

        // -- I/O: . (dot) needs host function because it does number-to-string --
        // We'll compile a word that pops and calls a host function.
        // The simplest approach: make DOT a host function that reads the stack
        // directly via memory + dsp.
        self.register_dot()?;

        // -- .S (print stack) --
        self.register_dot_s()?;

        // -- DEPTH --
        self.register_depth()?;

        Ok(())
    }

    /// Register the `.` (dot) word as a host function.
    fn register_dot(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let output = Arc::clone(&self.output);
        let base_cell = Arc::clone(&self.base_cell);

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Read top of data stack
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let value = i32::from_le_bytes(b);
                // Increment dsp (pop)
                dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))?;
                // Format number
                let base_val = *base_cell.lock().unwrap();
                let s = if base_val == 16 {
                    if value < 0 {
                        format!("-{:X} ", -(value as i64))
                    } else {
                        format!("{:X} ", value)
                    }
                } else {
                    format!("{} ", value)
                };
                output.lock().unwrap().push_str(&s);
                Ok(())
            },
        );

        self.register_host_primitive(".", false, func)?;
        Ok(())
    }

    /// Register `.S` (print stack without consuming).
    fn register_dot_s(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let output = Arc::clone(&self.output);

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let depth = (DATA_STACK_TOP - sp) / CELL_SIZE;
                let mut out = output.lock().unwrap();
                out.push_str(&format!("<{}> ", depth));
                // Print from bottom to top
                let mut addr = DATA_STACK_TOP - CELL_SIZE;
                while addr >= sp {
                    let b: [u8; 4] = data[addr as usize..addr as usize + 4].try_into().unwrap();
                    let v = i32::from_le_bytes(b);
                    out.push_str(&format!("{} ", v));
                    if addr < CELL_SIZE {
                        break;
                    }
                    addr -= CELL_SIZE;
                }
                Ok(())
            },
        );

        self.register_host_primitive(".S", false, func)?;
        Ok(())
    }

    /// Register DEPTH word.
    fn register_depth(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp_global = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp_global.get(&mut caller).unwrap_i32() as u32;
                let depth = ((DATA_STACK_TOP - sp) / CELL_SIZE) as i32;
                // Push depth onto stack
                let new_sp = sp - CELL_SIZE;
                let data = memory.data_mut(&mut caller);
                let bytes = depth.to_le_bytes();
                data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&bytes);
                dsp_global.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("DEPTH", false, func)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Priority 1: Loop support host functions
    // -----------------------------------------------------------------------

    /// Register J (outer loop counter) as a host function.
    /// During nested DO loops the return stack looks like:
    ///   ... outer_limit outer_index inner_limit inner_index  (inner_index on top)
    /// J reads the outer index = rsp + 8 (skip inner index and inner limit).
    fn register_j(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let rsp = self.rsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let rsp_val = rsp.get(&mut caller).unwrap_i32() as u32;
                // rsp points to inner_index, rsp+4 = inner_limit, rsp+8 = outer_index
                let addr = (rsp_val + 8) as usize;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[addr..addr + 4].try_into().unwrap();
                let value = i32::from_le_bytes(b);
                // Push onto data stack
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let new_sp = sp - CELL_SIZE;
                let data = memory.data_mut(&mut caller);
                let bytes = value.to_le_bytes();
                data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&bytes);
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("J", false, func)?;
        Ok(())
    }

    /// Register LEAVE as a host function.
    /// Sets the loop index equal to the limit so the loop exits on next iteration.
    fn register_leave(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let rsp = self.rsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let rsp_val = rsp.get(&mut caller).unwrap_i32() as u32;
                // rsp points to index, rsp+4 = limit
                let limit_addr = (rsp_val + 4) as usize;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[limit_addr..limit_addr + 4].try_into().unwrap();
                let limit = i32::from_le_bytes(b);
                // Set index = limit
                let index_addr = rsp_val as usize;
                let data = memory.data_mut(&mut caller);
                let bytes = limit.to_le_bytes();
                data[index_addr..index_addr + 4].copy_from_slice(&bytes);
                Ok(())
            },
        );

        self.register_host_primitive("LEAVE", false, func)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Priority 2: Defining words
    // -----------------------------------------------------------------------

    /// VARIABLE <name> -- create a variable with one cell of storage.
    fn define_variable(&mut self) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("VARIABLE: expected name"))?;

        // Create a dictionary entry; the word will push its parameter field address.
        let word_id = self
            .dictionary
            .create(&name, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // Allocate one cell in WASM memory for the variable's storage
        self.refresh_user_here();
        let var_addr = self.user_here;
        self.user_here += CELL_SIZE;

        // Initialize the cell to 0 in WASM memory
        let data = self.memory.data_mut(&mut self.store);
        data[var_addr as usize..var_addr as usize + 4].copy_from_slice(&0i32.to_le_bytes());

        // Compile a tiny word that pushes the variable's address
        let ir_body = vec![IrOp::PushI32(var_addr as i32)];
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
        };
        let compiled = compile_word(&name, &ir_body, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for VARIABLE {}: {}", name, e))?;

        self.instantiate_and_install(&compiled, word_id)?;
        self.dictionary.reveal();
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);
        self.sync_here_cell();

        Ok(())
    }

    /// CONSTANT <name> -- create a constant.
    fn define_constant(&mut self) -> anyhow::Result<()> {
        let value = self.pop_data_stack()?;
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("CONSTANT: expected name"))?;

        let word_id = self
            .dictionary
            .create(&name, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // Compile a word that pushes the constant value
        let ir_body = vec![IrOp::PushI32(value)];
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
        };
        let compiled = compile_word(&name, &ir_body, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for CONSTANT {}: {}", name, e))?;

        self.instantiate_and_install(&compiled, word_id)?;
        self.dictionary.reveal();
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);
        self.sync_here_cell();

        Ok(())
    }

    /// CREATE <name> -- create a word that pushes its parameter field address.
    /// The address points into WASM linear memory where user data can be stored.
    fn define_create(&mut self) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("CREATE: expected name"))?;

        let word_id = self
            .dictionary
            .create(&name, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // The parameter field address is the current user_here
        self.refresh_user_here();
        let pfa = self.user_here;

        // Compile a word that pushes the pfa
        let ir_body = vec![IrOp::PushI32(pfa as i32)];
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
        };
        let compiled = compile_word(&name, &ir_body, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for CREATE {}: {}", name, e))?;

        self.instantiate_and_install(&compiled, word_id)?;
        self.dictionary.reveal();
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);
        self.sync_here_cell();

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Priority 3: Memory/system host functions
    // -----------------------------------------------------------------------

    /// HERE -- push the current user data pointer.
    fn register_here(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        // Use a shared cell that tracks user_here.
        let here_cell = Arc::new(Mutex::new(self.user_here));
        self.here_cell = Some(Arc::clone(&here_cell));

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let here_val = *here_cell.lock().unwrap();
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let new_sp = sp - CELL_SIZE;
                let data = memory.data_mut(&mut caller);
                let bytes = (here_val as i32).to_le_bytes();
                data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&bytes);
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("HERE", false, func)?;
        Ok(())
    }

    /// Keep the here_cell in sync with user_here.
    fn sync_here_cell(&self) {
        if let Some(ref cell) = self.here_cell {
            *cell.lock().unwrap() = self.user_here;
        }
    }

    /// Update user_here from the shared cell and then write back.
    fn refresh_user_here(&mut self) {
        if let Some(ref cell) = self.here_cell {
            self.user_here = *cell.lock().unwrap();
        }
    }

    /// ALLOT -- ( n -- ) advance HERE by n bytes.
    fn register_allot(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let here_cell = self.here_cell.clone();

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Pop n from data stack
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let n = i32::from_le_bytes(b);
                dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))?;
                // Advance HERE
                if let Some(ref cell) = here_cell {
                    let mut h = cell.lock().unwrap();
                    *h = (*h as i32 + n) as u32;
                }
                Ok(())
            },
        );

        self.register_host_primitive("ALLOT", false, func)?;
        Ok(())
    }

    /// , (comma) -- ( x -- ) store x at HERE, advance HERE by cell.
    fn register_comma(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let here_cell = self.here_cell.clone();

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Pop value from data stack
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let value = i32::from_le_bytes(b);
                dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))?;
                // Store at HERE and advance
                if let Some(ref cell) = here_cell {
                    let mut h = cell.lock().unwrap();
                    let addr = *h as usize;
                    let data = memory.data_mut(&mut caller);
                    let bytes = value.to_le_bytes();
                    data[addr..addr + 4].copy_from_slice(&bytes);
                    *h += CELL_SIZE;
                }
                Ok(())
            },
        );

        self.register_host_primitive(",", false, func)?;
        Ok(())
    }

    /// C, -- ( char -- ) store byte at HERE, advance HERE by 1.
    fn register_c_comma(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let here_cell = self.here_cell.clone();

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let value = i32::from_le_bytes(b) as u8;
                dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))?;
                if let Some(ref cell) = here_cell {
                    let mut h = cell.lock().unwrap();
                    let addr = *h as usize;
                    let data = memory.data_mut(&mut caller);
                    data[addr] = value;
                    *h += 1;
                }
                Ok(())
            },
        );

        self.register_host_primitive("C,", false, func)?;
        Ok(())
    }

    /// ALIGN -- align HERE to cell boundary.
    fn register_align(&mut self) -> anyhow::Result<()> {
        let here_cell = self.here_cell.clone();

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |_caller, _params, _results| {
                if let Some(ref cell) = here_cell {
                    let mut h = cell.lock().unwrap();
                    *h = (*h + 3) & !3;
                }
                Ok(())
            },
        );

        self.register_host_primitive("ALIGN", false, func)?;
        Ok(())
    }

    /// ALIGNED -- ( addr -- aligned-addr ) align address to cell boundary.
    fn register_aligned(&mut self) -> anyhow::Result<()> {
        // Can be done purely in IR: (addr + 3) AND NOT(3)
        // addr 3 + 3 INVERT AND
        self.register_primitive(
            "ALIGNED",
            false,
            vec![
                IrOp::PushI32(3),
                IrOp::Add,
                IrOp::PushI32(!3),
                IrOp::And,
            ],
        )?;
        Ok(())
    }

    /// MOVE -- ( src dst n -- ) memory move.
    fn register_move(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                // Pop n
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let n = i32::from_le_bytes(b) as usize;
                // Pop dst
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let dst = i32::from_le_bytes(b) as usize;
                // Pop src
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let src = i32::from_le_bytes(b) as usize;
                dsp.set(&mut caller, Val::I32((sp + 12) as i32))?;
                // Perform copy (handle overlapping regions)
                let data = memory.data_mut(&mut caller);
                if src < dst && src + n > dst {
                    // Overlapping, copy backwards
                    for i in (0..n).rev() {
                        data[dst + i] = data[src + i];
                    }
                } else {
                    for i in 0..n {
                        data[dst + i] = data[src + i];
                    }
                }
                Ok(())
            },
        );

        self.register_host_primitive("MOVE", false, func)?;
        Ok(())
    }

    /// FILL -- ( addr n char -- ) fill memory.
    fn register_fill(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let ch = i32::from_le_bytes(b) as u8;
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let n = i32::from_le_bytes(b) as usize;
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let addr = i32::from_le_bytes(b) as usize;
                dsp.set(&mut caller, Val::I32((sp + 12) as i32))?;
                let data = memory.data_mut(&mut caller);
                for i in 0..n {
                    data[addr + i] = ch;
                }
                Ok(())
            },
        );

        self.register_host_primitive("FILL", false, func)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Priority 4: Stack/arithmetic host functions
    // -----------------------------------------------------------------------

    /// 2OVER -- ( a b c d -- a b c d a b ) copy second pair over top pair.
    fn register_2over(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                // Stack (top first): d at sp, c at sp+4, b at sp+8, a at sp+12
                // We want to copy a and b on top
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let val_b = i32::from_le_bytes(b);
                let b: [u8; 4] = data[(sp + 12) as usize..(sp + 16) as usize]
                    .try_into()
                    .unwrap();
                let val_a = i32::from_le_bytes(b);
                // Push a then b (a goes deeper, b on top)
                let new_sp = sp - 8;
                let data = memory.data_mut(&mut caller);
                // Write a at new_sp+4 (deeper), b at new_sp (top)
                data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                    .copy_from_slice(&val_a.to_le_bytes());
                data[new_sp as usize..(new_sp + 4) as usize]
                    .copy_from_slice(&val_b.to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("2OVER", false, func)?;
        Ok(())
    }

    /// ?DUP -- ( x -- 0 | x x ) duplicate if non-zero.
    fn register_qdup(&mut self) -> anyhow::Result<()> {
        self.register_primitive(
            "?DUP",
            false,
            vec![
                IrOp::Dup,
                IrOp::If {
                    then_body: vec![IrOp::Dup],
                    else_body: None,
                },
            ],
        )?;
        Ok(())
    }

    /// PICK -- ( xn ... x0 n -- xn ... x0 xn ) pick nth item.
    fn register_pick(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                // Read n from TOS
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let n = i32::from_le_bytes(b) as u32;
                // Read the nth item below TOS: at sp + (n+1)*CELL_SIZE
                let pick_addr = (sp + (n + 1) * CELL_SIZE) as usize;
                let b: [u8; 4] = data[pick_addr..pick_addr + 4].try_into().unwrap();
                let value = i32::from_le_bytes(b);
                // Replace TOS with picked value
                let data = memory.data_mut(&mut caller);
                let bytes = value.to_le_bytes();
                data[sp as usize..sp as usize + 4].copy_from_slice(&bytes);
                Ok(())
            },
        );

        self.register_host_primitive("PICK", false, func)?;
        Ok(())
    }

    /// MIN -- ( a b -- min )
    fn register_min(&mut self) -> anyhow::Result<()> {
        // 2DUP > IF SWAP THEN DROP
        self.register_primitive(
            "MIN",
            false,
            vec![
                IrOp::Over,
                IrOp::Over,
                IrOp::Gt,
                IrOp::If {
                    then_body: vec![IrOp::Swap],
                    else_body: None,
                },
                IrOp::Drop,
            ],
        )?;
        Ok(())
    }

    /// MAX -- ( a b -- max )
    fn register_max(&mut self) -> anyhow::Result<()> {
        // 2DUP < IF SWAP THEN DROP
        self.register_primitive(
            "MAX",
            false,
            vec![
                IrOp::Over,
                IrOp::Over,
                IrOp::Lt,
                IrOp::If {
                    then_body: vec![IrOp::Swap],
                    else_body: None,
                },
                IrOp::Drop,
            ],
        )?;
        Ok(())
    }

    /// WITHIN -- ( n lo hi -- flag )
    fn register_within(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let hi = i32::from_le_bytes(b);
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let lo = i32::from_le_bytes(b);
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let n = i32::from_le_bytes(b);
                // WITHIN: true if lo <= n < hi (unsigned subtraction trick)
                let result = ((n.wrapping_sub(lo)) as u32) < ((hi.wrapping_sub(lo)) as u32);
                let flag: i32 = if result { -1 } else { 0 };
                // Pop 3, push 1: net = sp + 8
                let new_sp = sp + 8;
                let data = memory.data_mut(&mut caller);
                let bytes = flag.to_le_bytes();
                data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&bytes);
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("WITHIN", false, func)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Priority 6: System/compiler host functions
    // -----------------------------------------------------------------------

    /// IMMEDIATE -- toggle immediate flag on the most recent word.
    fn register_immediate_word(&mut self) -> anyhow::Result<()> {
        // IMMEDIATE needs to call dictionary.toggle_immediate().
        // Since the host function can't access self.dictionary directly,
        // we use the WASM memory to track this... actually, we handle IMMEDIATE
        // as a special token in interpret_token instead.
        //
        // But we still want it in the dictionary so it can be found.
        // Let's make it a no-op host function and handle it in interpret_token.
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |_caller, _params, _results| Ok(()),
        );

        self.register_host_primitive("IMMEDIATE", false, func)?;
        Ok(())
    }

    /// DECIMAL -- set BASE to 10.
    fn register_decimal(&mut self) -> anyhow::Result<()> {
        // Similar to IMMEDIATE, we handle in interpret_token.
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |_caller, _params, _results| Ok(()),
        );

        self.register_host_primitive("DECIMAL", false, func)?;
        Ok(())
    }

    /// HEX -- set BASE to 16.
    fn register_hex(&mut self) -> anyhow::Result<()> {
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |_caller, _params, _results| Ok(()),
        );

        self.register_host_primitive("HEX", false, func)?;
        Ok(())
    }

    /// TYPE -- ( c-addr u -- ) output a string from memory.
    fn register_type_word(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let output = Arc::clone(&self.output);

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                // Pop u (length)
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let len = i32::from_le_bytes(b) as usize;
                // Pop c-addr
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let addr = i32::from_le_bytes(b) as usize;
                dsp.set(&mut caller, Val::I32((sp + 8) as i32))?;
                // Read string from memory and output
                let data = memory.data(&caller);
                let s = String::from_utf8_lossy(&data[addr..addr + len]).to_string();
                output.lock().unwrap().push_str(&s);
                Ok(())
            },
        );

        self.register_host_primitive("TYPE", false, func)?;
        Ok(())
    }

    /// SPACES -- ( n -- ) output n spaces.
    fn register_spaces(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let output = Arc::clone(&self.output);

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let n = i32::from_le_bytes(b);
                dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))?;
                if n > 0 {
                    let spaces = " ".repeat(n as usize);
                    output.lock().unwrap().push_str(&spaces);
                }
                Ok(())
            },
        );

        self.register_host_primitive("SPACES", false, func)?;
        Ok(())
    }

    /// ' (tick) in interpret mode -- push the xt (function table index) of the next word.
    fn register_tick(&mut self) -> anyhow::Result<()> {
        // Tick is handled as a special token in interpret_token_immediate.
        // But we still register it so it's in the dictionary for FIND etc.
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |_caller, _params, _results| Ok(()),
        );

        self.register_host_primitive("'", false, func)?;
        Ok(())
    }

    /// Interpret-mode tick: read next word, look it up, push its xt.
    fn interpret_tick(&mut self) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("': expected word name"))?;
        if let Some((_addr, word_id, _imm)) = self.dictionary.find(&name) {
            self.push_data_stack(word_id.0 as i32)?;
        } else {
            anyhow::bail!("': unknown word: {}", name);
        }
        Ok(())
    }

    /// Interpret-mode CHAR: read next word, push first character.
    fn interpret_char(&mut self) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("CHAR: expected word"))?;
        if let Some(ch) = name.chars().next() {
            self.push_data_stack(ch as i32)?;
        }
        Ok(())
    }

    /// >BODY -- ( xt -- addr ) given xt, return parameter field address.
    fn register_to_body(&mut self) -> anyhow::Result<()> {
        // For our system, >BODY is tricky since we'd need to map xt back to
        // a dictionary entry. For now, a stub that's unused in simple programs.
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |_caller, _params, _results| Ok(()),
        );

        self.register_host_primitive(">BODY", false, func)?;
        Ok(())
    }

    /// ENVIRONMENT? -- ( c-addr u -- false ) query system parameters.
    fn register_environment_q(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                // Pop two args (c-addr u), push FALSE
                let new_sp = sp + 4; // net: pop 2, push 1 = sp + 4
                let data = memory.data_mut(&mut caller);
                let bytes = 0i32.to_le_bytes();
                data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&bytes);
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("ENVIRONMENT?", false, func)?;
        Ok(())
    }

    /// SOURCE -- ( -- c-addr u ) push address and length of input buffer.
    fn register_source(&mut self) -> anyhow::Result<()> {
        // SOURCE is complex because the input buffer is in Rust-side state.
        // For now, return 0 0 as a stub.
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let new_sp = sp - 8; // push 2 values
                let data = memory.data_mut(&mut caller);
                // c-addr = 0
                data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&0i32.to_le_bytes());
                // u = 0
                data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                    .copy_from_slice(&0i32.to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("SOURCE", false, func)?;
        Ok(())
    }

    /// ABORT -- clear stacks and throw error.
    fn register_abort(&mut self) -> anyhow::Result<()> {
        let dsp = self.dsp;
        let rsp = self.rsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Reset stack pointers
                dsp.set(&mut caller, Val::I32(DATA_STACK_TOP as i32))?;
                rsp.set(&mut caller, Val::I32(RETURN_STACK_TOP as i32))?;
                Err(wasmtime::Error::msg("ABORT"))
            },
        );

        self.register_host_primitive("ABORT", false, func)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(input: &str) -> (Vec<i32>, String) {
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate(input).unwrap();
        let output = vm.take_output();
        let stack = vm.data_stack();
        (stack, output)
    }

    fn eval_output(input: &str) -> String {
        let (_, output) = eval(input);
        output
    }

    fn eval_stack(input: &str) -> Vec<i32> {
        let (stack, _) = eval(input);
        stack
    }

    // -- Basic stack operations --

    #[test]
    fn test_push_number() {
        assert_eq!(eval_stack("42"), vec![42]);
    }

    #[test]
    fn test_push_multiple() {
        assert_eq!(eval_stack("1 2 3"), vec![3, 2, 1]);
    }

    #[test]
    fn test_negative_number() {
        assert_eq!(eval_stack("-5"), vec![-5]);
    }

    #[test]
    fn test_hex_number() {
        assert_eq!(eval_stack("$FF"), vec![255]);
    }

    #[test]
    fn test_binary_number() {
        assert_eq!(eval_stack("%1010"), vec![10]);
    }

    // -- Arithmetic --

    #[test]
    fn test_add() {
        assert_eq!(eval_stack("2 3 +"), vec![5]);
    }

    #[test]
    fn test_sub() {
        assert_eq!(eval_stack("10 3 -"), vec![7]);
    }

    #[test]
    fn test_mul() {
        assert_eq!(eval_stack("6 7 *"), vec![42]);
    }

    #[test]
    fn test_div() {
        assert_eq!(eval_stack("10 3 /"), vec![3]);
    }

    #[test]
    fn test_mod() {
        assert_eq!(eval_stack("10 3 MOD"), vec![1]);
    }

    // -- I/O --

    #[test]
    fn test_dot() {
        assert_eq!(eval_output("42 ."), "42 ");
    }

    #[test]
    fn test_dot_negative() {
        assert_eq!(eval_output("-5 ."), "-5 ");
    }

    #[test]
    fn test_emit() {
        assert_eq!(eval_output("65 EMIT"), "A");
    }

    #[test]
    fn test_cr() {
        assert_eq!(eval_output("CR"), "\n");
    }

    // -- Colon definitions --

    #[test]
    fn test_square() {
        assert_eq!(eval_output(": SQUARE DUP * ; 7 SQUARE ."), "49 ");
    }

    #[test]
    fn test_two_plus_three() {
        assert_eq!(eval_output("2 3 + ."), "5 ");
    }

    #[test]
    fn test_colon_def_with_call() {
        assert_eq!(
            eval_output(": DOUBLE DUP + ; : QUAD DOUBLE DOUBLE ; 3 QUAD ."),
            "12 "
        );
    }

    // -- Control flow --

    #[test]
    fn test_if_then() {
        assert_eq!(eval_output(": TEST 1 > IF 42 THEN ; 5 TEST ."), "42 ");
    }

    #[test]
    fn test_if_else_then() {
        assert_eq!(
            eval_output(": ABS2 DUP 0< IF NEGATE THEN ; -5 ABS2 ."),
            "5 "
        );
    }

    #[test]
    fn test_begin_until() {
        // Count down from 3, push each value
        assert_eq!(
            eval_output(": COUNTDOWN BEGIN DUP . 1 - DUP 0= UNTIL DROP ; 3 COUNTDOWN"),
            "3 2 1 "
        );
    }

    #[test]
    fn test_do_loop() {
        assert_eq!(
            eval_output(": TEST 5 0 DO 42 . LOOP ; TEST"),
            "42 42 42 42 42 "
        );
    }

    // -- Recursion --

    #[test]
    fn test_factorial() {
        assert_eq!(
            eval_output(": FACT DUP 1 > IF DUP 1 - RECURSE * THEN ; 5 FACT ."),
            "120 "
        );
    }

    // -- Comments --

    #[test]
    fn test_paren_comment() {
        assert_eq!(eval_stack("1 ( this is a comment ) 2"), vec![2, 1]);
    }

    #[test]
    fn test_backslash_comment() {
        assert_eq!(eval_stack("1 2 \\ this is ignored"), vec![2, 1]);
    }

    // -- String output --

    #[test]
    fn test_dot_quote() {
        assert_eq!(eval_output(".\" Hello World\""), "Hello World");
    }

    // -- Stack words --

    #[test]
    fn test_dup() {
        assert_eq!(eval_stack("5 DUP"), vec![5, 5]);
    }

    #[test]
    fn test_drop() {
        assert_eq!(eval_stack("1 2 DROP"), vec![1]);
    }

    #[test]
    fn test_swap() {
        assert_eq!(eval_stack("1 2 SWAP"), vec![1, 2]);
    }

    #[test]
    fn test_over() {
        assert_eq!(eval_stack("1 2 OVER"), vec![1, 2, 1]);
    }

    #[test]
    fn test_rot() {
        // ( 1 2 3 -- 2 3 1 )  top-first: [1, 3, 2]
        assert_eq!(eval_stack("1 2 3 ROT"), vec![1, 3, 2]);
    }

    // -- Comparison --

    #[test]
    fn test_eq() {
        assert_eq!(eval_stack("5 5 ="), vec![-1]);
        assert_eq!(eval_stack("3 5 ="), vec![0]);
    }

    #[test]
    fn test_less_than() {
        assert_eq!(eval_stack("3 5 <"), vec![-1]);
        assert_eq!(eval_stack("5 3 <"), vec![0]);
    }

    #[test]
    fn test_greater_than() {
        assert_eq!(eval_stack("5 3 >"), vec![-1]);
        assert_eq!(eval_stack("3 5 >"), vec![0]);
    }

    // -- Logic --

    #[test]
    fn test_and() {
        assert_eq!(eval_stack("$FF $0F AND"), vec![0x0F]);
    }

    #[test]
    fn test_or() {
        assert_eq!(eval_stack("$F0 $0F OR"), vec![0xFF]);
    }

    #[test]
    fn test_invert() {
        assert_eq!(eval_stack("0 INVERT"), vec![-1]);
    }

    // -- Constants --

    #[test]
    fn test_true_false() {
        assert_eq!(eval_stack("TRUE"), vec![-1]);
        assert_eq!(eval_stack("FALSE"), vec![0]);
    }

    #[test]
    fn test_bl() {
        assert_eq!(eval_stack("BL"), vec![32]);
    }

    // -- Complex programs --

    #[test]
    fn test_fibonacci() {
        assert_eq!(
            eval_output(": FIB DUP 1 > IF DUP 1 - RECURSE SWAP 2 - RECURSE + THEN ; 10 FIB ."),
            "55 "
        );
    }

    #[test]
    fn test_begin_while_repeat() {
        assert_eq!(
            eval_output(": COUNTDOWN BEGIN DUP WHILE DUP . 1 - REPEAT DROP ; 3 COUNTDOWN"),
            "3 2 1 "
        );
    }

    #[test]
    fn test_nested_if() {
        assert_eq!(
            eval_output(
                ": CLASSIFY DUP 0< IF DROP .\" neg\" ELSE 0= IF .\" zero\" ELSE .\" pos\" THEN THEN ; -1 CLASSIFY"
            ),
            "neg"
        );
    }

    #[test]
    fn test_nested_if_zero() {
        assert_eq!(
            eval_output(
                ": CLASSIFY DUP 0< IF DROP .\" neg\" ELSE 0= IF .\" zero\" ELSE .\" pos\" THEN THEN ; 0 CLASSIFY"
            ),
            "zero"
        );
    }

    #[test]
    fn test_nested_if_pos() {
        assert_eq!(
            eval_output(
                ": CLASSIFY DUP 0< IF DROP .\" neg\" ELSE 0= IF .\" zero\" ELSE .\" pos\" THEN THEN ; 5 CLASSIFY"
            ),
            "pos"
        );
    }

    // -- Multiple evaluations (simulating REPL) --

    #[test]
    fn test_multi_eval() {
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate(": SQUARE DUP * ;").unwrap();
        let _ = vm.take_output();
        vm.evaluate("7 SQUARE .").unwrap();
        assert_eq!(vm.take_output(), "49 ");
    }

    // ===================================================================
    // New words: Priority 1 - Loop support
    // ===================================================================

    #[test]
    fn test_i_in_do_loop() {
        // : TEST 5 0 DO I . LOOP ; TEST
        assert_eq!(
            eval_output(": TEST 5 0 DO I . LOOP ; TEST"),
            "0 1 2 3 4 "
        );
    }

    #[test]
    fn test_j_in_nested_do_loop() {
        // Nested loops: outer 0..2, inner 0..3
        assert_eq!(
            eval_output(": TEST 3 0 DO 2 0 DO J . LOOP LOOP ; TEST"),
            "0 0 1 1 2 2 "
        );
    }

    #[test]
    fn test_unloop() {
        // UNLOOP removes loop params, EXIT leaves the word
        assert_eq!(
            eval_output(": TEST 5 0 DO I DUP 3 = IF . UNLOOP EXIT THEN DROP LOOP ; TEST"),
            "3 "
        );
    }

    #[test]
    fn test_leave() {
        // LEAVE sets index=limit so the loop exits on next iteration.
        // Note: LEAVE does not skip the rest of the current iteration's body.
        // So we print first, then check for the exit condition.
        assert_eq!(
            eval_output(": TEST 10 0 DO I . I 3 = IF LEAVE THEN LOOP ; TEST"),
            "0 1 2 3 "
        );
    }

    // ===================================================================
    // New words: Priority 2 - Defining words
    // ===================================================================

    #[test]
    fn test_variable() {
        assert_eq!(eval_output("VARIABLE X 42 X ! X @ ."), "42 ");
    }

    #[test]
    fn test_variable_default_zero() {
        assert_eq!(eval_output("VARIABLE X X @ ."), "0 ");
    }

    #[test]
    fn test_variable_multiple() {
        assert_eq!(
            eval_output("VARIABLE A VARIABLE B 10 A ! 20 B ! A @ B @ + ."),
            "30 "
        );
    }

    #[test]
    fn test_constant() {
        assert_eq!(eval_output("10 CONSTANT TEN TEN ."), "10 ");
    }

    #[test]
    fn test_constant_negative() {
        assert_eq!(eval_output("-42 CONSTANT NEG NEG ."), "-42 ");
    }

    #[test]
    fn test_create() {
        // CREATE makes a word that pushes its parameter field address
        // We can store a value there and fetch it
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("CREATE FOO").unwrap();
        // FOO pushes an address; we can read/write that location
        vm.evaluate("FOO").unwrap();
        let stack = vm.data_stack();
        assert!(!stack.is_empty());
        // The address should be a valid memory address
        assert!(stack[0] > 0);
    }

    // ===================================================================
    // New words: Priority 3 - Memory/system words
    // ===================================================================

    #[test]
    fn test_cells() {
        assert_eq!(eval_stack("3 CELLS"), vec![12]);
    }

    #[test]
    fn test_cell_plus() {
        assert_eq!(eval_stack("100 CELL+"), vec![104]);
    }

    #[test]
    fn test_chars_noop() {
        assert_eq!(eval_stack("5 CHARS"), vec![5]);
    }

    #[test]
    fn test_char_plus() {
        assert_eq!(eval_stack("100 CHAR+"), vec![101]);
    }

    #[test]
    fn test_here() {
        // HERE should push a valid address
        let stack = eval_stack("HERE");
        assert_eq!(stack.len(), 1);
        assert!(stack[0] > 0);
    }

    #[test]
    fn test_aligned() {
        assert_eq!(eval_stack("0 ALIGNED"), vec![0]);
        assert_eq!(eval_stack("1 ALIGNED"), vec![4]);
        assert_eq!(eval_stack("4 ALIGNED"), vec![4]);
        assert_eq!(eval_stack("5 ALIGNED"), vec![8]);
    }

    // ===================================================================
    // New words: Priority 4 - Stack/arithmetic
    // ===================================================================

    #[test]
    fn test_2dup() {
        assert_eq!(eval_stack("1 2 2DUP"), vec![2, 1, 2, 1]);
    }

    #[test]
    fn test_2drop() {
        assert_eq!(eval_stack("1 2 3 4 2DROP"), vec![2, 1]);
    }

    #[test]
    fn test_2swap() {
        // ( 1 2 3 4 -- 3 4 1 2 )
        assert_eq!(eval_stack("1 2 3 4 2SWAP"), vec![2, 1, 4, 3]);
    }

    #[test]
    fn test_2over() {
        // ( 1 2 3 4 -- 1 2 3 4 1 2 )
        assert_eq!(eval_stack("1 2 3 4 2OVER"), vec![2, 1, 4, 3, 2, 1]);
    }

    #[test]
    fn test_qdup_nonzero() {
        assert_eq!(eval_stack("5 ?DUP"), vec![5, 5]);
    }

    #[test]
    fn test_qdup_zero() {
        assert_eq!(eval_stack("0 ?DUP"), vec![0]);
    }

    #[test]
    fn test_min() {
        assert_eq!(eval_stack("3 5 MIN"), vec![3]);
        assert_eq!(eval_stack("5 3 MIN"), vec![3]);
        assert_eq!(eval_stack("-1 1 MIN"), vec![-1]);
    }

    #[test]
    fn test_max() {
        assert_eq!(eval_stack("3 5 MAX"), vec![5]);
        assert_eq!(eval_stack("5 3 MAX"), vec![5]);
        assert_eq!(eval_stack("-1 1 MAX"), vec![1]);
    }

    #[test]
    fn test_pick() {
        // 0 PICK = DUP
        assert_eq!(eval_stack("1 2 3 0 PICK"), vec![3, 3, 2, 1]);
        // 1 PICK = OVER
        assert_eq!(eval_stack("1 2 3 1 PICK"), vec![2, 3, 2, 1]);
        // 2 PICK
        assert_eq!(eval_stack("1 2 3 2 PICK"), vec![1, 3, 2, 1]);
    }

    // ===================================================================
    // New words: Priority 5 - Comparison
    // ===================================================================

    #[test]
    fn test_0_not_equal() {
        assert_eq!(eval_stack("5 0<>"), vec![-1]);
        assert_eq!(eval_stack("0 0<>"), vec![0]);
    }

    #[test]
    fn test_0_greater() {
        assert_eq!(eval_stack("5 0>"), vec![-1]);
        assert_eq!(eval_stack("0 0>"), vec![0]);
        assert_eq!(eval_stack("-1 0>"), vec![0]);
    }

    // ===================================================================
    // New words: Priority 6 - System/compiler
    // ===================================================================

    #[test]
    fn test_execute() {
        // ' word EXECUTE should execute the word
        assert_eq!(eval_output("42 ' . EXECUTE"), "42 ");
    }

    #[test]
    fn test_execute_in_colon() {
        assert_eq!(
            eval_output(": TEST ['] . EXECUTE ; 99 TEST"),
            "99 "
        );
    }

    #[test]
    fn test_hex_decimal() {
        assert_eq!(eval_output("HEX FF DECIMAL ."), "255 ");
    }

    #[test]
    fn test_hex_output() {
        assert_eq!(eval_output("HEX FF ."), "FF ");
    }

    #[test]
    fn test_decimal_default() {
        assert_eq!(eval_output("255 ."), "255 ");
    }

    #[test]
    fn test_immediate() {
        // Define a word, then mark it IMMEDIATE
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate(": MYWORD 42 ; IMMEDIATE").unwrap();
        // MYWORD is now immediate; when used in compile mode it executes
        vm.evaluate(": TEST MYWORD . ; TEST").unwrap();
        // During compilation of TEST, MYWORD executes immediately pushing 42,
        // then . prints it. After TEST is defined, running TEST does nothing
        // because MYWORD already ran during compilation.
        let out = vm.take_output();
        assert_eq!(out, "42 ");
    }

    #[test]
    fn test_char_word() {
        assert_eq!(eval_stack("CHAR A"), vec![65]);
        assert_eq!(eval_stack("CHAR Z"), vec![90]);
    }

    #[test]
    fn test_bracket_char() {
        assert_eq!(
            eval_output(": TEST [CHAR] A EMIT ; TEST"),
            "A"
        );
    }

    #[test]
    fn test_spaces() {
        assert_eq!(eval_output("3 SPACES"), "   ");
    }

    #[test]
    fn test_constant_in_colon_def() {
        assert_eq!(
            eval_output("10 CONSTANT TEN : TEST TEN . ; TEST"),
            "10 "
        );
    }

    #[test]
    fn test_variable_in_colon_def() {
        assert_eq!(
            eval_output("VARIABLE X 42 X ! : TEST X @ . ; TEST"),
            "42 "
        );
    }

    #[test]
    fn test_within() {
        assert_eq!(eval_stack("5 0 10 WITHIN"), vec![-1]);
        assert_eq!(eval_stack("0 0 10 WITHIN"), vec![-1]);
        assert_eq!(eval_stack("10 0 10 WITHIN"), vec![0]);
        assert_eq!(eval_stack("-1 0 10 WITHIN"), vec![0]);
    }

    #[test]
    fn test_do_loop_with_i_and_step() {
        // +LOOP with step of 2
        assert_eq!(
            eval_output(": TEST 10 0 DO I . 2 +LOOP ; TEST"),
            "0 2 4 6 8 "
        );
    }
}
