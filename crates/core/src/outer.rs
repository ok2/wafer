//! Outer interpreter: tokenizer, number parser, and interpret/compile dispatch.
//!
//! The outer interpreter is the main loop of Forth:
//! 1. Read a token (whitespace-delimited word)
//! 2. Look it up in the dictionary
//! 3. If found: execute (interpret mode) or compile (compile mode)
//! 4. If not found: try to parse as a number
//! 5. If number: push (interpret) or compile as literal (compile mode)
//! 6. If neither: error

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use wasmtime::{
    Engine, Func, FuncType, Global, Instance, Memory, Module, Mutability, Ref, RefType, Store,
    Table, Val, ValType,
};

use crate::codegen::{CodegenConfig, CompiledModule, compile_word};
use crate::dictionary::{Dictionary, WordId};
use crate::ir::IrOp;
use crate::memory::{
    CELL_SIZE, DATA_STACK_TOP, INPUT_BUFFER_BASE, INPUT_BUFFER_SIZE, RETURN_STACK_TOP,
    SYSVAR_BASE_VAR, SYSVAR_NUM_TIB, SYSVAR_STATE, SYSVAR_TO_IN,
};

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
// DOES> support
// ---------------------------------------------------------------------------

/// Stored definition for a DOES>-based defining word.
struct DoesDefinition {
    /// The IR for the create-part (code between CREATE and DOES>).
    create_ir: Vec<IrOp>,
    /// The word ID of the compiled does-action (code after DOES>).
    does_action_id: WordId,
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Number formatting helpers
// ---------------------------------------------------------------------------

/// Format a signed integer in the given base, followed by a space.
fn format_signed(value: i32, base: u32) -> String {
    if base == 10 {
        format!("{} ", value)
    } else if value < 0 {
        let abs = -(value as i64);
        format!("-{} ", format_unsigned_digits(abs as u32, base))
    } else {
        format!("{} ", format_unsigned_digits(value as u32, base))
    }
}

/// Format an unsigned integer in the given base, followed by a space.
fn format_unsigned(value: u32, base: u32) -> String {
    if base == 10 {
        format!("{} ", value)
    } else {
        format!("{} ", format_unsigned_digits(value, base))
    }
}

/// Convert an unsigned value to a digit string in the given base.
fn format_unsigned_digits(mut value: u32, base: u32) -> String {
    if value == 0 {
        return "0".to_string();
    }
    let mut digits = Vec::new();
    while value > 0 {
        let rem = (value % base) as u8;
        let ch = if rem < 10 {
            b'0' + rem
        } else {
            b'A' + rem - 10
        };
        digits.push(ch as char);
        value /= base;
    }
    digits.iter().rev().collect()
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
    // DOES> definitions: maps defining word ID to its DoesDefinition
    does_definitions: HashMap<WordId, DoesDefinition>,
    // Pending action from compiled defining/parsing words
    // 0 = none, 1 = CONSTANT, 2 = VARIABLE, 3 = CREATE, 4 = EVALUATE
    pending_define: Arc<Mutex<i32>>,
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
            does_definitions: HashMap::new(),
            pending_define: Arc::new(Mutex::new(0)),
        };

        vm.register_primitives()?;

        Ok(vm)
    }

    /// Evaluate a line of Forth input.
    pub fn evaluate(&mut self, input: &str) -> anyhow::Result<()> {
        self.input_buffer = input.to_string();
        self.input_pos = 0;
        self.sync_input_to_wasm();

        while let Some(token) = self.next_token() {
            self.sync_input_to_wasm();
            let wasm_to_in_before = self.input_pos;
            match self.interpret_token(&token) {
                Ok(()) => {}
                Err(e) => {
                    // Reset compile state on error to prevent cascading failures
                    self.state = 0;
                    self.compiling_name = None;
                    self.compiling_ir.clear();
                    self.control_stack.clear();
                    self.compiling_word_id = None;
                    return Err(e);
                }
            }
            // Read >IN back from WASM memory. Only apply if Forth code changed it
            // (i.e., the WASM value differs from what sync_input_to_wasm wrote).
            // This distinguishes Forth's `>IN !` from Rust-side parse_until changes.
            let data = self.memory.data(&self.store);
            let b: [u8; 4] = data[SYSVAR_TO_IN as usize..SYSVAR_TO_IN as usize + 4]
                .try_into()
                .unwrap();
            let wasm_to_in = u32::from_le_bytes(b) as usize;
            if wasm_to_in != wasm_to_in_before {
                self.input_pos = wasm_to_in;
            }
            // If >IN was set past the end of the input, stop processing
            if self.input_pos >= self.input_buffer.len() {
                break;
            }
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
            "]" => {
                // Switch to compile mode (can be used outside a colon definition)
                self.state = -1;
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
        if token_upper == ".(" {
            // Parse until closing paren and print
            if let Some(s) = self.parse_until(')') {
                self.output.lock().unwrap().push_str(&s);
            }
            return Ok(());
        }
        if token_upper == "S\"" {
            // Parse string, store in WASM memory, push (c-addr u) on stack
            if let Some(s) = self.parse_until('"') {
                self.refresh_user_here();
                let addr = self.user_here;
                let bytes = s.as_bytes();
                let len = bytes.len() as u32;
                let data = self.memory.data_mut(&mut self.store);
                data[addr as usize..addr as usize + len as usize].copy_from_slice(bytes);
                self.user_here += len;
                self.sync_here_cell();
                self.push_data_stack(addr as i32)?;
                self.push_data_stack(len as i32)?;
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
            "DOES>" => return self.interpret_does(),
            "'" => return self.interpret_tick(),
            "[CHAR]" => {
                // In interpret mode, CHAR is the standard word
                return self.interpret_char();
            }
            "CHAR" => return self.interpret_char(),
            "EVALUATE" => return self.interpret_evaluate(),
            "WORD" => return self.interpret_word(),
            "FIND" => return self.interpret_find(),
            _ => {}
        }

        // Look up in dictionary
        if let Some((_addr, word_id, _is_immediate)) = self.dictionary.find(token) {
            // Check if this is a DOES>-defining word
            if self.does_definitions.contains_key(&word_id) {
                return self.execute_does_defining(word_id);
            }
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
            // Store string at HERE, compile code to push (c-addr u)
            if let Some(s) = self.parse_until('"') {
                self.refresh_user_here();
                let addr = self.user_here;
                let bytes = s.as_bytes();
                let len = bytes.len() as u32;
                let data = self.memory.data_mut(&mut self.store);
                data[addr as usize..addr as usize + len as usize].copy_from_slice(bytes);
                self.user_here += len;
                self.sync_here_cell();
                self.push_ir(IrOp::PushI32(addr as i32));
                self.push_ir(IrOp::PushI32(len as i32));
            }
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

        // Handle ABORT" in compile mode
        if token_upper == "ABORT\"" {
            if let Some(s) = self.parse_until('"') {
                // Compile: IF <push-addr> <push-len> TYPE ABORT THEN
                // The flag is already on stack; compile the check
                self.refresh_user_here();
                let addr = self.user_here;
                let bytes = s.as_bytes();
                let len = bytes.len() as u32;
                let data = self.memory.data_mut(&mut self.store);
                data[addr as usize..addr as usize + len as usize].copy_from_slice(bytes);
                self.user_here += len;
                self.sync_here_cell();

                // Find TYPE and ABORT word IDs
                let type_call = self.dictionary.find("TYPE").map(|(_, id, _)| id);
                let abort_call = self.dictionary.find("ABORT").map(|(_, id, _)| id);
                let mut then_body = vec![IrOp::PushI32(addr as i32), IrOp::PushI32(len as i32)];
                if let Some(type_id) = type_call {
                    then_body.push(IrOp::Call(type_id));
                }
                if let Some(abort_id) = abort_call {
                    then_body.push(IrOp::Call(abort_id));
                }
                self.push_ir(IrOp::If {
                    then_body,
                    else_body: None,
                });
            }
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
            "DOES>" => {
                return self.compile_does();
            }
            "CREATE" => {
                // In compile mode, CREATE is a no-op marker for DOES> definitions.
                // The actual creation happens at runtime via the DOES> mechanism
                // or via the pending_define mechanism for non-DOES> patterns.
                return Ok(());
            }
            "VARIABLE" | "CONSTANT" => {
                // These are now in the dictionary as host functions.
                // Fall through to dictionary lookup to compile a call.
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
        // Check if the word changed BASE via WASM memory
        self.sync_base_from_wasm();
        // Handle pending defining actions (CONSTANT, VARIABLE, CREATE called at runtime)
        self.handle_pending_define()?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Data stack operations
    // -----------------------------------------------------------------------

    /// Push a value onto the data stack.
    fn push_data_stack(&mut self, value: i32) -> anyhow::Result<()> {
        let sp = self.dsp.get(&mut self.store).unwrap_i32() as u32;
        if sp < CELL_SIZE + crate::memory::DATA_STACK_BASE {
            anyhow::bail!("data stack overflow");
        }
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
        self.register_primitive("2/", false, vec![IrOp::PushI32(1), IrOp::ArithRshift])?;

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
            vec![IrOp::Rot, IrOp::ToR, IrOp::Rot, IrOp::FromR],
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
        self.register_dot()?;
        self.register_dot_s()?;
        self.register_depth()?;

        // -- Priority 7: New core words --
        self.register_count()?;
        self.register_s_to_d()?;
        self.register_cmove()?;
        self.register_cmove_up()?;
        self.register_find()?;
        self.register_to_in()?;
        self.register_state_var()?;
        self.register_base_var()?;

        // Double-cell arithmetic
        self.register_m_star()?;
        self.register_um_star()?;
        self.register_um_div_mod()?;
        self.register_fm_div_mod()?;
        self.register_sm_div_rem()?;

        // */ and */MOD
        self.register_star_slash()?;
        self.register_star_slash_mod()?;

        // U. (unsigned dot)
        self.register_u_dot()?;

        // >NUMBER
        self.register_to_number()?;

        // \ (backslash comment) as an immediate word so POSTPONE can find it
        self.register_backslash()?;

        // CONSTANT, VARIABLE, CREATE as callable words (for use inside colon defs)
        self.register_defining_words()?;

        // EVALUATE and WORD as callable words (for use inside colon defs)
        self.register_evaluate_word()?;
        self.register_word_word()?;

        // 2@ and 2!
        self.register_two_fetch()?;
        self.register_two_store()?;

        // Pictured numeric output
        self.register_pictured_numeric()?;

        Ok(())
    }

    /// Register the `.` (dot) word as a host function.
    fn register_dot(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let output = Arc::clone(&self.output);

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Read top of data stack
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let value = i32::from_le_bytes(b);
                // Read BASE from WASM memory
                let b: [u8; 4] = data[SYSVAR_BASE_VAR as usize..SYSVAR_BASE_VAR as usize + 4]
                    .try_into()
                    .unwrap();
                let base_val = u32::from_le_bytes(b);
                // Increment dsp (pop)
                dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))?;
                // Format number in current base
                let s = format_signed(value, base_val);
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
        // Store fn_index at 0x30 for DOES> to find
        self.store_latest_fn_index(word_id);
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
            vec![IrOp::PushI32(3), IrOp::Add, IrOp::PushI32(!3), IrOp::And],
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
                data[new_sp as usize..(new_sp + 4) as usize].copy_from_slice(&val_b.to_le_bytes());
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
        // DECIMAL stores 10 at BASE address in WASM memory
        self.register_primitive(
            "DECIMAL",
            false,
            vec![
                IrOp::PushI32(10),
                IrOp::PushI32(SYSVAR_BASE_VAR as i32),
                IrOp::Store,
            ],
        )?;
        Ok(())
    }

    /// HEX -- set BASE to 16.
    fn register_hex(&mut self) -> anyhow::Result<()> {
        // HEX stores 16 at BASE address in WASM memory
        self.register_primitive(
            "HEX",
            false,
            vec![
                IrOp::PushI32(16),
                IrOp::PushI32(SYSVAR_BASE_VAR as i32),
                IrOp::Store,
            ],
        )?;
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
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // The input buffer is synced to WASM memory at INPUT_BUFFER_BASE.
                // The length is stored at a known location. We read it from the
                // first 4 bytes before the buffer, or we use a different approach:
                // read the actual length from a sysvar.
                // For simplicity, read the buffer length from SYSVAR_NUM_TIB.
                let data = memory.data(&caller);
                let b: [u8; 4] = data[crate::memory::SYSVAR_NUM_TIB as usize
                    ..crate::memory::SYSVAR_NUM_TIB as usize + 4]
                    .try_into()
                    .unwrap();
                let len = i32::from_le_bytes(b);

                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let new_sp = sp - 8;
                let data = memory.data_mut(&mut caller);
                // c-addr (deeper)
                data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                    .copy_from_slice(&(INPUT_BUFFER_BASE as i32).to_le_bytes());
                // u (on top)
                data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&len.to_le_bytes());
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

    // -----------------------------------------------------------------------
    // EVALUATE -- save input, interpret string, restore input
    // -----------------------------------------------------------------------

    /// EVALUATE -- ( c-addr u -- ) interpret the given string.
    fn interpret_evaluate(&mut self) -> anyhow::Result<()> {
        // Pop length and address from data stack
        let len = self.pop_data_stack()? as u32;
        let addr = self.pop_data_stack()? as u32;

        // Bounds check
        let mem_len = self.memory.data(&self.store).len() as u32;
        if addr > mem_len || addr.wrapping_add(len) > mem_len {
            anyhow::bail!("EVALUATE: invalid address/length");
        }

        // Read the string from WASM memory
        let data = self.memory.data(&self.store);
        let s =
            String::from_utf8_lossy(&data[addr as usize..addr as usize + len as usize]).to_string();

        // Save current input state
        let saved_buffer = std::mem::take(&mut self.input_buffer);
        let saved_pos = self.input_pos;

        // Set new input
        self.input_buffer = s;
        self.input_pos = 0;

        // Interpret
        while let Some(token) = self.next_token() {
            self.interpret_token(&token)?;
        }

        // Restore input state
        self.input_buffer = saved_buffer;
        self.input_pos = saved_pos;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // WORD -- parse delimited word from input
    // -----------------------------------------------------------------------

    /// WORD ( char -- c-addr ) parse next word delimited by char.
    fn interpret_word(&mut self) -> anyhow::Result<()> {
        let delim = self.pop_data_stack()? as u8 as char;

        // Skip leading delimiters
        let bytes = self.input_buffer.as_bytes();
        while self.input_pos < bytes.len() && bytes[self.input_pos] == delim as u8 {
            self.input_pos += 1;
        }

        // Collect until delimiter or end
        let start = self.input_pos;
        while self.input_pos < bytes.len() && bytes[self.input_pos] != delim as u8 {
            self.input_pos += 1;
        }
        // Skip past delimiter
        if self.input_pos < bytes.len() {
            self.input_pos += 1;
        }

        let word_bytes = &bytes[start..self.input_pos.min(bytes.len())];
        // Trim trailing delimiter if present
        let word_bytes =
            if !word_bytes.is_empty() && word_bytes[word_bytes.len() - 1] == delim as u8 {
                &word_bytes[..word_bytes.len() - 1]
            } else {
                word_bytes
            };
        let word_len = word_bytes.len();

        // Store as counted string in WASM memory (at a transient buffer area)
        // Use PAD area for transient storage
        let buf_addr = crate::memory::PAD_BASE;
        let data = self.memory.data_mut(&mut self.store);
        data[buf_addr as usize] = word_len as u8;
        data[buf_addr as usize + 1..buf_addr as usize + 1 + word_len].copy_from_slice(word_bytes);

        self.push_data_stack(buf_addr as i32)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // DOES> -- compile-time and interpret-time
    // -----------------------------------------------------------------------

    /// DOES> in interpret mode (used in defining words like: CREATE xx DOES> @ )
    /// This implementation supports DOES> used after CREATE in the same definition.
    fn interpret_does(&mut self) -> anyhow::Result<()> {
        // In interpret mode, DOES> takes the code that follows it (rest of input)
        // and attaches it to the most recently CREATEd word.
        // Collect remaining tokens until ; or end of input as the DOES> body
        let mut does_ir: Vec<IrOp> = Vec::new();

        // The most recently defined word's address
        let latest = self.dictionary.latest();
        let pfa = self
            .dictionary
            .param_field_addr(latest)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        // Parse the rest as the does-body
        while let Some(token) = self.next_token() {
            let tu = token.to_ascii_uppercase();
            if tu == ";" {
                break;
            }
            // Simple: look up and compile calls
            if let Some((_addr, word_id, _imm)) = self.dictionary.find(&token) {
                does_ir.push(IrOp::Call(word_id));
            } else if let Some(n) = self.parse_number(&token) {
                does_ir.push(IrOp::PushI32(n));
            } else {
                anyhow::bail!("DOES>: unknown word: {}", token);
            }
        }

        // Compile the DOES> body: push PFA, then run the body
        let mut full_ir = vec![IrOp::PushI32(pfa as i32)];
        full_ir.extend(does_ir);

        // Get the existing word_id from the code field
        let fn_index = self
            .dictionary
            .code_field(latest)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let word_id = WordId(fn_index);

        // Compile and replace
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
        };
        let name = self
            .dictionary
            .word_name(latest)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let compiled = compile_word(&name, &full_ir, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for DOES>: {}", e))?;
        self.instantiate_and_install(&compiled, word_id)?;

        Ok(())
    }

    /// DOES> in compile mode -- handle the `: name CREATE ... DOES> ... ;` pattern.
    ///
    /// Strategy: compile the does-body as a separate WASM word, then create
    /// the defining word as a host function that:
    /// 1. Reads the next token from the input buffer
    /// 2. Creates a new word (via define_create logic)
    /// 3. Executes the create-part IR
    /// 4. Patches the new word to push PFA + call does-body
    fn compile_does(&mut self) -> anyhow::Result<()> {
        // The create-part is everything compiled so far in the current definition.
        let create_ir = std::mem::take(&mut self.compiling_ir);

        // Save the defining word's info before we modify the dictionary
        let defining_word_id = self
            .compiling_word_id
            .ok_or_else(|| anyhow::anyhow!("DOES>: not compiling"))?;
        let defining_name = self
            .compiling_name
            .clone()
            .ok_or_else(|| anyhow::anyhow!("DOES>: no word name"))?;
        // Save the dictionary address of the defining word so we can reveal it
        // even after intermediate dictionary entries are created.
        let defining_word_addr = self.dictionary.latest();

        // Collect the does-body tokens (everything after DOES> until ;)
        let mut does_tokens: Vec<String> = Vec::new();
        let mut depth = 0i32;
        while let Some(token) = self.next_token() {
            let tu = token.to_ascii_uppercase();
            if tu == ";" && depth == 0 {
                break;
            }
            if tu == "IF" || tu == "DO" || tu == "BEGIN" {
                depth += 1;
            }
            if tu == "THEN" || tu == "LOOP" || tu == "+LOOP" || tu == "UNTIL" || tu == "REPEAT" {
                depth -= 1;
            }
            does_tokens.push(token);
        }

        // Compile the does-body as a separate word
        let does_word_id = self
            .dictionary
            .create("_does_action_", false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        self.dictionary.reveal();
        self.next_table_index = self.next_table_index.max(does_word_id.0 + 1);

        // Save and compile does-body
        let saved_name = self.compiling_name.take();
        let saved_word_id = self.compiling_word_id.take();
        let saved_control = std::mem::take(&mut self.control_stack);

        self.compiling_ir.clear();
        self.compiling_name = Some("_does_action_".to_string());
        self.compiling_word_id = Some(does_word_id);

        for token in &does_tokens {
            self.compile_token(token)?;
        }

        let does_ir = std::mem::take(&mut self.compiling_ir);
        let config = CodegenConfig {
            base_fn_index: does_word_id.0,
            table_size: self.table_size(),
        };
        let compiled = compile_word("_does_action_", &does_ir, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for DOES> body: {}", e))?;
        self.instantiate_and_install(&compiled, does_word_id)?;

        // Restore compilation state
        self.compiling_name = saved_name;
        self.compiling_word_id = saved_word_id;
        self.control_stack = saved_control;

        // Register the defining word as a "does-defining" word.
        self.does_definitions.insert(
            defining_word_id,
            DoesDefinition {
                create_ir,
                does_action_id: does_word_id,
            },
        );

        // Compile the defining word as a no-op (the actual work is done
        // by the outer interpreter when it detects the does-definition).
        let config = CodegenConfig {
            base_fn_index: defining_word_id.0,
            table_size: self.table_size(),
        };
        let compiled = compile_word(&defining_name, &[], &config)
            .map_err(|e| anyhow::anyhow!("codegen error for defining word: {}", e))?;
        self.instantiate_and_install(&compiled, defining_word_id)?;

        // Reveal the defining word by its saved address (not LATEST, which
        // may have moved due to intermediate dictionary entries).
        self.dictionary.reveal_at(defining_word_addr);
        self.state = 0;
        self.compiling_name = None;
        self.compiling_word_id = None;
        self.compiling_ir.clear();
        self.sync_here_cell();

        Ok(())
    }

    /// Execute a DOES>-defining word (like CONST, VALUE, etc.).
    /// This handles the CREATE + create-part + DOES> patching at runtime.
    fn execute_does_defining(&mut self, defining_word_id: WordId) -> anyhow::Result<()> {
        // Get the does-definition info
        let def = self
            .does_definitions
            .get(&defining_word_id)
            .ok_or_else(|| anyhow::anyhow!("not a DOES> defining word"))?;
        let create_ir = def.create_ir.clone();
        let does_action_id = def.does_action_id;

        // Step 1: Read the name of the new word from the input stream
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("defining word: expected name"))?;

        // Step 2: Create the new word (like define_create)
        let new_word_id = self
            .dictionary
            .create(&name, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        self.refresh_user_here();
        let pfa = self.user_here;

        // Temporarily install a "push PFA" word (will be patched later)
        let ir_body = vec![IrOp::PushI32(pfa as i32)];
        let config = CodegenConfig {
            base_fn_index: new_word_id.0,
            table_size: self.table_size(),
        };
        let compiled = compile_word(&name, &ir_body, &config)
            .map_err(|e| anyhow::anyhow!("codegen: {}", e))?;
        self.instantiate_and_install(&compiled, new_word_id)?;
        self.dictionary.reveal();
        self.next_table_index = self.next_table_index.max(new_word_id.0 + 1);

        // Step 3: Execute the create-part IR
        // In standard Forth, CREATE does NOT push PFA onto the stack.
        // The create-part (e.g., `,`) operates on the data already on the stack.
        // For `: CONST CREATE , DOES> @ ;` with `42 CONST X`:
        //   stack has [42], CREATE reads "X", `,` pops 42 and stores at HERE (=PFA)
        if !create_ir.is_empty() {
            let tmp_word_id = self
                .dictionary
                .create("_create_part_", false)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            self.dictionary.reveal();
            self.next_table_index = self.next_table_index.max(tmp_word_id.0 + 1);

            let config = CodegenConfig {
                base_fn_index: tmp_word_id.0,
                table_size: self.table_size(),
            };
            let compiled = compile_word("_create_part_", &create_ir, &config)
                .map_err(|e| anyhow::anyhow!("codegen: {}", e))?;
            self.instantiate_and_install(&compiled, tmp_word_id)?;
            self.execute_word(tmp_word_id)?;
        }

        // Step 4: Patch the new word to push PFA and call does-action
        self.refresh_user_here();
        let patched_ir = vec![IrOp::PushI32(pfa as i32), IrOp::Call(does_action_id)];
        let config = CodegenConfig {
            base_fn_index: new_word_id.0,
            table_size: self.table_size(),
        };
        let compiled = compile_word(&name, &patched_ir, &config)
            .map_err(|e| anyhow::anyhow!("DOES> patch codegen: {}", e))?;
        self.instantiate_and_install(&compiled, new_word_id)?;
        self.sync_here_cell();

        Ok(())
    }

    // -----------------------------------------------------------------------
    // New core word registrations
    // -----------------------------------------------------------------------

    /// COUNT ( c-addr -- c-addr+1 u ) get counted string length.
    fn register_count(&mut self) -> anyhow::Result<()> {
        // DUP C@ SWAP 1+ SWAP  => but simpler: DUP 1+ SWAP C@
        // Actually: ( c-addr -- c-addr+1 u )
        // DUP C@ >R 1+ R>
        // Or even simpler with IR:
        // DUP 1+ SWAP C@
        self.register_primitive(
            "COUNT",
            false,
            vec![
                IrOp::Dup,
                IrOp::PushI32(1),
                IrOp::Add,
                IrOp::Swap,
                IrOp::CFetch,
            ],
        )?;
        Ok(())
    }

    /// S>D ( n -- d ) sign-extend single to double-cell.
    /// Pushes n, then 0 or -1 depending on sign.
    fn register_s_to_d(&mut self) -> anyhow::Result<()> {
        // ( n -- n sign ) where sign is 0 or -1
        // DUP 0< gives us 0 or -1
        self.register_primitive("S>D", false, vec![IrOp::Dup, IrOp::ZeroLt])?;
        Ok(())
    }

    /// CMOVE ( src dst u -- ) copy u bytes from src to dst, low-to-high.
    fn register_cmove(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let u = i32::from_le_bytes(b) as usize;
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let dst = i32::from_le_bytes(b) as usize;
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let src = i32::from_le_bytes(b) as usize;
                dsp.set(&mut caller, Val::I32((sp + 12) as i32))?;
                let data = memory.data_mut(&mut caller);
                for i in 0..u {
                    data[dst + i] = data[src + i];
                }
                Ok(())
            },
        );

        self.register_host_primitive("CMOVE", false, func)?;
        Ok(())
    }

    /// CMOVE> ( src dst u -- ) copy u bytes from src to dst, high-to-low.
    fn register_cmove_up(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let u = i32::from_le_bytes(b) as usize;
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let dst = i32::from_le_bytes(b) as usize;
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let src = i32::from_le_bytes(b) as usize;
                dsp.set(&mut caller, Val::I32((sp + 12) as i32))?;
                let data = memory.data_mut(&mut caller);
                for i in (0..u).rev() {
                    data[dst + i] = data[src + i];
                }
                Ok(())
            },
        );

        self.register_host_primitive("CMOVE>", false, func)?;
        Ok(())
    }

    /// FIND ( c-addr -- c-addr 0 | xt 1 | xt -1 ) look up counted string.
    fn register_find(&mut self) -> anyhow::Result<()> {
        let pending = Arc::clone(&self.pending_define);
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |_caller, _params, _results| {
                *pending.lock().unwrap() = 6;
                Ok(())
            },
        );

        self.register_host_primitive("FIND", false, func)?;
        Ok(())
    }

    /// >IN ( -- addr ) push address of the input position variable.
    fn register_to_in(&mut self) -> anyhow::Result<()> {
        // >IN is stored at SYSVAR_TO_IN in WASM memory
        self.register_primitive(">IN", false, vec![IrOp::PushI32(SYSVAR_TO_IN as i32)])?;
        Ok(())
    }

    /// STATE ( -- addr ) push address of the STATE variable.
    fn register_state_var(&mut self) -> anyhow::Result<()> {
        self.register_primitive("STATE", false, vec![IrOp::PushI32(SYSVAR_STATE as i32)])?;
        Ok(())
    }

    /// BASE ( -- addr ) push address of the BASE variable.
    fn register_base_var(&mut self) -> anyhow::Result<()> {
        // Initialize BASE in WASM memory
        let data = self.memory.data_mut(&mut self.store);
        data[SYSVAR_BASE_VAR as usize..SYSVAR_BASE_VAR as usize + 4]
            .copy_from_slice(&10u32.to_le_bytes());

        self.register_primitive("BASE", false, vec![IrOp::PushI32(SYSVAR_BASE_VAR as i32)])?;
        Ok(())
    }

    /// M* ( n1 n2 -- d ) signed multiply producing double-cell result.
    fn register_m_star(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let n2 = i32::from_le_bytes(b) as i64;
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let n1 = i32::from_le_bytes(b) as i64;
                let result = n1 * n2;
                // Store as double-cell: low cell deeper, high cell on top
                let lo = result as i32;
                let hi = (result >> 32) as i32;
                let data = memory.data_mut(&mut caller);
                // Overwrite the two stack slots (net: pop 2, push 2 = same sp)
                data[(sp + 4) as usize..(sp + 8) as usize].copy_from_slice(&lo.to_le_bytes());
                data[sp as usize..sp as usize + 4].copy_from_slice(&hi.to_le_bytes());
                Ok(())
            },
        );

        self.register_host_primitive("M*", false, func)?;
        Ok(())
    }

    /// UM* ( u1 u2 -- ud ) unsigned multiply producing double-cell result.
    fn register_um_star(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let u2 = u32::from_le_bytes(b) as u64;
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let u1 = u32::from_le_bytes(b) as u64;
                let result = u1 * u2;
                let lo = result as u32;
                let hi = (result >> 32) as u32;
                let data = memory.data_mut(&mut caller);
                data[(sp + 4) as usize..(sp + 8) as usize].copy_from_slice(&lo.to_le_bytes());
                data[sp as usize..sp as usize + 4].copy_from_slice(&hi.to_le_bytes());
                Ok(())
            },
        );

        self.register_host_primitive("UM*", false, func)?;
        Ok(())
    }

    /// UM/MOD ( ud u -- rem quot ) unsigned double-cell divide.
    fn register_um_div_mod(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                // Pop u (divisor)
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let divisor = u32::from_le_bytes(b) as u64;
                // Pop ud (double-cell): high at sp+4, low at sp+8
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let hi = u32::from_le_bytes(b) as u64;
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let lo = u32::from_le_bytes(b) as u64;
                let dividend = (hi << 32) | lo;

                if divisor == 0 {
                    return Err(wasmtime::Error::msg("division by zero"));
                }

                let quot = (dividend / divisor) as u32;
                let rem = (dividend % divisor) as u32;

                // Pop 3, push 2: net sp + 4
                let new_sp = sp + 4;
                let data = memory.data_mut(&mut caller);
                // rem deeper, quot on top
                data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                    .copy_from_slice(&(rem as i32).to_le_bytes());
                data[new_sp as usize..new_sp as usize + 4]
                    .copy_from_slice(&(quot as i32).to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("UM/MOD", false, func)?;
        Ok(())
    }

    /// FM/MOD ( d n -- rem quot ) floored division.
    fn register_fm_div_mod(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                // Pop n (divisor)
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let divisor = i32::from_le_bytes(b) as i64;
                // Pop d (double-cell): high at sp+4, low at sp+8
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let hi = i32::from_le_bytes(b) as i64;
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let lo = u32::from_le_bytes(b) as i64;
                let dividend = (hi << 32) | (lo & 0xFFFF_FFFF);

                if divisor == 0 {
                    return Err(wasmtime::Error::msg("division by zero"));
                }

                // Floored division: quotient is floor(dividend/divisor)
                let mut quot = dividend / divisor;
                let mut rem = dividend % divisor;
                // Adjust for floored semantics: if remainder != 0 and signs differ
                if rem != 0 && ((rem ^ divisor) < 0) {
                    quot -= 1;
                    rem += divisor;
                }

                let new_sp = sp + 4;
                let data = memory.data_mut(&mut caller);
                data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                    .copy_from_slice(&(rem as i32).to_le_bytes());
                data[new_sp as usize..new_sp as usize + 4]
                    .copy_from_slice(&(quot as i32).to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("FM/MOD", false, func)?;
        Ok(())
    }

    /// SM/REM ( d n -- rem quot ) symmetric division.
    fn register_sm_div_rem(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let divisor = i32::from_le_bytes(b) as i64;
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let hi = i32::from_le_bytes(b) as i64;
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let lo = u32::from_le_bytes(b) as i64;
                let dividend = (hi << 32) | (lo & 0xFFFF_FFFF);

                if divisor == 0 {
                    return Err(wasmtime::Error::msg("division by zero"));
                }

                // Symmetric (truncated) division -- this is Rust's default
                let quot = dividend / divisor;
                let rem = dividend % divisor;

                let new_sp = sp + 4;
                let data = memory.data_mut(&mut caller);
                data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                    .copy_from_slice(&(rem as i32).to_le_bytes());
                data[new_sp as usize..new_sp as usize + 4]
                    .copy_from_slice(&(quot as i32).to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("SM/REM", false, func)?;
        Ok(())
    }

    /// */ ( n1 n2 n3 -- n4 ) n1*n2/n3 with intermediate double-precision.
    fn register_star_slash(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let n3 = i32::from_le_bytes(b) as i64;
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let n2 = i32::from_le_bytes(b) as i64;
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let n1 = i32::from_le_bytes(b) as i64;

                if n3 == 0 {
                    return Err(wasmtime::Error::msg("division by zero"));
                }

                let result = (n1 * n2) / n3;
                // Pop 3, push 1: net sp + 8
                let new_sp = sp + 8;
                let data = memory.data_mut(&mut caller);
                data[new_sp as usize..new_sp as usize + 4]
                    .copy_from_slice(&(result as i32).to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("*/", false, func)?;
        Ok(())
    }

    /// */MOD ( n1 n2 n3 -- rem quot ) n1*n2/n3 with intermediate double-precision.
    fn register_star_slash_mod(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let n3 = i32::from_le_bytes(b) as i64;
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let n2 = i32::from_le_bytes(b) as i64;
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let n1 = i32::from_le_bytes(b) as i64;

                if n3 == 0 {
                    return Err(wasmtime::Error::msg("division by zero"));
                }

                let product = n1 * n2;
                let quot = product / n3;
                let rem = product % n3;

                // Pop 3, push 2: net sp + 4
                let new_sp = sp + 4;
                let data = memory.data_mut(&mut caller);
                data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                    .copy_from_slice(&(rem as i32).to_le_bytes());
                data[new_sp as usize..new_sp as usize + 4]
                    .copy_from_slice(&(quot as i32).to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("*/MOD", false, func)?;
        Ok(())
    }

    /// U. ( u -- ) unsigned dot.
    fn register_u_dot(&mut self) -> anyhow::Result<()> {
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
                let value = u32::from_le_bytes(b);
                // Read BASE from WASM memory
                let b: [u8; 4] = data[SYSVAR_BASE_VAR as usize..SYSVAR_BASE_VAR as usize + 4]
                    .try_into()
                    .unwrap();
                let base_val = u32::from_le_bytes(b);
                dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))?;
                let s = format_unsigned(value, base_val);
                output.lock().unwrap().push_str(&s);
                Ok(())
            },
        );

        self.register_host_primitive("U.", false, func)?;
        Ok(())
    }

    /// >NUMBER ( ud1 c-addr1 u1 -- ud2 c-addr2 u2 ) convert string to number.
    fn register_to_number(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let mem_len = memory.data(&caller).len() as u32;
                if sp.wrapping_add(16) > mem_len || sp > mem_len {
                    return Err(wasmtime::Error::msg("stack underflow in >NUMBER"));
                }
                let data = memory.data(&caller);
                // Stack: u1 at sp, c-addr1 at sp+4, ud1-hi at sp+8, ud1-lo at sp+12
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let mut u1 = i32::from_le_bytes(b) as u32;
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let mut c_addr = u32::from_le_bytes(b);
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let ud_hi = u32::from_le_bytes(b) as u64;
                let b: [u8; 4] = data[(sp + 12) as usize..(sp + 16) as usize]
                    .try_into()
                    .unwrap();
                let ud_lo = u32::from_le_bytes(b) as u64;
                let mut ud = (ud_hi << 32) | ud_lo;

                // Read BASE from WASM memory (not base_cell)
                let b: [u8; 4] = data[SYSVAR_BASE_VAR as usize..SYSVAR_BASE_VAR as usize + 4]
                    .try_into()
                    .unwrap();
                let base = u32::from_le_bytes(b) as u64;

                while u1 > 0 {
                    let data = memory.data(&caller);
                    let ch = data[c_addr as usize] as char;
                    let digit = match ch.to_digit(base as u32) {
                        Some(d) => d as u64,
                        None => break,
                    };
                    ud = ud * base + digit;
                    c_addr += 1;
                    u1 -= 1;
                }

                let ud_lo_new = ud as u32;
                let ud_hi_new = (ud >> 32) as u32;

                let data = memory.data_mut(&mut caller);
                data[sp as usize..sp as usize + 4].copy_from_slice(&(u1 as i32).to_le_bytes());
                data[(sp + 4) as usize..(sp + 8) as usize]
                    .copy_from_slice(&(c_addr as i32).to_le_bytes());
                data[(sp + 8) as usize..(sp + 12) as usize]
                    .copy_from_slice(&(ud_hi_new as i32).to_le_bytes());
                data[(sp + 12) as usize..(sp + 16) as usize]
                    .copy_from_slice(&(ud_lo_new as i32).to_le_bytes());
                Ok(())
            },
        );

        self.register_host_primitive(">NUMBER", false, func)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // CONSTANT, VARIABLE, CREATE as callable defining words
    // -----------------------------------------------------------------------

    /// Register CONSTANT, VARIABLE, CREATE as host functions so they can
    /// be compiled into colon definitions (e.g., `: EQU CONSTANT ;`).
    fn register_defining_words(&mut self) -> anyhow::Result<()> {
        // CONSTANT: sets pending_define to 1
        {
            let pending = Arc::clone(&self.pending_define);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |_caller, _params, _results| {
                    *pending.lock().unwrap() = 1;
                    Ok(())
                },
            );
            self.register_host_primitive("CONSTANT", false, func)?;
        }

        // VARIABLE: sets pending_define to 2
        {
            let pending = Arc::clone(&self.pending_define);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |_caller, _params, _results| {
                    *pending.lock().unwrap() = 2;
                    Ok(())
                },
            );
            self.register_host_primitive("VARIABLE", false, func)?;
        }

        // CREATE: sets pending_define to 3
        {
            let pending = Arc::clone(&self.pending_define);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |_caller, _params, _results| {
                    *pending.lock().unwrap() = 3;
                    Ok(())
                },
            );
            self.register_host_primitive("CREATE", false, func)?;
        }

        Ok(())
    }

    /// Register EVALUATE as a host function callable from compiled code.
    fn register_evaluate_word(&mut self) -> anyhow::Result<()> {
        let pending = Arc::clone(&self.pending_define);
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |_caller, _params, _results| {
                *pending.lock().unwrap() = 4;
                Ok(())
            },
        );
        self.register_host_primitive("EVALUATE", false, func)?;
        Ok(())
    }

    /// Register WORD as a host function callable from compiled code.
    fn register_word_word(&mut self) -> anyhow::Result<()> {
        let pending = Arc::clone(&self.pending_define);
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |_caller, _params, _results| {
                *pending.lock().unwrap() = 5;
                Ok(())
            },
        );
        self.register_host_primitive("WORD", false, func)?;
        Ok(())
    }

    /// FIND ( c-addr -- c-addr 0 | xt 1 | xt -1 ) Look up counted string in dictionary.
    fn interpret_find(&mut self) -> anyhow::Result<()> {
        // Pop counted string address
        let c_addr = self.pop_data_stack()? as u32;

        // Read counted string from WASM memory
        let data = self.memory.data(&self.store);
        let count = data[c_addr as usize] as usize;
        let name_start = (c_addr + 1) as usize;
        let name = String::from_utf8_lossy(&data[name_start..name_start + count]).to_string();

        // Look up in dictionary
        if let Some((_addr, word_id, is_immediate)) = self.dictionary.find(&name) {
            // Found: push xt and flag
            self.push_data_stack(word_id.0 as i32)?;
            self.push_data_stack(if is_immediate { 1 } else { -1 })?;
        } else {
            // Not found: push original c-addr and 0
            self.push_data_stack(c_addr as i32)?;
            self.push_data_stack(0)?;
        }

        Ok(())
    }

    /// Check for and handle pending defining actions after word execution.
    fn handle_pending_define(&mut self) -> anyhow::Result<()> {
        let action = {
            let mut pending = self.pending_define.lock().unwrap();
            let a = *pending;
            *pending = 0;
            a
        };
        match action {
            1 => self.define_constant(),
            2 => self.define_variable(),
            3 => self.define_create(),
            4 => self.interpret_evaluate(),
            5 => self.interpret_word(),
            6 => self.interpret_find(),
            _ => Ok(()),
        }
    }

    // -----------------------------------------------------------------------
    // Backslash comment as a compilable immediate word
    // -----------------------------------------------------------------------

    /// Register `\` as an immediate host function that sets >IN to end of input.
    fn register_backslash(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Read #TIB (input buffer length)
                let data = memory.data(&caller);
                let b: [u8; 4] = data[crate::memory::SYSVAR_NUM_TIB as usize
                    ..crate::memory::SYSVAR_NUM_TIB as usize + 4]
                    .try_into()
                    .unwrap();
                let num_tib = u32::from_le_bytes(b);
                // Set >IN to end of input
                let data = memory.data_mut(&mut caller);
                data[SYSVAR_TO_IN as usize..SYSVAR_TO_IN as usize + 4]
                    .copy_from_slice(&num_tib.to_le_bytes());
                Ok(())
            },
        );

        self.register_host_primitive("\\", true, func)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // 2@ and 2!
    // -----------------------------------------------------------------------

    /// 2@ ( addr -- x1 x2 ) Fetch two cells. x2 from addr, x1 from addr+CELL.
    fn register_two_fetch(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let addr = u32::from_le_bytes(b);
                // x2 is at addr, x1 is at addr+4
                let b: [u8; 4] = data[addr as usize..addr as usize + 4].try_into().unwrap();
                let x2 = i32::from_le_bytes(b);
                let b: [u8; 4] = data[(addr + 4) as usize..(addr + 8) as usize]
                    .try_into()
                    .unwrap();
                let x1 = i32::from_le_bytes(b);
                // Replace addr with x1, push x2
                let new_sp = sp - 4;
                let data = memory.data_mut(&mut caller);
                data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                    .copy_from_slice(&x1.to_le_bytes());
                data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&x2.to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("2@", false, func)?;
        Ok(())
    }

    /// 2! ( x1 x2 addr -- ) Store x2 at addr, x1 at addr+CELL.
    fn register_two_store(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let addr = u32::from_le_bytes(b);
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let x2 = i32::from_le_bytes(b);
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let x1 = i32::from_le_bytes(b);
                // Store x2 at addr, x1 at addr+4
                let data = memory.data_mut(&mut caller);
                data[addr as usize..addr as usize + 4].copy_from_slice(&x2.to_le_bytes());
                data[(addr + 4) as usize..(addr + 8) as usize].copy_from_slice(&x1.to_le_bytes());
                // Pop 3 cells
                dsp.set(&mut caller, Val::I32((sp + 12) as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("2!", false, func)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Pictured numeric output: <# # #S #> HOLD SIGN
    // -----------------------------------------------------------------------

    /// Register pictured numeric output words.
    fn register_pictured_numeric(&mut self) -> anyhow::Result<()> {
        use crate::memory::{PAD_BASE, PAD_SIZE, SYSVAR_HLD};

        // <# ( -- ) Initialize pictured numeric output
        {
            let memory = self.memory;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _params, _results| {
                    let data = memory.data_mut(&mut caller);
                    // HLD points to end of PAD area (we build string backwards)
                    let hld = PAD_BASE + PAD_SIZE;
                    data[SYSVAR_HLD as usize..SYSVAR_HLD as usize + 4]
                        .copy_from_slice(&hld.to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("<#", false, func)?;
        }

        // HOLD ( char -- ) Add character to pictured output
        {
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
                    // Read HLD
                    let b: [u8; 4] = data[SYSVAR_HLD as usize..SYSVAR_HLD as usize + 4]
                        .try_into()
                        .unwrap();
                    let mut hld = u32::from_le_bytes(b);
                    hld -= 1;
                    let data = memory.data_mut(&mut caller);
                    data[hld as usize] = ch;
                    data[SYSVAR_HLD as usize..SYSVAR_HLD as usize + 4]
                        .copy_from_slice(&hld.to_le_bytes());
                    dsp.set(&mut caller, Val::I32((sp + 4) as i32))?;
                    Ok(())
                },
            );
            self.register_host_primitive("HOLD", false, func)?;
        }

        // SIGN ( n -- ) If n is negative, add '-' to pictured output
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _params, _results| {
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let data = memory.data(&caller);
                    let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                    let n = i32::from_le_bytes(b);
                    // Pop n
                    dsp.set(&mut caller, Val::I32((sp + 4) as i32))?;
                    if n < 0 {
                        // Add '-' like HOLD would
                        let data = memory.data(&caller);
                        let b: [u8; 4] = data[SYSVAR_HLD as usize..SYSVAR_HLD as usize + 4]
                            .try_into()
                            .unwrap();
                        let mut hld = u32::from_le_bytes(b);
                        hld -= 1;
                        let data = memory.data_mut(&mut caller);
                        data[hld as usize] = b'-';
                        data[SYSVAR_HLD as usize..SYSVAR_HLD as usize + 4]
                            .copy_from_slice(&hld.to_le_bytes());
                    }
                    Ok(())
                },
            );
            self.register_host_primitive("SIGN", false, func)?;
        }

        // # ( ud1 -- ud2 ) Divide ud by BASE, convert remainder to char, HOLD it
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _params, _results| {
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let data = memory.data(&caller);
                    // ud is on the stack as two cells: hi at sp, lo at sp+4
                    // Stack: ud-hi at sp (TOS), ud-lo at sp+4
                    let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                    let ud_hi = u32::from_le_bytes(b) as u64;
                    let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                        .try_into()
                        .unwrap();
                    let ud_lo = u32::from_le_bytes(b) as u64;
                    let ud = (ud_hi << 32) | ud_lo;

                    // Read BASE from WASM memory (not base_cell)
                    let b: [u8; 4] = data[SYSVAR_BASE_VAR as usize..SYSVAR_BASE_VAR as usize + 4]
                        .try_into()
                        .unwrap();
                    let base = u32::from_le_bytes(b) as u64;
                    let rem = (ud % base) as u32;
                    let quot = ud / base;

                    // Convert remainder to digit character
                    let ch = if rem < 10 {
                        b'0' + rem as u8
                    } else {
                        b'A' + (rem as u8 - 10)
                    };

                    // HOLD the character
                    let data = memory.data(&caller);
                    let b: [u8; 4] = data[SYSVAR_HLD as usize..SYSVAR_HLD as usize + 4]
                        .try_into()
                        .unwrap();
                    let mut hld = u32::from_le_bytes(b);
                    hld -= 1;
                    let data = memory.data_mut(&mut caller);
                    data[hld as usize] = ch;
                    data[SYSVAR_HLD as usize..SYSVAR_HLD as usize + 4]
                        .copy_from_slice(&hld.to_le_bytes());

                    // Write quotient back
                    let new_hi = (quot >> 32) as u32;
                    let new_lo = quot as u32;
                    data[sp as usize..sp as usize + 4].copy_from_slice(&new_hi.to_le_bytes());
                    data[(sp + 4) as usize..(sp + 8) as usize]
                        .copy_from_slice(&new_lo.to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("#", false, func)?;
        }

        // #S ( ud1 -- 0 0 ) Convert all remaining digits
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _params, _results| {
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let data = memory.data(&caller);
                    let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                    let ud_hi = u32::from_le_bytes(b) as u64;
                    let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                        .try_into()
                        .unwrap();
                    let ud_lo = u32::from_le_bytes(b) as u64;
                    let mut ud = (ud_hi << 32) | ud_lo;

                    // Read BASE from WASM memory (not base_cell)
                    let b: [u8; 4] = data[SYSVAR_BASE_VAR as usize..SYSVAR_BASE_VAR as usize + 4]
                        .try_into()
                        .unwrap();
                    let base = u32::from_le_bytes(b) as u64;

                    loop {
                        let rem = (ud % base) as u32;
                        ud /= base;
                        let ch = if rem < 10 {
                            b'0' + rem as u8
                        } else {
                            b'A' + (rem as u8 - 10)
                        };
                        let data = memory.data(&caller);
                        let b: [u8; 4] = data[SYSVAR_HLD as usize..SYSVAR_HLD as usize + 4]
                            .try_into()
                            .unwrap();
                        let mut hld = u32::from_le_bytes(b);
                        hld -= 1;
                        let data = memory.data_mut(&mut caller);
                        data[hld as usize] = ch;
                        data[SYSVAR_HLD as usize..SYSVAR_HLD as usize + 4]
                            .copy_from_slice(&hld.to_le_bytes());
                        if ud == 0 {
                            break;
                        }
                    }

                    let data = memory.data_mut(&mut caller);
                    data[sp as usize..sp as usize + 4].copy_from_slice(&0u32.to_le_bytes());
                    data[(sp + 4) as usize..(sp + 8) as usize].copy_from_slice(&0u32.to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("#S", false, func)?;
        }

        // #> ( xd -- c-addr u ) Finish pictured output, return string
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _params, _results| {
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let data = memory.data(&caller);
                    // Drop the double-cell, read HLD
                    let b: [u8; 4] = data[SYSVAR_HLD as usize..SYSVAR_HLD as usize + 4]
                        .try_into()
                        .unwrap();
                    let hld = u32::from_le_bytes(b);
                    let end = PAD_BASE + PAD_SIZE;
                    let len = end - hld;
                    // Replace the double on stack with (c-addr u)
                    let data = memory.data_mut(&mut caller);
                    data[(sp + 4) as usize..(sp + 8) as usize]
                        .copy_from_slice(&(hld as i32).to_le_bytes());
                    data[sp as usize..sp as usize + 4].copy_from_slice(&(len as i32).to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("#>", false, func)?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Improved SOURCE
    // -----------------------------------------------------------------------

    // SOURCE is already registered above. We need to update it to write
    // the current input buffer into WASM memory and return real addresses.
    // This is handled by syncing input_buffer to WASM memory before calls.

    /// Sync the current input buffer to WASM memory and update >IN.
    fn sync_input_to_wasm(&mut self) {
        let bytes = self.input_buffer.as_bytes();
        let len = bytes.len().min(INPUT_BUFFER_SIZE as usize);
        let data = self.memory.data_mut(&mut self.store);
        data[INPUT_BUFFER_BASE as usize..INPUT_BUFFER_BASE as usize + len]
            .copy_from_slice(&bytes[..len]);
        // Write >IN
        data[SYSVAR_TO_IN as usize..SYSVAR_TO_IN as usize + 4]
            .copy_from_slice(&(self.input_pos as u32).to_le_bytes());
        // Write STATE
        data[SYSVAR_STATE as usize..SYSVAR_STATE as usize + 4]
            .copy_from_slice(&self.state.to_le_bytes());
        // Write BASE
        data[SYSVAR_BASE_VAR as usize..SYSVAR_BASE_VAR as usize + 4]
            .copy_from_slice(&self.base.to_le_bytes());
        // Write #TIB (input buffer length)
        data[SYSVAR_NUM_TIB as usize..SYSVAR_NUM_TIB as usize + 4]
            .copy_from_slice(&(len as u32).to_le_bytes());
    }

    /// Sync BASE from WASM memory back to Rust after executing a word.
    fn sync_base_from_wasm(&mut self) {
        // Check if BASE was changed via WASM memory write (e.g., `10 BASE !`)
        let data = self.memory.data(&self.store);
        let b: [u8; 4] = data[SYSVAR_BASE_VAR as usize..SYSVAR_BASE_VAR as usize + 4]
            .try_into()
            .unwrap();
        let wasm_base = u32::from_le_bytes(b);
        if wasm_base != self.base && (2..=36).contains(&wasm_base) {
            self.base = wasm_base;
            *self.base_cell.lock().unwrap() = wasm_base;
        }
    }

    // -----------------------------------------------------------------------
    // Update define_create to store fn_index for DOES>
    // -----------------------------------------------------------------------

    /// Store the fn_index of the most recently CREATEd word at address 0x30
    /// so the DOES> patcher can find it.
    fn store_latest_fn_index(&mut self, word_id: WordId) {
        let data = self.memory.data_mut(&mut self.store);
        data[0x30..0x34].copy_from_slice(&word_id.0.to_le_bytes());
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
        assert_eq!(eval_output(": TEST 5 0 DO I . LOOP ; TEST"), "0 1 2 3 4 ");
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
        assert_eq!(eval_output(": TEST ['] . EXECUTE ; 99 TEST"), "99 ");
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
        assert_eq!(eval_output(": TEST [CHAR] A EMIT ; TEST"), "A");
    }

    #[test]
    fn test_spaces() {
        assert_eq!(eval_output("3 SPACES"), "   ");
    }

    #[test]
    fn test_constant_in_colon_def() {
        assert_eq!(eval_output("10 CONSTANT TEN : TEST TEN . ; TEST"), "10 ");
    }

    #[test]
    fn test_variable_in_colon_def() {
        assert_eq!(eval_output("VARIABLE X 42 X ! : TEST X @ . ; TEST"), "42 ");
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

    // ===================================================================
    // New words: EVALUATE
    // ===================================================================

    #[test]
    fn test_evaluate_basic() {
        assert_eq!(eval_output("S\" 2 3 + .\" EVALUATE"), "5 ");
    }

    #[test]
    fn test_evaluate_nested() {
        assert_eq!(eval_output("S\" 42 .\" EVALUATE"), "42 ");
    }

    #[test]
    fn test_evaluate_define_word() {
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("S\" : DOUBLE DUP + ;\" EVALUATE").unwrap();
        vm.evaluate("5 DOUBLE .").unwrap();
        assert_eq!(vm.take_output(), "10 ");
    }

    // ===================================================================
    // New words: S" (string literal)
    // ===================================================================

    #[test]
    fn test_s_quote_interpret() {
        // S" in interpret mode pushes c-addr and u
        let stack = eval_stack("S\" hello\"");
        assert_eq!(stack.len(), 2);
        assert!(stack[0] > 0); // length = 5
        assert!(stack[1] > 0); // address > 0
    }

    #[test]
    fn test_s_quote_type() {
        assert_eq!(eval_output("S\" Hello\" TYPE"), "Hello");
    }

    #[test]
    fn test_s_quote_compile_mode() {
        assert_eq!(eval_output(": TEST S\" World\" TYPE ; TEST"), "World");
    }

    // ===================================================================
    // New words: COUNT
    // ===================================================================

    #[test]
    fn test_count() {
        // Create a counted string: length byte followed by characters
        let mut vm = ForthVM::new().unwrap();
        // Store counted string "AB" at HERE: 2 (length), 65 ('A'), 66 ('B')
        vm.evaluate("HERE 2 C, 65 C, 66 C,").unwrap();
        // COUNT should give: addr+1 and length
        vm.evaluate("COUNT TYPE").unwrap();
        assert_eq!(vm.take_output(), "AB");
    }

    // ===================================================================
    // New words: S>D
    // ===================================================================

    #[test]
    fn test_s_to_d_positive() {
        // S>D: 5 -> (5, 0) on stack as double
        assert_eq!(eval_stack("5 S>D"), vec![0, 5]);
    }

    #[test]
    fn test_s_to_d_negative() {
        // S>D: -1 -> (-1, -1) on stack as double
        assert_eq!(eval_stack("-1 S>D"), vec![-1, -1]);
    }

    #[test]
    fn test_s_to_d_zero() {
        assert_eq!(eval_stack("0 S>D"), vec![0, 0]);
    }

    // ===================================================================
    // New words: CMOVE, CMOVE>
    // ===================================================================

    #[test]
    fn test_cmove() {
        let mut vm = ForthVM::new().unwrap();
        // Store "ABC" at src, then copy to dst
        vm.evaluate("HERE").unwrap(); // src address on stack
        vm.evaluate("65 C, 66 C, 67 C,").unwrap();
        vm.evaluate("HERE").unwrap(); // dst address on stack
        vm.evaluate("0 C, 0 C, 0 C,").unwrap(); // allocate dst space
        // Stack has: src dst (dst on top)
        // CMOVE needs ( src dst u -- )
        vm.evaluate("3 CMOVE").unwrap();
        // Nothing left on stack; but we need dst to read back
        // Recalculate: dst was at src+3
        vm.evaluate("HERE 3 -").unwrap(); // points to dst
        vm.evaluate("DUP C@ SWAP 1+ DUP C@ SWAP 1+ C@").unwrap();
        let stack = vm.data_stack();
        assert_eq!(stack[0], 67); // 'C'
        assert_eq!(stack[1], 66); // 'B'
        assert_eq!(stack[2], 65); // 'A'
    }

    #[test]
    fn test_cmove_up() {
        // CMOVE> copies high-to-low for overlapping regions
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("HERE 65 C, 66 C, 67 C,").unwrap();
        let stack = vm.data_stack();
        let src = stack[0];
        // Copy 3 bytes from src to src+1
        vm.evaluate(&format!("{} {} 3 CMOVE>", src, src + 1))
            .unwrap();
        // Memory should now be: A A B C (first byte unchanged, rest shifted)
        vm.evaluate(&format!("{} C@", src + 1)).unwrap();
        assert_eq!(vm.data_stack()[0], 65); // 'A' was copied
    }

    // ===================================================================
    // New words: >IN, STATE, BASE
    // ===================================================================

    #[test]
    fn test_to_in() {
        // >IN should push a valid address
        let stack = eval_stack(">IN");
        assert_eq!(stack.len(), 1);
        assert_eq!(stack[0], SYSVAR_TO_IN as i32);
    }

    #[test]
    fn test_state_variable() {
        // STATE should push the address of the state variable
        let stack = eval_stack("STATE");
        assert_eq!(stack.len(), 1);
        assert_eq!(stack[0], SYSVAR_STATE as i32);
    }

    #[test]
    fn test_base_variable() {
        let stack = eval_stack("BASE");
        assert_eq!(stack.len(), 1);
        assert_eq!(stack[0], SYSVAR_BASE_VAR as i32);
    }

    // ===================================================================
    // New words: DOES>
    // ===================================================================

    #[test]
    fn test_does_constant_pattern() {
        // The classic DOES> test: define CONST using CREATE and DOES>
        assert_eq!(
            eval_output(": CONST CREATE , DOES> @ ; 42 CONST X X ."),
            "42 "
        );
    }

    #[test]
    fn test_does_multiple_instances() {
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate(": CONST CREATE , DOES> @ ;").unwrap();
        vm.evaluate("10 CONST TEN").unwrap();
        vm.evaluate("20 CONST TWENTY").unwrap();
        vm.evaluate("TEN . TWENTY .").unwrap();
        assert_eq!(vm.take_output(), "10 20 ");
    }

    // ===================================================================
    // New words: Double-cell arithmetic
    // ===================================================================

    #[test]
    fn test_m_star() {
        // M* ( n1 n2 -- d ) signed multiply to double
        // 3 * 4 = 12, fits in low cell, high = 0
        assert_eq!(eval_stack("3 4 M*"), vec![0, 12]);
    }

    #[test]
    fn test_m_star_negative() {
        // -3 * 4 = -12
        assert_eq!(eval_stack("-3 4 M*"), vec![-1, -12]);
    }

    #[test]
    fn test_um_star() {
        // UM* ( u1 u2 -- ud ) unsigned multiply to double
        assert_eq!(eval_stack("3 4 UM*"), vec![0, 12]);
    }

    #[test]
    fn test_um_div_mod() {
        // UM/MOD ( ud u -- rem quot )
        // 10 / 3 = 3 rem 1
        assert_eq!(eval_stack("10 0 3 UM/MOD"), vec![3, 1]);
    }

    #[test]
    fn test_fm_div_mod() {
        // FM/MOD ( d n -- rem quot ) floored division
        // 10 / 3 = 3 rem 1
        assert_eq!(eval_stack("10 0 3 FM/MOD"), vec![3, 1]);
    }

    #[test]
    fn test_fm_div_mod_negative() {
        // FM/MOD with negative dividend: -7 / 2
        // Floored: quot = -4, rem = 1 (because -4*2+1 = -7)
        assert_eq!(eval_stack("-7 -1 2 FM/MOD"), vec![-4, 1]);
    }

    #[test]
    fn test_sm_div_rem() {
        // SM/REM ( d n -- rem quot ) symmetric division
        // 10 / 3 = 3 rem 1
        assert_eq!(eval_stack("10 0 3 SM/REM"), vec![3, 1]);
    }

    #[test]
    fn test_sm_div_rem_negative() {
        // SM/REM with negative dividend: -7 / 2
        // Symmetric: quot = -3, rem = -1 (because -3*2+(-1) = -7)
        assert_eq!(eval_stack("-7 -1 2 SM/REM"), vec![-3, -1]);
    }

    // ===================================================================
    // New words: */ and */MOD
    // ===================================================================

    #[test]
    fn test_star_slash() {
        // */ ( n1 n2 n3 -- n4 ) = n1*n2/n3
        assert_eq!(eval_stack("10 3 2 */"), vec![15]);
    }

    #[test]
    fn test_star_slash_mod() {
        // */MOD ( n1 n2 n3 -- rem quot )
        assert_eq!(eval_stack("10 3 7 */MOD"), vec![4, 2]);
    }

    // ===================================================================
    // New words: U.
    // ===================================================================

    #[test]
    fn test_u_dot() {
        assert_eq!(eval_output("-1 U."), "4294967295 ");
    }

    // ===================================================================
    // New words: ABORT"
    // ===================================================================

    #[test]
    fn test_abort_quote_no_trigger() {
        // Flag is 0 (false), so ABORT" should NOT trigger
        assert_eq!(eval_output(": TEST 0 ABORT\" oops\" 42 . ; TEST"), "42 ");
    }

    #[test]
    fn test_abort_quote_trigger() {
        // Flag is non-zero (true), so ABORT" should trigger and throw
        let mut vm = ForthVM::new().unwrap();
        let result = vm.evaluate(": TEST -1 ABORT\" oops\" 42 . ; TEST");
        assert!(result.is_err());
    }

    // ===================================================================
    // New words: SOURCE
    // ===================================================================

    #[test]
    fn test_source() {
        // SOURCE should push (c-addr u) of the input buffer
        let stack = eval_stack("SOURCE");
        assert_eq!(stack.len(), 2);
        assert!(stack[0] > 0); // length > 0
    }

    // ===================================================================
    // New words: FIND (basic test via interpret mode)
    // ===================================================================

    #[test]
    fn test_find_exists() {
        // Test FIND with a known word. Create a counted string for "DUP".
        let stack = eval_stack("HERE 3 C, CHAR D C, CHAR U C, CHAR P C, FIND");
        // FIND should return (xt, -1) for a normal word
        assert_eq!(stack.len(), 2);
        assert_eq!(stack[0], -1); // flag: non-immediate
        assert!(stack[1] >= 0); // xt should be a valid word_id
    }

    // ===================================================================
    // New words: >NUMBER (basic test)
    // ===================================================================

    #[test]
    fn test_to_number_basic() {
        // >NUMBER ( ud1 c-addr1 u1 -- ud2 c-addr2 u2 )
        // Convert "123" starting from ud=0
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("S\" 123\"").unwrap(); // push c-addr u
        // Push ud1 = 0 0 underneath
        vm.evaluate("0 0 2SWAP").unwrap(); // stack: 0 0 c-addr u
        // But >NUMBER expects: ud-lo ud-hi c-addr u
        // Actually stack order: u (top), c-addr, ud-hi, ud-lo (bottom)
        vm.evaluate(">NUMBER").unwrap();
        let stack = vm.data_stack();
        // u2 should be 0 (all chars consumed)
        assert_eq!(stack[0], 0);
        // The ud2-lo should be 123
        assert_eq!(stack[3], 123);
    }

    // ===================================================================
    // New words: WORD (basic test)
    // ===================================================================

    #[test]
    fn test_word_basic() {
        // WORD ( char -- c-addr ) parse next word delimited by char
        // After "WORD" we push the delimiter char and call WORD
        // This is tricky to test since WORD reads from the input buffer
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("BL WORD HELLO").unwrap();
        let stack = vm.data_stack();
        assert!(!stack.is_empty());
        // The returned address should be a counted string at PAD
        let addr = stack[0] as u32;
        let data = vm.memory.data(&vm.store);
        let len = data[addr as usize];
        assert_eq!(len, 5); // "HELLO" is 5 chars
    }
}
