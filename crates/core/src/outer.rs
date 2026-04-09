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

use crate::codegen::{CodegenConfig, CompiledModule, compile_consolidated_module, compile_word};
use crate::config::WaferConfig;
use crate::dictionary::{Dictionary, DictionaryState, WordId};
use crate::ir::IrOp;
use crate::memory::{
    CELL_SIZE, DATA_STACK_TOP, FLOAT_SIZE, FLOAT_STACK_BASE, FLOAT_STACK_TOP, INPUT_BUFFER_BASE,
    INPUT_BUFFER_SIZE, RETURN_STACK_TOP, SYSVAR_BASE_VAR, SYSVAR_HERE, SYSVAR_LEAVE_FLAG,
    SYSVAR_NUM_TIB, SYSVAR_STATE, SYSVAR_TO_IN,
};
use crate::optimizer::optimize;

// ---------------------------------------------------------------------------
// Control-flow compilation state
// ---------------------------------------------------------------------------

/// Control-flow entry on the compile-time control stack.
#[derive(Debug, Clone)]
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
    /// Two WHILEs in a single BEGIN loop: BEGIN test1 WHILE test2 WHILE ...
    BeginWhileWhile {
        outer_test: Vec<IrOp>,
        inner_test: Vec<IrOp>,
        body: Vec<IrOp>,
    },
    /// After REPEAT resolves a double-WHILE loop. Holds the completed loop
    /// structure and collects the "`after_repeat`" code. ELSE/THEN close it.
    PostDoubleWhileRepeat {
        outer_test: Vec<IrOp>,
        inner_test: Vec<IrOp>,
        loop_body: Vec<IrOp>,
        prefix: Vec<IrOp>,
    },
    /// After ELSE in a double-WHILE structure. Holds everything and collects
    /// the else body. THEN closes it.
    PostDoubleWhileRepeatElse {
        outer_test: Vec<IrOp>,
        inner_test: Vec<IrOp>,
        loop_body: Vec<IrOp>,
        after_repeat: Vec<IrOp>,
        prefix: Vec<IrOp>,
    },
    /// CASE statement: holds prefix and the list of ENDOF forward branches
    Case {
        prefix: Vec<IrOp>,
        endof_branches: Vec<(Vec<IrOp>, Vec<IrOp>)>, // (of_condition, of_body) pairs
    },
    /// OF statement inside CASE: holds prefix and current partial Case state
    Of {
        prefix: Vec<IrOp>,
        endof_branches: Vec<(Vec<IrOp>, Vec<IrOp>)>,
        of_test: Vec<IrOp>, // code compiled between OF and the CASE's previous state
    },
    /// ?DO: wraps a Do frame with a skip check. When LOOP resolves the Do,
    /// it needs to also close the IF/ELSE wrapping.
    QDo {
        /// The prefix before the ?DO (including the OVER OVER = check)
        prefix: Vec<IrOp>,
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
#[derive(Clone)]
struct DoesDefinition {
    /// The IR for the create-part (code between CREATE and DOES>).
    create_ir: Vec<IrOp>,
    /// The word ID of the compiled does-action (code after DOES>).
    does_action_id: WordId,
    /// Whether the definition included CREATE before DOES>.
    has_create: bool,
}

/// Saved VM state for a MARKER word.
struct MarkerState {
    dict_state: DictionaryState,
    user_here: u32,
    next_table_index: u32,
    word_pfa_map: HashMap<u32, u32>,
    ir_bodies: HashMap<WordId, Vec<IrOp>>,
    does_definitions: HashMap<WordId, DoesDefinition>,
    host_word_names: HashMap<WordId, String>,
    two_value_words: std::collections::HashSet<u32>,
    fvalue_words: std::collections::HashSet<u32>,
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
    fsp: Global,
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
    // Map from WordId to name for host-function words (for export metadata).
    host_word_names: HashMap<WordId, String>,
    // Shared HERE value for host functions (synced with user_here)
    here_cell: Option<Arc<Mutex<u32>>>,
    // User data allocation pointer in WASM linear memory.
    // Variables and user data are allocated here (not in dictionary internal memory).
    user_here: u32,
    // Shared BASE value for host functions
    base_cell: Arc<Mutex<u32>>,
    // DOES> definitions: maps defining word ID to its DoesDefinition
    does_definitions: HashMap<WordId, DoesDefinition>,
    // Last word created by CREATE: (dictionary address, PFA in WASM memory), for DOES> patching
    last_created_info: Option<(u32, u32)>,
    // Map from word_id (xt) to PFA (for >BODY)
    word_pfa_map: HashMap<u32, u32>,
    // Shared copy of word_pfa_map for host function access
    word_pfa_map_shared: Option<Arc<Mutex<HashMap<u32, u32>>>>,
    // True when CREATE appeared in the current colon definition before DOES>
    saw_create_in_def: bool,
    // Pending action from compiled defining/parsing words
    // 0 = none, 1 = CONSTANT, 2 = VARIABLE, 3 = CREATE, 4 = EVALUATE
    pending_define: Arc<Mutex<Vec<i32>>>,
    // Pending word IDs to compile (used by COMPILE, / POSTPONE mechanism)
    pending_compile: Arc<Mutex<Vec<u32>>>,
    // Pending DOES> patch: (does_action_id) to apply after word execution
    pending_does_patch: Arc<Mutex<Option<u32>>>,
    // Exception word set: throw code shared between CATCH and THROW host functions
    throw_code: Arc<Mutex<Option<i32>>>,
    // Shared dictionary lookup: maps uppercase name -> (WordId, is_immediate)
    word_lookup: Arc<Mutex<HashMap<String, (u32, bool)>>>,
    // Set of word_ids that are 2VALUEs (need 2-cell TO semantics)
    two_value_words: std::collections::HashSet<u32>,
    // Set of word_ids that are FVALUEs (need float TO semantics)
    fvalue_words: std::collections::HashSet<u32>,
    // Float I/O precision (default 6)
    float_precision: Arc<Mutex<usize>>,
    /// Stored IR bodies for inlining optimization.
    ir_bodies: HashMap<WordId, Vec<IrOp>>,
    /// Optimization configuration.
    config: WaferConfig,
    /// Total WASM module bytes compiled.
    total_module_bytes: u64,
    /// When true, `register_primitive` defers WASM compilation for batch processing.
    batch_mode: bool,
    /// IR primitives deferred during `batch_mode` for single-module compilation.
    deferred_ir: Vec<(WordId, Vec<IrOp>)>,
    /// Recorded top-level IR from interpretation mode (for `wafer build`).
    toplevel_ir: Vec<IrOp>,
    /// When true, interpretation-mode execution is recorded into `toplevel_ir`.
    recording_toplevel: bool,
    /// Saved states for MARKER words: marker_id -> MarkerState
    marker_states: HashMap<u32, MarkerState>,
    /// Pending MARKER restore: after a marker word executes, restore this state
    pending_marker_restore: Arc<Mutex<Option<u32>>>,
    /// Conditional compilation skip depth: >0 means we're skipping tokens for [IF]/[ELSE]
    conditional_skip_depth: u32,
    /// Local variable names for the current definition ({: ... :} syntax)
    compiling_locals: Vec<String>,
    /// Substitution table for SUBSTITUTE/REPLACES (String word set)
    substitutions: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    /// Search order: list of wordlist IDs (first = top of search order).
    /// Shared via Arc so host functions can modify it directly.
    search_order: Arc<Mutex<Vec<u32>>>,
    /// Next wordlist ID to allocate (shared).
    next_wid: Arc<Mutex<u32>>,
}

impl ForthVM {
    /// Boot a new Forth VM with all primitives registered.
    pub fn new() -> anyhow::Result<Self> {
        Self::new_with_config(WaferConfig::default())
    }

    /// Boot a new Forth VM with custom optimization configuration.
    pub fn new_with_config(wafer_config: WaferConfig) -> anyhow::Result<Self> {
        let mut config = wasmtime::Config::new();
        config.cranelift_nan_canonicalization(false);
        // Best-effort module caching
        let _ = config.cache_config_load_default();
        let engine = Engine::new(&config)?;
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

        // Float stack pointer global
        let fsp = Global::new(
            &mut store,
            wasmtime::GlobalType::new(ValType::I32, Mutability::Var),
            Val::I32(FLOAT_STACK_TOP as i32),
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

        let dictionary = Dictionary::new();

        let mut vm = ForthVM {
            dictionary,
            engine,
            store,
            memory,
            table,
            dsp,
            rsp,
            fsp,
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
            host_word_names: HashMap::new(),
            here_cell: None,
            // User data starts at 64K in WASM memory, well clear of all system regions
            user_here: 0x10000,
            base_cell: Arc::new(Mutex::new(10)),
            does_definitions: HashMap::new(),
            last_created_info: None,
            saw_create_in_def: false,
            word_pfa_map: HashMap::new(),
            word_pfa_map_shared: None,
            pending_define: Arc::new(Mutex::new(Vec::new())),
            pending_compile: Arc::new(Mutex::new(Vec::new())),
            pending_does_patch: Arc::new(Mutex::new(None)),
            throw_code: Arc::new(Mutex::new(None)),
            word_lookup: Arc::new(Mutex::new(HashMap::new())),
            two_value_words: std::collections::HashSet::new(),
            fvalue_words: std::collections::HashSet::new(),
            float_precision: Arc::new(Mutex::new(6)),
            ir_bodies: HashMap::new(),
            config: wafer_config,
            total_module_bytes: 0,
            batch_mode: false,
            deferred_ir: Vec::new(),
            toplevel_ir: Vec::new(),
            recording_toplevel: false,
            marker_states: HashMap::new(),
            pending_marker_restore: Arc::new(Mutex::new(None)),
            conditional_skip_depth: 0,
            compiling_locals: Vec::new(),
            substitutions: Arc::new(Mutex::new(HashMap::new())),
            search_order: Arc::new(Mutex::new(vec![1])),
            next_wid: Arc::new(Mutex::new(2)),
        };

        vm.register_primitives()?;

        Ok(vm)
    }

    /// Evaluate a line of Forth input.
    pub fn evaluate(&mut self, input: &str) -> anyhow::Result<()> {
        self.input_buffer = input.to_string();
        self.input_pos = 0;
        self.sync_input_to_wasm();
        self.sync_here_to_wasm();

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
                    self.compiling_locals.clear();
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
        let mem_len = data.len() as u32;
        let mut stack = Vec::new();
        let mut addr = sp;
        while addr < DATA_STACK_TOP && addr < mem_len {
            let b: [u8; 4] = data[addr as usize..addr as usize + 4].try_into().unwrap();
            stack.push(i32::from_le_bytes(b));
            addr += CELL_SIZE;
        }
        stack
    }

    /// Total WASM module bytes compiled so far.
    pub fn total_module_bytes(&self) -> u64 {
        self.total_module_bytes
    }

    // -----------------------------------------------------------------------
    // Export support: public accessors for `wafer build`
    // -----------------------------------------------------------------------

    /// Enable or disable top-level execution recording.
    ///
    /// When enabled, interpretation-mode word calls and literal pushes are
    /// captured into an IR body that becomes the `_start` entry point in
    /// exported WASM modules.
    pub fn set_recording(&mut self, on: bool) {
        self.recording_toplevel = on;
    }

    /// Return the recorded top-level IR (empty if recording was not enabled).
    pub fn toplevel_ir(&self) -> &[IrOp] {
        &self.toplevel_ir
    }

    /// Snapshot WASM linear memory from byte 0 through `user_here`.
    ///
    /// The returned bytes contain system variables, stack regions, and all
    /// user-allocated data (VARIABLEs, strings, etc.). This becomes the
    /// WASM data section in exported modules.
    pub fn memory_snapshot(&mut self) -> Vec<u8> {
        self.refresh_user_here();
        let data = self.memory.data(&self.store);
        let end = self.user_here as usize;
        data[..end].to_vec()
    }

    /// Return all IR-based word bodies, sorted by `WordId`.
    pub fn ir_words(&self) -> Vec<(WordId, Vec<IrOp>)> {
        let mut words: Vec<(WordId, Vec<IrOp>)> = self
            .ir_bodies
            .iter()
            .map(|(&id, body)| (id, body.clone()))
            .collect();
        words.sort_by_key(|(id, _)| id.0);
        words
    }

    /// Map of host-function `WordId`s to their Forth names.
    pub fn host_function_names(&self) -> &HashMap<WordId, String> {
        &self.host_word_names
    }

    /// Resolve a word name to its `WordId`. Returns `None` if not found.
    pub fn resolve_word(&self, name: &str) -> Option<WordId> {
        self.dictionary
            .find(&name.to_ascii_uppercase())
            .map(|(_, id, _)| id)
    }

    /// Current function table size.
    pub fn current_table_size(&self) -> u32 {
        self.table.size(&self.store) as u32
    }

    /// Initial stack pointer values: (dsp, rsp, fsp).
    pub fn stack_pointer_inits(&self) -> (u32, u32, u32) {
        (DATA_STACK_TOP, RETURN_STACK_TOP, FLOAT_STACK_TOP)
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

        // Conditional compilation skip: when conditional_skip_depth > 0,
        // only process [IF]/[ELSE]/[THEN] for depth tracking, skip everything else.
        if self.conditional_skip_depth > 0 {
            match token_upper.as_str() {
                "[IF]" => self.conditional_skip_depth += 1,
                "[ELSE]" if self.conditional_skip_depth == 1 => {
                    self.conditional_skip_depth = 0;
                }
                "[THEN]" => {
                    self.conditional_skip_depth -= 1;
                }
                _ => {} // All other tokens are parsed and discarded
            }
            return Ok(());
        }

        // Handle colon definition start
        if token_upper == ":" {
            return self.start_colon_def();
        }
        // Handle :NONAME definition
        if token_upper == ":NONAME" {
            return self.start_noname_def();
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
            "]" => {
                // Switch to compile mode (can be used outside a colon definition)
                self.state = -1;
                return Ok(());
            }
            "[IF]" => {
                let flag = self.pop_data_stack()?;
                if flag == 0 {
                    self.conditional_skip_depth = 1;
                }
                return Ok(());
            }
            "[ELSE]" => {
                // We're in the TRUE branch; skip to matching [THEN]
                self.conditional_skip_depth = 1;
                return Ok(());
            }
            "[THEN]" => {
                // No-op — marks end of conditional
                return Ok(());
            }
            "[DEFINED]" => {
                let name = self
                    .next_token()
                    .ok_or_else(|| anyhow::anyhow!("[DEFINED]: expected name"))?;
                let found = self.dictionary.find(&name).is_some();
                self.push_data_stack(if found { -1 } else { 0 })?;
                return Ok(());
            }
            "[UNDEFINED]" => {
                let name = self
                    .next_token()
                    .ok_or_else(|| anyhow::anyhow!("[UNDEFINED]: expected name"))?;
                let found = self.dictionary.find(&name).is_some();
                self.push_data_stack(if found { 0 } else { -1 })?;
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
        if token_upper == "S\\\"" {
            // S\" with escape sequences in interpret mode
            if let Some(raw) = self.parse_s_escape() {
                self.refresh_user_here();
                let addr = self.user_here;
                let len = raw.len() as u32;
                let data = self.memory.data_mut(&mut self.store);
                data[addr as usize..addr as usize + len as usize].copy_from_slice(&raw);
                self.user_here += len;
                self.sync_here_cell();
                self.push_data_stack(addr as i32)?;
                self.push_data_stack(len as i32)?;
            }
            return Ok(());
        }
        if token_upper == "C\"" {
            // C" in interpret mode: store counted string at transient area
            if let Some(s) = self.parse_until('"') {
                self.refresh_user_here();
                let addr = self.user_here;
                let bytes = s.as_bytes();
                let len = bytes.len() as u8;
                let data = self.memory.data_mut(&mut self.store);
                data[addr as usize] = len;
                data[addr as usize + 1..addr as usize + 1 + len as usize].copy_from_slice(bytes);
                self.user_here += 1 + len as u32;
                self.sync_here_cell();
                self.push_data_stack(addr as i32)?;
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
            "VALUE" => return self.define_value(),
            "DOES>" => return self.interpret_does(),
            "'" => return self.interpret_tick(),
            "[CHAR]" => {
                // In interpret mode, CHAR is the standard word
                return self.interpret_char();
            }
            "CHAR" => return self.interpret_char(),
            "EVALUATE" => return self.interpret_evaluate(),
            "WORD" => return self.interpret_word(),
            "TO" => return self.interpret_to(),
            "IS" => return self.interpret_is(),
            "ACTION-OF" => return self.interpret_action_of(),
            "PARSE" => return self.interpret_parse(),
            "PARSE-NAME" => return self.interpret_parse_name(),
            "REFILL" => {
                // In piped/string mode, REFILL returns FALSE
                self.push_data_stack(0)?;
                return Ok(());
            }
            "BUFFER:" => return self.define_buffer(),
            "MARKER" => return self.define_marker(),
            "2CONSTANT" => return self.define_2constant(),
            "2VARIABLE" => return self.define_2variable(),
            "2VALUE" => return self.define_2value(),
            "FVARIABLE" => return self.define_fvariable(),
            "FCONSTANT" => return self.define_fconstant(),
            "FVALUE" => return self.define_fvalue(),
            "CONSOLIDATE" => return self.consolidate(),
            "SYNONYM" => return self.define_synonym(),
            "ORDER" => {
                let so = self.search_order.lock().unwrap();
                let output = format!(
                    "Search order: {:?} Compilation: {}\n",
                    *so,
                    self.dictionary.current_wid()
                );
                self.output.lock().unwrap().push_str(&output);
                return Ok(());
            }
            _ => {}
        }

        // Look up in dictionary
        if let Some((_addr, word_id, _is_immediate)) = self.dictionary.find(token) {
            // Check if this is a DOES>-defining word
            if self.does_definitions.contains_key(&word_id) {
                return self.execute_does_defining(word_id);
            }
            self.execute_word(word_id)?;
            if self.recording_toplevel && self.state == 0 {
                self.toplevel_ir.push(IrOp::Call(word_id));
            }
            return Ok(());
        }

        // Try to parse as double-number (trailing dot)
        if let Some((lo, hi)) = self.parse_double_number(token) {
            self.push_data_stack(lo)?;
            self.push_data_stack(hi)?;
            if self.recording_toplevel && self.state == 0 {
                self.toplevel_ir.push(IrOp::PushI32(lo));
                self.toplevel_ir.push(IrOp::PushI32(hi));
            }
            return Ok(());
        }

        // Try to parse as number
        if let Some(n) = self.parse_number(token) {
            self.push_data_stack(n)?;
            if self.recording_toplevel && self.state == 0 {
                self.toplevel_ir.push(IrOp::PushI32(n));
            }
            return Ok(());
        }

        // Try to parse as float literal (contains 'E' or 'e')
        if let Some(f) = self.parse_float_literal(token) {
            self.fpush(f)?;
            if self.recording_toplevel && self.state == 0 {
                self.toplevel_ir.push(IrOp::PushF64(f));
            }
            return Ok(());
        }

        anyhow::bail!("unknown word: {token}");
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
        if token_upper == "C\"" {
            // C" in compile mode: store counted string at HERE, compile literal
            if let Some(s) = self.parse_until('"') {
                self.refresh_user_here();
                let addr = self.user_here;
                let bytes = s.as_bytes();
                let len = bytes.len() as u8;
                let data = self.memory.data_mut(&mut self.store);
                data[addr as usize] = len;
                data[addr as usize + 1..addr as usize + 1 + len as usize].copy_from_slice(bytes);
                self.user_here += 1 + len as u32;
                self.sync_here_cell();
                self.push_ir(IrOp::PushI32(addr as i32));
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

                // ABORT" throws -2 without displaying the message.
                // The message (addr, len) is saved but not typed here.
                let throw_call = self.dictionary.find("THROW").map(|(_, id, _)| id);
                let mut then_body = vec![IrOp::PushI32(-2)];
                if let Some(throw_id) = throw_call {
                    then_body.push(IrOp::Call(throw_id));
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
            "AGAIN" => return self.compile_again(),
            "WHILE" => return self.compile_while(),
            "REPEAT" => return self.compile_repeat(),
            "?DO" => return self.compile_qdo(),
            "CASE" => return self.compile_case(),
            "OF" => return self.compile_of(),
            "ENDOF" => return self.compile_endof(),
            "ENDCASE" => return self.compile_endcase(),
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
            "2LITERAL" => {
                // compile-time: pop two cells from data stack, compile as literals
                let stack = self.data_stack();
                if stack.len() >= 2 {
                    let hi = self.pop_data_stack()?;
                    let lo = self.pop_data_stack()?;
                    self.push_ir(IrOp::PushI32(lo));
                    self.push_ir(IrOp::PushI32(hi));
                }
                return Ok(());
            }
            "FLITERAL" => {
                // compile-time: pop from float stack, compile as float literal
                let f = self.fpop()?;
                self.compile_float_literal(f)?;
                return Ok(());
            }
            "SLITERAL" => {
                // compile-time: pop (c-addr u) from data stack, copy string,
                // compile code to push the new (c-addr u)
                let stack = self.data_stack();
                if stack.len() >= 2 {
                    let u = self.pop_data_stack()? as u32;
                    let c_addr = self.pop_data_stack()? as u32;
                    // Copy string to a new location in HERE space
                    self.refresh_user_here();
                    let new_addr = self.user_here;
                    let data = self.memory.data(&self.store);
                    let end = (c_addr as usize).saturating_add(u as usize);
                    if end <= data.len() {
                        let bytes: Vec<u8> = data[c_addr as usize..end].to_vec();
                        let data = self.memory.data_mut(&mut self.store);
                        data[new_addr as usize..new_addr as usize + u as usize]
                            .copy_from_slice(&bytes);
                        self.user_here += u;
                        self.sync_here_cell();
                    }
                    self.push_ir(IrOp::PushI32(new_addr as i32));
                    self.push_ir(IrOp::PushI32(u as i32));
                }
                return Ok(());
            }
            "POSTPONE" => {
                // Forth 2012 POSTPONE semantics:
                // - Immediate word: compile a call (so it executes at runtime,
                //   i.e., during compilation of the enclosing definition)
                // - Non-immediate word: compile code that, when executed,
                //   appends Call(word_id) to the current compilation.
                //   This uses COMPILE, to signal the outer interpreter.
                if let Some(next) = self.next_token() {
                    if let Some((_addr, word_id, is_imm)) = self.dictionary.find(&next) {
                        if is_imm {
                            // Immediate: just compile a call to it
                            self.push_ir(IrOp::Call(word_id));
                        } else {
                            // Non-immediate: compile code to push xt and call COMPILE,
                            let compile_comma_id = self
                                .dictionary
                                .find("COMPILE,")
                                .map(|(_, id, _)| id)
                                .ok_or_else(|| anyhow::anyhow!("POSTPONE: COMPILE, not found"))?;
                            self.push_ir(IrOp::PushI32(word_id.0 as i32));
                            self.push_ir(IrOp::Call(compile_comma_id));
                        }
                    } else {
                        anyhow::bail!("POSTPONE: unknown word: {next}");
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
                        anyhow::bail!("['] unknown word: {next}");
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
                self.saw_create_in_def = true;
                return Ok(());
            }
            "VARIABLE" | "CONSTANT" => {
                // These are now in the dictionary as host functions.
                // Fall through to dictionary lookup to compile a call.
            }
            "TO" => {
                return self.compile_to();
            }
            "IS" => {
                return self.compile_is();
            }
            "ACTION-OF" => {
                return self.compile_action_of();
            }
            "S\\\"" => {
                // S\" with escape sequences
                if let Some(raw) = self.parse_s_escape() {
                    self.refresh_user_here();
                    let addr = self.user_here;
                    let len = raw.len() as u32;
                    let data = self.memory.data_mut(&mut self.store);
                    data[addr as usize..addr as usize + len as usize].copy_from_slice(&raw);
                    self.user_here += len;
                    self.sync_here_cell();
                    self.push_ir(IrOp::PushI32(addr as i32));
                    self.push_ir(IrOp::PushI32(len as i32));
                }
                return Ok(());
            }
            "{:" => {
                return self.compile_locals_block();
            }
            _ => {}
        }

        // Check for local variable reference (locals supersede dictionary words)
        if let Some(idx) = self
            .compiling_locals
            .iter()
            .position(|n| n.eq_ignore_ascii_case(token))
        {
            self.push_ir(IrOp::ForthLocalGet(idx as u32));
            return Ok(());
        }

        // Look up in dictionary (search order, then fallback to all wordlists)
        if let Some((_addr, word_id, is_immediate)) = self.dictionary.find(token) {
            if is_immediate {
                // Execute immediately even in compile mode
                self.execute_word(word_id)?;
                // Handle any pending COMPILE, operations from POSTPONE
                self.handle_pending_compile();
            } else {
                self.push_ir(IrOp::Call(word_id));
            }
            return Ok(());
        }

        // Try to parse as double-number (trailing dot)
        if let Some((lo, hi)) = self.parse_double_number(token) {
            self.push_ir(IrOp::PushI32(lo));
            self.push_ir(IrOp::PushI32(hi));
            return Ok(());
        }

        // Try to parse as number
        if let Some(n) = self.parse_number(token) {
            self.push_ir(IrOp::PushI32(n));
            return Ok(());
        }

        // Try to parse as float literal -- compile as FLITERAL
        if let Some(f) = self.parse_float_literal(token) {
            self.compile_float_literal(f)?;
            return Ok(());
        }

        anyhow::bail!("unknown word: {token}");
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
            Some(ControlEntry::IfElse {
                then_body,
                else_body: mut prefix,
            }) => {
                // Multiple ELSE: save the condition flag on the return stack
                // so subsequent IFs can re-test it with R@.
                let first_else = std::mem::take(&mut self.compiling_ir);
                prefix.push(IrOp::ToR); // save flag to return stack
                prefix.push(IrOp::RFetch); // copy for first If test
                prefix.push(IrOp::If {
                    then_body,
                    else_body: Some(first_else),
                });
                // R-stack still holds the flag; push R@ for next If test
                prefix.push(IrOp::RFetch);
                // Push an If entry — the next code will be the "then" body
                // of the next branch pair (e.g., code "3" in IF 1 ELSE 2 ELSE 3 ELSE 4)
                self.control_stack
                    .push(ControlEntry::If { then_body: prefix });
                // compiling_ir is empty, collects the next then-code
            }
            Some(ControlEntry::PostDoubleWhileRepeat {
                outer_test,
                inner_test,
                loop_body,
                prefix,
            }) => {
                // ELSE after REPEAT in double-WHILE: collect after_repeat code
                let after_repeat = std::mem::take(&mut self.compiling_ir);
                self.control_stack
                    .push(ControlEntry::PostDoubleWhileRepeatElse {
                        outer_test,
                        inner_test,
                        loop_body,
                        after_repeat,
                        prefix,
                    });
                // compiling_ir now empty, collects the else body
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
                // Check if this was created by a multi-ELSE desugaring
                // (prefix ends with RFetch which pushed the flag for this If)
                let multi_else = matches!(prefix.last(), Some(IrOp::RFetch));
                self.compiling_ir = prefix;
                self.compiling_ir.push(IrOp::If {
                    then_body,
                    else_body: None,
                });
                if multi_else {
                    self.compiling_ir.push(IrOp::FromR);
                    self.compiling_ir.push(IrOp::Drop);
                }
            }
            Some(ControlEntry::IfElse {
                then_body,
                else_body: prefix,
            }) => {
                // compiling_ir has the else_body ops
                let else_body = std::mem::take(&mut self.compiling_ir);
                // Check if this was created by a multi-ELSE desugaring
                let multi_else = matches!(prefix.last(), Some(IrOp::RFetch));
                self.compiling_ir = prefix;
                self.compiling_ir.push(IrOp::If {
                    then_body,
                    else_body: Some(else_body),
                });
                if multi_else {
                    self.compiling_ir.push(IrOp::FromR);
                    self.compiling_ir.push(IrOp::Drop);
                }
            }
            Some(ControlEntry::PostDoubleWhileRepeat {
                outer_test,
                inner_test,
                loop_body,
                prefix,
            }) => {
                // THEN directly after REPEAT (no ELSE): collect after_repeat
                let after_repeat = std::mem::take(&mut self.compiling_ir);
                self.compiling_ir = prefix;
                self.compiling_ir.push(IrOp::BeginDoubleWhileRepeat {
                    outer_test,
                    inner_test,
                    body: loop_body,
                    after_repeat,
                    else_body: None,
                });
            }
            Some(ControlEntry::PostDoubleWhileRepeatElse {
                outer_test,
                inner_test,
                loop_body,
                after_repeat,
                prefix,
            }) => {
                // THEN after ELSE in double-WHILE: collect else body, emit IR
                let else_body = std::mem::take(&mut self.compiling_ir);
                self.compiling_ir = prefix;
                self.compiling_ir.push(IrOp::BeginDoubleWhileRepeat {
                    outer_test,
                    inner_test,
                    body: loop_body,
                    after_repeat,
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

                // Check if this was a ?DO: resolve the wrapping IF/ELSE too
                if matches!(self.control_stack.last(), Some(ControlEntry::QDo { .. })) {
                    let Some(ControlEntry::QDo { prefix: qdo_prefix }) = self.control_stack.pop()
                    else {
                        unreachable!()
                    };
                    // The do_loop IR is now in compiling_ir.
                    // Build: prefix + IF { 2DROP } ELSE { do_loop } THEN
                    let else_body = std::mem::take(&mut self.compiling_ir);
                    let then_body = vec![IrOp::Drop, IrOp::Drop];
                    self.compiling_ir = qdo_prefix;
                    self.compiling_ir.push(IrOp::If {
                        then_body,
                        else_body: Some(else_body),
                    });
                }
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
            Some(ControlEntry::BeginWhile {
                test: outer_test,
                body: prefix,
            }) => {
                // Second WHILE in the same BEGIN loop
                let inner_test = std::mem::take(&mut self.compiling_ir);
                self.control_stack.push(ControlEntry::BeginWhileWhile {
                    outer_test,
                    inner_test,
                    body: prefix, // stash original prefix
                });
                // compiling_ir now empty, collects the inner loop body
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
            Some(ControlEntry::BeginWhileWhile {
                outer_test,
                inner_test,
                body: prefix,
            }) => {
                // REPEAT in a double-WHILE: closes the inner loop.
                // Code after REPEAT (before ELSE/THEN) still needs to be collected.
                let loop_body = std::mem::take(&mut self.compiling_ir);
                self.control_stack
                    .push(ControlEntry::PostDoubleWhileRepeat {
                        outer_test,
                        inner_test,
                        loop_body,
                        prefix,
                    });
                // compiling_ir is now empty, collects the after_repeat code
            }
            Some(ControlEntry::Begin { body: prefix }) => {
                // BEGIN...REPEAT (no WHILE) — treat as BEGIN...AGAIN (infinite loop)
                let body = std::mem::take(&mut self.compiling_ir);
                self.compiling_ir = prefix;
                self.compiling_ir.push(IrOp::BeginAgain { body });
            }
            _ => anyhow::bail!("REPEAT without matching BEGIN...WHILE"),
        }
        Ok(())
    }

    fn compile_again(&mut self) -> anyhow::Result<()> {
        match self.control_stack.pop() {
            Some(ControlEntry::Begin { body: prefix }) => {
                let body = std::mem::take(&mut self.compiling_ir);
                self.compiling_ir = prefix;
                self.compiling_ir.push(IrOp::BeginAgain { body });
            }
            _ => anyhow::bail!("AGAIN without matching BEGIN"),
        }
        Ok(())
    }

    fn compile_qdo(&mut self) -> anyhow::Result<()> {
        // ?DO is like DO but skips the loop body if limit == index.
        // Emit: OVER OVER = IF 2DROP ELSE <DO body LOOP> THEN
        //
        // We use a QDo control entry to track that LOOP needs to close
        // the IF/ELSE wrapper too.

        // Emit the equality check as part of the current compiling_ir
        self.push_ir(IrOp::Over);
        self.push_ir(IrOp::Over);
        self.push_ir(IrOp::Eq);

        // Save the prefix (including the check)
        let prefix = std::mem::take(&mut self.compiling_ir);

        // Push QDo frame (bottom), then Do frame (top)
        self.control_stack.push(ControlEntry::QDo { prefix });
        self.control_stack.push(ControlEntry::Do {
            body: Vec::new(), // Do's "prefix" is empty since we're inside the else branch
        });
        // compiling_ir is now empty, collecting the loop body

        Ok(())
    }

    fn compile_case(&mut self) -> anyhow::Result<()> {
        let prefix = std::mem::take(&mut self.compiling_ir);
        self.control_stack.push(ControlEntry::Case {
            prefix,
            endof_branches: Vec::new(),
        });
        // compiling_ir now empty, collects default/fallthrough code or the first OF
        Ok(())
    }

    fn compile_of(&mut self) -> anyhow::Result<()> {
        // OF: compile `OVER = IF DROP`
        // The code between CASE (or last ENDOF) and OF is part of the test
        match self.control_stack.pop() {
            Some(ControlEntry::Case {
                prefix,
                endof_branches,
            }) => {
                let of_test = std::mem::take(&mut self.compiling_ir);
                self.control_stack.push(ControlEntry::Of {
                    prefix,
                    endof_branches,
                    of_test,
                });
                // compiling_ir now empty, collects the OF body (code until ENDOF)
            }
            _ => anyhow::bail!("OF without matching CASE"),
        }
        Ok(())
    }

    fn compile_endof(&mut self) -> anyhow::Result<()> {
        match self.control_stack.pop() {
            Some(ControlEntry::Of {
                prefix,
                mut endof_branches,
                of_test,
            }) => {
                let of_body = std::mem::take(&mut self.compiling_ir);
                endof_branches.push((of_test, of_body));
                self.control_stack.push(ControlEntry::Case {
                    prefix,
                    endof_branches,
                });
                // compiling_ir now empty, collects the next OF or default code
            }
            _ => anyhow::bail!("ENDOF without matching OF"),
        }
        Ok(())
    }

    fn compile_endcase(&mut self) -> anyhow::Result<()> {
        // ENDCASE: compile DROP then resolve all branches
        match self.control_stack.pop() {
            Some(ControlEntry::Case {
                prefix,
                endof_branches,
            }) => {
                let default_code = std::mem::take(&mut self.compiling_ir);
                self.compiling_ir = prefix;

                // Build nested IF/ELSE structure:
                // OVER = IF DROP <body1> ELSE OVER = IF DROP <body2> ELSE ... DROP <default> THEN ... THEN
                self.compile_case_ir(&endof_branches, &default_code);
            }
            _ => anyhow::bail!("ENDCASE without matching CASE"),
        }
        Ok(())
    }

    /// Build the nested IR for a CASE statement.
    fn compile_case_ir(&mut self, branches: &[(Vec<IrOp>, Vec<IrOp>)], default_code: &[IrOp]) {
        if branches.is_empty() {
            // Default case: emit default code first, then DROP the selector
            self.compiling_ir.extend(default_code.iter().cloned());
            self.compiling_ir.push(IrOp::Drop);
            return;
        }

        let (ref test_code, ref body) = branches[0];
        let remaining = &branches[1..];

        // Emit test_code (if any -- usually empty for simple CASE n OF patterns)
        self.compiling_ir.extend(test_code.iter().cloned());

        // OVER = IF DROP <body>
        let mut then_body = vec![IrOp::Drop];
        then_body.extend(body.iter().cloned());

        // Build else body recursively
        let mut else_ir = Vec::new();
        let saved = std::mem::take(&mut self.compiling_ir);
        self.compiling_ir = else_ir;
        self.compile_case_ir(remaining, default_code);
        else_ir = std::mem::take(&mut self.compiling_ir);
        self.compiling_ir = saved;

        // Emit: OVER = IF DROP <body> ELSE <rest> THEN
        self.compiling_ir.push(IrOp::Over);
        self.compiling_ir.push(IrOp::Eq);
        self.compiling_ir.push(IrOp::If {
            then_body,
            else_body: Some(else_ir),
        });
    }

    // -----------------------------------------------------------------------
    // Colon definition
    // -----------------------------------------------------------------------

    fn start_noname_def(&mut self) -> anyhow::Result<()> {
        if self.state != 0 {
            anyhow::bail!("nested colon definitions not allowed");
        }

        // Allocate a word ID for the anonymous definition
        let name = format!("_noname_{}_", self.next_table_index);
        let word_id = self
            .dictionary
            .create(&name, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        // Reveal immediately so it gets an xt but isn't findable by name
        // (since the name is internal)
        self.dictionary.reveal();

        self.compiling_name = Some(name);
        self.compiling_word_id = Some(word_id);
        self.compiling_ir.clear();
        self.control_stack.clear();
        self.state = -1;
        self.saw_create_in_def = false;
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);

        // Push the xt onto the data stack (so caller can use it)
        self.push_data_stack(word_id.0 as i32)?;

        Ok(())
    }

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
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        self.compiling_name = Some(name);
        self.compiling_word_id = Some(word_id);
        self.compiling_ir.clear();
        self.control_stack.clear();
        self.state = -1;
        self.saw_create_in_def = false;
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);

        Ok(())
    }

    /// Run all enabled optimization passes on an IR sequence.
    fn optimize_ir(&self, ir: Vec<IrOp>, bodies: &HashMap<WordId, Vec<IrOp>>) -> Vec<IrOp> {
        optimize(ir, &self.config.opt, bodies)
    }

    /// Parse a `{: args | locals -- comment :}` block and compile local initializations.
    fn compile_locals_block(&mut self) -> anyhow::Result<()> {
        let mut args: Vec<String> = Vec::new();
        let mut in_comment = false;
        let mut in_uninit = false;

        loop {
            let tok = self
                .next_token()
                .ok_or_else(|| anyhow::anyhow!("{{: missing :}}"))?;
            let tok_upper = tok.to_ascii_uppercase();
            match tok_upper.as_str() {
                ":}" => break,
                "--" => {
                    in_comment = true;
                }
                "|" => {
                    in_uninit = true;
                }
                _ => {
                    if in_comment {
                        continue; // Skip comment tokens
                    }
                    if in_uninit {
                        // Uninitialized local — just add to the map, no stack pop
                        self.compiling_locals.push(tok_upper);
                    } else {
                        // Stack-initialized arg
                        args.push(tok_upper);
                    }
                }
            }
        }

        // Add args to locals map (they go first)
        let base = self.compiling_locals.len();
        for arg in &args {
            self.compiling_locals.insert(base, arg.clone());
        }
        // Actually, args should be at the start of the locals list
        // with the first arg having the lowest index
        let n_args = args.len();
        let mut new_locals = args;
        // Append any already-added uninit locals
        new_locals.extend(self.compiling_locals.drain(base..));
        self.compiling_locals.splice(base..base, new_locals);

        // Compile: pop args from data stack into locals (in reverse order)
        // The first arg is deepest on the stack, last arg is on top
        for i in (0..n_args).rev() {
            self.push_ir(IrOp::ForthLocalSet((base + i) as u32));
        }

        Ok(())
    }

    fn finish_colon_def(&mut self) -> anyhow::Result<()> {
        if self.state == 0 {
            anyhow::bail!("not in compile mode");
        }
        // Auto-close unclosed IF structures (supports unstructured control flow)
        while let Some(entry) = self.control_stack.last() {
            match entry {
                ControlEntry::If { .. } | ControlEntry::IfElse { .. } => {
                    // Treat as implicit THEN at end of definition
                    self.compile_then()?;
                }
                _ => {
                    anyhow::bail!("unresolved control structure");
                }
            }
        }

        self.compiling_locals.clear();

        let name = self
            .compiling_name
            .take()
            .ok_or_else(|| anyhow::anyhow!("no word being compiled"))?;
        let word_id = self
            .compiling_word_id
            .take()
            .ok_or_else(|| anyhow::anyhow!("no word being compiled"))?;
        let ir = std::mem::take(&mut self.compiling_ir);
        let bodies = self.ir_bodies.clone();
        let ir = self.optimize_ir(ir, &bodies);
        self.ir_bodies.insert(word_id, ir.clone());

        // Compile to WASM
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
        };
        let compiled =
            compile_word(&name, &ir, &config).map_err(|e| anyhow::anyhow!("codegen error: {e}"))?;

        // Instantiate and install in the table
        self.instantiate_and_install(&compiled, word_id)?;

        // Reveal the word
        self.dictionary.reveal();
        // Check if IMMEDIATE was toggled (the word might be immediate)
        let is_immediate = self.dictionary.find(&name).is_some_and(|(_, _, imm)| imm);
        self.sync_word_lookup(&name, word_id, is_immediate);
        self.state = 0;
        // Refresh user_here from the shared cell before syncing back,
        // so that host-function advances (ALLOT, , etc.) are preserved.
        self.refresh_user_here();
        self.sync_here_cell();

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Consolidation
    // -----------------------------------------------------------------------

    /// Recompile all IR-based words into a single WASM module with direct calls.
    ///
    /// After consolidation, `call_indirect` between IR-based words is replaced
    /// with direct `call` instructions, enabling Cranelift to optimize across
    /// word boundaries. Host functions are unaffected and still use indirect
    /// calls.
    fn consolidate(&mut self) -> anyhow::Result<()> {
        // Collect all words with IR bodies
        let mut words: Vec<(WordId, Vec<IrOp>)> = self
            .ir_bodies
            .iter()
            .map(|(&id, body)| (id, body.clone()))
            .collect();
        words.sort_by_key(|(id, _)| id.0);

        if words.is_empty() {
            return Ok(());
        }

        // Build local function map: WordId -> module-internal function index.
        // Imported functions: emit (idx 0). Defined functions start at idx 1.
        let mut local_fn_map = HashMap::new();
        for (i, (word_id, _)) in words.iter().enumerate() {
            local_fn_map.insert(*word_id, (i as u32) + 1);
        }

        let table_size = self.table_size();

        // Compile the consolidated module
        let module_bytes = compile_consolidated_module(&words, &local_fn_map, table_size)
            .map_err(|e| anyhow::anyhow!("consolidation codegen error: {e}"))?;

        // Instantiate
        let module = Module::new(&self.engine, &module_bytes)?;
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

        // Update function table with new exports
        for (i, (word_id, _)) in words.iter().enumerate() {
            let export_name = format!("fn_{i}");
            let func = instance
                .get_func(&mut self.store, &export_name)
                .ok_or_else(|| anyhow::anyhow!("missing export {export_name}"))?;
            self.table
                .set(&mut self.store, word_id.0 as u64, Ref::Func(Some(func)))?;
        }

        Ok(())
    }

    /// Batch-compile all deferred IR primitives into a single WASM module.
    fn batch_compile_deferred(&mut self) -> anyhow::Result<()> {
        let words = std::mem::take(&mut self.deferred_ir);
        if words.is_empty() {
            return Ok(());
        }

        let mut local_fn_map = HashMap::new();
        for (i, (word_id, _)) in words.iter().enumerate() {
            local_fn_map.insert(*word_id, (i as u32) + 1);
        }

        self.ensure_table_size(self.next_table_index)?;
        let table_size = self.table_size();

        let module_bytes = compile_consolidated_module(&words, &local_fn_map, table_size)
            .map_err(|e| anyhow::anyhow!("batch compile error: {e}"))?;

        self.total_module_bytes += module_bytes.len() as u64;
        let module = Module::new(&self.engine, &module_bytes)?;
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

        for (i, (word_id, _)) in words.iter().enumerate() {
            let func = instance
                .get_func(&mut self.store, &format!("fn_{i}"))
                .ok_or_else(|| anyhow::anyhow!("missing batch export fn_{i}"))?;
            self.table
                .set(&mut self.store, word_id.0 as u64, Ref::Func(Some(func)))?;
        }

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
        self.total_module_bytes += compiled.bytes.len() as u64;

        let module = Module::new(&self.engine, &compiled.bytes)?;
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

    /// Execute a word by its `WordId` (calls through the function table).
    fn execute_word(&mut self, word_id: WordId) -> anyhow::Result<()> {
        // Rebuild word lookup so inline FIND host function has latest data
        self.rebuild_word_lookup();

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
        // Handle pending DOES> patch (runtime DOES> from double-DOES> words)
        self.handle_pending_does_patch()?;
        // Handle pending COMPILE, operations (used by [ ... ] sequences)
        self.handle_pending_compile();
        // Handle pending MARKER restore
        self.handle_pending_marker_restore()?;
        // Sync search order from shared state to dictionary
        let so = self.search_order.lock().unwrap().clone();
        self.dictionary.set_search_order(&so);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Data stack operations
    // -----------------------------------------------------------------------

    /// Push a value onto the data stack.
    fn push_data_stack(&mut self, value: i32) -> anyhow::Result<()> {
        let sp = self.dsp.get(&mut self.store).unwrap_i32() as u32;
        let mem_len = self.memory.data(&self.store).len() as u32;
        if sp < CELL_SIZE + crate::memory::DATA_STACK_BASE || sp > mem_len {
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
        let mem_len = self.memory.data(&self.store).len() as u32;
        if sp >= DATA_STACK_TOP || sp > mem_len {
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
    // Float stack operations
    // -----------------------------------------------------------------------

    /// Push a value onto the float stack.
    fn fpush(&mut self, val: f64) -> anyhow::Result<()> {
        let sp = self.fsp.get(&mut self.store).unwrap_i32() as u32;
        let new_sp = sp - FLOAT_SIZE;
        if new_sp < FLOAT_STACK_BASE {
            anyhow::bail!("float stack overflow");
        }
        self.fsp.set(&mut self.store, Val::I32(new_sp as i32))?;
        let mem = self.memory.data_mut(&mut self.store);
        mem[new_sp as usize..new_sp as usize + 8].copy_from_slice(&val.to_le_bytes());
        Ok(())
    }

    /// Pop a value from the float stack.
    fn fpop(&mut self) -> anyhow::Result<f64> {
        let sp = self.fsp.get(&mut self.store).unwrap_i32() as u32;
        if sp >= FLOAT_STACK_TOP {
            anyhow::bail!("float stack underflow");
        }
        let mem = self.memory.data(&self.store);
        let bytes: [u8; 8] = mem[sp as usize..sp as usize + 8].try_into().unwrap();
        self.fsp.set(&mut self.store, Val::I32((sp + 8) as i32))?;
        Ok(f64::from_le_bytes(bytes))
    }

    /// Read the current float stack contents (top-first).
    #[cfg(test)]
    fn float_stack(&mut self) -> Vec<f64> {
        let sp = self.fsp.get(&mut self.store).unwrap_i32() as u32;
        let data = self.memory.data(&self.store);
        let mut stack = Vec::new();
        let mut addr = sp;
        while addr < FLOAT_STACK_TOP {
            let b: [u8; 8] = data[addr as usize..addr as usize + 8].try_into().unwrap();
            stack.push(f64::from_le_bytes(b));
            addr += FLOAT_SIZE;
        }
        stack
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
        } else if rest.len() == 3 && rest.as_bytes()[0] == b'\'' && rest.as_bytes()[2] == b'\'' {
            // Character literal: 'x' → ASCII value of x
            Some(rest.as_bytes()[1] as i64)
        } else {
            i64::from_str_radix(rest, self.base).ok()
        };

        result.map(|n| if negative { -(n as i32) } else { n as i32 })
    }

    /// Try to parse a token as a double-number (token ends with `.`).
    /// Returns (lo, hi) where the double-cell value is (hi << 32) | lo.
    fn parse_double_number(&self, token: &str) -> Option<(i32, i32)> {
        let token = token.trim();
        if token.is_empty() {
            return None;
        }

        // Check for trailing dot (double-number indicator)
        let without_dot = token.strip_suffix('.')?;
        if without_dot.is_empty() {
            return None;
        }

        // Check for negative prefix
        let (negative, rest) = if let Some(stripped) = without_dot.strip_prefix('-') {
            (true, stripped)
        } else {
            (false, without_dot)
        };

        if rest.is_empty() {
            return None;
        }

        // Parse based on prefix -- use i128 to handle the full u64 range
        let result: Option<i128> = if let Some(hex) = rest.strip_prefix('$') {
            i128::from_str_radix(hex, 16).ok()
        } else if let Some(dec) = rest.strip_prefix('#') {
            dec.parse::<i128>().ok()
        } else if let Some(bin) = rest.strip_prefix('%') {
            i128::from_str_radix(bin, 2).ok()
        } else {
            i128::from_str_radix(rest, self.base).ok()
        };

        result.map(|n| {
            let val: i64 = if negative { -(n as i64) } else { n as i64 };
            let lo = val as i32;
            let hi = (val >> 32) as i32;
            (lo, hi)
        })
    }

    // -----------------------------------------------------------------------
    // Float literal parsing
    // -----------------------------------------------------------------------

    /// Try to parse a token as a floating-point literal (Forth 2012 format).
    /// Forth float literals contain 'E' or 'e', e.g. `1E`, `1.5E0`, `-3.14E2`, `1E-3`.
    #[allow(clippy::unused_self)]
    fn parse_float_literal(&self, token: &str) -> Option<f64> {
        if token.is_empty() {
            return None;
        }
        let upper = token.to_ascii_uppercase();
        // Must contain 'E' or 'D' (Forth sometimes uses D for double-float exponent)
        if !upper.contains('E') && !upper.contains('D') {
            return None;
        }
        // Replace D with E for Rust parsing
        let normalized = upper.replace('D', "E");
        // Forth allows trailing E without exponent: "1E" means "1E0"
        // Also "1E+" or "1E-" mean "1E+0" and "1E-0"
        let s = if normalized.ends_with('E')
            || normalized.ends_with("E+")
            || normalized.ends_with("E-")
        {
            format!("{normalized}0")
        } else {
            normalized
        };
        s.parse::<f64>().ok()
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
        let bodies = self.ir_bodies.clone();
        let ir_body = self.optimize_ir(ir_body, &bodies);
        let word_id = self
            .dictionary
            .create(name, immediate)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        self.ir_bodies.insert(word_id, ir_body.clone());
        self.dictionary.reveal();
        self.sync_word_lookup(name, word_id, immediate);
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);

        if self.batch_mode {
            // Defer WASM compilation for batch processing
            self.deferred_ir.push((word_id, ir_body));
        } else {
            let config = CodegenConfig {
                base_fn_index: word_id.0,
                table_size: self.table_size(),
                stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
            };
            let compiled = compile_word(name, &ir_body, &config)
                .map_err(|e| anyhow::anyhow!("codegen error for {name}: {e}"))?;
            self.instantiate_and_install(&compiled, word_id)?;
        }

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
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        self.ensure_table_size(word_id.0)?;
        self.table
            .set(&mut self.store, word_id.0 as u64, Ref::Func(Some(func)))?;
        self.dictionary.reveal();
        self.sync_word_lookup(name, word_id, immediate);
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);
        self.host_word_names
            .insert(word_id, name.to_ascii_uppercase());

        Ok(word_id)
    }

    /// Register all built-in primitive words.
    fn register_primitives(&mut self) -> anyhow::Result<()> {
        self.batch_mode = true;

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
        // J -- outer loop counter
        self.register_primitive("J", false, vec![IrOp::LoopJ])?;
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
        // HERE: defined in boot.fth (reads SYSVAR_HERE from WASM memory).
        // Initialize the here_cell for host functions that still need it.
        self.here_cell = Some(Arc::new(Mutex::new(self.user_here)));
        // ALLOT, comma, C-comma: defined in boot.fth
        self.register_primitive("CELLS", false, vec![IrOp::PushI32(4), IrOp::Mul])?;
        self.register_primitive("CELL+", false, vec![IrOp::PushI32(4), IrOp::Add])?;
        // CHARS is a no-op (byte addressed)
        self.register_primitive("CHARS", false, vec![])?;
        self.register_primitive("CHAR+", false, vec![IrOp::PushI32(1), IrOp::Add])?;
        // ALIGN: defined in boot.fth
        self.register_aligned()?;
        // MOVE, FILL: defined in boot.fth

        // -- Priority 4: Stack/arithmetic --
        self.register_primitive("2DUP", false, vec![IrOp::Over, IrOp::Over])?;
        self.register_primitive("2DROP", false, vec![IrOp::Drop, IrOp::Drop])?;
        self.register_primitive(
            "2SWAP",
            false,
            vec![IrOp::Rot, IrOp::ToR, IrOp::Rot, IrOp::FromR],
        )?;
        // 2OVER: defined in boot.fth
        // PICK: defined in boot.fth
        self.register_roll()?;
        self.register_qdup()?;
        // PICK: defined in boot.fth (uses SP@ IR op)
        self.register_min()?;
        self.register_max()?;
        // WITHIN: defined in boot.fth

        // -- Priority 5: Comparison --
        self.register_primitive("0<>", false, vec![IrOp::ZeroEq, IrOp::ZeroEq])?;
        self.register_primitive("0>", false, vec![IrOp::PushI32(0), IrOp::Gt])?;

        // -- Priority 6: System/compiler --
        self.register_primitive("EXECUTE", false, vec![IrOp::Execute])?;
        self.register_primitive("SP@", false, vec![IrOp::SpFetch])?;
        self.register_immediate_word()?;
        self.register_decimal()?;
        self.register_hex()?;
        // TYPE, SPACES: defined in boot.fth
        self.register_tick()?;
        self.register_to_body()?;
        self.register_environment_q()?;
        // SOURCE: defined in boot.fth
        self.register_abort()?;

        // . (dot): defined in boot.fth
        self.register_dot_s()?;
        // DEPTH: defined in boot.fth (uses SP@ IR op)

        // -- Priority 7: New core words --
        self.register_count()?;
        self.register_s_to_d()?;
        // CMOVE, CMOVE>: defined in boot.fth
        self.register_find()?;
        self.register_to_in()?;
        self.register_state_var()?;
        self.register_base_var()?;

        // Double-cell arithmetic
        self.register_m_star()?;
        self.register_um_star()?;
        self.register_um_div_mod()?;
        // FM/MOD, SM/REM, */, */MOD: defined in boot.fth

        // U. (unsigned dot)
        // U.: defined in boot.fth

        // >NUMBER
        self.register_to_number()?;

        // \ (backslash comment) as an immediate word so POSTPONE can find it
        self.register_backslash()?;

        // COMPILE, (compile-comma) for POSTPONE mechanism
        self.register_compile_comma()?;

        // Runtime DOES> patch for double-DOES> support
        self.register_does_patch()?;

        // CONSTANT, VARIABLE, CREATE as callable words (for use inside colon defs)
        self.register_defining_words()?;

        // EVALUATE and WORD as callable words (for use inside colon defs)
        self.register_evaluate_word()?;
        self.register_word_word()?;

        // MARKER restore host function
        self.register_marker_restore()?;

        // 2@, 2!: defined in boot.fth

        // Pictured numeric output
        // Pictured numeric output (<# # #S #> HOLD SIGN): defined in boot.fth

        // Exception word set: CATCH and THROW
        self.register_catch_throw()?;

        // SOURCE-ID ( -- 0 ) always 0 for user input
        self.register_primitive(
            "SOURCE-ID",
            false,
            vec![
                IrOp::PushI32(crate::memory::SYSVAR_SOURCE_ID as i32),
                IrOp::Fetch,
            ],
        )?;

        // -- Core Extension words --
        // 2>R, 2R>, 2R@
        self.register_primitive("2>R", false, vec![IrOp::Swap, IrOp::ToR, IrOp::ToR])?;
        self.register_primitive("2R>", false, vec![IrOp::FromR, IrOp::FromR, IrOp::Swap])?;
        self.register_2r_fetch()?;

        // U>
        self.register_primitive("U>", false, vec![IrOp::Swap, IrOp::LtUnsigned])?;

        // PAD
        self.register_primitive(
            "PAD",
            false,
            vec![IrOp::PushI32(crate::memory::PAD_BASE as i32)],
        )?;

        // ERASE: defined in boot.fth

        // .R and U.R
        // .R, U.R: defined in boot.fth

        // UNUSED
        self.register_unused()?;

        // UTIME ( -- ud ) microseconds since epoch as double-cell
        self.register_utime()?;

        // HOLDS
        // HOLDS: defined in boot.fth

        // PARSE as a host function (for compiled code)
        self.register_parse_host()?;

        // PARSE-NAME as a host function (for compiled code)
        self.register_parse_name_host()?;

        // REFILL as a host function (always returns FALSE in piped mode)
        self.register_refill()?;

        // Memory-Allocation word set
        self.register_memory_alloc()?;

        // S\" (string with escape sequences)
        // Handled as a special token in compile_token/interpret_token

        // BUFFER: ( u "name" -- ) like CREATE + ALLOT
        // Handled as a special token in interpret_token_immediate

        // MARKER -- stub
        // Handled as a special token in interpret_token_immediate

        // DEFER!, DEFER@ (standard aliases)
        // DEFER!, DEFER@: defined in boot.fth

        // FALSE and TRUE are already registered in core
        // NIP, TUCK already registered
        // 0<>, 0>, <> already registered
        // HEX already registered
        // .( already handled
        // \ already registered

        // -- Double-Number word set --
        // D+, D-, DNEGATE, DABS, D0=, D0<, D=, D<, D2*, D2/,
        // DMAX, DMIN, M+, DU<, 2ROT: defined in boot.fth
        self.register_d_to_s()?;
        self.register_m_star_slash()?;
        // D., D.R: defined in boot.fth

        // -- String word set --
        // COMPARE: defined in boot.fth
        self.register_search()?;
        // /STRING, BLANK, -TRAILING: defined in boot.fth
        self.register_string_substitution()?;

        // -- Programming-Tools word set --
        self.register_n_to_r()?;

        // -- Search-Order word set --
        self.register_search_order()?;

        // -- Floating-Point word set --
        self.register_float_words()?;

        // Batch-compile all deferred IR primitives into a single WASM module
        self.batch_mode = false;
        self.batch_compile_deferred()?;

        // Load Forth bootstrap definitions (replaces many host functions).
        // Evaluate line-by-line so `\` comments work correctly.
        let boot = include_str!("../boot.fth");
        for line in boot.lines() {
            self.evaluate(line)?;
        }

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
                let mut out = output.lock().unwrap();
                if sp >= DATA_STACK_TOP {
                    out.push_str("<0> ");
                    return Ok(());
                }
                let data = memory.data(&caller);
                let depth = (DATA_STACK_TOP - sp) / CELL_SIZE;
                out.push_str(&format!("<{depth}> "));
                // Print from bottom to top
                let mut addr = DATA_STACK_TOP - CELL_SIZE;
                while addr >= sp {
                    let b: [u8; 4] = data[addr as usize..addr as usize + 4].try_into().unwrap();
                    let v = i32::from_le_bytes(b);
                    out.push_str(&format!("{v} "));
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

    // -----------------------------------------------------------------------
    // Priority 1: Loop support host functions
    // -----------------------------------------------------------------------

    /// Register LEAVE as a host function.
    /// Sets the loop index equal to the limit and sets the leave flag
    /// so the loop exits on the next +LOOP/LOOP check.
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
                // Set leave flag so +LOOP exits even with step=0
                data[SYSVAR_LEAVE_FLAG as usize..SYSVAR_LEAVE_FLAG as usize + 4]
                    .copy_from_slice(&1i32.to_le_bytes());
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
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Allocate one cell in WASM memory for the variable's storage
        self.refresh_user_here();
        let var_addr = self.user_here;
        self.user_here += CELL_SIZE;

        // Initialize the cell to 0 in WASM memory
        let data = self.memory.data_mut(&mut self.store);
        data[var_addr as usize..var_addr as usize + 4].copy_from_slice(&0i32.to_le_bytes());

        // Compile a tiny word that pushes the variable's address
        let ir_body = vec![IrOp::PushI32(var_addr as i32)];
        self.ir_bodies.insert(word_id, ir_body.clone());
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
        };
        let compiled = compile_word(&name, &ir_body, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for VARIABLE {name}: {e}"))?;

        self.instantiate_and_install(&compiled, word_id)?;
        self.dictionary.reveal();
        self.sync_word_lookup(&name, word_id, false);
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
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Compile a word that pushes the constant value
        let ir_body = vec![IrOp::PushI32(value)];
        self.ir_bodies.insert(word_id, ir_body.clone());
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
        };
        let compiled = compile_word(&name, &ir_body, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for CONSTANT {name}: {e}"))?;

        self.instantiate_and_install(&compiled, word_id)?;
        self.dictionary.reveal();
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);
        // Refresh before sync to preserve host-function-side changes (C,, ALLOT, etc.)
        self.refresh_user_here();
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
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // The parameter field address is the current user_here
        self.refresh_user_here();
        let pfa = self.user_here;

        // Compile a word that pushes the pfa
        let ir_body = vec![IrOp::PushI32(pfa as i32)];
        self.ir_bodies.insert(word_id, ir_body.clone());
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
        };
        let compiled = compile_word(&name, &ir_body, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for CREATE {name}: {e}"))?;

        self.instantiate_and_install(&compiled, word_id)?;
        self.dictionary.reveal();
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);
        // Store fn_index at 0x30 for DOES> to find
        self.store_latest_fn_index(word_id);
        // Track for DOES> patching (used when DOES> has no CREATE)
        self.last_created_info = Some((self.dictionary.latest(), pfa));
        // Map xt -> PFA for >BODY
        self.word_pfa_map.insert(word_id.0, pfa);
        self.sync_pfa_map(word_id.0, pfa);
        self.sync_here_cell();

        Ok(())
    }

    /// VALUE <name> -- ( x -- ) create a value that pushes x when invoked.
    fn define_value(&mut self) -> anyhow::Result<()> {
        let value = self.pop_data_stack()?;
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("VALUE: expected name"))?;

        let word_id = self
            .dictionary
            .create(&name, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Allocate one cell in WASM memory for the value's storage
        self.refresh_user_here();
        let val_addr = self.user_here;
        self.user_here += CELL_SIZE;

        // Initialize the cell with the given value
        let data = self.memory.data_mut(&mut self.store);
        data[val_addr as usize..val_addr as usize + 4].copy_from_slice(&value.to_le_bytes());

        // Compile a word that fetches from the value's address
        let ir_body = vec![IrOp::PushI32(val_addr as i32), IrOp::Fetch];
        self.ir_bodies.insert(word_id, ir_body.clone());
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
        };
        let compiled = compile_word(&name, &ir_body, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for VALUE {name}: {e}"))?;

        self.instantiate_and_install(&compiled, word_id)?;
        self.dictionary.reveal();
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);
        // Map xt -> PFA for TO and >BODY
        self.word_pfa_map.insert(word_id.0, val_addr);
        self.sync_pfa_map(word_id.0, val_addr);
        self.sync_here_cell();

        Ok(())
    }

    /// DEFER <name> -- create a deferred execution word.
    fn define_defer(&mut self) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("DEFER: expected name"))?;

        let word_id = self
            .dictionary
            .create(&name, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Allocate one cell to hold the xt
        self.refresh_user_here();
        let defer_addr = self.user_here;
        self.user_here += CELL_SIZE;

        // Default: find ABORT and use its xt, or use 0
        let default_xt = self.dictionary.find("ABORT").map_or(0, |(_, id, _)| id.0);

        let data = self.memory.data_mut(&mut self.store);
        data[defer_addr as usize..defer_addr as usize + 4]
            .copy_from_slice(&default_xt.to_le_bytes());

        // Compile a word that fetches the xt and executes it
        let ir_body = vec![IrOp::PushI32(defer_addr as i32), IrOp::Fetch, IrOp::Execute];
        self.ir_bodies.insert(word_id, ir_body.clone());
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
        };
        let compiled = compile_word(&name, &ir_body, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for DEFER {name}: {e}"))?;

        self.instantiate_and_install(&compiled, word_id)?;
        self.dictionary.reveal();
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);
        // Map xt -> PFA for IS and ACTION-OF
        self.word_pfa_map.insert(word_id.0, defer_addr);
        self.sync_pfa_map(word_id.0, defer_addr);
        self.sync_here_cell();

        Ok(())
    }

    /// SYNONYM ( "newname" "oldname" -- ) create an alias.
    fn define_synonym(&mut self) -> anyhow::Result<()> {
        let new_name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("SYNONYM: expected newname"))?;
        let old_name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("SYNONYM: expected oldname"))?;

        if let Some((_addr, word_id, is_imm)) = self.dictionary.find(&old_name) {
            // Create a new word that calls the old one
            let new_word_id = self
                .dictionary
                .create(&new_name, is_imm)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            let ir_body = vec![IrOp::Call(word_id)];
            self.ir_bodies.insert(new_word_id, ir_body.clone());
            let config = CodegenConfig {
                base_fn_index: new_word_id.0,
                table_size: self.table_size(),
                stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
            };
            let compiled = compile_word(&new_name, &ir_body, &config)
                .map_err(|e| anyhow::anyhow!("codegen error for SYNONYM: {e}"))?;
            self.instantiate_and_install(&compiled, new_word_id)?;
            self.dictionary.reveal();
            self.next_table_index = self.next_table_index.max(new_word_id.0 + 1);
        } else {
            anyhow::bail!("SYNONYM: unknown word: {old_name}");
        }
        Ok(())
    }

    /// IMMEDIATE -- toggle the immediate flag on the most recently defined word.
    /// Called via pending_define when IMMEDIATE is executed from compiled code.
    fn set_immediate(&mut self) -> anyhow::Result<()> {
        self.dictionary
            .set_immediate()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let latest = self.dictionary.latest();
        if let Ok(name) = self.dictionary.word_name(latest)
            && let Some((_, word_id, is_imm)) = self.dictionary.find(&name)
        {
            self.sync_word_lookup(&name, word_id, is_imm);
        }
        Ok(())
    }

    /// BUFFER: ( u "name" -- ) create a named buffer of u bytes.
    fn define_buffer(&mut self) -> anyhow::Result<()> {
        let size = self.pop_data_stack()? as u32;
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("BUFFER:: expected name"))?;

        let word_id = self
            .dictionary
            .create(&name, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Allocate the buffer in WASM memory (aligned to cell boundary)
        self.refresh_user_here();
        self.user_here = (self.user_here + 3) & !3; // ALIGN
        let buf_addr = self.user_here;
        self.user_here += size;

        // Compile a word that pushes the buffer address
        let ir_body = vec![IrOp::PushI32(buf_addr as i32)];
        self.ir_bodies.insert(word_id, ir_body.clone());
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
        };
        let compiled = compile_word(&name, &ir_body, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for BUFFER: {name}: {e}"))?;

        self.instantiate_and_install(&compiled, word_id)?;
        self.dictionary.reveal();
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);
        self.word_pfa_map.insert(word_id.0, buf_addr);
        self.sync_pfa_map(word_id.0, buf_addr);
        self.sync_here_cell();

        Ok(())
    }

    /// MARKER <name> -- create a marker that restores dictionary state.
    /// Saves a snapshot of the VM; when the marker word is executed, restores it.
    fn define_marker(&mut self) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("MARKER: expected name"))?;

        // Save state BEFORE creating the marker word itself
        let saved = MarkerState {
            dict_state: self.dictionary.save_state(),
            user_here: self.user_here,
            next_table_index: self.next_table_index,
            word_pfa_map: self.word_pfa_map.clone(),
            ir_bodies: self.ir_bodies.clone(),
            does_definitions: self.does_definitions.clone(),
            host_word_names: self.host_word_names.clone(),
            two_value_words: self.two_value_words.clone(),
            fvalue_words: self.fvalue_words.clone(),
        };

        let word_id = self
            .dictionary
            .create(&name, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Store the saved state keyed by word_id
        self.marker_states.insert(word_id.0, saved);

        // Compile the marker word: push marker_id, call _MARKER_RESTORE_
        let restore_id = self
            .dictionary
            .find("_MARKER_RESTORE_")
            .map(|(_, id, _)| id)
            .ok_or_else(|| anyhow::anyhow!("_MARKER_RESTORE_ not found"))?;
        let ir_body = vec![IrOp::PushI32(word_id.0 as i32), IrOp::Call(restore_id)];
        self.ir_bodies.insert(word_id, ir_body.clone());
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
        };
        let compiled = compile_word(&name, &ir_body, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for MARKER {name}: {e}"))?;

        self.instantiate_and_install(&compiled, word_id)?;
        self.dictionary.reveal();
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);

        Ok(())
    }

    /// Register `_MARKER_RESTORE_` host function.
    /// ( marker_id -- ) Signals the outer interpreter to restore state.
    fn register_marker_restore(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let pending = Arc::clone(&self.pending_marker_restore);

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Pop marker_id from data stack
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let marker_id = u32::from_le_bytes(b);
                let new_sp = sp + 4;
                dsp.set(&mut caller, Val::I32(new_sp as i32)).unwrap();
                *pending.lock().unwrap() = Some(marker_id);
                Ok(())
            },
        );

        self.register_host_primitive("_MARKER_RESTORE_", false, func)?;
        Ok(())
    }

    /// TO <name> -- ( x -- ) store x into the value named by <name>.
    fn interpret_to(&mut self) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("TO: expected name"))?;

        if let Some((_addr, word_id, _imm)) = self.dictionary.find(&name) {
            if let Some(&pfa) = self.word_pfa_map.get(&word_id.0) {
                if self.fvalue_words.contains(&word_id.0) {
                    // FVALUE: pop from float stack, store 8 bytes
                    let value = self.fpop()?;
                    let data = self.memory.data_mut(&mut self.store);
                    data[pfa as usize..pfa as usize + 8].copy_from_slice(&value.to_le_bytes());
                } else if self.two_value_words.contains(&word_id.0) {
                    // 2VALUE: pop two cells
                    let hi = self.pop_data_stack()?;
                    let lo = self.pop_data_stack()?;
                    let data = self.memory.data_mut(&mut self.store);
                    data[pfa as usize..pfa as usize + 4].copy_from_slice(&lo.to_le_bytes());
                    data[pfa as usize + 4..pfa as usize + 8].copy_from_slice(&hi.to_le_bytes());
                } else {
                    let value = self.pop_data_stack()?;
                    let data = self.memory.data_mut(&mut self.store);
                    data[pfa as usize..pfa as usize + 4].copy_from_slice(&value.to_le_bytes());
                }
            } else {
                anyhow::bail!("TO: {name} has no parameter field");
            }
        } else {
            anyhow::bail!("TO: unknown word: {name}");
        }
        Ok(())
    }

    /// IS <name> -- ( xt -- ) set the deferred word to xt.
    fn interpret_is(&mut self) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("IS: expected name"))?;
        let xt = self.pop_data_stack()?;

        if let Some((_addr, word_id, _imm)) = self.dictionary.find(&name) {
            if let Some(&pfa) = self.word_pfa_map.get(&word_id.0) {
                let data = self.memory.data_mut(&mut self.store);
                data[pfa as usize..pfa as usize + 4].copy_from_slice(&xt.to_le_bytes());
            } else {
                anyhow::bail!("IS: {name} has no parameter field");
            }
        } else {
            anyhow::bail!("IS: unknown word: {name}");
        }
        Ok(())
    }

    /// ACTION-OF <name> -- ( -- xt ) retrieve the xt from a deferred word.
    fn interpret_action_of(&mut self) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("ACTION-OF: expected name"))?;

        if let Some((_addr, word_id, _imm)) = self.dictionary.find(&name) {
            if let Some(&pfa) = self.word_pfa_map.get(&word_id.0) {
                let data = self.memory.data(&self.store);
                let b: [u8; 4] = data[pfa as usize..pfa as usize + 4].try_into().unwrap();
                let xt = i32::from_le_bytes(b);
                self.push_data_stack(xt)?;
            } else {
                anyhow::bail!("ACTION-OF: {name} has no parameter field");
            }
        } else {
            anyhow::bail!("ACTION-OF: unknown word: {name}");
        }
        Ok(())
    }

    /// TO in compile mode: read next word, find its PFA, compile a store.
    fn compile_to(&mut self) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("TO: expected name"))?;

        // Check if target is a local variable
        if let Some(idx) = self
            .compiling_locals
            .iter()
            .position(|n| n.eq_ignore_ascii_case(&name))
        {
            self.push_ir(IrOp::ForthLocalSet(idx as u32));
            return Ok(());
        }

        if let Some((_addr, word_id, _imm)) = self.dictionary.find(&name) {
            if let Some(&pfa) = self.word_pfa_map.get(&word_id.0) {
                if self.fvalue_words.contains(&word_id.0) {
                    // FVALUE: compile a call to a host function that pops
                    // from the float stack and stores at pfa
                    let store_word = self.make_fvalue_store(pfa)?;
                    self.push_ir(IrOp::Call(store_word));
                } else if self.two_value_words.contains(&word_id.0) {
                    // 2VALUE: ( x1 x2 -- ) store two cells
                    self.push_ir(IrOp::PushI32((pfa + 4) as i32));
                    self.push_ir(IrOp::Store); // stores x2 at pfa+4
                    self.push_ir(IrOp::PushI32(pfa as i32));
                    self.push_ir(IrOp::Store); // stores x1 at pfa
                } else {
                    self.push_ir(IrOp::PushI32(pfa as i32));
                    self.push_ir(IrOp::Store);
                }
            } else {
                anyhow::bail!("TO: {name} has no parameter field");
            }
        } else {
            anyhow::bail!("TO: unknown word: {name}");
        }
        Ok(())
    }

    /// IS in compile mode: read next word, find its PFA, compile a store.
    fn compile_is(&mut self) -> anyhow::Result<()> {
        // IS is the same as TO for DEFER words
        self.compile_to()
    }

    /// ACTION-OF in compile mode: read next word, compile fetch from PFA.
    fn compile_action_of(&mut self) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("ACTION-OF: expected name"))?;

        if let Some((_addr, word_id, _imm)) = self.dictionary.find(&name) {
            if let Some(&pfa) = self.word_pfa_map.get(&word_id.0) {
                self.push_ir(IrOp::PushI32(pfa as i32));
                self.push_ir(IrOp::Fetch);
            } else {
                anyhow::bail!("ACTION-OF: {name} has no parameter field");
            }
        } else {
            anyhow::bail!("ACTION-OF: unknown word: {name}");
        }
        Ok(())
    }

    /// PARSE ( char "text" -- c-addr u ) parse input delimited by char.
    fn interpret_parse(&mut self) -> anyhow::Result<()> {
        let delim = self.pop_data_stack()? as u8 as char;

        let bytes = self.input_buffer.as_bytes();
        // Skip one leading space (the delimiter between the parsed word and its argument)
        if self.input_pos < bytes.len() && bytes[self.input_pos] == b' ' {
            self.input_pos += 1;
        }
        let start = self.input_pos;
        while self.input_pos < bytes.len() && bytes[self.input_pos] != delim as u8 {
            self.input_pos += 1;
        }
        let end = self.input_pos;
        // Skip past delimiter
        if self.input_pos < bytes.len() {
            self.input_pos += 1;
        }

        // Store the parsed text in WASM memory at PAD area
        let text = &bytes[start..end];
        let text_len = text.len() as u32;
        let buf_addr = INPUT_BUFFER_BASE + start as u32;

        self.push_data_stack(buf_addr as i32)?;
        self.push_data_stack(text_len as i32)?;
        Ok(())
    }

    /// PARSE-NAME ( "name" -- c-addr u ) parse next whitespace-delimited name.
    fn interpret_parse_name(&mut self) -> anyhow::Result<()> {
        let bytes = self.input_buffer.as_bytes();
        // Skip leading whitespace
        while self.input_pos < bytes.len() && bytes[self.input_pos].is_ascii_whitespace() {
            self.input_pos += 1;
        }
        let start = self.input_pos;
        while self.input_pos < bytes.len() && !bytes[self.input_pos].is_ascii_whitespace() {
            self.input_pos += 1;
        }
        let end = self.input_pos;

        let buf_addr = INPUT_BUFFER_BASE + start as u32;
        let text_len = (end - start) as u32;

        self.push_data_stack(buf_addr as i32)?;
        self.push_data_stack(text_len as i32)?;
        Ok(())
    }

    /// Parse a string with escape sequences for S\".
    fn parse_s_escape(&mut self) -> Option<Vec<u8>> {
        let bytes = self.input_buffer.as_bytes();
        // Skip one leading space if present
        if self.input_pos < bytes.len() && bytes[self.input_pos] == b' ' {
            self.input_pos += 1;
        }
        let mut result = Vec::new();
        while self.input_pos < bytes.len() && bytes[self.input_pos] != b'"' {
            if bytes[self.input_pos] == b'\\' {
                self.input_pos += 1;
                if self.input_pos < bytes.len() {
                    let ch = bytes[self.input_pos];
                    match ch {
                        b'a' => result.push(7),  // BEL
                        b'b' => result.push(8),  // BS
                        b'e' => result.push(27), // ESC
                        b'f' => result.push(12), // FF
                        b'l' => result.push(10), // LF
                        b'm' => {
                            result.push(13);
                            result.push(10);
                        } // CR/LF
                        b'n' => result.push(10), // newline
                        b'q' => result.push(b'"'), // quote
                        b'r' => result.push(13), // CR
                        b't' => result.push(9),  // TAB
                        b'v' => result.push(11), // VT
                        b'z' => result.push(0),  // NUL
                        b'\\' => result.push(b'\\'),
                        b'"' => result.push(b'"'),
                        b'x' | b'X' => {
                            // Hex escape: \xNN
                            self.input_pos += 1;
                            let mut hex_val = 0u8;
                            for _ in 0..2 {
                                if self.input_pos < bytes.len() {
                                    if let Some(d) = (bytes[self.input_pos] as char).to_digit(16) {
                                        hex_val = hex_val * 16 + d as u8;
                                        self.input_pos += 1;
                                    } else {
                                        break;
                                    }
                                }
                            }
                            result.push(hex_val);
                            continue; // already advanced past the hex digits
                        }
                        _ => result.push(ch),
                    }
                }
            } else {
                result.push(bytes[self.input_pos]);
            }
            self.input_pos += 1;
        }
        // Skip past closing quote
        if self.input_pos < bytes.len() {
            self.input_pos += 1;
        }
        Some(result)
    }

    // -----------------------------------------------------------------------
    // Priority 3: Memory/system host functions
    // -----------------------------------------------------------------------

    /// Keep the `here_cell` and WASM `memory[SYSVAR_HERE]` in sync with `user_here`.
    fn sync_here_cell(&mut self) {
        if let Some(ref cell) = self.here_cell {
            *cell.lock().unwrap() = self.user_here;
        }
        self.sync_here_to_wasm();
    }

    /// Sync a new `word_pfa_map` entry to the shared copy (for >BODY host function).
    fn sync_pfa_map(&self, word_id: u32, pfa: u32) {
        if let Some(ref shared) = self.word_pfa_map_shared {
            shared.lock().unwrap().insert(word_id, pfa);
        }
    }

    /// Update `user_here` from the shared cell and WASM memory.
    ///
    /// Reads both `here_cell` (modified by Rust host functions) and
    /// `memory[SYSVAR_HERE]` (modified by Forth ALLOT/`,`/`C,`/ALIGN).
    /// Takes the maximum to ensure no allocation is lost.
    fn refresh_user_here(&mut self) {
        if let Some(ref cell) = self.here_cell {
            self.user_here = *cell.lock().unwrap();
        }
        let data = self.memory.data(&self.store);
        let mem_len = data.len() as u32;
        let mem_here = u32::from_le_bytes(
            data[SYSVAR_HERE as usize..SYSVAR_HERE as usize + 4]
                .try_into()
                .unwrap(),
        );
        // Only accept mem_here if it's within valid memory bounds.
        // A corrupted SYSVAR_HERE (e.g., from stack overflow into the sysvar area)
        // would otherwise propagate as a garbage user_here.
        if mem_here > self.user_here && mem_here < mem_len {
            self.user_here = mem_here;
            if let Some(ref cell) = self.here_cell {
                *cell.lock().unwrap() = mem_here;
            }
        }
    }

    /// Write `user_here` to WASM `memory[SYSVAR_HERE]` so Forth code can read it.
    /// Refreshes from `here_cell` first in case a host function updated it.
    fn sync_here_to_wasm(&mut self) {
        self.refresh_user_here();
        let data = self.memory.data_mut(&mut self.store);
        data[SYSVAR_HERE as usize..SYSVAR_HERE as usize + 4]
            .copy_from_slice(&self.user_here.to_le_bytes());
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

    // -----------------------------------------------------------------------
    // Priority 4: Stack/arithmetic host functions
    // -----------------------------------------------------------------------

    /// ROLL -- ( xu xu-1 ... x0 u -- xu-1 ... x0 xu ) rotate u+1 items.
    fn register_roll(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Pop u from stack
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let u = i32::from_le_bytes(b) as u32;
                let sp = sp + CELL_SIZE; // pop u

                if u == 0 {
                    // 0 ROLL is a no-op
                    dsp.set(&mut caller, Val::I32(sp as i32))?;
                    return Ok(());
                }

                // Save xu (the deep item to bring to top)
                let xu_addr = sp + u * CELL_SIZE;
                let data = memory.data(&caller);
                let saved: [u8; 4] = data[xu_addr as usize..xu_addr as usize + 4]
                    .try_into()
                    .unwrap();

                // Shift items from sp to sp+(u-1)*4 toward higher addresses by one cell
                // (i.e., move each item one position deeper)
                let data = memory.data_mut(&mut caller);
                let src_start = sp as usize;
                let count = (u * CELL_SIZE) as usize;
                // Copy backward to handle overlap correctly
                for i in (0..count).rev() {
                    data[src_start + CELL_SIZE as usize + i] = data[src_start + i];
                }

                // Write saved xu at new TOS
                data[sp as usize..sp as usize + 4].copy_from_slice(&saved);

                dsp.set(&mut caller, Val::I32(sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("ROLL", false, func)?;
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

    // -----------------------------------------------------------------------
    // Priority 6: System/compiler host functions
    // -----------------------------------------------------------------------

    /// IMMEDIATE -- toggle immediate flag on the most recent word.
    fn register_immediate_word(&mut self) -> anyhow::Result<()> {
        // IMMEDIATE needs to call dictionary.set_immediate().
        // Since the host function can't access self.dictionary directly,
        // we use the WASM memory to track this... actually, we handle IMMEDIATE
        // as a special token in interpret_token instead.
        //
        // Use pending_define mechanism so IMMEDIATE works from compiled code.
        let pending = Arc::clone(&self.pending_define);
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |_caller, _params, _results| {
                pending.lock().unwrap().push(12);
                Ok(())
            },
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
            anyhow::bail!("': unknown word: {name}");
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
        let memory = self.memory;
        let dsp = self.dsp;
        // Share the PFA map with the host function via Arc<Mutex<>>
        let pfa_map = Arc::new(Mutex::new(self.word_pfa_map.clone()));
        // Store the Arc for later updates
        self.word_pfa_map_shared = Some(Arc::clone(&pfa_map));

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Pop xt from data stack
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let xt = u32::from_le_bytes(b);

                // Look up PFA for this xt
                let map = pfa_map.lock().unwrap();
                let pfa = map.get(&xt).copied().unwrap_or(0);
                drop(map);

                // Replace TOS with PFA
                let data = memory.data_mut(&mut caller);
                data[sp as usize..sp as usize + 4].copy_from_slice(&(pfa as i32).to_le_bytes());
                Ok(())
            },
        );

        self.register_host_primitive(">BODY", false, func)?;
        Ok(())
    }

    /// ENVIRONMENT? -- ( c-addr u -- false | value true ) query system parameters.
    fn register_environment_q(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let u = u32::from_le_bytes(b);
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let addr = u32::from_le_bytes(b);
                let query = String::from_utf8_lossy(&data[addr as usize..(addr + u) as usize])
                    .to_ascii_uppercase();

                match query.as_str() {
                    "#LOCALS" => {
                        // Return (16 TRUE) — support at least 16 locals
                        let data = memory.data_mut(&mut caller);
                        data[(sp + 4) as usize..(sp + 8) as usize]
                            .copy_from_slice(&16i32.to_le_bytes());
                        data[sp as usize..sp as usize + 4].copy_from_slice(&(-1i32).to_le_bytes()); // TRUE
                        dsp.set(&mut caller, Val::I32(sp as i32))?;
                    }
                    _ => {
                        // Unknown: pop 2, push FALSE
                        let new_sp = sp + 4;
                        let data = memory.data_mut(&mut caller);
                        data[new_sp as usize..new_sp as usize + 4]
                            .copy_from_slice(&0i32.to_le_bytes());
                        dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                    }
                }
                Ok(())
            },
        );

        self.register_host_primitive("ENVIRONMENT?", false, func)?;
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
    // Exception word set: CATCH and THROW
    // -----------------------------------------------------------------------

    /// Register CATCH and THROW (Forth 2012 Exception word set).
    ///
    /// CATCH ( xt -- exception# | 0 ) executes xt. If it completes normally,
    /// pushes 0. If THROW is called, restores stacks and pushes the throw code.
    ///
    /// THROW ( exception# -- ) if non-zero, unwinds execution back to the
    /// nearest CATCH, passing the exception code.
    fn register_catch_throw(&mut self) -> anyhow::Result<()> {
        let throw_code = Arc::clone(&self.throw_code);
        let memory = self.memory;
        let dsp = self.dsp;
        let rsp = self.rsp;
        let table = self.table;

        // THROW ( exception# -- )
        let throw_code_for_throw = Arc::clone(&throw_code);
        let throw_func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Pop throw code from data stack
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                if sp >= DATA_STACK_TOP {
                    return Err(wasmtime::Error::msg("THROW: stack underflow"));
                }
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let code = i32::from_le_bytes(b);
                // Pop TOS
                dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))?;

                if code == 0 {
                    return Ok(());
                }

                // Store the throw code and trigger a trap to unwind back to CATCH
                *throw_code_for_throw.lock().unwrap() = Some(code);
                Err(wasmtime::Error::msg("forth-throw"))
            },
        );
        self.register_host_primitive("THROW", false, throw_func)?;

        // CATCH ( xt -- exception# | 0 )
        let throw_code_for_catch = Arc::clone(&throw_code);
        let catch_func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Pop xt from data stack
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                if sp >= DATA_STACK_TOP {
                    return Err(wasmtime::Error::msg("CATCH: stack underflow"));
                }
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let xt = u32::from_le_bytes(b);
                // Pop TOS (remove xt)
                let sp_after_pop = sp + CELL_SIZE;
                dsp.set(&mut caller, Val::I32(sp_after_pop as i32))?;

                // Save stack depths for restoration on THROW
                let saved_dsp = sp_after_pop;
                let saved_rsp = rsp.get(&mut caller).unwrap_i32() as u32;

                // Look up the function in the table
                let func_ref = table
                    .get(&mut caller, xt as u64)
                    .ok_or_else(|| wasmtime::Error::msg("CATCH: invalid xt"))?;
                let func = *func_ref
                    .unwrap_func()
                    .ok_or_else(|| wasmtime::Error::msg("CATCH: null funcref"))?;

                // Call the word -- if THROW is invoked, func.call returns Err
                match func.call(&mut caller, &[], &mut []) {
                    Ok(()) => {
                        // Normal completion: push 0
                        let current_sp = dsp.get(&mut caller).unwrap_i32() as u32;
                        let mem_len = memory.data(&caller).len() as u32;
                        let new_sp = current_sp.wrapping_sub(CELL_SIZE);
                        if new_sp >= mem_len {
                            return Err(wasmtime::Error::msg("stack overflow in CATCH"));
                        }
                        let data = memory.data_mut(&mut caller);
                        data[new_sp as usize..new_sp as usize + 4]
                            .copy_from_slice(&0_i32.to_le_bytes());
                        dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                        Ok(())
                    }
                    Err(_) => {
                        // Check if this was a THROW (vs some other trap)
                        let mut tc = throw_code_for_catch.lock().unwrap();
                        let code = tc.take().unwrap_or(-1);
                        drop(tc);

                        // Restore stack pointers to saved depths
                        dsp.set(&mut caller, Val::I32(saved_dsp as i32))?;
                        rsp.set(&mut caller, Val::I32(saved_rsp as i32))?;

                        // Push the throw code onto the restored stack
                        let mem_len = memory.data(&caller).len() as u32;
                        let new_sp = saved_dsp.wrapping_sub(CELL_SIZE);
                        if new_sp >= mem_len {
                            return Err(wasmtime::Error::msg("stack overflow in CATCH"));
                        }
                        let data = memory.data_mut(&mut caller);
                        data[new_sp as usize..new_sp as usize + 4]
                            .copy_from_slice(&code.to_le_bytes());
                        dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                        Ok(())
                    }
                }
            },
        );
        self.register_host_primitive("CATCH", false, catch_func)?;

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

        // Save current input state and SOURCE-ID
        let saved_buffer = std::mem::take(&mut self.input_buffer);
        let saved_pos = self.input_pos;
        let data = self.memory.data(&self.store);
        let saved_source_id = i32::from_le_bytes(
            data[crate::memory::SYSVAR_SOURCE_ID as usize
                ..crate::memory::SYSVAR_SOURCE_ID as usize + 4]
                .try_into()
                .unwrap(),
        );

        // Set new input and SOURCE-ID = -1 (string source)
        self.input_buffer = s;
        self.input_pos = 0;
        let data = self.memory.data_mut(&mut self.store);
        data[crate::memory::SYSVAR_SOURCE_ID as usize
            ..crate::memory::SYSVAR_SOURCE_ID as usize + 4]
            .copy_from_slice(&(-1i32).to_le_bytes());

        // Sync input buffer, >IN, and #TIB to WASM (for SOURCE and WORD)
        {
            let bytes = self.input_buffer.as_bytes();
            let len = bytes.len().min(INPUT_BUFFER_SIZE as usize);
            let data = self.memory.data_mut(&mut self.store);
            data[INPUT_BUFFER_BASE as usize..INPUT_BUFFER_BASE as usize + len]
                .copy_from_slice(&bytes[..len]);
            data[SYSVAR_TO_IN as usize..SYSVAR_TO_IN as usize + 4]
                .copy_from_slice(&0u32.to_le_bytes());
            data[SYSVAR_NUM_TIB as usize..SYSVAR_NUM_TIB as usize + 4]
                .copy_from_slice(&(len as u32).to_le_bytes());
        }

        // Interpret with >IN sync (supports >IN manipulation)
        while let Some(token) = self.next_token() {
            {
                let data = self.memory.data_mut(&mut self.store);
                data[SYSVAR_TO_IN as usize..SYSVAR_TO_IN as usize + 4]
                    .copy_from_slice(&(self.input_pos as u32).to_le_bytes());
            }
            let wasm_to_in_before = self.input_pos;
            self.interpret_token(&token)?;
            let data = self.memory.data(&self.store);
            let b: [u8; 4] = data[SYSVAR_TO_IN as usize..SYSVAR_TO_IN as usize + 4]
                .try_into()
                .unwrap();
            let wasm_to_in = u32::from_le_bytes(b) as usize;
            if wasm_to_in != wasm_to_in_before {
                self.input_pos = wasm_to_in;
            }
            if self.input_pos >= self.input_buffer.len() {
                break;
            }
        }

        // Restore input state, SOURCE-ID, and sync back to WASM
        self.input_buffer = saved_buffer;
        self.input_pos = saved_pos;
        {
            let bytes = self.input_buffer.as_bytes();
            let len = bytes.len().min(INPUT_BUFFER_SIZE as usize);
            let data = self.memory.data_mut(&mut self.store);
            data[INPUT_BUFFER_BASE as usize..INPUT_BUFFER_BASE as usize + len]
                .copy_from_slice(&bytes[..len]);
            data[SYSVAR_TO_IN as usize..SYSVAR_TO_IN as usize + 4]
                .copy_from_slice(&(self.input_pos as u32).to_le_bytes());
            data[SYSVAR_NUM_TIB as usize..SYSVAR_NUM_TIB as usize + 4]
                .copy_from_slice(&(len as u32).to_le_bytes());
            data[crate::memory::SYSVAR_SOURCE_ID as usize
                ..crate::memory::SYSVAR_SOURCE_ID as usize + 4]
                .copy_from_slice(&saved_source_id.to_le_bytes());
        }

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

        // Store as counted string in WASM memory (at a dedicated WORD buffer)
        let buf_addr = crate::memory::WORD_BUF_BASE;
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
            .map_err(|e| anyhow::anyhow!("{e}"))?;

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
                anyhow::bail!("DOES>: unknown word: {token}");
            }
        }

        // Compile the DOES> body: push PFA, then run the body
        let mut full_ir = vec![IrOp::PushI32(pfa as i32)];
        full_ir.extend(does_ir);

        // Get the existing word_id from the code field
        let fn_index = self
            .dictionary
            .code_field(latest)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let word_id = WordId(fn_index);

        // Compile and replace
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
        };
        let name = self
            .dictionary
            .word_name(latest)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let compiled = compile_word(&name, &full_ir, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for DOES>: {e}"))?;
        self.instantiate_and_install(&compiled, word_id)?;

        Ok(())
    }

    /// DOES> in compile mode -- handle the `: name CREATE ... DOES> ... ;` pattern.
    ///
    /// Strategy: compile the does-body as a separate WASM word, then create
    /// the defining word as a host function that:
    /// 1. Reads the next token from the input buffer
    /// 2. Creates a new word (via `define_create` logic)
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

        // Check for a second DOES> in the does-body (double-DOES> pattern).
        // If found, split: first part is the first does-action, second part
        // becomes a separate does-action that gets patched in at runtime.
        let does_split = does_tokens
            .iter()
            .position(|t| t.eq_ignore_ascii_case("DOES>"));
        let (first_tokens, second_does_tokens) = if let Some(pos) = does_split {
            (
                does_tokens[..pos].to_vec(),
                Some(does_tokens[pos + 1..].to_vec()),
            )
        } else {
            (does_tokens, None)
        };

        // If there's a second DOES>, compile its body first as a separate word
        let second_does_action_id = if let Some(ref second_tokens) = second_does_tokens {
            let second_word_id = self
                .dictionary
                .create("_does_action2_", false)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            self.dictionary.reveal();
            self.next_table_index = self.next_table_index.max(second_word_id.0 + 1);

            let saved_name2 = self.compiling_name.take();
            let saved_word_id2 = self.compiling_word_id.take();
            let saved_control2 = std::mem::take(&mut self.control_stack);

            self.compiling_ir.clear();
            self.compiling_name = Some("_does_action2_".to_string());
            self.compiling_word_id = Some(second_word_id);

            for token in second_tokens {
                self.compile_token(token)?;
            }

            let second_ir = std::mem::take(&mut self.compiling_ir);
            let config = CodegenConfig {
                base_fn_index: second_word_id.0,
                table_size: self.table_size(),
                stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
            };
            let compiled = compile_word("_does_action2_", &second_ir, &config)
                .map_err(|e| anyhow::anyhow!("codegen error for DOES> body 2: {e}"))?;
            self.instantiate_and_install(&compiled, second_word_id)?;

            self.compiling_name = saved_name2;
            self.compiling_word_id = saved_word_id2;
            self.control_stack = saved_control2;

            Some(second_word_id)
        } else {
            None
        };

        // Compile the first does-body as a separate word
        let does_word_id = self
            .dictionary
            .create("_does_action_", false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        self.dictionary.reveal();
        self.next_table_index = self.next_table_index.max(does_word_id.0 + 1);

        // Save and compile does-body
        let saved_name = self.compiling_name.take();
        let saved_word_id = self.compiling_word_id.take();
        let saved_control = std::mem::take(&mut self.control_stack);
        let saved_locals = std::mem::take(&mut self.compiling_locals);

        self.compiling_ir.clear();
        self.compiling_name = Some("_does_action_".to_string());
        self.compiling_word_id = Some(does_word_id);

        // Replay does-body tokens via the input buffer so that words like {: can
        // use next_token() to read subsequent tokens (e.g., local names up to :}).
        let saved_input = std::mem::take(&mut self.input_buffer);
        let saved_pos = self.input_pos;
        self.input_buffer = first_tokens.join(" ");
        self.input_pos = 0;
        while let Some(token) = self.next_token() {
            self.compile_token(&token)?;
        }
        self.input_buffer = saved_input;
        self.input_pos = saved_pos;

        // If there's a second DOES>, append code to patch the word at runtime
        if let Some(second_action_id) = second_does_action_id {
            let does_patch_id = self
                .dictionary
                .find("_DOES_PATCH_")
                .map(|(_, id, _)| id)
                .ok_or_else(|| anyhow::anyhow!("_DOES_PATCH_ not found"))?;
            self.push_ir(IrOp::PushI32(second_action_id.0 as i32));
            self.push_ir(IrOp::Call(does_patch_id));
        }

        let does_ir = std::mem::take(&mut self.compiling_ir);
        let config = CodegenConfig {
            base_fn_index: does_word_id.0,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
        };
        let compiled = compile_word("_does_action_", &does_ir, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for DOES> body: {e}"))?;
        self.instantiate_and_install(&compiled, does_word_id)?;

        // Restore compilation state
        self.compiling_name = saved_name;
        self.compiling_word_id = saved_word_id;
        self.control_stack = saved_control;
        self.compiling_locals = saved_locals;

        // Register the defining word as a "does-defining" word.
        let has_create = self.saw_create_in_def;
        self.does_definitions.insert(
            defining_word_id,
            DoesDefinition {
                create_ir,
                does_action_id: does_word_id,
                has_create,
            },
        );

        // Compile the defining word as a no-op (the actual work is done
        // by the outer interpreter when it detects the does-definition).
        let config = CodegenConfig {
            base_fn_index: defining_word_id.0,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
        };
        let compiled = compile_word(&defining_name, &[], &config)
            .map_err(|e| anyhow::anyhow!("codegen error for defining word: {e}"))?;
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
    ///
    /// Two cases:
    /// - With create-part (e.g., `: MYDEF CREATE , DOES> @ ;`): reads a name,
    ///   creates a new word, runs the create-part, then patches the new word.
    /// - Without create-part (e.g., `: DOES1 DOES> @ 1 + ;`): simply patches
    ///   the most recently defined word with the DOES> action.
    fn execute_does_defining(&mut self, defining_word_id: WordId) -> anyhow::Result<()> {
        // Get the does-definition info
        let def = self
            .does_definitions
            .get(&defining_word_id)
            .ok_or_else(|| anyhow::anyhow!("not a DOES> defining word"))?;
        let create_ir = def.create_ir.clone();
        let does_action_id = def.does_action_id;

        // Check if the definition included CREATE. If not, the word just
        // patches the most recently CREATEd word without reading a new name.
        let has_create = def.has_create;

        if has_create {
            // Full defining-word pattern: read name, create word, run create-part
            // Step 1: Read the name of the new word from the input stream
            let name = self
                .next_token()
                .ok_or_else(|| anyhow::anyhow!("defining word: expected name"))?;

            // Step 2: Create the new word (like define_create)
            let new_word_id = self
                .dictionary
                .create(&name, false)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            self.refresh_user_here();
            let pfa = self.user_here;

            // Temporarily install a "push PFA" word (will be patched later)
            let ir_body = vec![IrOp::PushI32(pfa as i32)];
            self.ir_bodies.insert(new_word_id, ir_body.clone());
            let config = CodegenConfig {
                base_fn_index: new_word_id.0,
                table_size: self.table_size(),
                stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
            };
            let compiled = compile_word(&name, &ir_body, &config)
                .map_err(|e| anyhow::anyhow!("codegen: {e}"))?;
            self.instantiate_and_install(&compiled, new_word_id)?;
            self.dictionary.reveal();
            self.next_table_index = self.next_table_index.max(new_word_id.0 + 1);
            // Track PFA for >BODY
            self.word_pfa_map.insert(new_word_id.0, pfa);
            self.sync_pfa_map(new_word_id.0, pfa);
            // Track for DOES> patching
            self.last_created_info = Some((self.dictionary.latest(), pfa));

            // Step 3: Execute the create-part IR using a reserved fn index
            // (don't create a dictionary entry — that would change `latest()`)
            let tmp_fn_idx = self.dictionary.next_fn_index();
            self.dictionary.reserve_fn_index();
            let tmp_word_id = WordId(tmp_fn_idx);
            self.next_table_index = self.next_table_index.max(tmp_fn_idx + 1);

            let config = CodegenConfig {
                base_fn_index: tmp_word_id.0,
                table_size: self.table_size(),
                stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
            };
            let compiled = compile_word("_create_part_", &create_ir, &config)
                .map_err(|e| anyhow::anyhow!("codegen: {e}"))?;
            self.instantiate_and_install(&compiled, tmp_word_id)?;
            self.execute_word(tmp_word_id)?;

            // Step 4: Patch the new word to push PFA and call does-action
            self.refresh_user_here();
            let patched_ir = vec![IrOp::PushI32(pfa as i32), IrOp::Call(does_action_id)];
            let config = CodegenConfig {
                base_fn_index: new_word_id.0,
                table_size: self.table_size(),
                stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
            };
            let compiled = compile_word(&name, &patched_ir, &config)
                .map_err(|e| anyhow::anyhow!("DOES> patch codegen: {e}"))?;
            self.instantiate_and_install(&compiled, new_word_id)?;
            self.sync_here_cell();
        } else {
            // No create-part: just patch the most recently CREATEd word.
            // This handles patterns like `: DOES1 DOES> @ 1 + ;`
            let (target_addr, pfa) = self
                .last_created_info
                .ok_or_else(|| anyhow::anyhow!("DOES>: no CREATEd word to patch"))?;

            let fn_index = self
                .dictionary
                .code_field(target_addr)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let target_word_id = WordId(fn_index);

            let name = self
                .dictionary
                .word_name(target_addr)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            let patched_ir = vec![IrOp::PushI32(pfa as i32), IrOp::Call(does_action_id)];
            let config = CodegenConfig {
                base_fn_index: target_word_id.0,
                table_size: self.table_size(),
                stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
            };
            let compiled = compile_word(&name, &patched_ir, &config)
                .map_err(|e| anyhow::anyhow!("DOES> patch codegen: {e}"))?;
            self.instantiate_and_install(&compiled, target_word_id)?;
        }

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

    /// FIND ( c-addr -- c-addr 0 | xt 1 | xt -1 ) look up counted string.
    fn register_find(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let word_lookup = Arc::clone(&self.word_lookup);

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let mem_len = data.len() as u32;

                // Stack pointer sanity check
                if sp < CELL_SIZE || sp > mem_len {
                    return Err(wasmtime::Error::msg("stack error in FIND"));
                }

                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let c_addr = u32::from_le_bytes(b);

                // Bounds check
                if c_addr >= mem_len {
                    // Push c-addr and 0 (not found)
                    let new_sp = sp - CELL_SIZE;
                    let data = memory.data_mut(&mut caller);
                    data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&0i32.to_le_bytes());
                    dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                    return Ok(());
                }

                let count = data[c_addr as usize] as usize;
                let name_start = (c_addr + 1) as usize;
                if name_start + count > mem_len as usize {
                    let new_sp = sp - CELL_SIZE;
                    let data = memory.data_mut(&mut caller);
                    data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&0i32.to_le_bytes());
                    dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                    return Ok(());
                }

                let name_bytes = &data[name_start..name_start + count];
                let name = String::from_utf8_lossy(name_bytes).to_ascii_uppercase();

                let lookup = word_lookup.lock().unwrap();
                if let Some(&(xt, is_imm)) = lookup.get(&name) {
                    // Found: replace c-addr with xt, push flag
                    let new_sp = sp - CELL_SIZE;
                    let flag: i32 = if is_imm { 1 } else { -1 };
                    let data = memory.data_mut(&mut caller);
                    // Replace c-addr with xt
                    data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                        .copy_from_slice(&(xt as i32).to_le_bytes());
                    // Push flag
                    data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&flag.to_le_bytes());
                    dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                } else {
                    // Not found: push c-addr and 0
                    let new_sp = sp - CELL_SIZE;
                    let data = memory.data_mut(&mut caller);
                    data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&0i32.to_le_bytes());
                    dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                }

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

    /// Register COMPILE, as a host function.
    /// COMPILE, ( xt -- ) appends a call to xt into the current compilation.
    /// Used internally by POSTPONE for non-immediate words.
    fn register_compile_comma(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let pending_compile = Arc::clone(&self.pending_compile);

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Pop xt from data stack
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let xt = u32::from_le_bytes(b);
                // Drop top of stack
                let new_sp = sp + 4;
                dsp.set(&mut caller, Val::I32(new_sp as i32)).unwrap();
                // Signal the outer interpreter to compile a call to this xt
                pending_compile.lock().unwrap().push(xt);
                Ok(())
            },
        );

        self.register_host_primitive("COMPILE,", false, func)?;
        Ok(())
    }

    /// Register `_does_patch_` as a host function for runtime DOES> patching.
    /// ( `does_action_id` -- ) Signals the outer interpreter to patch the most
    /// recently `CREATEd` word with a new DOES> action.
    fn register_does_patch(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let pending_does_patch = Arc::clone(&self.pending_does_patch);

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Pop does_action_id from data stack
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let does_action_id = u32::from_le_bytes(b);
                let new_sp = sp + 4;
                dsp.set(&mut caller, Val::I32(new_sp as i32)).unwrap();
                *pending_does_patch.lock().unwrap() = Some(does_action_id);
                Ok(())
            },
        );

        self.register_host_primitive("_DOES_PATCH_", false, func)?;
        Ok(())
    }

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
                    pending.lock().unwrap().push(1);
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
                    pending.lock().unwrap().push(2);
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
                    pending.lock().unwrap().push(3);
                    Ok(())
                },
            );
            self.register_host_primitive("CREATE", false, func)?;
        }

        // 2CONSTANT: sets pending_define to 9
        {
            let pending = Arc::clone(&self.pending_define);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |_caller, _params, _results| {
                    pending.lock().unwrap().push(9);
                    Ok(())
                },
            );
            self.register_host_primitive("2CONSTANT", false, func)?;
        }

        // 2VARIABLE: sets pending_define to 10
        {
            let pending = Arc::clone(&self.pending_define);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |_caller, _params, _results| {
                    pending.lock().unwrap().push(10);
                    Ok(())
                },
            );
            self.register_host_primitive("2VARIABLE", false, func)?;
        }

        // DEFER: sets pending_define to 11
        {
            let pending = Arc::clone(&self.pending_define);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |_caller, _params, _results| {
                    pending.lock().unwrap().push(11);
                    Ok(())
                },
            );
            self.register_host_primitive("DEFER", false, func)?;
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
                pending.lock().unwrap().push(4);
                Ok(())
            },
        );
        self.register_host_primitive("EVALUATE", false, func)?;
        Ok(())
    }

    /// Register WORD as a host function callable from compiled code.
    /// WORD ( char -- c-addr ) reads from the WASM input buffer and updates >IN.
    fn register_word_word(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Pop delimiter from data stack
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let delim = i32::from_le_bytes(b) as u8;
                dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))?;

                // Read >IN and #TIB from WASM memory
                let data = memory.data(&caller);
                let b: [u8; 4] = data[SYSVAR_TO_IN as usize..SYSVAR_TO_IN as usize + 4]
                    .try_into()
                    .unwrap();
                let mut to_in = u32::from_le_bytes(b);
                let b: [u8; 4] = data[SYSVAR_NUM_TIB as usize..SYSVAR_NUM_TIB as usize + 4]
                    .try_into()
                    .unwrap();
                let num_tib = u32::from_le_bytes(b);

                // Skip leading delimiters (also skip spaces when delimiter != space)
                while to_in < num_tib {
                    let data = memory.data(&caller);
                    let ch = data[(INPUT_BUFFER_BASE + to_in) as usize];
                    if ch == delim || (delim != b' ' && ch == b' ') {
                        to_in += 1;
                    } else {
                        break;
                    }
                }

                // Collect word
                let start = to_in;
                while to_in < num_tib {
                    let data = memory.data(&caller);
                    if data[(INPUT_BUFFER_BASE + to_in) as usize] == delim {
                        break;
                    }
                    to_in += 1;
                }
                let word_len = to_in - start;

                // Skip past delimiter
                if to_in < num_tib {
                    to_in += 1;
                }

                // Update >IN in WASM memory
                let data = memory.data_mut(&mut caller);
                data[SYSVAR_TO_IN as usize..SYSVAR_TO_IN as usize + 4]
                    .copy_from_slice(&to_in.to_le_bytes());

                // Store counted string at dedicated WORD buffer
                let buf_addr = crate::memory::WORD_BUF_BASE;
                data[buf_addr as usize] = word_len as u8;
                let src_start = (INPUT_BUFFER_BASE + start) as usize;
                let dst_start = buf_addr as usize + 1;
                for i in 0..word_len as usize {
                    data[dst_start + i] = data[src_start + i];
                }

                // Push c-addr onto data stack
                let new_sp = sp; // We already popped delim, now push c-addr
                let data = memory.data_mut(&mut caller);
                data[(new_sp) as usize..(new_sp + 4) as usize]
                    .copy_from_slice(&(buf_addr as i32).to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;

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

        // Bounds check: c_addr must be within WASM memory
        let mem_len = self.memory.data(&self.store).len() as u32;
        if c_addr >= mem_len {
            // Invalid address -- push original address and 0 (not found)
            self.push_data_stack(c_addr as i32)?;
            self.push_data_stack(0)?;
            return Ok(());
        }

        // Read counted string from WASM memory
        let data = self.memory.data(&self.store);
        let count = data[c_addr as usize] as usize;
        let name_start = (c_addr + 1) as usize;
        if name_start + count > mem_len as usize {
            // String extends past memory -- push original address and 0
            self.push_data_stack(c_addr as i32)?;
            self.push_data_stack(0)?;
            return Ok(());
        }
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
        let actions: Vec<i32> = {
            let mut pending = self.pending_define.lock().unwrap();
            std::mem::take(&mut *pending)
        };
        for action in actions {
            match action {
                1 => self.define_constant()?,
                2 => self.define_variable()?,
                3 => self.define_create()?,
                4 => self.interpret_evaluate()?,
                5 => self.interpret_word()?,
                6 => self.interpret_find()?,
                7 => self.interpret_parse()?,
                8 => self.interpret_parse_name()?,
                9 => self.define_2constant()?,
                10 => self.define_2variable()?,
                11 => self.define_defer()?,
                12 => self.set_immediate()?,
                20 => self.do_get_current()?,
                21 => self.do_set_current()?,
                25 => self.do_search_wordlist()?,
                33 => {
                    // DEFINITIONS: set current_wid to top of search order
                    let so = self.search_order.lock().unwrap();
                    if let Some(&top) = so.first() {
                        self.dictionary.set_current_wid(top);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Drain `pending_compile` and push `IrOp::Call` for each entry into `compiling_ir`.
    /// Called after executing an immediate word during compilation.
    fn handle_pending_compile(&mut self) {
        let pending: Vec<u32> = {
            let mut v = self.pending_compile.lock().unwrap();
            std::mem::take(&mut *v)
        };
        for xt in pending {
            self.push_ir(IrOp::Call(WordId(xt)));
        }
    }

    /// Handle a pending runtime DOES> patch.
    /// When a DOES> body contains another DOES>, the inner DOES> signals via
    /// `_DOES_PATCH_` to replace the most recently `CREATEd` word's behavior.
    fn handle_pending_does_patch(&mut self) -> anyhow::Result<()> {
        let does_action_id = {
            let mut p = self.pending_does_patch.lock().unwrap();
            p.take()
        };
        if let Some(action_id) = does_action_id {
            let (target_addr, pfa) = self
                .last_created_info
                .ok_or_else(|| anyhow::anyhow!("runtime DOES>: no CREATEd word to patch"))?;

            let fn_index = self
                .dictionary
                .code_field(target_addr)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let target_word_id = WordId(fn_index);

            let name = self
                .dictionary
                .word_name(target_addr)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            let patched_ir = vec![IrOp::PushI32(pfa as i32), IrOp::Call(WordId(action_id))];
            let config = CodegenConfig {
                base_fn_index: target_word_id.0,
                table_size: self.table_size(),
                stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
            };
            let compiled = compile_word(&name, &patched_ir, &config)
                .map_err(|e| anyhow::anyhow!("runtime DOES> patch codegen: {e}"))?;
            self.instantiate_and_install(&compiled, target_word_id)?;
        }
        Ok(())
    }

    /// Handle a pending MARKER restore.
    /// When a marker word executes, it signals via `pending_marker_restore`
    /// to roll back the dictionary and VM state to when the marker was created.
    fn handle_pending_marker_restore(&mut self) -> anyhow::Result<()> {
        let marker_id = {
            let mut p = self.pending_marker_restore.lock().unwrap();
            p.take()
        };
        if let Some(id) = marker_id {
            if let Some(state) = self.marker_states.remove(&id) {
                self.dictionary.restore_state(state.dict_state);
                self.user_here = state.user_here;
                self.next_table_index = state.next_table_index;
                self.word_pfa_map = state.word_pfa_map;
                self.ir_bodies = state.ir_bodies;
                self.does_definitions = state.does_definitions;
                self.host_word_names = state.host_word_names;
                self.two_value_words = state.two_value_words;
                self.fvalue_words = state.fvalue_words;
                self.sync_here_cell();
                self.rebuild_word_lookup();
                // Remove any marker states that were created after this one
                self.marker_states.retain(|&k, _| k < id);
            }
        }
        Ok(())
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
                let b: [u8; 4] = data[SYSVAR_NUM_TIB as usize..SYSVAR_NUM_TIB as usize + 4]
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

        // .( is an immediate word that prints until closing paren.
        // Register as no-op in dictionary so FIND can discover it as immediate.
        // The actual parsing is handled by interpret_token_immediate/compile_token.
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            |_caller, _params, _results| Ok(()),
        );
        self.register_host_primitive(".(", true, func)?;

        // ( is an immediate word (comment). Register in dictionary for FIND.
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            |_caller, _params, _results| Ok(()),
        );
        self.register_host_primitive("(", true, func)?;

        // Register [IF], [ELSE], [THEN], [DEFINED], [UNDEFINED] as immediate no-ops
        // so they are findable by WORD+FIND. Actual logic is in interpret_token.
        for name in &["[IF]", "[ELSE]", "[THEN]", "[DEFINED]", "[UNDEFINED]"] {
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                |_caller, _params, _results| Ok(()),
            );
            self.register_host_primitive(name, true, func)?;
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

    /// Store the `fn_index` of the most recently `CREATEd` word at address 0x30
    /// so the DOES> patcher can find it.
    fn store_latest_fn_index(&mut self, word_id: WordId) {
        let data = self.memory.data_mut(&mut self.store);
        data[0x30..0x34].copy_from_slice(&word_id.0.to_le_bytes());
    }

    /// Sync a word to the shared `word_lookup` for inline FIND access.
    fn sync_word_lookup(&self, name: &str, word_id: WordId, is_immediate: bool) {
        let mut lookup = self.word_lookup.lock().unwrap();
        lookup.insert(name.to_ascii_uppercase(), (word_id.0, is_immediate));
    }

    /// Rebuild the entire `word_lookup` from the dictionary.
    /// This iterates all visible words and populates the shared lookup table.
    fn rebuild_word_lookup(&self) {
        let mut lookup = self.word_lookup.lock().unwrap();
        lookup.clear();
        // Use dictionary.find for each known word is too slow.
        // Instead, iterate through the dictionary's linked list.
        // We use the dictionary's public API to traverse:
        let mut addr = self.dictionary.latest();
        while addr != 0 {
            if let Ok(name) = self.dictionary.word_name(addr)
                && let Some((_, word_id, is_imm)) = self.dictionary.find(&name)
            {
                lookup.insert(name.to_ascii_uppercase(), (word_id.0, is_imm));
            }
            // The link field is at the start of the entry (first 4 bytes)
            let prev = self.dictionary.read_link(addr);
            if prev == addr {
                break; // Prevent infinite loop
            }
            addr = prev;
        }
    }

    // -----------------------------------------------------------------------
    // Core Extension words: register functions
    // -----------------------------------------------------------------------

    /// 2R@ ( -- x1 x2 ) ( R: x1 x2 -- x1 x2 ) copy two cells from return stack.
    fn register_2r_fetch(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let rsp = self.rsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let rsp_val = rsp.get(&mut caller).unwrap_i32() as u32;
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                // Return stack: x2 at rsp, x1 at rsp+4
                let b: [u8; 4] = data[rsp_val as usize..rsp_val as usize + 4]
                    .try_into()
                    .unwrap();
                let x2 = i32::from_le_bytes(b);
                let b: [u8; 4] = data[(rsp_val + 4) as usize..(rsp_val + 8) as usize]
                    .try_into()
                    .unwrap();
                let x1 = i32::from_le_bytes(b);
                // Push x1 then x2 onto data stack
                let mem_len = memory.data(&caller).len() as u32;
                if sp < 8 || sp > mem_len {
                    return Err(wasmtime::Error::msg("data stack overflow in 2R@"));
                }
                let new_sp = sp - 8;
                let data = memory.data_mut(&mut caller);
                data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                    .copy_from_slice(&x1.to_le_bytes());
                data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&x2.to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("2R@", false, func)?;
        Ok(())
    }

    /// UNUSED ( -- u ) return available dictionary space.
    fn register_unused(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let here_cell = self.here_cell.clone();

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let mut here_val = here_cell.as_ref().map_or(0, |c| *c.lock().unwrap());
                let data = memory.data(&caller);
                let mem_size = data.len() as u32;
                // Also read SYSVAR_HERE from WASM (Forth ALLOT/,/C, update it directly)
                let mem_here = u32::from_le_bytes(
                    data[SYSVAR_HERE as usize..SYSVAR_HERE as usize + 4]
                        .try_into()
                        .unwrap(),
                );
                if mem_here > here_val && mem_here < mem_size {
                    here_val = mem_here;
                }
                let unused = mem_size.saturating_sub(here_val);
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                if sp < CELL_SIZE || sp > mem_size {
                    return Err(wasmtime::Error::msg("data stack overflow in UNUSED"));
                }
                let new_sp = sp - CELL_SIZE;
                let data = memory.data_mut(&mut caller);
                data[new_sp as usize..new_sp as usize + 4]
                    .copy_from_slice(&(unused as i32).to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("UNUSED", false, func)?;
        Ok(())
    }

    /// UTIME ( -- ud ) push microseconds since epoch as a double-cell value.
    fn register_utime(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                use std::time::{SystemTime, UNIX_EPOCH};
                let us = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                let lo = us as i32;
                let hi = (us >> 32) as i32;
                // Push double: lo first (deeper), then hi on top
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let new_sp = sp - 2 * CELL_SIZE;
                let data = memory.data_mut(&mut caller);
                data[new_sp as usize..new_sp as usize + 4]
                    .copy_from_slice(&hi.to_le_bytes());
                data[new_sp as usize + 4..new_sp as usize + 8]
                    .copy_from_slice(&lo.to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("UTIME", false, func)?;
        Ok(())
    }

    /// PARSE ( char "ccc<char>" -- c-addr u ) as inline host function.
    fn register_parse_host(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Pop delimiter from data stack
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let delim = i32::from_le_bytes(b) as u8;
                let sp = sp + CELL_SIZE; // pop delimiter

                // Read >IN and #TIB from WASM memory
                let data = memory.data(&caller);
                let b: [u8; 4] = data[SYSVAR_TO_IN as usize..SYSVAR_TO_IN as usize + 4]
                    .try_into()
                    .unwrap();
                let mut to_in = u32::from_le_bytes(b);
                let b: [u8; 4] = data[SYSVAR_NUM_TIB as usize..SYSVAR_NUM_TIB as usize + 4]
                    .try_into()
                    .unwrap();
                let num_tib = u32::from_le_bytes(b);

                // Skip one leading space (outer interpreter's trailing delimiter)
                if to_in < num_tib {
                    let data = memory.data(&caller);
                    if data[(INPUT_BUFFER_BASE + to_in) as usize] == b' ' {
                        to_in += 1;
                    }
                }

                // Parse until delimiter
                let start = to_in;
                while to_in < num_tib {
                    let data = memory.data(&caller);
                    if data[(INPUT_BUFFER_BASE + to_in) as usize] == delim {
                        break;
                    }
                    to_in += 1;
                }
                let parsed_len = to_in - start;

                // Skip past delimiter
                if to_in < num_tib {
                    to_in += 1;
                }

                // Update >IN in WASM memory
                let data = memory.data_mut(&mut caller);
                data[SYSVAR_TO_IN as usize..SYSVAR_TO_IN as usize + 4]
                    .copy_from_slice(&to_in.to_le_bytes());

                // Push (c-addr u) to data stack
                let c_addr = INPUT_BUFFER_BASE + start;
                let new_sp = sp - 2 * CELL_SIZE;
                data[new_sp as usize..new_sp as usize + 4]
                    .copy_from_slice(&(parsed_len as i32).to_le_bytes());
                data[(new_sp + CELL_SIZE) as usize..(new_sp + 2 * CELL_SIZE) as usize]
                    .copy_from_slice(&(c_addr as i32).to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;

                Ok(())
            },
        );

        self.register_host_primitive("PARSE", false, func)?;
        Ok(())
    }

    /// PARSE-NAME ( "<spaces>name<space>" -- c-addr u ) as inline host function.
    fn register_parse_name_host(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Read >IN and #TIB from WASM memory
                let data = memory.data(&caller);
                let b: [u8; 4] = data[SYSVAR_TO_IN as usize..SYSVAR_TO_IN as usize + 4]
                    .try_into()
                    .unwrap();
                let mut to_in = u32::from_le_bytes(b);
                let b: [u8; 4] = data[SYSVAR_NUM_TIB as usize..SYSVAR_NUM_TIB as usize + 4]
                    .try_into()
                    .unwrap();
                let num_tib = u32::from_le_bytes(b);

                // Skip leading whitespace
                while to_in < num_tib {
                    let data = memory.data(&caller);
                    if !data[(INPUT_BUFFER_BASE + to_in) as usize].is_ascii_whitespace() {
                        break;
                    }
                    to_in += 1;
                }
                let start = to_in;

                // Parse until whitespace
                while to_in < num_tib {
                    let data = memory.data(&caller);
                    if data[(INPUT_BUFFER_BASE + to_in) as usize].is_ascii_whitespace() {
                        break;
                    }
                    to_in += 1;
                }
                let parsed_len = to_in - start;

                // Update >IN
                let data = memory.data_mut(&mut caller);
                data[SYSVAR_TO_IN as usize..SYSVAR_TO_IN as usize + 4]
                    .copy_from_slice(&to_in.to_le_bytes());

                // Push (c-addr u) to data stack
                let c_addr = INPUT_BUFFER_BASE + start;
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let new_sp = sp - 2 * CELL_SIZE;
                let data = memory.data_mut(&mut caller);
                data[new_sp as usize..new_sp as usize + 4]
                    .copy_from_slice(&(parsed_len as i32).to_le_bytes());
                data[(new_sp + CELL_SIZE) as usize..(new_sp + 2 * CELL_SIZE) as usize]
                    .copy_from_slice(&(c_addr as i32).to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;

                Ok(())
            },
        );

        self.register_host_primitive("PARSE-NAME", false, func)?;
        Ok(())
    }

    /// REFILL ( -- flag ) in piped/string mode, always returns FALSE.
    fn register_refill(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let mem_len = memory.data(&caller).len() as u32;
                if sp < CELL_SIZE || sp > mem_len {
                    return Err(wasmtime::Error::msg("data stack overflow in REFILL"));
                }
                let new_sp = sp - CELL_SIZE;
                let data = memory.data_mut(&mut caller);
                data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&0i32.to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("REFILL", false, func)?;

        // ACCEPT ( c-addr +n1 -- +n2 )  receive up to +n1 characters.
        // In non-interactive mode, return 0 (no input).
        let memory = self.memory;
        let dsp = self.dsp;
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Pop +n1 (max count) and c-addr from stack
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let new_sp = sp + CELL_SIZE; // pop +n1
                let new_sp = new_sp + CELL_SIZE; // pop c-addr
                // Push 0 (no characters received)
                let result_sp = new_sp - CELL_SIZE;
                let data = memory.data_mut(&mut caller);
                data[result_sp as usize..result_sp as usize + 4]
                    .copy_from_slice(&0i32.to_le_bytes());
                dsp.set(&mut caller, Val::I32(result_sp as i32))?;
                Ok(())
            },
        );
        self.register_host_primitive("ACCEPT", false, func)?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Double-Number word set
    // -----------------------------------------------------------------------

    /// Memory-Allocation word set: ALLOCATE, FREE, RESIZE.
    ///
    /// Uses a simple arena allocator at the top of WASM linear memory.
    /// Each allocated block has a 4-byte header storing its size.
    fn register_memory_alloc(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        // ALLOCATE ( u -- a-addr ior )
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let size = u32::from_le_bytes(b);

                let mem_len = data.len() as u32;

                // Reject obviously impossible sizes (> available memory)
                if size > mem_len / 2 {
                    let data = memory.data_mut(&mut caller);
                    data[sp as usize..sp as usize + 4].copy_from_slice(&0i32.to_le_bytes());
                    let new_sp = sp - CELL_SIZE;
                    data[new_sp as usize..new_sp as usize + 4]
                        .copy_from_slice(&(-1i32).to_le_bytes());
                    dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                    return Ok(());
                }

                // Allocate from top of memory, growing downward
                // Use last 4 bytes of memory as the allocation pointer
                let alloc_ptr_addr = mem_len - 4;
                let b: [u8; 4] = data[alloc_ptr_addr as usize..mem_len as usize]
                    .try_into()
                    .unwrap();
                let mut alloc_top = u32::from_le_bytes(b);
                if alloc_top == 0 {
                    alloc_top = mem_len - 8; // Initialize: leave room for pointer
                }

                // Block: [size(4)] [data(size)] — aligned to 4 bytes
                let aligned_size = (size + 3) & !3;
                let block_size = 4 + aligned_size;

                if alloc_top < block_size + 0x20000 {
                    // Not enough memory (leave some space for dictionary growth)
                    let data = memory.data_mut(&mut caller);
                    // Replace u with a-addr=0, push ior=-1
                    data[sp as usize..sp as usize + 4].copy_from_slice(&0i32.to_le_bytes());
                    let new_sp = sp - CELL_SIZE;
                    data[new_sp as usize..new_sp as usize + 4]
                        .copy_from_slice(&(-1i32).to_le_bytes());
                    dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                    return Ok(());
                }

                let block_start = alloc_top - block_size;
                let data_addr = block_start + 4; // skip size header

                let data = memory.data_mut(&mut caller);
                // Write size header
                data[block_start as usize..block_start as usize + 4]
                    .copy_from_slice(&size.to_le_bytes());
                // Zero the allocated area
                for i in 0..aligned_size as usize {
                    data[data_addr as usize + i] = 0;
                }
                // Update allocation pointer
                data[alloc_ptr_addr as usize..mem_len as usize]
                    .copy_from_slice(&block_start.to_le_bytes());

                // Replace u with a-addr, push ior=0
                data[sp as usize..sp as usize + 4]
                    .copy_from_slice(&(data_addr as i32).to_le_bytes());
                let new_sp = sp - CELL_SIZE;
                data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&0i32.to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );
        self.register_host_primitive("ALLOCATE", false, func)?;

        // FREE ( a-addr -- ior )
        let memory = self.memory;
        let dsp = self.dsp;
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                // Simple allocator: FREE is a no-op (arena style), return ior=0
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data_mut(&mut caller);
                // Replace a-addr with ior=0
                data[sp as usize..sp as usize + 4].copy_from_slice(&0i32.to_le_bytes());
                Ok(())
            },
        );
        self.register_host_primitive("FREE", false, func)?;

        // RESIZE ( a-addr u -- a-addr2 ior )
        let memory = self.memory;
        let dsp = self.dsp;
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let new_size = u32::from_le_bytes(b);
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let old_addr = u32::from_le_bytes(b);

                let mem_len = data.len() as u32;

                // Reject obviously impossible sizes
                if new_size > mem_len / 2 {
                    let data = memory.data_mut(&mut caller);
                    data[(sp + 4) as usize..(sp + 8) as usize]
                        .copy_from_slice(&(old_addr as i32).to_le_bytes());
                    data[sp as usize..sp as usize + 4].copy_from_slice(&(-1i32).to_le_bytes());
                    return Ok(());
                }

                // Read old size from header (4 bytes before old_addr)
                let data = memory.data(&caller);
                let old_size = if old_addr >= 4 {
                    let b: [u8; 4] = data[(old_addr - 4) as usize..old_addr as usize]
                        .try_into()
                        .unwrap();
                    u32::from_le_bytes(b)
                } else {
                    0
                };

                let alloc_ptr_addr = mem_len - 4;
                let b: [u8; 4] = data[alloc_ptr_addr as usize..mem_len as usize]
                    .try_into()
                    .unwrap();
                let mut alloc_top = u32::from_le_bytes(b);
                if alloc_top == 0 {
                    alloc_top = mem_len - 8;
                }

                let aligned_size = (new_size + 3) & !3;
                let block_size = 4 + aligned_size;

                if alloc_top < block_size + 0x20000 {
                    // Allocation failure
                    let data = memory.data_mut(&mut caller);
                    // Keep old a-addr, push ior=-1
                    let new_sp = sp + CELL_SIZE; // pop new_size
                    data[(new_sp) as usize..(new_sp + 4) as usize]
                        .copy_from_slice(&(old_addr as i32).to_le_bytes());
                    let new_sp = new_sp - CELL_SIZE;
                    data[new_sp as usize..new_sp as usize + 4]
                        .copy_from_slice(&(-1i32).to_le_bytes());
                    dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                    return Ok(());
                }

                let block_start = alloc_top - block_size;
                let new_addr = block_start + 4;

                // Copy old data to new location
                let copy_len = old_size.min(new_size) as usize;
                let data = memory.data_mut(&mut caller);
                for i in 0..copy_len {
                    data[new_addr as usize + i] = data[old_addr as usize + i];
                }
                // Zero any extra space
                for i in copy_len..aligned_size as usize {
                    data[new_addr as usize + i] = 0;
                }
                // Write size header
                data[block_start as usize..block_start as usize + 4]
                    .copy_from_slice(&new_size.to_le_bytes());
                // Update allocation pointer
                data[alloc_ptr_addr as usize..mem_len as usize]
                    .copy_from_slice(&block_start.to_le_bytes());

                // Replace (a-addr u) with (a-addr2 ior)
                data[(sp + 4) as usize..(sp + 8) as usize]
                    .copy_from_slice(&(new_addr as i32).to_le_bytes());
                data[sp as usize..sp as usize + 4].copy_from_slice(&0i32.to_le_bytes());
                Ok(())
            },
        );
        self.register_host_primitive("RESIZE", false, func)?;

        Ok(())
    }

    /// D>S ( d -- n ) convert double to single (just drop high cell).
    fn register_d_to_s(&mut self) -> anyhow::Result<()> {
        // D>S just drops the high cell
        self.register_primitive("D>S", false, vec![IrOp::Drop])?;
        Ok(())
    }

    // -- Search-Order pending handlers --

    /// GET-CURRENT ( -- wid )
    fn do_get_current(&mut self) -> anyhow::Result<()> {
        let wid = self.dictionary.current_wid() as i32;
        self.push_data_stack(wid)
    }

    /// SET-CURRENT ( wid -- )
    fn do_set_current(&mut self) -> anyhow::Result<()> {
        let wid = self.pop_data_stack()? as u32;
        self.dictionary.set_current_wid(wid);
        Ok(())
    }

    /// SEARCH-WORDLIST ( c-addr u wid -- 0 | xt 1 | xt -1 )
    fn do_search_wordlist(&mut self) -> anyhow::Result<()> {
        let wid = self.pop_data_stack()? as u32;
        let u = self.pop_data_stack()? as u32;
        let addr = self.pop_data_stack()? as u32;

        let data = self.memory.data(&self.store);
        let name = String::from_utf8_lossy(&data[addr as usize..(addr + u) as usize]).to_string();

        if let Some((_word_addr, word_id, is_imm)) = self.dictionary.find_in_wid(&name, wid) {
            self.push_data_stack(word_id.0 as i32)?;
            self.push_data_stack(if is_imm { 1 } else { -1 })?;
        } else {
            self.push_data_stack(0)?;
        }
        Ok(())
    }

    /// Register Search-Order word set words.
    fn register_search_order(&mut self) -> anyhow::Result<()> {
        // FORTH-WORDLIST ( -- wid )
        self.register_primitive("FORTH-WORDLIST", false, vec![IrOp::PushI32(1)])?;

        // GET-CURRENT ( -- wid )
        // Returns the current compilation wordlist from pending mechanism
        let pending = Arc::clone(&self.pending_define);
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |_caller, _params, _results| {
                pending.lock().unwrap().push(20); // GET-CURRENT action
                Ok(())
            },
        );
        self.register_host_primitive("GET-CURRENT", false, func)?;

        // SET-CURRENT ( wid -- )
        let pending = Arc::clone(&self.pending_define);
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |_caller, _params, _results| {
                pending.lock().unwrap().push(21); // SET-CURRENT action
                Ok(())
            },
        );
        self.register_host_primitive("SET-CURRENT", false, func)?;

        // WORDLIST ( -- wid ) — directly allocates and pushes
        {
            let nw = Arc::clone(&self.next_wid);
            let memory = self.memory;
            let dsp_g = self.dsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _params, _results| {
                    let mut nw_val = nw.lock().unwrap();
                    let wid = *nw_val;
                    *nw_val += 1;
                    drop(nw_val);
                    let sp = dsp_g.get(&mut caller).unwrap_i32() as u32;
                    let new_sp = sp - CELL_SIZE;
                    let data = memory.data_mut(&mut caller);
                    data[new_sp as usize..new_sp as usize + 4]
                        .copy_from_slice(&(wid as i32).to_le_bytes());
                    dsp_g.set(&mut caller, Val::I32(new_sp as i32))?;
                    Ok(())
                },
            );
            self.register_host_primitive("WORDLIST", false, func)?;
        }

        // GET-ORDER ( -- widn ... wid1 n ) — directly pushes search order
        {
            let so = Arc::clone(&self.search_order);
            let memory = self.memory;
            let dsp_g = self.dsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _params, _results| {
                    let order = so.lock().unwrap().clone();
                    let n = order.len() as u32;
                    let sp = dsp_g.get(&mut caller).unwrap_i32() as u32;
                    let new_sp = sp - (n + 1) * CELL_SIZE;
                    let data = memory.data_mut(&mut caller);
                    // wid1 (top of search order) = closest to n on stack
                    // widn (bottom of search order) = deepest on stack
                    for (i, &wid) in order.iter().enumerate() {
                        let addr = (new_sp + CELL_SIZE + i as u32 * CELL_SIZE) as usize;
                        data[addr..addr + 4].copy_from_slice(&(wid as i32).to_le_bytes());
                    }
                    data[new_sp as usize..new_sp as usize + 4]
                        .copy_from_slice(&(n as i32).to_le_bytes());
                    dsp_g.set(&mut caller, Val::I32(new_sp as i32))?;
                    Ok(())
                },
            );
            self.register_host_primitive("GET-ORDER", false, func)?;
        }

        // SET-ORDER ( widn ... wid1 n -- ) — directly pops and sets search order
        {
            let so = Arc::clone(&self.search_order);
            let memory = self.memory;
            let dsp_g = self.dsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _params, _results| {
                    let sp = dsp_g.get(&mut caller).unwrap_i32() as u32;
                    let data = memory.data(&caller);
                    let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                    let n = i32::from_le_bytes(b);

                    if n == -1 {
                        *so.lock().unwrap() = vec![1];
                        dsp_g.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))?;
                    } else {
                        let n = n as u32;
                        let mut order = Vec::new();
                        // wid1 is just above n on stack, widn is deepest
                        for i in 0..n {
                            let addr = (sp + CELL_SIZE + i * CELL_SIZE) as usize;
                            let data = memory.data(&caller);
                            let b: [u8; 4] = data[addr..addr + 4].try_into().unwrap();
                            order.push(u32::from_le_bytes(b));
                        }
                        *so.lock().unwrap() = order;
                        dsp_g.set(&mut caller, Val::I32((sp + (1 + n) * CELL_SIZE) as i32))?;
                    }
                    Ok(())
                },
            );
            self.register_host_primitive("SET-ORDER", false, func)?;
        }

        // ONLY ( -- ) set minimum search order
        {
            let so = Arc::clone(&self.search_order);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |_caller, _params, _results| {
                    *so.lock().unwrap() = vec![1];
                    Ok(())
                },
            );
            self.register_host_primitive("ONLY", false, func)?;
        }

        // ALSO ( -- ) duplicate top of search order
        {
            let so = Arc::clone(&self.search_order);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |_caller, _params, _results| {
                    let mut order = so.lock().unwrap();
                    if let Some(&top) = order.first() {
                        order.insert(0, top);
                    }
                    Ok(())
                },
            );
            self.register_host_primitive("ALSO", false, func)?;
        }

        // PREVIOUS ( -- ) remove top of search order
        {
            let so = Arc::clone(&self.search_order);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |_caller, _params, _results| {
                    let mut order = so.lock().unwrap();
                    if !order.is_empty() {
                        order.remove(0);
                    }
                    Ok(())
                },
            );
            self.register_host_primitive("PREVIOUS", false, func)?;
        }

        // DEFINITIONS ( -- ) set compilation wordlist to top of search order
        {
            let so = Arc::clone(&self.search_order);
            let pending = Arc::clone(&self.pending_define);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |_caller, _params, _results| {
                    let order = so.lock().unwrap();
                    if order.first().is_some() {
                        // Use pending to set current_wid (needs dictionary access)
                        drop(order);
                        pending.lock().unwrap().push(33);
                    }
                    Ok(())
                },
            );
            self.register_host_primitive("DEFINITIONS", false, func)?;
        }

        // FORTH ( -- ) replace top of search order with FORTH wordlist
        {
            let so = Arc::clone(&self.search_order);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |_caller, _params, _results| {
                    let mut order = so.lock().unwrap();
                    if !order.is_empty() {
                        order[0] = 1;
                    } else {
                        order.push(1);
                    }
                    Ok(())
                },
            );
            self.register_host_primitive("FORTH", false, func)?;
        }

        // SEARCH-WORDLIST ( c-addr u wid -- 0 | xt 1 | xt -1 )
        let pending = Arc::clone(&self.pending_define);
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |_caller, _params, _results| {
                pending.lock().unwrap().push(25); // SEARCH-WORDLIST action
                Ok(())
            },
        );
        self.register_host_primitive("SEARCH-WORDLIST", false, func)?;

        Ok(())
    }

    /// Register N>R and NR> for the Programming-Tools word set.
    fn register_n_to_r(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;
        let rsp = self.rsp;

        // N>R ( xn..x1 n -- ; R: -- x1..xn n )
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let n = i32::from_le_bytes(b) as u32;

                let mut rsp_val = rsp.get(&mut caller).unwrap_i32() as u32;

                // Move n items from data stack to return stack, plus n itself
                // Data stack: x1(deepest)..xn(just below n), n(top)
                // Need to push x1 first (deepest on R), then x2, ..., xn, then n
                let items_base = sp + 4; // past n
                for i in (0..n).rev() {
                    let addr = (items_base + i * 4) as usize;
                    let data = memory.data(&caller);
                    let val = i32::from_le_bytes(data[addr..addr + 4].try_into().unwrap());
                    rsp_val -= 4;
                    let data = memory.data_mut(&mut caller);
                    data[rsp_val as usize..rsp_val as usize + 4]
                        .copy_from_slice(&val.to_le_bytes());
                }
                // Push n to return stack
                rsp_val -= 4;
                let data = memory.data_mut(&mut caller);
                data[rsp_val as usize..rsp_val as usize + 4]
                    .copy_from_slice(&(n as i32).to_le_bytes());
                rsp.set(&mut caller, Val::I32(rsp_val as i32))?;

                // Pop n+1 items from data stack
                let new_sp = sp + (n + 1) * 4;
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );
        self.register_host_primitive("N>R", false, func)?;

        // NR> ( -- xn..x1 n ; R: x1..xn n -- )
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let mut rsp_val = rsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                // Pop n from return stack
                let b: [u8; 4] = data[rsp_val as usize..rsp_val as usize + 4]
                    .try_into()
                    .unwrap();
                let n = i32::from_le_bytes(b) as u32;
                rsp_val += 4;

                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                // Make space for n+1 items on data stack
                let new_sp = sp - (n + 1) * 4;

                // Pop n items from return stack to data stack
                // R-stack has x1(deepest)..xn(top after n)
                // Data stack needs xn..x1 n (with n on top)
                for i in 0..n {
                    let data = memory.data(&caller);
                    let val = i32::from_le_bytes(
                        data[rsp_val as usize..rsp_val as usize + 4]
                            .try_into()
                            .unwrap(),
                    );
                    rsp_val += 4;
                    let addr = (new_sp + 4 + i * 4) as usize;
                    let data = memory.data_mut(&mut caller);
                    data[addr..addr + 4].copy_from_slice(&val.to_le_bytes());
                }
                rsp.set(&mut caller, Val::I32(rsp_val as i32))?;

                // Push n on top of data stack
                let data = memory.data_mut(&mut caller);
                data[new_sp as usize..new_sp as usize + 4]
                    .copy_from_slice(&(n as i32).to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );
        self.register_host_primitive("NR>", false, func)?;

        Ok(())
    }

    /// Register UNESCAPE, SUBSTITUTE, REPLACES for the String word set.
    fn register_string_substitution(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        // UNESCAPE ( c-addr1 u1 c-addr2 -- c-addr2 u2 )
        // Copy string escaping each % as %%
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let dest = u32::from_le_bytes(b);
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let u1 = u32::from_le_bytes(b);
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let src = u32::from_le_bytes(b);

                // Read source
                let src_bytes: Vec<u8> = data[src as usize..(src + u1) as usize].to_vec();

                // Escape: each % becomes %%
                let mut result = Vec::with_capacity(u1 as usize * 2);
                for &ch in &src_bytes {
                    if ch == b'%' {
                        result.push(b'%');
                        result.push(b'%');
                    } else {
                        result.push(ch);
                    }
                }

                // Write to dest
                let u2 = result.len() as u32;
                let data = memory.data_mut(&mut caller);
                data[dest as usize..(dest + u2) as usize].copy_from_slice(&result);

                // Pop 3, push 2: net sp + 4
                let new_sp = sp + 4;
                let data = memory.data_mut(&mut caller);
                data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                    .copy_from_slice(&(dest as i32).to_le_bytes());
                data[new_sp as usize..(new_sp + 4) as usize]
                    .copy_from_slice(&(u2 as i32).to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );
        self.register_host_primitive("UNESCAPE", false, func)?;

        // REPLACES ( c-addr1 u1 c-addr2 u2 -- )
        // Define substitution: name (c-addr2 u2) → replacement (c-addr1 u1)
        let subs = Arc::clone(&self.substitutions);
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                // Stack: u2(sp), c-addr2(sp+4), u1(sp+8), c-addr1(sp+12)
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let u2 = u32::from_le_bytes(b);
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let name_addr = u32::from_le_bytes(b);
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let u1 = u32::from_le_bytes(b);
                let b: [u8; 4] = data[(sp + 12) as usize..(sp + 16) as usize]
                    .try_into()
                    .unwrap();
                let repl_addr = u32::from_le_bytes(b);

                let name =
                    String::from_utf8_lossy(&data[name_addr as usize..(name_addr + u2) as usize])
                        .to_ascii_uppercase();

                // Copy replacement string to Rust-side storage (WASM addresses are transient)
                let repl_bytes = data[repl_addr as usize..(repl_addr + u1) as usize].to_vec();
                subs.lock().unwrap().insert(name, repl_bytes);

                // Pop 4 items
                let new_sp = sp + 16;
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );
        self.register_host_primitive("REPLACES", false, func)?;

        // SUBSTITUTE ( c-addr1 u1 c-addr2 u2 -- c-addr2 u2 n )
        // Replace %name% patterns, %% → %
        let subs = Arc::clone(&self.substitutions);
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                // Stack: u2/capacity(sp), c-addr2/dest(sp+4), u1(sp+8), c-addr1(sp+12)
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let capacity = u32::from_le_bytes(b) as usize;
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let dest = u32::from_le_bytes(b);
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let u1 = u32::from_le_bytes(b);
                let b: [u8; 4] = data[(sp + 12) as usize..(sp + 16) as usize]
                    .try_into()
                    .unwrap();
                let src = u32::from_le_bytes(b);

                let src_bytes: Vec<u8> = data[src as usize..(src + u1) as usize].to_vec();

                let subs_map = subs.lock().unwrap();
                let mut result = Vec::with_capacity(capacity);
                let mut sub_count: i32 = 0;
                let mut i = 0;
                let mut overflow = false;

                while i < src_bytes.len() {
                    if src_bytes[i] == b'%' {
                        if i + 1 < src_bytes.len() && src_bytes[i + 1] == b'%' {
                            // %% → %
                            result.push(b'%');
                            i += 2;
                        } else {
                            // Look for closing %
                            if let Some(end) = src_bytes[i + 1..].iter().position(|&c| c == b'%') {
                                let name_bytes = &src_bytes[i + 1..i + 1 + end];
                                let name = String::from_utf8_lossy(name_bytes).to_ascii_uppercase();
                                if let Some(repl_bytes) = subs_map.get(&name) {
                                    // Substitute
                                    let avail = capacity - result.len();
                                    let copy_len = repl_bytes.len().min(avail);
                                    result.extend_from_slice(&repl_bytes[..copy_len]);
                                    sub_count += 1;
                                } else {
                                    // Unknown name: keep %name% as-is
                                    let avail = capacity - result.len();
                                    let chunk = &src_bytes[i..i + 1 + end + 1];
                                    let copy_len = chunk.len().min(avail);
                                    result.extend_from_slice(&chunk[..copy_len]);
                                }
                                i += 1 + end + 1; // skip past closing %
                            } else {
                                // No closing % — copy rest as-is
                                let avail = capacity - result.len();
                                let chunk = &src_bytes[i..];
                                let copy_len = chunk.len().min(avail);
                                result.extend_from_slice(&chunk[..copy_len]);
                                i = src_bytes.len();
                            }
                        }
                    } else {
                        result.push(src_bytes[i]);
                        i += 1;
                    }
                }
                drop(subs_map);

                // Check overflow
                if result.len() > capacity {
                    overflow = true;
                    result.truncate(capacity);
                }
                if overflow {
                    sub_count = if sub_count > 0 { -sub_count } else { -1 };
                }

                // Write result to dest
                let u2 = result.len() as u32;
                let data = memory.data_mut(&mut caller);
                data[dest as usize..(dest + u2) as usize].copy_from_slice(&result);

                // Pop 4, push 3: net sp + 4
                let new_sp = sp + 4;
                let data = memory.data_mut(&mut caller);
                data[(new_sp + 8) as usize..(new_sp + 12) as usize]
                    .copy_from_slice(&(dest as i32).to_le_bytes());
                data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                    .copy_from_slice(&(u2 as i32).to_le_bytes());
                data[new_sp as usize..(new_sp + 4) as usize]
                    .copy_from_slice(&sub_count.to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );
        self.register_host_primitive("SUBSTITUTE", false, func)?;

        Ok(())
    }

    /// M*/ ( d n1 n2 -- d ) multiply d by n1, divide by n2.
    fn register_m_star_slash(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                // Stack: n2(sp), n1(sp+4), d-hi(sp+8), d-lo(sp+12)
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let n2 = i32::from_le_bytes(b) as i128;
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let n1 = i32::from_le_bytes(b) as i128;
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let d_hi = i32::from_le_bytes(b) as i64;
                let b: [u8; 4] = data[(sp + 12) as usize..(sp + 16) as usize]
                    .try_into()
                    .unwrap();
                let d_lo = u32::from_le_bytes(b) as i64;
                let d = ((d_hi << 32) | (d_lo & 0xFFFF_FFFF)) as i128;

                if n2 == 0 {
                    return Err(wasmtime::Error::msg("M*/: division by zero"));
                }

                // Symmetric (truncating) division to match WAFER's / behavior
                let product = d * n1;
                let quot = product / n2;

                let result = quot as i64;
                let lo = result as i32;
                let hi = (result >> 32) as i32;
                // Pop 4, push 2: net sp + 8
                let new_sp = sp + 8;
                let data = memory.data_mut(&mut caller);
                data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                    .copy_from_slice(&lo.to_le_bytes());
                data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&hi.to_le_bytes());
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("M*/", false, func)?;
        Ok(())
    }

    /// 2CONSTANT ( x1 x2 "name" -- ) define a double-cell constant.
    fn define_2constant(&mut self) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("2CONSTANT: expected name"))?;
        let hi = self.pop_data_stack()?;
        let lo = self.pop_data_stack()?;

        let word_id = self.dictionary.create(&name, false)?;
        self.dictionary.reveal();

        let ir = vec![IrOp::PushI32(lo), IrOp::PushI32(hi)];
        self.ir_bodies.insert(word_id, ir.clone());
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
        };
        let compiled = compile_word(&name, &ir, &config)
            .map_err(|e| anyhow::anyhow!("2CONSTANT codegen: {e}"))?;
        self.instantiate_and_install(&compiled, word_id)?;
        self.sync_word_lookup(&name, word_id, false);
        Ok(())
    }

    /// 2VARIABLE ( "name" -- ) define a double-cell variable.
    fn define_2variable(&mut self) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("2VARIABLE: expected name"))?;

        self.refresh_user_here();
        let addr = self.user_here;
        // Initialize 8 bytes to zero
        let data = self.memory.data_mut(&mut self.store);
        data[addr as usize..addr as usize + 8].copy_from_slice(&[0u8; 8]);
        self.user_here += 8;
        self.sync_here_cell();

        let word_id = self.dictionary.create(&name, false)?;
        self.dictionary.reveal();

        let ir = vec![IrOp::PushI32(addr as i32)];
        self.ir_bodies.insert(word_id, ir.clone());
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
        };
        let compiled = compile_word(&name, &ir, &config)
            .map_err(|e| anyhow::anyhow!("2VARIABLE codegen: {e}"))?;
        self.instantiate_and_install(&compiled, word_id)?;
        self.word_pfa_map.insert(word_id.0, addr);
        if let Some(ref shared) = self.word_pfa_map_shared {
            shared.lock().unwrap().insert(word_id.0, addr);
        }
        self.sync_word_lookup(&name, word_id, false);
        Ok(())
    }

    /// 2VALUE ( x1 x2 "name" -- ) define a double-cell value.
    fn define_2value(&mut self) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("2VALUE: expected name"))?;
        let hi = self.pop_data_stack()?;
        let lo = self.pop_data_stack()?;

        self.refresh_user_here();
        let addr = self.user_here;
        let data = self.memory.data_mut(&mut self.store);
        data[addr as usize..addr as usize + 4].copy_from_slice(&lo.to_le_bytes());
        data[addr as usize + 4..addr as usize + 8].copy_from_slice(&hi.to_le_bytes());
        self.user_here += 8;
        self.sync_here_cell();

        let word_id = self.dictionary.create(&name, false)?;
        self.dictionary.reveal();

        // 2VALUE pushes two cells from the stored address
        // PFA @ PFA+4 @
        let ir = vec![
            IrOp::PushI32(addr as i32),
            IrOp::Fetch,
            IrOp::PushI32((addr + 4) as i32),
            IrOp::Fetch,
        ];
        self.ir_bodies.insert(word_id, ir.clone());
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
        };
        let compiled = compile_word(&name, &ir, &config)
            .map_err(|e| anyhow::anyhow!("2VALUE codegen: {e}"))?;
        self.instantiate_and_install(&compiled, word_id)?;
        self.word_pfa_map.insert(word_id.0, addr);
        if let Some(ref shared) = self.word_pfa_map_shared {
            shared.lock().unwrap().insert(word_id.0, addr);
        }
        self.two_value_words.insert(word_id.0);
        self.sync_word_lookup(&name, word_id, false);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // String word set
    // -----------------------------------------------------------------------

    /// SEARCH ( c-addr1 u1 c-addr2 u2 -- c-addr3 u3 flag ) search for substring.
    fn register_search(&mut self) -> anyhow::Result<()> {
        let memory = self.memory;
        let dsp = self.dsp;

        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _params, _results| {
                let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                let data = memory.data(&caller);
                // Stack: u2(sp), c-addr2(sp+4), u1(sp+8), c-addr1(sp+12)
                let b: [u8; 4] = data[sp as usize..sp as usize + 4].try_into().unwrap();
                let u2 = i32::from_le_bytes(b) as usize;
                let b: [u8; 4] = data[(sp + 4) as usize..(sp + 8) as usize]
                    .try_into()
                    .unwrap();
                let addr2 = u32::from_le_bytes(b) as usize;
                let b: [u8; 4] = data[(sp + 8) as usize..(sp + 12) as usize]
                    .try_into()
                    .unwrap();
                let u1 = i32::from_le_bytes(b) as usize;
                let b: [u8; 4] = data[(sp + 12) as usize..(sp + 16) as usize]
                    .try_into()
                    .unwrap();
                let addr1 = u32::from_le_bytes(b) as usize;

                let mem_len = data.len();

                // If needle is empty, always found at start
                if u2 == 0 {
                    // Return (c-addr1 u1 true)
                    // Pop 4, push 3: net sp + 4
                    let new_sp = sp + 4;
                    let data = memory.data_mut(&mut caller);
                    data[(new_sp + 8) as usize..(new_sp + 12) as usize]
                        .copy_from_slice(&(addr1 as i32).to_le_bytes());
                    data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                        .copy_from_slice(&(u1 as i32).to_le_bytes());
                    data[new_sp as usize..new_sp as usize + 4]
                        .copy_from_slice(&(-1i32).to_le_bytes());
                    dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                    return Ok(());
                }

                if u2 > u1 {
                    // Can't find, return (c-addr1 u1 false)
                    let new_sp = sp + 4;
                    let data = memory.data_mut(&mut caller);
                    data[(new_sp + 8) as usize..(new_sp + 12) as usize]
                        .copy_from_slice(&(addr1 as i32).to_le_bytes());
                    data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                        .copy_from_slice(&(u1 as i32).to_le_bytes());
                    data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&0i32.to_le_bytes());
                    dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                    return Ok(());
                }

                // Search for needle in haystack
                let mut found = false;
                let mut found_offset = 0usize;
                for i in 0..=(u1 - u2) {
                    let mut matched = true;
                    for j in 0..u2 {
                        let h = if addr1 + i + j < mem_len {
                            data[addr1 + i + j]
                        } else {
                            0
                        };
                        let n = if addr2 + j < mem_len {
                            data[addr2 + j]
                        } else {
                            0
                        };
                        if h != n {
                            matched = false;
                            break;
                        }
                    }
                    if matched {
                        found = true;
                        found_offset = i;
                        break;
                    }
                }

                let new_sp = sp + 4;
                let data = memory.data_mut(&mut caller);
                if found {
                    let new_addr = (addr1 + found_offset) as i32;
                    let new_len = (u1 - found_offset) as i32;
                    data[(new_sp + 8) as usize..(new_sp + 12) as usize]
                        .copy_from_slice(&new_addr.to_le_bytes());
                    data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                        .copy_from_slice(&new_len.to_le_bytes());
                    data[new_sp as usize..new_sp as usize + 4]
                        .copy_from_slice(&(-1i32).to_le_bytes());
                } else {
                    data[(new_sp + 8) as usize..(new_sp + 12) as usize]
                        .copy_from_slice(&(addr1 as i32).to_le_bytes());
                    data[(new_sp + 4) as usize..(new_sp + 8) as usize]
                        .copy_from_slice(&(u1 as i32).to_le_bytes());
                    data[new_sp as usize..new_sp as usize + 4].copy_from_slice(&0i32.to_le_bytes());
                }
                dsp.set(&mut caller, Val::I32(new_sp as i32))?;
                Ok(())
            },
        );

        self.register_host_primitive("SEARCH", false, func)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Floating-Point word set
    // -----------------------------------------------------------------------

    /// Register all floating-point words.
    fn register_float_words(&mut self) -> anyhow::Result<()> {
        self.register_float_stack_ops()?;
        self.register_float_arithmetic()?;
        self.register_float_comparisons()?;
        self.register_float_memory()?;
        self.register_float_conversions()?;
        self.register_float_trig()?;
        self.register_float_exp_log()?;
        self.register_float_hyperbolic()?;
        self.register_float_io()?;
        self.register_float_misc()?;
        Ok(())
    }

    /// Helper: create a host function that takes no data-stack args
    /// and operates on the float stack via fsp/memory closures.
    /// Pattern for unary float ops: pop one float, compute, push result.
    fn register_float_unary(&mut self, name: &str, op: fn(f64) -> f64) -> anyhow::Result<()> {
        let memory = self.memory;
        let fsp = self.fsp;
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _, _| {
                let sp = fsp.get(&mut caller).unwrap_i32() as u32;
                if sp >= FLOAT_STACK_TOP {
                    return Err(wasmtime::Error::msg("float stack underflow"));
                }
                let mem = memory.data(&caller);
                let bytes: [u8; 8] = mem[sp as usize..sp as usize + 8].try_into().unwrap();
                let a = f64::from_le_bytes(bytes);
                let result = op(a);
                let mem = memory.data_mut(&mut caller);
                mem[sp as usize..sp as usize + 8].copy_from_slice(&result.to_le_bytes());
                Ok(())
            },
        );
        self.register_host_primitive(name, false, func)?;
        Ok(())
    }

    /// Pattern for binary float ops: pop two floats (b then a), compute, push result.
    fn register_float_binary(&mut self, name: &str, op: fn(f64, f64) -> f64) -> anyhow::Result<()> {
        let memory = self.memory;
        let fsp = self.fsp;
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _, _| {
                let sp = fsp.get(&mut caller).unwrap_i32() as u32;
                if sp + 8 >= FLOAT_STACK_TOP {
                    return Err(wasmtime::Error::msg("float stack underflow"));
                }
                let mem = memory.data(&caller);
                let b_bytes: [u8; 8] = mem[sp as usize..sp as usize + 8].try_into().unwrap();
                let a_bytes: [u8; 8] = mem[sp as usize + 8..sp as usize + 16].try_into().unwrap();
                let b = f64::from_le_bytes(b_bytes);
                let a = f64::from_le_bytes(a_bytes);
                let result = op(a, b);
                let new_sp = sp + 8;
                fsp.set(&mut caller, Val::I32(new_sp as i32)).unwrap();
                let mem = memory.data_mut(&mut caller);
                mem[new_sp as usize..new_sp as usize + 8].copy_from_slice(&result.to_le_bytes());
                Ok(())
            },
        );
        self.register_host_primitive(name, false, func)?;
        Ok(())
    }

    /// Float stack manipulation words.
    fn register_float_stack_ops(&mut self) -> anyhow::Result<()> {
        self.register_primitive("FDROP", false, vec![IrOp::FDrop])?;
        self.register_primitive("FDUP", false, vec![IrOp::FDup])?;
        self.register_primitive("FSWAP", false, vec![IrOp::FSwap])?;
        self.register_primitive("FOVER", false, vec![IrOp::FOver])?;

        // FROT ( F: r1 r2 r3 -- r2 r3 r1 )
        {
            let memory = self.memory;
            let fsp = self.fsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = fsp.get(&mut caller).unwrap_i32() as u32;
                    let mem = memory.data(&caller);
                    let c: [u8; 8] = mem[sp as usize..sp as usize + 8].try_into().unwrap();
                    let b: [u8; 8] = mem[sp as usize + 8..sp as usize + 16].try_into().unwrap();
                    let a: [u8; 8] = mem[sp as usize + 16..sp as usize + 24].try_into().unwrap();
                    let mem = memory.data_mut(&mut caller);
                    mem[sp as usize..sp as usize + 8].copy_from_slice(&a);
                    mem[sp as usize + 8..sp as usize + 16].copy_from_slice(&c);
                    mem[sp as usize + 16..sp as usize + 24].copy_from_slice(&b);
                    Ok(())
                },
            );
            self.register_host_primitive("FROT", false, func)?;
        }

        // FDEPTH ( -- +n ) number of floats on the float stack, pushed onto DATA stack
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let fsp = self.fsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let fsp_val = fsp.get(&mut caller).unwrap_i32() as u32;
                    let depth = if fsp_val <= FLOAT_STACK_TOP {
                        ((FLOAT_STACK_TOP - fsp_val) / FLOAT_SIZE) as i32
                    } else {
                        0
                    };
                    // Push onto data stack
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let new_sp = sp - CELL_SIZE;
                    dsp.set(&mut caller, Val::I32(new_sp as i32)).unwrap();
                    let mem = memory.data_mut(&mut caller);
                    mem[new_sp as usize..new_sp as usize + 4].copy_from_slice(&depth.to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("FDEPTH", false, func)?;
        }

        self.register_primitive("FNIP", false, vec![IrOp::FSwap, IrOp::FDrop])?;
        self.register_primitive("FTUCK", false, vec![IrOp::FSwap, IrOp::FOver])?;

        Ok(())
    }

    /// Float arithmetic words.
    fn register_float_arithmetic(&mut self) -> anyhow::Result<()> {
        self.register_primitive("F+", false, vec![IrOp::FAdd])?;
        self.register_primitive("F-", false, vec![IrOp::FSub])?;
        self.register_primitive("F*", false, vec![IrOp::FMul])?;
        self.register_primitive("F/", false, vec![IrOp::FDiv])?;
        self.register_primitive("FNEGATE", false, vec![IrOp::FNegate])?;
        self.register_primitive("FABS", false, vec![IrOp::FAbs])?;
        self.register_primitive("FMAX", false, vec![IrOp::FMax])?;
        self.register_primitive("FMIN", false, vec![IrOp::FMin])?;
        self.register_primitive("FSQRT", false, vec![IrOp::FSqrt])?;
        self.register_primitive("FLOOR", false, vec![IrOp::FFloor])?;
        self.register_primitive("FROUND", false, vec![IrOp::FRound])?;
        self.register_float_binary("F**", f64::powf)?;
        Ok(())
    }

    /// Float comparison words. Results go on the DATA stack.
    fn register_float_comparisons(&mut self) -> anyhow::Result<()> {
        self.register_primitive("F0=", false, vec![IrOp::FZeroEq])?;
        self.register_primitive("F0<", false, vec![IrOp::FZeroLt])?;
        self.register_primitive("F=", false, vec![IrOp::FEq])?;
        self.register_primitive("F<", false, vec![IrOp::FLt])?;

        // F~ ( -- flag ) ( F: r1 r2 r3 -- ) approximate float comparison
        // If r3 > 0: true if |r1-r2| < r3
        // If r3 = 0: true if r1 and r2 are exactly equal (bitwise)
        // If r3 < 0: true if |r1-r2| < |r3|*(|r1|+|r2|)
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let fsp = self.fsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = fsp.get(&mut caller).unwrap_i32() as u32;
                    let mem = memory.data(&caller);
                    let r3_bytes: [u8; 8] = mem[sp as usize..sp as usize + 8].try_into().unwrap();
                    let r2_bytes: [u8; 8] =
                        mem[sp as usize + 8..sp as usize + 16].try_into().unwrap();
                    let r1_bytes: [u8; 8] =
                        mem[sp as usize + 16..sp as usize + 24].try_into().unwrap();
                    let r3 = f64::from_le_bytes(r3_bytes);
                    let r2 = f64::from_le_bytes(r2_bytes);
                    let r1 = f64::from_le_bytes(r1_bytes);
                    fsp.set(&mut caller, Val::I32((sp + 24) as i32)).unwrap();

                    let result = if r3 > 0.0 {
                        (r1 - r2).abs() < r3
                    } else if r3 == 0.0 {
                        r1.to_bits() == r2.to_bits()
                    } else {
                        // r3 < 0: relative comparison
                        (r1 - r2).abs() < r3.abs() * (r1.abs() + r2.abs())
                    };

                    let flag: i32 = if result { -1 } else { 0 };
                    let dsp_val = dsp.get(&mut caller).unwrap_i32() as u32;
                    let new_dsp = dsp_val
                        .checked_sub(CELL_SIZE)
                        .ok_or_else(|| wasmtime::Error::msg("data stack overflow in F~"))?;
                    dsp.set(&mut caller, Val::I32(new_dsp as i32)).unwrap();
                    let mem = memory.data_mut(&mut caller);
                    mem[new_dsp as usize..new_dsp as usize + 4]
                        .copy_from_slice(&flag.to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("F~", false, func)?;
        }

        Ok(())
    }

    /// Float memory words.
    fn register_float_memory(&mut self) -> anyhow::Result<()> {
        self.register_primitive("F@", false, vec![IrOp::FetchFloat])?;
        self.register_primitive("F!", false, vec![IrOp::StoreFloat])?;

        // FLOAT+ ( f-addr1 -- f-addr2 ) add float size to address
        self.register_primitive(
            "FLOAT+",
            false,
            vec![IrOp::PushI32(FLOAT_SIZE as i32), IrOp::Add],
        )?;

        // FLOATS ( n1 -- n2 ) multiply by float size
        self.register_primitive(
            "FLOATS",
            false,
            vec![IrOp::PushI32(FLOAT_SIZE as i32), IrOp::Mul],
        )?;

        // FALIGNED ( addr -- f-addr ) align to float boundary (8 bytes)
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let mem = memory.data(&caller);
                    let b: [u8; 4] = mem[sp as usize..sp as usize + 4].try_into().unwrap();
                    let addr = u32::from_le_bytes(b);
                    let aligned = (addr + 7) & !7;
                    let mem = memory.data_mut(&mut caller);
                    mem[sp as usize..sp as usize + 4].copy_from_slice(&aligned.to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("FALIGNED", false, func)?;
        }

        // SFLOATS ( n -- n*sfloat_size ) single-float size (same as FLOATS for us)
        self.register_primitive(
            "SFLOATS",
            false,
            vec![IrOp::PushI32(FLOAT_SIZE as i32), IrOp::Mul],
        )?;

        // SFLOAT+ ( addr -- addr+sfloat_size )
        self.register_primitive(
            "SFLOAT+",
            false,
            vec![IrOp::PushI32(FLOAT_SIZE as i32), IrOp::Add],
        )?;

        // DFLOATS ( n -- n*dfloat_size )
        self.register_primitive(
            "DFLOATS",
            false,
            vec![IrOp::PushI32(FLOAT_SIZE as i32), IrOp::Mul],
        )?;

        // DFLOAT+ ( addr -- addr+dfloat_size )
        self.register_primitive(
            "DFLOAT+",
            false,
            vec![IrOp::PushI32(FLOAT_SIZE as i32), IrOp::Add],
        )?;

        Ok(())
    }

    /// Float conversion words.
    fn register_float_conversions(&mut self) -> anyhow::Result<()> {
        // D>F ( d -- ) ( F: -- r ) convert double-cell integer to float
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let fsp = self.fsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let mem = memory.data(&caller);
                    // Double-cell: hi on top, lo below
                    let hi_bytes: [u8; 4] = mem[sp as usize..sp as usize + 4].try_into().unwrap();
                    let lo_bytes: [u8; 4] =
                        mem[sp as usize + 4..sp as usize + 8].try_into().unwrap();
                    let hi = i32::from_le_bytes(hi_bytes);
                    let lo = i32::from_le_bytes(lo_bytes);
                    let d = ((hi as i64) << 32) | (lo as u32 as i64);
                    let f = d as f64;
                    // Pop two cells from data stack
                    dsp.set(&mut caller, Val::I32((sp + 8) as i32)).unwrap();
                    // Push onto float stack
                    let fsp_val = fsp.get(&mut caller).unwrap_i32() as u32;
                    let new_fsp = fsp_val - FLOAT_SIZE;
                    fsp.set(&mut caller, Val::I32(new_fsp as i32)).unwrap();
                    let mem = memory.data_mut(&mut caller);
                    mem[new_fsp as usize..new_fsp as usize + 8].copy_from_slice(&f.to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("D>F", false, func)?;
        }

        // F>D ( -- d ) ( F: r -- ) convert float to double-cell integer
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let fsp = self.fsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    // Pop from float stack
                    let fsp_val = fsp.get(&mut caller).unwrap_i32() as u32;
                    let mem = memory.data(&caller);
                    let bytes: [u8; 8] = mem[fsp_val as usize..fsp_val as usize + 8]
                        .try_into()
                        .unwrap();
                    let f = f64::from_le_bytes(bytes);
                    fsp.set(&mut caller, Val::I32((fsp_val + FLOAT_SIZE) as i32))
                        .unwrap();
                    // Convert to i64
                    let d = f as i64;
                    let lo = d as i32;
                    let hi = (d >> 32) as i32;
                    // Push lo then hi onto data stack
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let new_sp = sp - 8; // two cells
                    dsp.set(&mut caller, Val::I32(new_sp as i32)).unwrap();
                    let mem = memory.data_mut(&mut caller);
                    mem[new_sp as usize + 4..new_sp as usize + 8]
                        .copy_from_slice(&lo.to_le_bytes());
                    mem[new_sp as usize..new_sp as usize + 4].copy_from_slice(&hi.to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("F>D", false, func)?;
        }

        self.register_primitive("S>F", false, vec![IrOp::StoF])?;
        self.register_primitive("F>S", false, vec![IrOp::FtoS])?;

        Ok(())
    }

    /// Trigonometric functions.
    fn register_float_trig(&mut self) -> anyhow::Result<()> {
        self.register_float_unary("FSIN", f64::sin)?;
        self.register_float_unary("FCOS", f64::cos)?;
        self.register_float_unary("FTAN", f64::tan)?;
        self.register_float_unary("FASIN", f64::asin)?;
        self.register_float_unary("FACOS", f64::acos)?;
        self.register_float_unary("FATAN", f64::atan)?;
        self.register_float_binary("FATAN2", f64::atan2)?;

        // FSINCOS ( F: r1 -- r2 r3 ) r2=sin(r1) r3=cos(r1)
        {
            let memory = self.memory;
            let fsp = self.fsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = fsp.get(&mut caller).unwrap_i32() as u32;
                    let mem = memory.data(&caller);
                    let bytes: [u8; 8] = mem[sp as usize..sp as usize + 8].try_into().unwrap();
                    let val = f64::from_le_bytes(bytes);
                    let sin_val = val.sin();
                    let cos_val = val.cos();
                    // Replace TOS with sin, push cos on top
                    // Result: sin deeper, cos on top
                    let new_sp = sp - 8; // one more item
                    if new_sp < FLOAT_STACK_BASE {
                        return Err(wasmtime::Error::msg("float stack overflow"));
                    }
                    fsp.set(&mut caller, Val::I32(new_sp as i32)).unwrap();
                    let mem = memory.data_mut(&mut caller);
                    mem[new_sp as usize + 8..new_sp as usize + 16]
                        .copy_from_slice(&sin_val.to_le_bytes());
                    mem[new_sp as usize..new_sp as usize + 8]
                        .copy_from_slice(&cos_val.to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("FSINCOS", false, func)?;
        }

        Ok(())
    }

    /// Exponential and logarithmic functions.
    fn register_float_exp_log(&mut self) -> anyhow::Result<()> {
        self.register_float_unary("FEXP", f64::exp)?;
        self.register_float_unary("FEXPM1", f64::exp_m1)?;
        self.register_float_unary("FLN", f64::ln)?;
        self.register_float_unary("FLNP1", f64::ln_1p)?;
        self.register_float_unary("FLOG", f64::log10)?;
        self.register_float_unary("FALOG", |x| 10.0_f64.powf(x))?;
        Ok(())
    }

    /// Hyperbolic functions.
    fn register_float_hyperbolic(&mut self) -> anyhow::Result<()> {
        self.register_float_unary("FSINH", f64::sinh)?;
        self.register_float_unary("FCOSH", f64::cosh)?;
        self.register_float_unary("FTANH", f64::tanh)?;
        self.register_float_unary("FASINH", f64::asinh)?;
        self.register_float_unary("FACOSH", f64::acosh)?;
        self.register_float_unary("FATANH", f64::atanh)?;
        Ok(())
    }

    /// Float I/O words.
    fn register_float_io(&mut self) -> anyhow::Result<()> {
        // F. ( F: r -- ) print float followed by space
        {
            let memory = self.memory;
            let fsp = self.fsp;
            let output = Arc::clone(&self.output);
            let precision = Arc::clone(&self.float_precision);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = fsp.get(&mut caller).unwrap_i32() as u32;
                    let mem = memory.data(&caller);
                    let bytes: [u8; 8] = mem[sp as usize..sp as usize + 8].try_into().unwrap();
                    let val = f64::from_le_bytes(bytes);
                    fsp.set(&mut caller, Val::I32((sp + 8) as i32)).unwrap();
                    let prec = *precision.lock().unwrap();
                    let s = format!("{val:.prec$} ");
                    output.lock().unwrap().push_str(&s);
                    Ok(())
                },
            );
            self.register_host_primitive("F.", false, func)?;
        }

        // FE. ( F: r -- ) print float in engineering notation
        {
            let memory = self.memory;
            let fsp = self.fsp;
            let output = Arc::clone(&self.output);
            let precision = Arc::clone(&self.float_precision);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = fsp.get(&mut caller).unwrap_i32() as u32;
                    let mem = memory.data(&caller);
                    let bytes: [u8; 8] = mem[sp as usize..sp as usize + 8].try_into().unwrap();
                    let val = f64::from_le_bytes(bytes);
                    fsp.set(&mut caller, Val::I32((sp + 8) as i32)).unwrap();
                    let prec = *precision.lock().unwrap();
                    let s = format_engineering(val, prec);
                    output.lock().unwrap().push_str(&s);
                    Ok(())
                },
            );
            self.register_host_primitive("FE.", false, func)?;
        }

        // FS. ( F: r -- ) print float in scientific notation
        {
            let memory = self.memory;
            let fsp = self.fsp;
            let output = Arc::clone(&self.output);
            let precision = Arc::clone(&self.float_precision);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = fsp.get(&mut caller).unwrap_i32() as u32;
                    let mem = memory.data(&caller);
                    let bytes: [u8; 8] = mem[sp as usize..sp as usize + 8].try_into().unwrap();
                    let val = f64::from_le_bytes(bytes);
                    fsp.set(&mut caller, Val::I32((sp + 8) as i32)).unwrap();
                    let prec = *precision.lock().unwrap();
                    let s = format!("{val:.prec$E} ");
                    output.lock().unwrap().push_str(&s);
                    Ok(())
                },
            );
            self.register_host_primitive("FS.", false, func)?;
        }

        // PRECISION ( -- u ) get current float output precision
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let precision = Arc::clone(&self.float_precision);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let prec = *precision.lock().unwrap() as i32;
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let new_sp = sp - CELL_SIZE;
                    dsp.set(&mut caller, Val::I32(new_sp as i32)).unwrap();
                    let mem = memory.data_mut(&mut caller);
                    mem[new_sp as usize..new_sp as usize + 4].copy_from_slice(&prec.to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("PRECISION", false, func)?;
        }

        // SET-PRECISION ( u -- ) set float output precision
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let precision = Arc::clone(&self.float_precision);
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let mem = memory.data(&caller);
                    let b: [u8; 4] = mem[sp as usize..sp as usize + 4].try_into().unwrap();
                    let n = i32::from_le_bytes(b) as usize;
                    dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))
                        .unwrap();
                    *precision.lock().unwrap() = n;
                    Ok(())
                },
            );
            self.register_host_primitive("SET-PRECISION", false, func)?;
        }

        // REPRESENT ( c-addr u -- n flag1 flag2 ) ( F: r -- )
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let fsp = self.fsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    // Read all values from memory first
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let fsp_val = fsp.get(&mut caller).unwrap_i32() as u32;
                    let (u, c_addr, val) = {
                        let mem = memory.data(&caller);
                        let u_bytes: [u8; 4] =
                            mem[sp as usize..sp as usize + 4].try_into().unwrap();
                        let addr_bytes: [u8; 4] =
                            mem[sp as usize + 4..sp as usize + 8].try_into().unwrap();
                        let u = i32::from_le_bytes(u_bytes) as usize;
                        let c_addr = u32::from_le_bytes(addr_bytes) as usize;
                        let f_bytes: [u8; 8] = mem[fsp_val as usize..fsp_val as usize + 8]
                            .try_into()
                            .unwrap();
                        (u, c_addr, f64::from_le_bytes(f_bytes))
                    };

                    // Update stack pointers: pop 2 data cells, pop 1 float
                    dsp.set(&mut caller, Val::I32((sp + 8) as i32)).unwrap();
                    fsp.set(&mut caller, Val::I32((fsp_val + FLOAT_SIZE) as i32))
                        .unwrap();

                    let (digits, exp, is_negative, is_valid) = represent_float(val, u);

                    // Store digits at c-addr, then push results
                    let digit_bytes = digits.as_bytes();
                    let copy_len = digit_bytes.len().min(u);
                    // Push n, flag1 (sign), flag2 (valid) onto data stack
                    let cur_sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let new_sp = cur_sp - 12;
                    dsp.set(&mut caller, Val::I32(new_sp as i32)).unwrap();
                    let mem = memory.data_mut(&mut caller);
                    mem[c_addr..c_addr + copy_len].copy_from_slice(&digit_bytes[..copy_len]);
                    // Bottom: n (exponent)
                    mem[new_sp as usize + 8..new_sp as usize + 12]
                        .copy_from_slice(&exp.to_le_bytes());
                    // Middle: flag1 (is_negative => true flag)
                    let sign_flag: i32 = if is_negative { -1 } else { 0 };
                    mem[new_sp as usize + 4..new_sp as usize + 8]
                        .copy_from_slice(&sign_flag.to_le_bytes());
                    // Top: flag2 (is_valid => true flag)
                    let valid_flag: i32 = if is_valid { -1 } else { 0 };
                    mem[new_sp as usize..new_sp as usize + 4]
                        .copy_from_slice(&valid_flag.to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("REPRESENT", false, func)?;
        }

        // >FLOAT ( c-addr u -- flag ) ( F: -- r | ) parse string as float
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let fsp = self.fsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let (u, c_addr, s_owned) = {
                        let mem = memory.data(&caller);
                        let u_bytes: [u8; 4] =
                            mem[sp as usize..sp as usize + 4].try_into().unwrap();
                        let addr_bytes: [u8; 4] =
                            mem[sp as usize + 4..sp as usize + 8].try_into().unwrap();
                        let u = i32::from_le_bytes(u_bytes) as usize;
                        let c_addr = u32::from_le_bytes(addr_bytes) as usize;
                        let s = std::str::from_utf8(&mem[c_addr..c_addr + u])
                            .unwrap_or("")
                            .to_string();
                        (u, c_addr, s)
                    };
                    let _ = (u, c_addr);
                    // Pop u and c-addr (2 cells), will push back 1 cell (flag)
                    dsp.set(&mut caller, Val::I32((sp + 4) as i32)).unwrap();

                    let result = parse_forth_float(&s_owned);

                    match result {
                        Some(f) => {
                            // Push float onto float stack
                            let fsp_val = fsp.get(&mut caller).unwrap_i32() as u32;
                            let new_fsp = fsp_val - FLOAT_SIZE;
                            fsp.set(&mut caller, Val::I32(new_fsp as i32)).unwrap();
                            let flag_sp = dsp.get(&mut caller).unwrap_i32() as u32;
                            let mem = memory.data_mut(&mut caller);
                            mem[new_fsp as usize..new_fsp as usize + 8]
                                .copy_from_slice(&f.to_le_bytes());
                            mem[flag_sp as usize..flag_sp as usize + 4]
                                .copy_from_slice(&(-1_i32).to_le_bytes());
                        }
                        None => {
                            let flag_sp = dsp.get(&mut caller).unwrap_i32() as u32;
                            let mem = memory.data_mut(&mut caller);
                            mem[flag_sp as usize..flag_sp as usize + 4]
                                .copy_from_slice(&0_i32.to_le_bytes());
                        }
                    }
                    Ok(())
                },
            );
            self.register_host_primitive(">FLOAT", false, func)?;
        }

        Ok(())
    }

    /// Miscellaneous float words: FVARIABLE, FCONSTANT, FVALUE, >FLOAT parsing.
    fn register_float_misc(&mut self) -> anyhow::Result<()> {
        // FVARIABLE, FCONSTANT, FVALUE are handled in interpret_token_immediate
        // as special tokens (like VARIABLE/CONSTANT/VALUE).

        // SF! ( sf-addr -- ) ( F: r -- ) store as single-precision float (f32)
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let fsp = self.fsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let fsp_val = fsp.get(&mut caller).unwrap_i32() as u32;
                    let (addr, f32_bytes) = {
                        let mem = memory.data(&caller);
                        let addr_bytes: [u8; 4] =
                            mem[sp as usize..sp as usize + 4].try_into().unwrap();
                        let addr = u32::from_le_bytes(addr_bytes) as usize;
                        let f_bytes: [u8; 8] = mem[fsp_val as usize..fsp_val as usize + 8]
                            .try_into()
                            .unwrap();
                        let val = f64::from_le_bytes(f_bytes);
                        (addr, (val as f32).to_le_bytes())
                    };
                    dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))
                        .unwrap();
                    fsp.set(&mut caller, Val::I32((fsp_val + FLOAT_SIZE) as i32))
                        .unwrap();
                    let mem = memory.data_mut(&mut caller);
                    mem[addr..addr + 4].copy_from_slice(&f32_bytes);
                    Ok(())
                },
            );
            self.register_host_primitive("SF!", false, func)?;
        }

        // SF@ ( sf-addr -- ) ( F: -- r ) fetch single-precision float (f32)
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let fsp = self.fsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let fsp_val = fsp.get(&mut caller).unwrap_i32() as u32;
                    let val = {
                        let mem = memory.data(&caller);
                        let addr_bytes: [u8; 4] =
                            mem[sp as usize..sp as usize + 4].try_into().unwrap();
                        let addr = u32::from_le_bytes(addr_bytes) as usize;
                        let f32_bytes: [u8; 4] = mem[addr..addr + 4].try_into().unwrap();
                        f32::from_le_bytes(f32_bytes) as f64
                    };
                    dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))
                        .unwrap();
                    let new_fsp = fsp_val - FLOAT_SIZE;
                    fsp.set(&mut caller, Val::I32(new_fsp as i32)).unwrap();
                    let mem = memory.data_mut(&mut caller);
                    mem[new_fsp as usize..new_fsp as usize + 8].copy_from_slice(&val.to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("SF@", false, func)?;
        }

        // DF! ( df-addr -- ) ( F: r -- ) same as F! (our floats are already f64)
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let fsp = self.fsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let fsp_val = fsp.get(&mut caller).unwrap_i32() as u32;
                    let (addr, float_bytes) = {
                        let mem = memory.data(&caller);
                        let addr_bytes: [u8; 4] =
                            mem[sp as usize..sp as usize + 4].try_into().unwrap();
                        let addr = u32::from_le_bytes(addr_bytes) as usize;
                        let float_bytes: [u8; 8] = mem[fsp_val as usize..fsp_val as usize + 8]
                            .try_into()
                            .unwrap();
                        (addr, float_bytes)
                    };
                    dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))
                        .unwrap();
                    fsp.set(&mut caller, Val::I32((fsp_val + FLOAT_SIZE) as i32))
                        .unwrap();
                    let mem = memory.data_mut(&mut caller);
                    mem[addr..addr + 8].copy_from_slice(&float_bytes);
                    Ok(())
                },
            );
            self.register_host_primitive("DF!", false, func)?;
        }

        // DF@ ( df-addr -- ) ( F: -- r ) same as F@ (our floats are already f64)
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let fsp = self.fsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let fsp_val = fsp.get(&mut caller).unwrap_i32() as u32;
                    let val = {
                        let mem = memory.data(&caller);
                        let addr_bytes: [u8; 4] =
                            mem[sp as usize..sp as usize + 4].try_into().unwrap();
                        let addr = u32::from_le_bytes(addr_bytes) as usize;
                        let float_bytes: [u8; 8] = mem[addr..addr + 8].try_into().unwrap();
                        f64::from_le_bytes(float_bytes)
                    };
                    dsp.set(&mut caller, Val::I32((sp + CELL_SIZE) as i32))
                        .unwrap();
                    let new_fsp = fsp_val - FLOAT_SIZE;
                    fsp.set(&mut caller, Val::I32(new_fsp as i32)).unwrap();
                    let mem = memory.data_mut(&mut caller);
                    mem[new_fsp as usize..new_fsp as usize + 8].copy_from_slice(&val.to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("DF@", false, func)?;
        }

        // SFALIGNED, DFALIGNED (alignment words for single/double floats)
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let mem = memory.data(&caller);
                    let b: [u8; 4] = mem[sp as usize..sp as usize + 4].try_into().unwrap();
                    let addr = u32::from_le_bytes(b);
                    let aligned = (addr + 3) & !3; // 4-byte alignment for single float
                    let mem = memory.data_mut(&mut caller);
                    mem[sp as usize..sp as usize + 4].copy_from_slice(&aligned.to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("SFALIGNED", false, func)?;
        }

        // DFALIGNED is the same as FALIGNED (8-byte alignment)
        {
            let memory = self.memory;
            let dsp = self.dsp;
            let func = Func::new(
                &mut self.store,
                FuncType::new(&self.engine, [], []),
                move |mut caller, _, _| {
                    let sp = dsp.get(&mut caller).unwrap_i32() as u32;
                    let mem = memory.data(&caller);
                    let b: [u8; 4] = mem[sp as usize..sp as usize + 4].try_into().unwrap();
                    let addr = u32::from_le_bytes(b);
                    let aligned = (addr + 7) & !7;
                    let mem = memory.data_mut(&mut caller);
                    mem[sp as usize..sp as usize + 4].copy_from_slice(&aligned.to_le_bytes());
                    Ok(())
                },
            );
            self.register_host_primitive("DFALIGNED", false, func)?;
        }

        Ok(())
    }

    /// Allocate a function table slot for an anonymous host function.
    /// Returns a `WordId` that can be used in `IrOp::Call`.
    /// Does NOT touch the dictionary, so it's safe during colon compilation.
    fn install_anon_func(&mut self, func: Func) -> anyhow::Result<WordId> {
        let idx = self.next_table_index;
        self.next_table_index += 1;
        // Also advance the dictionary's fn index counter to stay in sync
        self.dictionary.reserve_fn_index();
        self.ensure_table_size(idx)?;
        self.table
            .set(&mut self.store, idx as u64, Ref::Func(Some(func)))?;
        Ok(WordId(idx))
    }

    /// Compile a float literal for use inside a colon definition.
    /// Emits `PushF64` IR op which compiles directly to WASM f64.const + float stack push.
    fn compile_float_literal(&mut self, val: f64) -> anyhow::Result<()> {
        self.push_ir(IrOp::PushF64(val));
        Ok(())
    }

    /// Create a host function that pops from float stack and stores at the given address.
    /// Used for `TO <fvalue>` in compile mode.
    fn make_fvalue_store(&mut self, pfa: u32) -> anyhow::Result<WordId> {
        let memory = self.memory;
        let fsp = self.fsp;
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _, _| {
                let sp = fsp.get(&mut caller).unwrap_i32() as u32;
                let mem = memory.data(&caller);
                let bytes: [u8; 8] = mem[sp as usize..sp as usize + 8].try_into().unwrap();
                fsp.set(&mut caller, Val::I32((sp + FLOAT_SIZE) as i32))
                    .unwrap();
                let mem = memory.data_mut(&mut caller);
                mem[pfa as usize..pfa as usize + 8].copy_from_slice(&bytes);
                Ok(())
            },
        );
        self.install_anon_func(func)
    }

    /// FVARIABLE <name> -- allocate 8 bytes, word pushes address
    fn define_fvariable(&mut self) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("FVARIABLE: expected name"))?;

        let word_id = self
            .dictionary
            .create(&name, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Allocate 8 bytes aligned
        self.refresh_user_here();
        let addr = (self.user_here + 7) & !7;
        self.user_here = addr + FLOAT_SIZE;

        // Initialize to zero
        let data = self.memory.data_mut(&mut self.store);
        data[addr as usize..addr as usize + 8].copy_from_slice(&0.0_f64.to_le_bytes());

        // Compile a word that pushes the address onto the DATA stack
        let ir_body = vec![IrOp::PushI32(addr as i32)];
        self.ir_bodies.insert(word_id, ir_body.clone());
        let config = CodegenConfig {
            base_fn_index: word_id.0,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
        };
        let compiled = compile_word(&name, &ir_body, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for FVARIABLE {name}: {e}"))?;

        self.instantiate_and_install(&compiled, word_id)?;
        self.dictionary.reveal();
        self.sync_word_lookup(&name, word_id, false);
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);
        self.sync_here_cell();

        Ok(())
    }

    /// FCONSTANT <name> ( F: r -- ) -- create a word that pushes r onto float stack
    fn define_fconstant(&mut self) -> anyhow::Result<()> {
        let val = self.fpop()?;
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("FCONSTANT: expected name"))?;

        let word_id = self
            .dictionary
            .create(&name, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Create a host function that pushes the constant onto float stack
        let memory = self.memory;
        let fsp = self.fsp;
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _, _| {
                let sp = fsp.get(&mut caller).unwrap_i32() as u32;
                let new_sp = sp - FLOAT_SIZE;
                if new_sp < FLOAT_STACK_BASE {
                    return Err(wasmtime::Error::msg("float stack overflow"));
                }
                fsp.set(&mut caller, Val::I32(new_sp as i32)).unwrap();
                let mem = memory.data_mut(&mut caller);
                mem[new_sp as usize..new_sp as usize + 8].copy_from_slice(&val.to_le_bytes());
                Ok(())
            },
        );

        self.ensure_table_size(word_id.0)?;
        self.table
            .set(&mut self.store, word_id.0 as u64, Ref::Func(Some(func)))?;
        self.dictionary.reveal();
        self.sync_word_lookup(&name, word_id, false);
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);

        Ok(())
    }

    /// FVALUE <name> ( F: r -- ) -- create a word that fetches r from storage
    fn define_fvalue(&mut self) -> anyhow::Result<()> {
        let val = self.fpop()?;
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("FVALUE: expected name"))?;

        let word_id = self
            .dictionary
            .create(&name, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Allocate 8 bytes aligned for the value's storage
        self.refresh_user_here();
        let val_addr = (self.user_here + 7) & !7;
        self.user_here = val_addr + FLOAT_SIZE;

        // Initialize the storage with the given value
        let data = self.memory.data_mut(&mut self.store);
        data[val_addr as usize..val_addr as usize + 8].copy_from_slice(&val.to_le_bytes());

        // Create a host function that fetches from storage and pushes onto float stack
        let memory = self.memory;
        let fsp = self.fsp;
        let func = Func::new(
            &mut self.store,
            FuncType::new(&self.engine, [], []),
            move |mut caller, _, _| {
                let mem = memory.data(&caller);
                let bytes: [u8; 8] = mem[val_addr as usize..val_addr as usize + 8]
                    .try_into()
                    .unwrap();
                let sp = fsp.get(&mut caller).unwrap_i32() as u32;
                let new_sp = sp - FLOAT_SIZE;
                if new_sp < FLOAT_STACK_BASE {
                    return Err(wasmtime::Error::msg("float stack overflow"));
                }
                fsp.set(&mut caller, Val::I32(new_sp as i32)).unwrap();
                let mem = memory.data_mut(&mut caller);
                mem[new_sp as usize..new_sp as usize + 8].copy_from_slice(&bytes);
                Ok(())
            },
        );

        self.ensure_table_size(word_id.0)?;
        self.table
            .set(&mut self.store, word_id.0 as u64, Ref::Func(Some(func)))?;
        self.dictionary.reveal();
        self.sync_word_lookup(&name, word_id, false);
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);
        // Map xt -> PFA for TO
        self.word_pfa_map.insert(word_id.0, val_addr);
        self.sync_pfa_map(word_id.0, val_addr);
        self.fvalue_words.insert(word_id.0);
        self.sync_here_cell();

        Ok(())
    }
}

/// Format a float in engineering notation (exponent is multiple of 3).
fn format_engineering(val: f64, prec: usize) -> String {
    if val == 0.0 {
        return format!("0.{:0>width$}E0 ", "", width = prec);
    }
    let abs_val = val.abs();
    let exp = abs_val.log10().floor() as i32;
    let eng_exp = exp - exp.rem_euclid(3);
    let mantissa = val / 10.0_f64.powi(eng_exp);
    format!("{mantissa:.prec$}E{eng_exp} ")
}

/// Parse a Forth float format string into f64.
fn parse_forth_float(s: &str) -> Option<f64> {
    let s = s.trim();
    // Empty string or all spaces = 0.0 (Forth 2012 >FLOAT special case)
    if s.is_empty() {
        return Some(0.0);
    }
    let upper = s.to_ascii_uppercase();

    // Reject anything with letters other than E or D
    for c in upper.chars() {
        if c.is_ascii_alphabetic() && c != 'E' && c != 'D' {
            return None;
        }
    }

    // Replace 'D' with 'E' for Rust parsing
    let normalized = upper.replace('D', "E");

    // Check that there's at least one digit somewhere
    let has_digit = normalized.chars().any(|c| c.is_ascii_digit());
    if !has_digit {
        return None;
    }

    // Must contain 'E' or a '.' to be a valid float
    if !normalized.contains('E') {
        if normalized.contains('.') {
            return normalized.parse::<f64>().ok();
        }
        // Just digits with no E and no dot -- not a valid float for >FLOAT
        return None;
    }

    // Must not have multiple E's
    if normalized.matches('E').count() > 1 {
        return None;
    }

    // Must not contain spaces within the number
    if normalized.contains(' ') {
        return None;
    }

    // Split on E, verify the mantissa part has digits
    let parts: Vec<&str> = normalized.splitn(2, 'E').collect();
    let mantissa = parts[0];
    // Strip sign from mantissa
    let mantissa_stripped = mantissa.trim_start_matches(['+', '-']);
    // Must have at least one digit in mantissa
    if !mantissa_stripped.chars().any(|c| c.is_ascii_digit()) {
        return None;
    }

    // Trailing E without exponent: "1E" means "1E0"
    let s = if normalized.ends_with('E') || normalized.ends_with("E+") || normalized.ends_with("E-")
    {
        format!("{normalized}0")
    } else {
        normalized
    };

    s.parse::<f64>().ok()
}

/// REPRESENT helper: convert f64 to digit string.
fn represent_float(val: f64, buf_len: usize) -> (String, i32, bool, bool) {
    if buf_len == 0 {
        return (String::new(), 0, val.is_sign_negative(), false);
    }
    if val.is_nan() {
        return ("0".repeat(buf_len), 0, false, false);
    }
    if val.is_infinite() {
        return ("0".repeat(buf_len), 0, val < 0.0, false);
    }
    let is_negative = val.is_sign_negative();
    let abs_val = val.abs();
    if abs_val == 0.0 {
        return ("0".repeat(buf_len), 0, is_negative, true);
    }
    let exp = abs_val.log10().floor() as i32 + 1;
    let scaled = abs_val / 10.0_f64.powi(exp - buf_len as i32);
    let digits = format!("{:.0}", scaled.round());
    // Handle carry (e.g., 9.95 with buf_len=2 -> "100")
    if digits.len() > buf_len {
        // Rounding caused overflow; increment exponent
        let truncated = &digits[..buf_len];
        return (truncated.to_string(), exp + 1, is_negative, true);
    }
    let padded = format!("{digits:0>buf_len$}");
    (padded, exp, is_negative, true)
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
    fn test_inline_tailcall_rstack_interaction() {
        // Regression: inlining a word that had a TailCall inside an If branch
        // caused the TailCall's Return to exit the *caller*, corrupting the
        // return stack. The fix: detailcall() recursively converts TailCall
        // back to Call inside all nested control-flow bodies when inlining.
        assert_eq!(
            eval_stack(": T 42 >R 99 >R -7 -1 DABS R> R> ; T"),
            vec![42, 99, 0, 7]
        );
    }

    #[test]
    fn test_do_loop_with_i_and_step() {
        // +LOOP with step of 2
        assert_eq!(
            eval_output(": TEST 10 0 DO I . 2 +LOOP ; TEST"),
            "0 2 4 6 8 "
        );
    }

    #[test]
    fn test_plus_loop_leave_with_zero_step() {
        // Regression: LEAVE inside +LOOP with step=0 caused infinite loop.
        // LEAVE sets index=limit, but the XOR termination check yields 0 XOR 0 = 0
        // (not negative), so the loop never exited without the leave flag.
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("VARIABLE INCRMNT  VARIABLE ITERS").unwrap();
        vm.evaluate(
            ": QD6 INCRMNT ! 0 ITERS ! ?DO 1 ITERS +! I ITERS @ 6 = IF LEAVE THEN INCRMNT @ +LOOP ITERS @ ;"
        ).unwrap();
        vm.evaluate("-1 2 0 QD6").unwrap();
        let stack = vm.data_stack();
        // Expected: 2 2 2 2 2 2 6 (6 iterations of I=2, then ITERS@=6)
        assert_eq!(stack, vec![6, 2, 2, 2, 2, 2, 2]);
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

    // ===================================================================
    // Exception word set: CATCH and THROW
    // ===================================================================

    #[test]
    fn test_catch_no_throw() {
        // CATCH with a word that doesn't throw should push 0
        assert_eq!(eval_output(": TEST ['] DUP CATCH . ; 5 TEST"), "0 ");
    }

    #[test]
    fn test_catch_no_throw_stack() {
        // After CATCH of a non-throwing word, TOS should be 0 and the
        // word's effect should be visible underneath
        assert_eq!(eval_stack("5 ' DUP CATCH"), vec![0, 5, 5]);
    }

    #[test]
    fn test_throw_zero_is_noop() {
        // THROW with 0 should do nothing
        assert_eq!(eval_output(": TEST 0 THROW 123 . ; TEST"), "123 ");
    }

    #[test]
    fn test_catch_throw_basic() {
        // CATCH with a word that throws should push the throw code
        assert_eq!(
            eval_output(": THROWER 42 THROW ; : TEST ['] THROWER CATCH . ; TEST"),
            "42 "
        );
    }

    #[test]
    fn test_catch_stack_restore() {
        // THROW should restore the data stack to the depth saved by CATCH
        // Before CATCH: stack is (10 20), CATCH pops xt, saves depth (10 20)
        // THROWER pushes 1 2 3 then throws 99
        // CATCH restores to (10 20) and pushes 99
        let stack = eval_stack(": THROWER 1 2 3 99 THROW ; 10 20 ' THROWER CATCH");
        assert_eq!(stack, vec![99, 20, 10]);
    }

    #[test]
    fn test_nested_catch() {
        // Nested CATCH: inner CATCH catches the throw, outer CATCH sees success
        assert_eq!(
            eval_output(
                ": INNER 5 THROW ; : OUTER ['] INNER CATCH . ; : TEST ['] OUTER CATCH . ; TEST"
            ),
            "5 0 "
        );
    }

    #[test]
    fn test_catch_negative_throw() {
        // Standard throw codes are negative
        assert_eq!(
            eval_output(": THROWER -1 THROW ; : TEST ['] THROWER CATCH . ; TEST"),
            "-1 "
        );
    }

    #[test]
    fn test_catch_preserves_output() {
        // Output before THROW should still be visible
        assert_eq!(
            eval_output(": THROWER 65 EMIT 1 THROW ; : TEST ['] THROWER CATCH DROP ; TEST"),
            "A"
        );
    }

    #[test]
    fn test_catch_in_colon_def() {
        // CATCH can be used inside a colon definition
        assert_eq!(
            eval_output(": ERR 10 THROW ; : SAFE ['] ERR CATCH ; SAFE ."),
            "10 "
        );
    }

    #[test]
    fn test_throw_skips_rest_of_word() {
        // After THROW, remaining code in the throwing word should not execute
        assert_eq!(
            eval_output(": BAD 1 THROW 999 . ; : TEST ['] BAD CATCH . ; TEST"),
            "1 "
        );
    }

    // ===================================================================
    // POSTPONE: Forth 2012 GT5/GT7 tests
    // ===================================================================

    #[test]
    fn test_postpone_non_immediate_gt5() {
        // : GT1 123 ;
        // : GT4 POSTPONE GT1 ; IMMEDIATE
        // : GT5 GT4 ;
        // GT5 -> 123
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate(": GT1 123 ;").unwrap();
        vm.evaluate(": GT4 POSTPONE GT1 ; IMMEDIATE").unwrap();
        vm.evaluate(": GT5 GT4 ;").unwrap();
        vm.evaluate("GT5").unwrap();
        assert_eq!(vm.data_stack(), vec![123]);
    }

    #[test]
    fn test_postpone_immediate_gt7() {
        // : GT6 345 ; IMMEDIATE
        // : GT7 POSTPONE GT6 ;
        // GT7 -> 345
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate(": GT6 345 ; IMMEDIATE").unwrap();
        vm.evaluate(": GT7 POSTPONE GT6 ;").unwrap();
        vm.evaluate("GT7").unwrap();
        assert_eq!(vm.data_stack(), vec![345]);
    }

    // ===================================================================
    // Double DOES>: Forth 2012 WEIRD: W1 test
    // ===================================================================

    #[test]
    fn test_double_does() {
        // : WEIRD: CREATE DOES> 1 + DOES> 2 + ;
        // WEIRD: W1
        // W1 first call: PFA 1 + (first DOES> behavior, then patches to second)
        // W1 second call: PFA 2 + (second DOES> behavior)
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate(": WEIRD: CREATE DOES> 1 + DOES> 2 + ;")
            .unwrap();
        vm.evaluate("WEIRD: W1").unwrap();
        // Get HERE (which is the PFA of W1)
        vm.evaluate("' W1 >BODY").unwrap();
        let pfa = vm.data_stack()[0];
        vm.evaluate("DROP").unwrap();
        // First call: PFA 1 +
        vm.evaluate("W1").unwrap();
        assert_eq!(vm.data_stack(), vec![pfa + 1]);
        vm.evaluate("DROP").unwrap();
        // Second call: PFA 2 +
        vm.evaluate("W1").unwrap();
        assert_eq!(vm.data_stack(), vec![pfa + 2]);
    }

    // ===================================================================
    // Core Extension words
    // ===================================================================

    #[test]
    fn test_value_basic() {
        assert_eq!(eval_output("10 VALUE FOO FOO ."), "10 ");
    }

    #[test]
    fn test_value_to() {
        assert_eq!(eval_output("10 VALUE FOO 20 TO FOO FOO ."), "20 ");
    }

    #[test]
    fn test_value_in_colon() {
        assert_eq!(eval_output("10 VALUE FOO : TEST FOO . ; TEST"), "10 ");
    }

    #[test]
    fn test_value_to_in_colon() {
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("10 VALUE FOO").unwrap();
        vm.evaluate(": SETFOO TO FOO ;").unwrap();
        vm.evaluate("20 SETFOO FOO .").unwrap();
        assert_eq!(vm.take_output(), "20 ");
    }

    #[test]
    fn test_defer_basic() {
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("DEFER MY-DEFER").unwrap();
        vm.evaluate("' DUP IS MY-DEFER").unwrap();
        vm.evaluate("5 MY-DEFER .S").unwrap();
        assert_eq!(vm.take_output(), "<2> 5 5 ");
    }

    #[test]
    fn test_defer_action_of() {
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("DEFER MY-DEFER").unwrap();
        vm.evaluate("' DUP IS MY-DEFER").unwrap();
        vm.evaluate("ACTION-OF MY-DEFER ' DUP =").unwrap();
        assert_eq!(vm.data_stack(), vec![-1]); // TRUE
    }

    #[test]
    fn test_2r_operations() {
        assert_eq!(eval_stack(": TEST 1 2 2>R 2R> ; TEST"), vec![2, 1]);
        assert_eq!(
            eval_stack(": TEST 1 2 2>R 2R@ 2R> 2DROP ; TEST"),
            vec![2, 1]
        );
    }

    #[test]
    fn test_again() {
        // AGAIN creates an infinite loop; use EXIT to break out
        assert_eq!(
            eval_output(": TEST BEGIN DUP . 1+ DUP 5 > IF EXIT THEN AGAIN ; 1 TEST"),
            "1 2 3 4 5 "
        );
    }

    #[test]
    fn test_case_of_endof_endcase() {
        assert_eq!(
            eval_output(
                ": TEST CASE 1 OF 10 ENDOF 2 OF 20 ENDOF 0 SWAP ENDCASE ; 1 TEST . 2 TEST . 3 TEST ."
            ),
            "10 20 0 "
        );
    }

    #[test]
    fn test_case_empty() {
        // Empty CASE with just DROP
        assert_eq!(eval_output(": TEST CASE ENDCASE ; 5 TEST"), "");
    }

    #[test]
    fn test_u_greater() {
        assert_eq!(eval_stack("2 1 U>"), vec![-1]);
        assert_eq!(eval_stack("1 2 U>"), vec![0]);
        assert_eq!(eval_stack("-1 1 U>"), vec![-1]); // -1 as unsigned > 1
    }

    #[test]
    fn test_qdo_basic() {
        assert_eq!(
            eval_output(": TEST 10 0 ?DO I . LOOP ; TEST"),
            "0 1 2 3 4 5 6 7 8 9 "
        );
    }

    #[test]
    fn test_qdo_skip() {
        // ?DO should skip the loop body when limit == index
        assert_eq!(eval_output(": TEST 0 0 ?DO I . LOOP ; TEST"), "");
    }

    #[test]
    fn test_pad() {
        let stack = eval_stack("PAD");
        assert_eq!(stack.len(), 1);
        assert_eq!(stack[0], crate::memory::PAD_BASE as i32);
    }

    #[test]
    fn test_erase() {
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("HERE 65 C, 66 C, 67 C,").unwrap(); // write ABC, stack: addr
        vm.evaluate("DUP 3 ERASE").unwrap(); // erase 3 bytes at addr
        vm.evaluate("DUP C@ SWAP 1+ C@").unwrap();
        assert_eq!(vm.data_stack(), vec![0, 0]);
    }

    #[test]
    fn test_dot_r() {
        assert_eq!(eval_output("123 6 .R"), "   123");
    }

    #[test]
    fn test_u_dot_r() {
        assert_eq!(eval_output("123 6 U.R"), "   123");
    }

    #[test]
    fn test_unused() {
        let stack = eval_stack("UNUSED");
        assert_eq!(stack.len(), 1);
        assert!(stack[0] > 0); // Should have some available space
    }

    #[test]
    fn test_noname() {
        assert_eq!(eval_output(":NONAME 42 . ; EXECUTE"), "42 ");
    }

    #[test]
    fn test_noname_constant() {
        assert_eq!(
            eval_output(":NONAME DUP + ; CONSTANT DUP+ 5 DUP+ EXECUTE ."),
            "10 "
        );
    }

    #[test]
    fn test_parse() {
        // PARSE ( char -- c-addr u ) in interpret mode
        // Skips one leading space (outer interpreter's trailing delimiter)
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("CHAR ) PARSE hello)").unwrap();
        let stack = vm.data_stack();
        assert_eq!(stack.len(), 2);
        assert_eq!(stack[0], 5); // length of "hello"
    }

    #[test]
    fn test_parse_name() {
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("PARSE-NAME hello").unwrap();
        let stack = vm.data_stack();
        assert_eq!(stack.len(), 2);
        assert_eq!(stack[0], 5); // length of "hello"
    }

    #[test]
    fn test_buffer_colon() {
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("100 BUFFER: BUF").unwrap();
        vm.evaluate("BUF").unwrap();
        let stack = vm.data_stack();
        assert_eq!(stack.len(), 1);
        assert!(stack[0] > 0); // Address should be valid
    }

    #[test]
    fn test_source_id() {
        // SOURCE-ID should return 0 for user input
        assert_eq!(eval_stack("SOURCE-ID"), vec![0]);
    }

    #[test]
    fn test_c_quote() {
        assert_eq!(eval_output("C\" hello\" COUNT TYPE"), "hello");
    }

    #[test]
    fn test_refill() {
        // REFILL should return FALSE in piped mode
        assert_eq!(eval_stack("REFILL"), vec![0]);
    }

    #[test]
    fn test_marker() {
        // MARKER should create a word without errors
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("MARKER MARK1").unwrap();
        // MARK1 should exist and be callable
        vm.evaluate("MARK1").unwrap();
    }

    #[test]
    fn test_holds() {
        // HOLDS adds string to pictured output
        assert_eq!(
            eval_output(": TEST 0 <# S\" xyz\" HOLDS 0 #> TYPE ; TEST"),
            "xyz"
        );
    }

    #[test]
    fn test_defer_store_fetch() {
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("DEFER MY-DEF").unwrap();
        vm.evaluate("' DUP ' MY-DEF DEFER!").unwrap();
        vm.evaluate("' MY-DEF DEFER@").unwrap();
        let dup_xt = {
            vm.evaluate("' DUP").unwrap();
            vm.data_stack()[0]
        };
        // The DEFER@ result should match DUP's xt
        let stack = vm.data_stack();
        assert_eq!(stack[0], dup_xt);
    }

    // -- Floating-Point word set tests --

    fn eval_float_stack(input: &str) -> Vec<f64> {
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate(input).unwrap();
        vm.float_stack()
    }

    #[test]
    fn test_float_literal_interpret() {
        let fs = eval_float_stack("1E");
        assert_eq!(fs.len(), 1);
        assert!((fs[0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_float_literal_with_exponent() {
        let fs = eval_float_stack("1.5E2");
        assert!((fs[0] - 150.0).abs() < 1e-10);
    }

    #[test]
    fn test_float_add() {
        assert_eq!(eval_output("1E 2E F+ F."), "3.000000 ");
    }

    #[test]
    fn test_float_sub() {
        assert_eq!(eval_output("5E 3E F- F."), "2.000000 ");
    }

    #[test]
    fn test_float_mul() {
        assert_eq!(eval_output("3E 4E F* F."), "12.000000 ");
    }

    #[test]
    fn test_float_div() {
        assert_eq!(eval_output("10E 4E F/ F."), "2.500000 ");
    }

    #[test]
    fn test_float_negate() {
        assert_eq!(eval_output("3E FNEGATE F."), "-3.000000 ");
    }

    #[test]
    fn test_float_abs() {
        assert_eq!(eval_output("-5E FABS F."), "5.000000 ");
    }

    #[test]
    fn test_fdepth() {
        assert_eq!(eval_stack("FDEPTH"), vec![0]);
        assert_eq!(eval_stack("1E FDEPTH"), vec![1]);
        assert_eq!(eval_stack("1E 2E FDEPTH"), vec![2]);
    }

    #[test]
    fn test_fdrop() {
        assert_eq!(eval_stack("1E 2E FDROP FDEPTH"), vec![1]);
    }

    #[test]
    fn test_fdup() {
        assert_eq!(eval_stack("3E FDUP FDEPTH"), vec![2]);
    }

    #[test]
    fn test_fswap() {
        assert_eq!(eval_output("1E 2E FSWAP F. F."), "1.000000 2.000000 ");
    }

    #[test]
    fn test_fover() {
        assert_eq!(
            eval_output("1E 2E FOVER F. F. F."),
            "1.000000 2.000000 1.000000 "
        );
    }

    #[test]
    fn test_frot() {
        assert_eq!(
            eval_output("1E 2E 3E FROT F. F. F."),
            "1.000000 3.000000 2.000000 "
        );
    }

    #[test]
    fn test_f0_eq() {
        assert_eq!(eval_stack("0E F0="), vec![-1]);
        assert_eq!(eval_stack("1E F0="), vec![0]);
    }

    #[test]
    fn test_f0_lt() {
        assert_eq!(eval_stack("-1E F0<"), vec![-1]);
        assert_eq!(eval_stack("0E F0<"), vec![0]);
        assert_eq!(eval_stack("1E F0<"), vec![0]);
    }

    #[test]
    fn test_f_eq() {
        assert_eq!(eval_stack("1E 1E F="), vec![-1]);
        assert_eq!(eval_stack("1E 2E F="), vec![0]);
    }

    #[test]
    fn test_f_lt() {
        assert_eq!(eval_stack("1E 2E F<"), vec![-1]);
        assert_eq!(eval_stack("2E 1E F<"), vec![0]);
    }

    #[test]
    fn test_s_to_f_f_to_s() {
        assert_eq!(eval_stack("42 S>F F>S"), vec![42]);
        assert_eq!(eval_stack("-7 S>F F>S"), vec![-7]);
    }

    #[test]
    fn test_d_to_f_f_to_d() {
        assert_eq!(eval_stack("1. D>F F>D"), vec![0, 1]); // 1. = lo=1, hi=0
    }

    #[test]
    fn test_float_literal_compile_mode() {
        assert_eq!(eval_stack(": TEST 3.14E0 F>S ; TEST"), vec![3]);
    }

    #[test]
    fn test_float_compile_fplus() {
        assert_eq!(eval_output(": FTEST 1E 2E F+ ; FTEST F."), "3.000000 ");
    }

    #[test]
    fn test_fvariable() {
        assert_eq!(eval_output("FVARIABLE X 3.14E0 X F! X F@ F."), "3.140000 ");
    }

    #[test]
    fn test_fconstant() {
        assert_eq!(eval_output("3.14E0 FCONSTANT PI PI F."), "3.140000 ");
    }

    #[test]
    fn test_fvalue_and_to() {
        assert_eq!(
            eval_output("1E FVALUE V V F. 2E TO V V F."),
            "1.000000 2.000000 "
        );
    }

    #[test]
    fn test_fliteral() {
        assert_eq!(eval_output(": FT [ -2E ] FLITERAL F. ; FT"), "-2.000000 ");
    }

    #[test]
    fn test_fsqrt() {
        assert_eq!(eval_output("4E FSQRT F."), "2.000000 ");
    }

    #[test]
    fn test_fsin_cos() {
        // sin(0) = 0, cos(0) = 1
        assert_eq!(eval_stack("0E FSIN F>S"), vec![0]);
        assert_eq!(eval_stack("0E FCOS F>S"), vec![1]);
    }

    #[test]
    fn test_fexp_fln() {
        assert_eq!(eval_stack("0E FEXP F>S"), vec![1]); // e^0 = 1
        assert_eq!(eval_stack("1E FLN F>S"), vec![0]); // ln(1) = 0
    }

    #[test]
    fn test_floor_fround() {
        assert_eq!(eval_output("1.7E FLOOR F."), "1.000000 ");
        assert_eq!(eval_output("-1.3E FLOOR F."), "-2.000000 ");
    }

    #[test]
    fn test_fpower() {
        assert_eq!(eval_output("2E 3E F** F."), "8.000000 ");
    }

    #[test]
    fn test_fmax_fmin() {
        assert_eq!(eval_output("3E 5E FMAX F."), "5.000000 ");
        assert_eq!(eval_output("3E 5E FMIN F."), "3.000000 ");
    }

    #[test]
    fn test_precision() {
        assert_eq!(eval_output("3 SET-PRECISION 1E F."), "1.000 ");
    }

    #[test]
    fn test_f_store_fetch() {
        assert_eq!(
            eval_output("VARIABLE BUF 2 CELLS ALLOT 42E BUF F! BUF F@ F."),
            "42.000000 "
        );
    }

    #[test]
    fn test_float_plus_floats() {
        assert_eq!(eval_stack("0 FLOAT+"), vec![8]);
        assert_eq!(eval_stack("3 FLOATS"), vec![24]);
    }

    #[test]
    fn test_represent() {
        // 1E with 5 digits should give "10000" and exponent 1
        let mut vm = ForthVM::new().unwrap();
        vm.evaluate("CREATE FBUF 20 ALLOT").unwrap();
        vm.evaluate("1E FBUF 5 REPRESENT").unwrap();
        let stack = vm.data_stack();
        // Stack should be: exponent=1, sign=0 (not negative), valid=-1 (true)
        // Top first: valid, sign, exponent
        assert_eq!(stack[0], -1); // valid = true
        assert_eq!(stack[1], 0); // not negative
        assert_eq!(stack[2], 1); // exponent
    }

    #[test]
    fn test_to_float() {
        // >FLOAT with "1E" should return true and push 1.0
        assert_eq!(eval_stack(r#"S" 1E" >FLOAT"#), vec![-1]);
        // >FLOAT with "." should return false
        assert_eq!(eval_stack(r#"S" ." >FLOAT"#), vec![0]);
    }

    #[test]
    fn test_f_tilde() {
        // Exact comparison: F~ with 0E
        assert_eq!(eval_stack("1E 1E 0E F~"), vec![-1]);
        assert_eq!(eval_stack("1E 2E 0E F~"), vec![0]);
        // Absolute comparison
        assert_eq!(eval_stack("1E 1.5E 1E F~"), vec![-1]); // |1-1.5| < 1
        assert_eq!(eval_stack("1E 2.5E 1E F~"), vec![0]); // |1-2.5| = 1.5 >= 1
    }

    #[test]
    fn optimizer_doesnt_break_basic_arithmetic() {
        assert_eq!(eval_stack("5 3 +"), vec![8]);
        assert_eq!(eval_stack("10 3 -"), vec![7]);
        assert_eq!(eval_stack(": SQUARE DUP * ; 7 SQUARE"), vec![49]);
    }

    #[test]
    fn optimizer_doesnt_break_control_flow() {
        assert_eq!(eval_stack(": T1 1 IF 42 ELSE 0 THEN ; T1"), vec![42]);
        assert_eq!(eval_stack(": T2 0 IF 42 ELSE 0 THEN ; T2"), vec![0]);
        assert_eq!(eval_stack(": SUM 0 SWAP 0 DO I + LOOP ; 10 SUM"), vec![45]);
    }

    // -- CONSOLIDATE tests --

    #[test]
    fn consolidate_basic() {
        assert_eq!(eval_stack(": A 1 ; : B A 2 + ; CONSOLIDATE B"), vec![3]);
    }

    #[test]
    fn consolidate_preserves_host_functions() {
        assert_eq!(
            eval_output(": HELLO 72 EMIT 73 EMIT ; CONSOLIDATE HELLO"),
            "HI"
        );
    }

    #[test]
    fn consolidate_no_op_when_empty() {
        // CONSOLIDATE with no user words should not error
        let (stack, _) = eval("CONSOLIDATE 42");
        assert_eq!(stack, vec![42]);
    }

    #[test]
    fn consolidate_multiple_words() {
        assert_eq!(
            eval_stack(": X 10 ; : Y 20 ; : Z X Y + ; CONSOLIDATE Z"),
            vec![30]
        );
    }

    #[test]
    fn consolidate_with_control_flow() {
        assert_eq!(
            eval_stack(": ABS2 DUP 0< IF NEGATE THEN ; CONSOLIDATE -5 ABS2"),
            vec![5]
        );
    }

    #[test]
    fn consolidate_with_loop() {
        assert_eq!(
            eval_stack(": SUM2 0 SWAP 0 DO I + LOOP ; CONSOLIDATE 5 SUM2"),
            vec![10]
        );
    }

    #[test]
    fn consolidate_preserves_variables() {
        assert_eq!(
            eval_stack("VARIABLE V 42 V ! : RV V @ ; CONSOLIDATE RV"),
            vec![42]
        );
    }

    #[test]
    fn consolidate_nested_calls() {
        // A calls B which calls C -- all should use direct calls after consolidation
        assert_eq!(
            eval_stack(": C 1 ; : B C C + ; : A B B + ; CONSOLIDATE A"),
            vec![4]
        );
    }

    #[test]
    fn consolidate_words_still_work_individually() {
        assert_eq!(eval_stack(": P 3 ; : Q 4 ; CONSOLIDATE P Q +"), vec![7]);
    }

    #[test]
    fn consolidate_with_begin_until() {
        // Countdown: start at 5, subtract 1 until 0
        assert_eq!(
            eval_stack(": CD BEGIN 1- DUP 0= UNTIL ; CONSOLIDATE 5 CD"),
            vec![0]
        );
    }

    #[test]
    fn consolidate_with_begin_while_repeat() {
        assert_eq!(
            eval_stack(": CW BEGIN DUP WHILE 1- REPEAT ; CONSOLIDATE 3 CW"),
            vec![0]
        );
    }

    // ===================================================================
    // End-to-end optimization verification tests
    // ===================================================================

    #[test]
    fn verify_peephole_active() {
        // PushI32(0) + Add should be removed by peephole
        assert_eq!(eval_stack(": T 0 + ; 5 T"), vec![5]);
    }

    #[test]
    fn verify_constant_folding_active() {
        // 3 4 + should fold to 7 at compile time
        assert_eq!(eval_stack(": T 3 4 + ; T"), vec![7]);
    }

    #[test]
    fn verify_strength_reduction_active() {
        // 4 * should become 2 LSHIFT
        assert_eq!(eval_stack(": T 4 * ; 3 T"), vec![12]);
    }

    #[test]
    fn verify_dce_active() {
        // Code after EXIT should be eliminated
        assert_eq!(eval_stack(": T 42 EXIT 99 ; T"), vec![42]);
    }

    #[test]
    fn verify_tail_call_active() {
        // Recursive word in tail position should work (tail call prevents stack overflow)
        assert_eq!(
            eval_stack(": DEC1 DUP 0= IF EXIT THEN 1- RECURSE ; 1000 DEC1"),
            vec![0],
        );
    }

    #[test]
    fn verify_inlining_active() {
        // Small word should be inlined: 5 + 3 should fold to 8 after inline + fold
        assert_eq!(eval_stack(": ADD3 3 + ; : T ADD3 ; 5 T"), vec![8]);
    }

    #[test]
    fn verify_compound_ops_active() {
        // 2DUP (Over Over -> TwoDup) should work
        assert_eq!(eval_stack(": T 2DUP + ; 3 4 T"), vec![7, 4, 3]);
    }

    #[test]
    fn verify_dsp_caching_active() {
        // Complex word should work with DSP caching
        assert_eq!(
            eval_stack(": FACT DUP 1 > IF DUP 1- RECURSE * ELSE DROP 1 THEN ; 5 FACT"),
            vec![120],
        );
    }

    #[test]
    fn verify_consolidation_active() {
        assert_eq!(
            eval_stack(": A 10 ; : B 20 ; : C A B + ; CONSOLIDATE C"),
            vec![30],
        );
    }

    #[test]
    fn verify_stack_promotion_square() {
        // DUP * is promotable (no control flow, no calls) -- should use locals
        assert_eq!(eval_stack(": SQUARE DUP * ; 7 SQUARE"), vec![49]);
    }

    #[test]
    fn verify_stack_promotion_arithmetic() {
        // Pure arithmetic promotion
        assert_eq!(eval_stack(": T OVER OVER + ; 3 4 T"), vec![7, 4, 3]);
    }

    #[test]
    fn verify_stack_promotion_swap() {
        // SWAP is a zero-instruction op in promoted path
        assert_eq!(eval_stack(": T SWAP ; 1 2 T"), vec![1, 2]);
    }

    #[test]
    fn verify_stack_promotion_rot() {
        // ROT is a zero-instruction op in promoted path
        assert_eq!(eval_stack(": T ROT ; 1 2 3 T"), vec![1, 3, 2]);
    }

    #[test]
    fn verify_stack_promotion_nip_tuck() {
        assert_eq!(eval_stack(": T NIP ; 1 2 T"), vec![2]);
        assert_eq!(eval_stack(": T TUCK ; 1 2 T"), vec![2, 1, 2]);
    }

    #[test]
    fn verify_stack_promotion_memory_ops() {
        // Memory fetch/store should work in promoted path
        assert_eq!(eval_stack("VARIABLE X 42 X ! : T X @ 10 + ; T"), vec![52],);
    }

    #[test]
    fn verify_stack_promotion_comparison() {
        assert_eq!(eval_stack(": T = ; 5 5 T"), vec![-1]);
        assert_eq!(eval_stack(": T < ; 3 5 T"), vec![-1]);
    }

    // ===================================================================
    // Float IR tests
    // ===================================================================

    #[test]
    fn float_ir_add() {
        assert_eq!(eval_output("1E 2E F+ F."), "3.000000 ");
    }

    #[test]
    fn float_ir_literal_in_colon() {
        assert_eq!(eval_output(": T 1.5E0 2.5E0 F+ F. ; T"), "4.000000 ");
    }

    #[test]
    fn float_ir_conversions() {
        assert_eq!(eval_stack("42 S>F F>S"), vec![42]);
    }

    #[test]
    fn float_ir_memory() {
        assert_eq!(eval_output("FVARIABLE X 3.14E0 X F! X F@ F."), "3.140000 ");
    }

    #[test]
    fn float_ir_comparisons() {
        assert_eq!(eval_stack("1E 2E F<"), vec![-1]);
        assert_eq!(eval_stack("2E 1E F<"), vec![0]);
        assert_eq!(eval_stack("3E 3E F="), vec![-1]);
        assert_eq!(eval_stack("0E F0="), vec![-1]);
        assert_eq!(eval_stack("1E F0="), vec![0]);
        assert_eq!(eval_stack("-1E F0<"), vec![-1]);
        assert_eq!(eval_stack("1E F0<"), vec![0]);
    }

    #[test]
    fn float_ir_stack_ops() {
        assert_eq!(eval_output("1E FDUP F. F."), "1.000000 1.000000 ");
        assert_eq!(eval_output("1E 2E FSWAP F. F."), "1.000000 2.000000 ");
        assert_eq!(
            eval_output("1E 2E FOVER F. F. F."),
            "1.000000 2.000000 1.000000 "
        );
    }

    #[test]
    fn float_ir_arithmetic() {
        assert_eq!(eval_output("10E 3E F- F."), "7.000000 ");
        assert_eq!(eval_output("3E 4E F* F."), "12.000000 ");
        assert_eq!(eval_output("10E 4E F/ F."), "2.500000 ");
        assert_eq!(eval_output("3E FNEGATE F."), "-3.000000 ");
        assert_eq!(eval_output("-7E FABS F."), "7.000000 ");
        assert_eq!(eval_output("9E FSQRT F."), "3.000000 ");
    }
}
