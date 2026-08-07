//! Outer interpreter: tokenizer, number parser, and interpret/compile dispatch.
// Allow trivial casts from the Runtime trait refactoring — the mechanical conversion
// from wasmtime's Val::I32/unwrap_i32 patterns left redundant `as u32`/`as i32` casts.
// These are correct and harmless; cleaning them up is a separate task.
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

use crate::runtime::{HostAccess, HostFn, Runtime};

use crate::codegen::{CodegenConfig, CompiledModule, compile_consolidated_module, compile_word};
use crate::config::WaferConfig;
use crate::dictionary::{Dictionary, DictionaryState, WordId};
use crate::ir::IrOp;
#[cfg(feature = "crypto")]
use crate::memory::HASH_SCRATCH_BASE;
use crate::memory::{
    CELL_SIZE, DATA_STACK_TOP, DPL_INIT, FLOAT_SIZE, FLOAT_STACK_BASE, FLOAT_STACK_TOP,
    INPUT_BUFFER_BASE, INPUT_BUFFER_SIZE, RETURN_STACK_TOP, SYSVAR_BASE_VAR, SYSVAR_DPL,
    SYSVAR_FAULT_CODE, SYSVAR_HERE, SYSVAR_LEAVE_FLAG, SYSVAR_NH, SYSVAR_NUM_TIB, SYSVAR_STATE,
    SYSVAR_TO_IN,
};
use crate::optimizer::optimize;

// ---------------------------------------------------------------------------
// Number conversion
// ---------------------------------------------------------------------------

/// Characters that force double-cell conversion, following `SwiftForth`'s
/// input number conversion rules.
///
/// A leading `-` is the one exception: it binds as a sign, which keeps `-1`
/// a single-cell number while `1-2` converts as a double.
const DOUBLE_PUNCTUATION: [u8; 6] = *b",.+-/:";

/// Split a leading minus off a token, returning whether it was negative.
///
/// Only `-` is a sign. A leading `+` stays punctuation, matching `sf64`,
/// where `+7` converts as the double 7 with `DPL` = 1.
///
/// The sign may sit between a base-override prefix and the digits (`$-FF`,
/// the Forth 2012 spelling) or, as a WAFER extension, before the prefix
/// (`-$FF`), so this runs at both positions.
fn strip_sign(s: &str) -> (bool, &str) {
    match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        _ => (false, s),
    }
}

/// A numeric token that converted successfully.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NumberLiteral {
    /// The accumulated 64-bit value, sign applied.
    value: i64,
    /// Digits right of the rightmost punctuation character. Negative when the
    /// token carried no punctuation, which is how `DPL` reports "single-cell".
    dpl: i32,
    /// Whether punctuation forced double-cell conversion.
    is_double: bool,
}

impl NumberLiteral {
    /// Low-order cell — the value a single-cell conversion leaves on the stack.
    fn lo(self) -> i32 {
        self.value as i32
    }

    /// High-order cell. For a single-cell conversion this is what `NH` holds,
    /// letting an out-of-range token be recovered as a double.
    fn hi(self) -> i32 {
        (self.value >> 32) as i32
    }
}

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
    /// AHEAD: unconditional forward branch — code between AHEAD and THEN is skipped.
    Ahead {
        prefix: Vec<IrOp>,
    },
    /// CS-PICK'd reference to a Begin dest. UNTIL resolves this by emitting
    /// `LoopRestartIfFalse` instead of creating a full `BeginUntil`.
    BeginRef,
    /// Flat forward block from CS-ROLL'd IF linearization.
    /// THEN resolves this by emitting `EndBlock(label)`.
    ForwardBlock {
        label: u32,
    },
}

/// Pending actions from host functions executed during immediate-word evaluation.
/// Processed in order after the immediate word returns.
#[derive(Debug)]
enum PendingAction {
    /// Compile a call to the given word (from COMPILE,).
    CompileCall(u32),
    /// CS-PICK with the given n.
    CsPick(u32),
    /// CS-ROLL with the given n.
    CsRoll(u32),
    /// Compile a control-flow operation (from POSTPONE of compile-time keywords).
    CompileControl(i32),
    /// Forth 2012 §13.6.1.0086 `(LOCAL)` non-sentinel: declare a local of the
    /// given name. Name is already ASCII-uppercased by the host primitive.
    DeclareLocal(String),
    /// Forth 2012 §13.6.1.0086 `(LOCAL)` sentinel (`0 0 (LOCAL)`): emit the
    /// init code for locals declared since the last sentinel (or start of
    /// the current colon definition).
    DeclareLocalEnd,
}

/// Forth 2012 throw code for QUIT (table 9.1). Unlike every other code it is
/// not an exception: CATCH lets it through and the interpreter reports nothing.
const QUIT_THROW: i32 = -56;

/// Forth 2012 throw code for ABORT. CATCH sees it like any other exception,
/// but an uncaught one prints nothing: ABORT is specified as "empty the data
/// stack and perform the function of QUIT", and QUIT displays no message.
const ABORT_THROW: i32 = -1;

// Control-flow action codes for PendingAction::CompileControl
const CTRL_IF: i32 = 1;
const CTRL_ELSE: i32 = 2;
const CTRL_THEN: i32 = 3;
const CTRL_BEGIN: i32 = 4;
const CTRL_UNTIL: i32 = 5;
const CTRL_WHILE: i32 = 6;
const CTRL_REPEAT: i32 = 7;
const CTRL_AGAIN: i32 = 8;
const CTRL_DO: i32 = 9;
const CTRL_LOOP: i32 = 10;
const CTRL_PLUS_LOOP: i32 = 11;
const CTRL_AHEAD: i32 = 12;

// ---------------------------------------------------------------------------
// VM state stored in the wasmtime Store
// ---------------------------------------------------------------------------

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

/// Tokens handled directly by the outer interpreter (`interpret_token`,
/// `interpret_token_immediate`, `compile_token` hardcoded match arms).
/// Several have no dictionary entry at all; SEE/SEE-IR/HELP explain them
/// instead of erroring. Keep in sync with those match arms.
pub(crate) const INTERPRETER_TOKENS: &[&str] = &[
    // Definition structure
    ":",
    ":NONAME",
    ";",
    "[:",
    ";]",
    "[",
    "]",
    "{:",
    // Conditional compilation
    "[IF]",
    "[ELSE]",
    "[THEN]",
    "[DEFINED]",
    "[UNDEFINED]",
    // Strings + comments
    ".\"",
    ".(",
    "S\"",
    "S\\\"",
    "C\"",
    "S",
    "(",
    "\\",
    "ABORT\"",
    // Defining words
    "VARIABLE",
    "CONSTANT",
    "CREATE",
    "VALUE",
    "DOES>",
    "2CONSTANT",
    "2VARIABLE",
    "2VALUE",
    "FVARIABLE",
    "FCONSTANT",
    "FVALUE",
    "BUFFER:",
    "MARKER",
    "REMEMBER",
    "GILD",
    "EMPTY",
    "SYNONYM",
    "CONSOLIDATE",
    // Parsing words
    "'",
    "[']",
    "CHAR",
    "[CHAR]",
    "EVALUATE",
    "WORD",
    "TO",
    "IS",
    "ACTION-OF",
    "PARSE",
    "PARSE-NAME",
    "REFILL",
    "ORDER",
    // Compile-mode control flow
    "IF",
    "ELSE",
    "THEN",
    "DO",
    "?DO",
    "LOOP",
    "+LOOP",
    "BEGIN",
    "UNTIL",
    "AGAIN",
    "WHILE",
    "REPEAT",
    "AHEAD",
    "CASE",
    "OF",
    "ENDOF",
    "ENDCASE",
    "RECURSE",
    "EXIT",
    "LITERAL",
    "2LITERAL",
    "FLITERAL",
    "SLITERAL",
    "POSTPONE",
];

/// Source loader injected for INCLUDE/INCLUDED: resolved path -> text.
pub type SourceLoader = Box<dyn Fn(&str) -> anyhow::Result<String> + Send + Sync>;

/// Append names space-separated, wrapped at 78 columns, ending with a
/// newline (also when the list is empty -- WORDS' historic shape).
fn push_wrapped(out: &mut String, names: &[&str]) {
    let mut col = 0usize;
    for name in names {
        if col + name.len() + 1 > 78 && col > 0 {
            out.push('\n');
            col = 0;
        }
        out.push_str(name);
        out.push(' ');
        col += name.len() + 1;
    }
    out.push('\n');
}

/// Saved VM state for a MARKER word.
#[derive(Clone)]
struct MarkerState {
    dict_state: DictionaryState,
    user_here: u32,
    next_table_index: u32,
    word_pfa_map: HashMap<u32, u32>,
    ir_bodies: HashMap<WordId, Vec<IrOp>>,
    does_definitions: HashMap<WordId, DoesDefinition>,
    host_word_names: HashMap<WordId, String>,
    word_sources: HashMap<WordId, String>,
    two_value_words: std::collections::HashSet<u32>,
    fvalue_words: std::collections::HashSet<u32>,
    // Namespace + text state: search order, wordlist allocation,
    // REPLACES table, ABORT" texts
    search_order: Vec<u32>,
    next_wid: u32,
    current_wid: u32,
    substitutions: HashMap<String, Vec<u8>>,
    abort_messages_len: usize,
    // REMEMBER-style marker: state was saved just AFTER the marker word
    // was defined, and the entry survives its own execution (re-runnable)
    keep: bool,
}

// ---------------------------------------------------------------------------
// ForthVM
// ---------------------------------------------------------------------------

/// The complete Forth virtual machine -- owns dictionary, WASM runtime, and state.
pub struct ForthVM<R: Runtime> {
    dictionary: Dictionary,
    rt: R,
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
    // SEE source capture: verbatim text of the colon definition in progress
    // (accumulated across evaluate() calls), the position in the CURRENT
    // input buffer where capture (re)starts, the byte offset of the most
    // recently read token, and completed sources by word id.
    compiling_source: String,
    source_capture_from: Option<usize>,
    last_token_start: usize,
    word_sources: HashMap<WordId, String>,
    // INCLUDE machinery: injected source loader (CLI: filesystem; web:
    // virtual or absent) and the stack of files being included --
    // (resolved path, current 1-based line) -- for cycle/depth checks,
    // relative-path resolution, and file:line error context.
    source_loader: Option<SourceLoader>,
    include_frames: Vec<(String, usize)>,
    // Output buffer
    output: Arc<Mutex<String>>,
    // Next table index (mirrors dictionary.next_fn_index conceptually,
    // but we track what's actually in the wasmtime table)
    next_table_index: u32,
    // The emit function (shared across all instantiated modules)
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
    // 0 = none, 1 = CONSTANT, 2 = VARIABLE, 3 = CREATE, 4 = EVALUATE,
    // 5 = WORD, 6 = FIND, 7 = PARSE, 8 = PARSE-NAME, 9 = 2CONSTANT,
    // 10 = 2VARIABLE, 11 = DEFER, 12 = IMMEDIATE, 20 = GET-CURRENT,
    // 21 = SET-CURRENT, 25 = SEARCH-WORDLIST, 33 = DEFINITIONS,
    // 40 = WORDS, 41 = SEE, 42 = SEE-IR, 43 = HELP, 44 = INCLUDED,
    // 45 = INCLUDE
    pending_define: Arc<Mutex<Vec<i32>>>,
    /// Pending actions from host functions (COMPILE,, CS-PICK, CS-ROLL, POSTPONE of control words).
    pending_actions: Arc<Mutex<Vec<PendingAction>>>,
    // Pending DOES> patch: (does_action_id) to apply after word execution
    pending_does_patch: Arc<Mutex<Option<u32>>>,
    // Exception word set: throw code shared between CATCH and THROW host functions
    throw_code: Arc<Mutex<Option<i32>>>,
    // ABORT" texts by compiled index, plus the not-yet-reported payload of
    // the most recent ABORT" (printed only if the -2 throw goes uncaught)
    abort_messages: Arc<Mutex<Vec<String>>>,
    abort_message: Arc<Mutex<Option<String>>>,
    // Set by BYE: the embedding REPL/driver should exit
    bye: Arc<std::sync::atomic::AtomicBool>,
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
    /// Saved states for MARKER words: `marker_id` -> `MarkerState`
    marker_states: HashMap<u32, MarkerState>,
    /// EMPTY's rollback target: boot state, or wherever GILD re-baselined.
    gild_state: Option<Box<MarkerState>>,
    /// Table index of `_STACK_FAULT_` (compiled stack-guard target).
    stack_fault_id: u32,
    /// Pending MARKER restore: after a marker word executes, restore this state
    pending_marker_restore: Arc<Mutex<Option<u32>>>,
    /// Conditional compilation skip depth: >0 means we're skipping tokens for [IF]/[ELSE]
    conditional_skip_depth: u32,
    /// Next label ID for flat forward blocks (CS-ROLL'd IF/THEN patterns)
    next_block_label: u32,
    /// Local variable names for the current definition ({: ... :} syntax)
    compiling_locals: Vec<String>,
    /// Parallel to `compiling_locals`: kind of each local (Int or Float).
    compiling_local_kinds: Vec<LocalKind>,
    /// Forth 2012 §13.6.1.0086 `(LOCAL)` batch base: index into
    /// `compiling_locals` where the current `(LOCAL)` batch started.
    /// `None` means no pending batch. Set on the first `DeclareLocal` of a
    /// batch, cleared on `DeclareLocalEnd`, reset on `finish_colon_def`.
    local_batch_base: Option<usize>,
    /// Substitution table for SUBSTITUTE/REPLACES (String word set)
    substitutions: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    /// Search order: list of wordlist IDs (first = top of search order).
    /// Shared via Arc so host functions can modify it directly.
    search_order: Arc<Mutex<Vec<u32>>>,
    /// Next wordlist ID to allocate (shared).
    next_wid: Arc<Mutex<u32>>,
    /// xorshift64 PRNG state for RANDOM / RND-SEED.
    rng_state: Arc<Mutex<u64>>,
    /// Stacked compile state for nested definitions (quotations `[: ;]`).
    compile_frames: Vec<CompileFrame>,
    /// Dictionary address of the word currently being compiled. Set by
    /// `start_colon_def` / `start_noname_def` / `start_quotation` so that
    /// `finish_colon_def` can use `reveal_at` instead of `reveal()` — the
    /// latter breaks when intermediate dictionary entries (quotations,
    /// `DOES>` actions) have moved `latest`.
    compiling_word_addr: u32,
}

/// Snapshot of one compilation context. Pushed by `[:`, popped by `;]`.
struct CompileFrame {
    compiling_name: Option<String>,
    compiling_word_id: Option<WordId>,
    compiling_word_addr: u32,
    compiling_ir: Vec<IrOp>,
    control_stack: Vec<ControlEntry>,
    saw_create_in_def: bool,
    compiling_locals: Vec<String>,
    compiling_local_kinds: Vec<LocalKind>,
    local_batch_base: Option<usize>,
    state: i32,
}

/// Type of a Forth local. Int locals live on the data stack and use
/// `ForthLocalGet/Set`. Float locals live on the float stack and use
/// `ForthFLocalGet/Set`. Their WASM local index spaces are independent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalKind {
    Int,
    Float,
}

/// Standard message for a Forth 2012 THROW code (subset wafer can raise).
fn throw_message(code: i32) -> Option<&'static str> {
    Some(match code {
        -1 => "ABORT",
        -2 => "ABORT\"",
        -3 => "Stack overflow",
        -4 => "Stack underflow",
        -5 => "Return stack overflow",
        -6 => "Return stack underflow",
        -8 => "Dictionary overflow",
        -9 => "Invalid memory address",
        -10 => "Division by zero",
        -11 => "Result out of range",
        -13 => "Undefined word",
        -14 => "Interpreting a compile-only word",
        -16 => "Attempt to use zero-length string as a name",
        -18 => "Parsed string overflow",
        -22 => "Control structure mismatch",
        -24 => "Invalid numeric argument",
        -28 => "User interrupt",
        -31 => "Word not defined by CREATE",
        -42 => "Floating-point divide by zero",
        -43 => "Floating-point result out of range",
        -44 => "Floating-point stack overflow",
        -45 => "Floating-point stack underflow",
        -56 => "QUIT",
        _ => return None,
    })
}

/// Format a cell as Forth `.` would: signed, in the given base (2..=36).
fn fmt_in_base(v: i32, base: u32) -> String {
    let base = if (2..=36).contains(&base) { base } else { 10 };
    if base == 10 {
        return v.to_string();
    }
    let neg = v < 0;
    let mut m = (v as i64).unsigned_abs();
    let mut digits = Vec::new();
    loop {
        digits.push(char::from_digit((m % u64::from(base)) as u32, base).unwrap());
        m /= u64::from(base);
        if m == 0 {
            break;
        }
    }
    if neg {
        digits.push('-');
    }
    digits.iter().rev().collect::<String>().to_ascii_uppercase()
}

/// Pop the data-stack top from host-function context.
fn host_pop(ctx: &mut dyn HostAccess) -> anyhow::Result<i32> {
    let sp = ctx.get_dsp();
    if sp >= DATA_STACK_TOP {
        anyhow::bail!("Stack underflow");
    }
    let v = ctx.mem_read_i32(sp);
    ctx.set_dsp(sp + CELL_SIZE);
    Ok(v)
}

/// Ensure the data stack holds at least `n` cells; returns the stack
/// pointer for the caller's reads. Host words must check before raw
/// pointer arithmetic — compiled-code guards do not cover them.
fn host_need(ctx: &mut dyn HostAccess, n: u32) -> anyhow::Result<u32> {
    let sp = ctx.get_dsp();
    match n.checked_mul(CELL_SIZE).and_then(|b| sp.checked_add(b)) {
        Some(end) if end <= DATA_STACK_TOP => Ok(sp),
        _ => anyhow::bail!("Stack underflow"),
    }
}

/// Ensure the float stack holds at least `n` floats; returns the pointer.
fn host_fneed(ctx: &mut dyn HostAccess, n: u32) -> anyhow::Result<u32> {
    let sp = ctx.get_fsp();
    match n.checked_mul(FLOAT_SIZE).and_then(|b| sp.checked_add(b)) {
        Some(end) if end <= FLOAT_STACK_TOP => Ok(sp),
        _ => anyhow::bail!("Float stack underflow"),
    }
}

/// Checked float-stack pop for host words.
fn host_fpop(ctx: &mut dyn HostAccess) -> anyhow::Result<f64> {
    let sp = ctx.get_fsp();
    if sp >= FLOAT_STACK_TOP {
        anyhow::bail!("Float stack underflow");
    }
    let bytes: [u8; 8] = ctx
        .mem_read_slice(sp, 8)
        .try_into()
        .map_err(|_| anyhow::anyhow!("float stack read failed"))?;
    ctx.set_fsp(sp + FLOAT_SIZE);
    Ok(f64::from_le_bytes(bytes))
}

/// Advance past the next `\n` in `buf`, starting at `from`. Returns the
/// byte index of the first character on the next line (or `buf.len()` if
/// there's no more newline). Used by the `\` line-comment handler per
/// Forth 2012 §6.2.2535 to correctly stop at end-of-line instead of
/// end-of-input when the input buffer spans multiple lines.
fn skip_to_end_of_line(buf: &str, from: usize) -> usize {
    let bytes = buf.as_bytes();
    let mut i = from;
    while i < bytes.len() {
        let ch = bytes[i];
        i += 1;
        if ch == b'\n' {
            break;
        }
    }
    i
}

impl<R: Runtime> ForthVM<R> {
    /// Boot a new Forth VM with all primitives registered.
    pub fn new() -> anyhow::Result<Self> {
        Self::new_with_config(WaferConfig::default())
    }

    /// Boot a new Forth VM with custom optimization configuration.
    pub fn new_with_config(wafer_config: WaferConfig) -> anyhow::Result<Self> {
        let output = Arc::new(Mutex::new(String::new()));
        let rt = R::new(
            16,
            256,
            DATA_STACK_TOP,
            RETURN_STACK_TOP,
            FLOAT_STACK_TOP,
            Arc::clone(&output),
        )?;
        let dictionary = Dictionary::new();

        let mut vm = ForthVM {
            dictionary,
            rt,
            state: 0,
            base: 10,
            input_buffer: String::new(),
            input_pos: 0,
            compiling_name: None,
            compiling_ir: Vec::new(),
            control_stack: Vec::new(),
            compiling_word_id: None,
            compiling_source: String::new(),
            source_capture_from: None,
            last_token_start: 0,
            word_sources: HashMap::new(),
            source_loader: None,
            include_frames: Vec::new(),
            output,
            next_table_index: 0,
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
            pending_actions: Arc::new(Mutex::new(Vec::new())),
            pending_does_patch: Arc::new(Mutex::new(None)),
            throw_code: Arc::new(Mutex::new(None)),
            abort_messages: Arc::new(Mutex::new(Vec::new())),
            abort_message: Arc::new(Mutex::new(None)),
            bye: Arc::new(std::sync::atomic::AtomicBool::new(false)),
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
            gild_state: None,
            stack_fault_id: 0,
            pending_marker_restore: Arc::new(Mutex::new(None)),
            conditional_skip_depth: 0,
            next_block_label: 0,
            compiling_locals: Vec::new(),
            compiling_local_kinds: Vec::new(),
            local_batch_base: None,
            substitutions: Arc::new(Mutex::new(HashMap::new())),
            search_order: Arc::new(Mutex::new(vec![1])),
            next_wid: Arc::new(Mutex::new(2)),
            // SystemTime::now() PANICS on wasm32-unknown-unknown (no time
            // source), which turned VM construction into an `unreachable`
            // trap in the browser. Seed from the wall clock only where one
            // exists; wasm hosts start deterministic and reseed via RND-SEED.
            #[cfg(not(target_arch = "wasm32"))]
            rng_state: {
                use std::time::{SystemTime, UNIX_EPOCH};
                let seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0xDEAD_BEEF_CAFE_BABE, |d| d.as_nanos() as u64);
                Arc::new(Mutex::new(if seed == 0 {
                    0xDEAD_BEEF_CAFE_BABE
                } else {
                    seed
                }))
            },
            #[cfg(target_arch = "wasm32")]
            rng_state: Arc::new(Mutex::new(0xDEAD_BEEF_CAFE_BABE)),
            compile_frames: Vec::new(),
            compiling_word_addr: 0,
        };

        vm.register_primitives()?;

        // Boot state is the default EMPTY target (until GILD re-baselines)
        vm.gild_state = Some(Box::new(vm.snapshot_marker_state(true)));

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
                    self.compiling_local_kinds.clear();
                    self.local_batch_base = None;
                    self.compile_frames.clear();
                    self.compiling_source.clear();
                    self.source_capture_from = None;
                    // QUIT and ABORT are not errors: the wipe above IS their
                    // "enter interpretation state", and the standard asks for
                    // the user input device back and no message at all (ABORT
                    // is defined as emptying the data stack and then doing
                    // QUIT; only ABORT" prints, and that is code -2). The rest
                    // of this input -- and any EVALUATE / INCLUDE frame it
                    // unwound through -- is abandoned by returning here.
                    let mut tc = self.throw_code.lock().unwrap();
                    if matches!(*tc, Some(QUIT_THROW | ABORT_THROW)) {
                        *tc = None;
                        drop(tc);
                        self.rt.mem_write_i32(crate::memory::SYSVAR_SOURCE_ID, 0);
                        self.include_frames.clear();
                        return Ok(());
                    }
                    drop(tc);
                    return Err(self.describe_uncaught(e));
                }
            }
            if self.bye.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            // Read >IN back from WASM memory. Only apply if Forth code changed it
            // (i.e., the WASM value differs from what sync_input_to_wasm wrote).
            // This distinguishes Forth's `>IN !` from Rust-side parse_until changes.
            let wasm_to_in = self.rt.mem_read_i32(SYSVAR_TO_IN) as u32 as usize;
            if wasm_to_in != wasm_to_in_before {
                self.input_pos = wasm_to_in;
            }
            // If >IN was set past the end of the input, stop processing
            if self.input_pos >= self.input_buffer.len() {
                break;
            }
        }

        // Multi-line definition: bank this buffer's tail into the capture
        // and continue from the start of the next buffer.
        if self.state != 0
            && let Some(from) = self.source_capture_from
        {
            let from = from.min(self.input_buffer.len());
            self.compiling_source.push_str(&self.input_buffer[from..]);
            self.compiling_source.push('\n');
            self.source_capture_from = Some(0);
        }

        Ok(())
    }

    /// Check if the VM is currently in compile mode.
    pub fn is_compiling(&self) -> bool {
        self.state != 0
    }

    /// True once BYE has executed; the embedding REPL/driver should exit.
    pub fn bye_requested(&self) -> bool {
        self.bye.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Turn an internal error into the message shown to the user.
    ///
    /// An uncaught THROW leaves its code in `throw_code` (the "forth-throw"
    /// sentinel error itself may get wrapped in a wasmtime trap when the
    /// throw crosses compiled code, so the code — not the message — is the
    /// reliable signal). Map the code to its standard message, or to the
    /// recorded text for `ABORT"`. Always drains the payload so a stale
    /// code can't leak into a later, unrelated error.
    fn describe_uncaught(&mut self, e: anyhow::Error) -> anyhow::Error {
        let code = self.throw_code.lock().unwrap().take();
        let text = self.abort_message.lock().unwrap().take();
        let (code, message) = match (code, text) {
            (Some(-2), Some(t)) => (-2, t),
            (Some(code), _) => match throw_message(code) {
                Some(m) => (code, format!("{m} (throw {code})")),
                None => (code, format!("Catch = {code}")),
            },
            (None, _) => return e,
        };
        // Typed carrier: display text unchanged, THROW code reachable via
        // downcast_ref::<WaferError>() for CLI/web consumers.
        anyhow::Error::new(crate::error::WaferError::UncaughtThrow { code, message })
    }

    /// Get and clear the output buffer.
    pub fn take_output(&mut self) -> String {
        let mut out = self.output.lock().unwrap();
        let s = out.clone();
        out.clear();
        s
    }

    /// Mutable access to the underlying runtime — useful for tests and for
    /// host shims that need to read or write WAFER linear memory directly.
    pub fn runtime_mut(&mut self) -> &mut R {
        &mut self.rt
    }

    /// Read the current data stack contents (top-first).
    pub fn data_stack(&mut self) -> Vec<i32> {
        let sp = self.rt.get_dsp();
        let mut stack = Vec::new();
        let mut addr = sp;
        while addr < DATA_STACK_TOP {
            stack.push(self.rt.mem_read_i32(addr));
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
        self.rt.mem_read_slice(0, self.user_here as usize)
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

    /// Names of all user-facing words (visible, non-internal), newest
    /// first — the completion/browsing view of the dictionary.
    pub fn word_names(&self) -> Vec<String> {
        self.dictionary.visible_words(false)
    }

    /// Resolve a word name to its `WordId`. Returns `None` if not found.
    pub fn resolve_word(&self, name: &str) -> Option<WordId> {
        self.dictionary
            .find(&name.to_ascii_uppercase())
            .map(|(_, id, _)| id)
    }

    /// Current function table size.
    pub fn current_table_size(&mut self) -> u32 {
        self.rt.table_size()
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
        self.last_token_start = start;
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

        // Quotations `[: ... ;]` — state-smart anonymous xt, nestable inside
        // colon definitions via the compile-frame stack.
        if token_upper == "[:" {
            return self.start_quotation();
        }
        if token_upper == ";]" {
            return self.finish_quotation();
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
                self.rt.mem_write_slice(addr as u32, bytes);
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
                self.rt.mem_write_slice(addr as u32, &raw);
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
                self.rt.mem_write_u8(addr as u32, len);
                self.rt.mem_write_slice(addr as u32 + 1, bytes);
                self.user_here += 1 + len as u32;
                self.sync_here_cell();
                self.push_data_stack(addr as i32)?;
            }
            return Ok(());
        }
        if token_upper == "S" {
            // State-smart string literal for the next whitespace-delimited token.
            // Interpret mode: copy token bytes to HERE-space (stable across REFILL),
            // push ( c-addr u ). Compile-mode branch lives in compile_token.
            if let Some(name) = self.next_token() {
                self.refresh_user_here();
                let addr = self.user_here;
                let bytes = name.as_bytes();
                let len = bytes.len() as u32;
                self.rt.mem_write_slice(addr, bytes);
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
            // Forth 2012 §6.2.2535: `\` parses and discards the remainder
            // of the *line*, not the remainder of the input buffer. Stop
            // at the first `\n`; fall through to end-of-buffer otherwise.
            self.input_pos = skip_to_end_of_line(&self.input_buffer, self.input_pos);
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
            "MARKER" => return self.define_marker(false),
            "REMEMBER" => return self.define_marker(true),
            "GILD" => {
                self.gild_state = Some(Box::new(self.snapshot_marker_state(true)));
                return Ok(());
            }
            "EMPTY" => {
                if let Some(state) = self.gild_state.clone() {
                    self.apply_marker_state(*state);
                }
                return Ok(());
            }
            "2CONSTANT" => return self.define_2constant(),
            "2VARIABLE" => return self.define_2variable(),
            "2VALUE" => return self.define_2value(),
            "FVARIABLE" => return self.define_fvariable(),
            "FCONSTANT" => return self.define_fconstant(),
            "FVALUE" => return self.define_fvalue(),
            "CONSOLIDATE" => return self.consolidate(),
            "SYNONYM" => return self.define_synonym(),
            "ORDER" => {
                // wid 1 is FORTH-WORDLIST; other wids are anonymous.
                let wid_name = |wid: u32| {
                    if wid == 1 {
                        "FORTH".to_string()
                    } else {
                        format!("wid#{wid}")
                    }
                };
                let so = self.search_order.lock().unwrap();
                let names: Vec<String> = so.iter().map(|&w| wid_name(w)).collect();
                let output = format!(
                    "Search order: {}  Compilation: {}\n",
                    names.join(" "),
                    wid_name(self.dictionary.current_wid())
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

        // Try to convert as a number; punctuation makes it double-cell
        if let Some(lit) = self.parse_numeric_literal(token) {
            self.record_number_conversion(lit);
            self.push_data_stack(lit.lo())?;
            if self.recording_toplevel && self.state == 0 {
                self.toplevel_ir.push(IrOp::PushI32(lit.lo()));
            }
            if lit.is_double {
                self.push_data_stack(lit.hi())?;
                if self.recording_toplevel && self.state == 0 {
                    self.toplevel_ir.push(IrOp::PushI32(lit.hi()));
                }
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

        // Constructs the outer interpreter only knows how to compile. Forth
        // 2012 leaves their interpretation semantics undefined and both gforth
        // and SwiftForth name the standard condition, so say what is wrong
        // instead of claiming the word does not exist.
        if INTERPRETER_TOKENS.contains(&token.to_uppercase().as_str()) {
            anyhow::bail!("interpreting a compile-only word: {token} (throw -14)");
        }

        anyhow::bail!("unknown word: {token}");
    }

    /// Compile a token in compile mode.
    fn compile_token(&mut self, token: &str) -> anyhow::Result<()> {
        let token_upper = token.to_ascii_uppercase();

        // Forth 2012 §13.3.3.2 — locals supersede dictionary names (and,
        // by extension, hardcoded compile-mode shortcuts) within their
        // declaration scope. Checked here, before any hardcoded token
        // handling, to keep that precedence uniform — otherwise e.g. a
        // local named `s` would be hijacked by the `S` string shortcut
        // below.
        if let Some(idx) = self
            .compiling_locals
            .iter()
            .position(|n| n.eq_ignore_ascii_case(token))
        {
            let kind = self.compiling_local_kinds[idx];
            let kind_idx = self.compiling_local_kinds[0..idx]
                .iter()
                .filter(|k| **k == kind)
                .count() as u32;
            match kind {
                LocalKind::Int => self.push_ir(IrOp::ForthLocalGet(kind_idx)),
                LocalKind::Float => self.push_ir(IrOp::ForthFLocalGet(kind_idx)),
            }
            return Ok(());
        }

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
                self.rt.mem_write_slice(addr as u32, bytes);
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
                self.rt.mem_write_u8(addr as u32, len);
                self.rt.mem_write_slice(addr as u32 + 1, bytes);
                self.user_here += 1 + len as u32;
                self.sync_here_cell();
                self.push_ir(IrOp::PushI32(addr as i32));
            }
            return Ok(());
        }
        if token_upper == "S" {
            // Compile-mode twin of the interpret-mode S handler: parse next
            // whitespace-delimited token, copy into HERE, compile ( c-addr u )
            // literals. Bit-identical to writing S" name" inline.
            if let Some(name) = self.next_token() {
                self.refresh_user_here();
                let addr = self.user_here;
                let bytes = name.as_bytes();
                let len = bytes.len() as u32;
                self.rt.mem_write_slice(addr, bytes);
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
            // See interpret-mode branch: `\` ends at `\n`, not at `#TIB`.
            self.input_pos = skip_to_end_of_line(&self.input_buffer, self.input_pos);
            return Ok(());
        }

        // Handle ABORT" in compile mode
        if token_upper == "ABORT\"" {
            if let Some(s) = self.parse_until('"') {
                // Record the text and compile: IF <idx> _ABORT_Q_ THEN.
                // _ABORT_Q_ stashes the text as the pending abort payload
                // and throws -2; the text is shown only if nothing CATCHes.
                let idx = {
                    let mut msgs = self.abort_messages.lock().unwrap();
                    msgs.push(s);
                    (msgs.len() - 1) as i32
                };
                let abort_q = self.dictionary.find("_ABORT_Q_").map(|(_, id, _)| id);
                let mut then_body = vec![IrOp::PushI32(idx)];
                if let Some(id) = abort_q {
                    then_body.push(IrOp::Call(id));
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
            "AHEAD" => return self.compile_ahead(),
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
                    let end = (c_addr as usize).saturating_add(u as usize);
                    if end <= self.rt.mem_len() {
                        let bytes: Vec<u8> = self
                            .rt
                            .mem_read_slice(c_addr as u32, (end - c_addr as usize) as usize);
                        self.rt.mem_write_slice(new_addr as u32, &bytes);
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
                    let upper = next.to_uppercase();
                    // Check for compile-time control-flow keywords first
                    let ctrl_code = match upper.as_str() {
                        "IF" => Some(CTRL_IF),
                        "ELSE" => Some(CTRL_ELSE),
                        "THEN" => Some(CTRL_THEN),
                        "BEGIN" => Some(CTRL_BEGIN),
                        "UNTIL" => Some(CTRL_UNTIL),
                        "WHILE" => Some(CTRL_WHILE),
                        "REPEAT" => Some(CTRL_REPEAT),
                        "AGAIN" => Some(CTRL_AGAIN),
                        "DO" => Some(CTRL_DO),
                        "LOOP" => Some(CTRL_LOOP),
                        "+LOOP" => Some(CTRL_PLUS_LOOP),
                        "AHEAD" => Some(CTRL_AHEAD),
                        _ => None,
                    };
                    if let Some(code) = ctrl_code {
                        // Compile code that pushes the action code and calls __CTRL__
                        let ctrl_id = self
                            .dictionary
                            .find("__CTRL__")
                            .map(|(_, id, _)| id)
                            .ok_or_else(|| anyhow::anyhow!("POSTPONE: __CTRL__ not found"))?;
                        self.push_ir(IrOp::PushI32(code));
                        self.push_ir(IrOp::Call(ctrl_id));
                    } else if let Some((_addr, word_id, is_imm)) = self.dictionary.find(&next) {
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
                    self.rt.mem_write_slice(addr as u32, &raw);
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

        // Look up in dictionary (search order, then fallback to all wordlists)
        if let Some((_addr, word_id, is_immediate)) = self.dictionary.find(token) {
            if is_immediate {
                // Execute immediately even in compile mode
                self.execute_word(word_id)?;
                // Handle any pending COMPILE, operations from POSTPONE
                self.handle_pending_actions()?;
            } else {
                self.push_ir(IrOp::Call(word_id));
            }
            return Ok(());
        }

        // Try to convert as a number; punctuation makes it double-cell
        if let Some(lit) = self.parse_numeric_literal(token) {
            self.record_number_conversion(lit);
            self.push_ir(IrOp::PushI32(lit.lo()));
            if lit.is_double {
                self.push_ir(IrOp::PushI32(lit.hi()));
            }
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
            Some(ControlEntry::ForwardBlock { label }) => {
                // CS-ROLL'd flat forward block: just emit EndBlock
                self.compiling_ir.push(IrOp::EndBlock(label));
            }
            Some(ControlEntry::Ahead {
                prefix: ahead_prefix,
            }) => {
                // AHEAD...THEN: code between is skipped (dead code).
                let skipped = std::mem::take(&mut self.compiling_ir);

                // Check if a Begin is on the stack (AHEAD + CS-ROLL into a loop).
                // In that case, the skipped code becomes "skip on first iteration."
                let begin_idx = self
                    .control_stack
                    .iter()
                    .rposition(|e| matches!(e, ControlEntry::Begin { .. }));

                if let Some(bi) = begin_idx {
                    if !skipped.is_empty() {
                        // Replace Begin's prefix (which is dead code between AHEAD and BEGIN)
                        // with AHEAD's prefix (code before AHEAD that should execute).
                        if let ControlEntry::Begin { body: ref mut bp } = self.control_stack[bi] {
                            *bp = ahead_prefix;
                        }
                        // Emit a first-iteration guard: allocate a local flag.
                        // This is an Int local; its kind-local-index is the count of
                        // existing Int entries.
                        let flag_idx = self
                            .compiling_local_kinds
                            .iter()
                            .filter(|k| **k == LocalKind::Int)
                            .count() as u32;
                        self.compiling_locals.push("__first_iter__".to_string());
                        self.compiling_local_kinds.push(LocalKind::Int);
                        // Push flag init into the Begin's prefix (before the loop)
                        if let ControlEntry::Begin { body: ref mut bp } = self.control_stack[bi] {
                            bp.push(IrOp::PushI32(1));
                            bp.push(IrOp::ForthLocalSet(flag_idx));
                        }
                        // In the loop body: if flag==0 execute skipped code, else clear flag
                        self.compiling_ir.push(IrOp::ForthLocalGet(flag_idx));
                        self.compiling_ir.push(IrOp::ZeroEq);
                        self.compiling_ir.push(IrOp::If {
                            then_body: skipped,
                            else_body: Some(vec![IrOp::PushI32(0), IrOp::ForthLocalSet(flag_idx)]),
                        });
                    } else {
                        // No code to skip — replace Begin's dead-code prefix
                        if let ControlEntry::Begin { body: ref mut bp } = self.control_stack[bi] {
                            *bp = ahead_prefix;
                        }
                    }
                } else {
                    // Simple case: no loop context, discard skipped code
                    self.compiling_ir = ahead_prefix;
                }
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
                let mut body = std::mem::take(&mut self.compiling_ir);
                // Desugar any LoopRestartIfFalse markers from CS-PICK'd UNTIL
                body = Self::desugar_loop_restarts(body);
                self.compiling_ir = prefix;
                self.compiling_ir.push(IrOp::BeginUntil { body });
            }
            Some(ControlEntry::BeginRef) => {
                // CS-PICK'd BEGIN: emit inline conditional restart instead of a full loop
                self.compiling_ir.push(IrOp::LoopRestartIfFalse);
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
    // AHEAD, CS-PICK, CS-ROLL (Programming-Tools)
    // -----------------------------------------------------------------------

    /// AHEAD — unconditional forward branch. Code between AHEAD and THEN is skipped.
    fn compile_ahead(&mut self) -> anyhow::Result<()> {
        let prefix = std::mem::take(&mut self.compiling_ir);
        self.control_stack.push(ControlEntry::Ahead { prefix });
        // compiling_ir is now empty — collects dead code between AHEAD and THEN
        Ok(())
    }

    /// CS-PICK — ( n -- ) Copy the n-th control-flow stack entry to the top.
    fn cs_pick(&mut self, n: u32) -> anyhow::Result<()> {
        let len = self.control_stack.len();
        if (n as usize) >= len {
            anyhow::bail!("CS-PICK: index {n} out of range (control stack depth {len})");
        }
        let idx = len - 1 - n as usize;
        let entry = &self.control_stack[idx];
        match entry {
            ControlEntry::Begin { .. } => {
                // CS-PICK of a BEGIN dest: push a reference marker.
                // When UNTIL resolves this, it emits LoopRestartIfFalse
                // instead of creating a full BeginUntil.
                self.control_stack.push(ControlEntry::BeginRef);
            }
            _ => {
                // Clone the entry for all other types
                let cloned = entry.clone();
                self.control_stack.push(cloned);
            }
        }
        Ok(())
    }

    /// CS-ROLL — ( n -- ) Rotate the top n+1 control-flow stack entries.
    /// 1 CS-ROLL = swap top two entries.
    /// 2 CS-ROLL = rotate top three (bring 3rd to top).
    fn cs_roll(&mut self, n: u32) -> anyhow::Result<()> {
        let len = self.control_stack.len();
        if (n as usize) >= len {
            anyhow::bail!("CS-ROLL: index {n} out of range (control stack depth {len})");
        }
        if n == 0 {
            return Ok(());
        }

        // Check how many If entries are in the top n+1 entries
        let start = len - 1 - n as usize;
        let if_count = self.control_stack[start..]
            .iter()
            .filter(|e| matches!(e, ControlEntry::If { .. }))
            .count();

        if if_count >= 2 {
            // Multiple If entries being reordered: linearize into Block/BranchIfFalse.
            self.linearize_if_entries(n)?;
        } else if n == 1 {
            // 1 CS-ROLL = swap. Check for IF + BEGIN pattern (= WHILE equivalent).
            let top = self.control_stack.pop();
            let second = self.control_stack.pop();
            match (second, top) {
                (
                    Some(ControlEntry::Begin { body: prefix }),
                    Some(ControlEntry::If { then_body: test }),
                ) => {
                    // Begin below + If on top → 1 CS-ROLL = WHILE equivalent
                    self.control_stack
                        .push(ControlEntry::BeginWhile { test, body: prefix });
                }
                (Some(s), Some(t)) => {
                    // Generic swap
                    self.control_stack.push(t);
                    self.control_stack.push(s);
                }
                _ => anyhow::bail!("CS-ROLL: control stack underflow"),
            }
        } else {
            // General rotation
            let idx = len - 1 - n as usize;
            let entry = self.control_stack.remove(idx);
            self.control_stack.push(entry);
        }
        Ok(())
    }

    /// Linearize If entries from the control stack into flat Block/BranchIfFalse code.
    /// Called when CS-ROLL reorders multiple If entries.
    ///
    /// Converts nested If prefixes into a linear sequence:
    /// `Block(label_n) ... Block(label_1) prefix1 BranchIfFalse(label_1) prefix2 BranchIfFalse(label_2) ...`
    /// Then THENs emit `EndBlock(label)` to close each block.
    fn linearize_if_entries(&mut self, n: u32) -> anyhow::Result<()> {
        let len = self.control_stack.len();
        let start = len - 1 - n as usize;

        // Pop the top n+1 entries
        let entries: Vec<ControlEntry> = self.control_stack.drain(start..).collect();

        // Assign a label to each If entry, extract its prefix, build linear code
        let mut labels = Vec::new(); // label per entry (0 for non-If)
        let mut linear_code = Vec::new();
        for entry in &entries {
            if let ControlEntry::If { then_body: prefix } = entry {
                let label = self.next_block_label;
                self.next_block_label += 1;
                labels.push(label);
                linear_code.extend(prefix.iter().cloned());
                linear_code.push(IrOp::BranchIfFalse(label));
            } else {
                labels.push(u32::MAX); // sentinel for non-If
            }
        }
        // Append current compiling_ir (code compiled after the last IF)
        linear_code.extend(std::mem::take(&mut self.compiling_ir));

        // Rotate: bring entry at index 0 (= deepest of the n+1) to the top
        let mut rotated_labels = labels.clone();
        let first = rotated_labels.remove(0);
        rotated_labels.push(first);

        // Build Block openings in the order entries appear after rotation
        // (first entry = outermost block = last to close)
        let mut blocks = Vec::new();
        for &label in &rotated_labels {
            if label != u32::MAX {
                blocks.push(IrOp::Block(label));
            }
        }

        // Final compiling_ir: Block openings + linearized code
        blocks.extend(linear_code);
        self.compiling_ir = blocks;

        // Push rotated ForwardBlock entries onto control stack
        for &label in &rotated_labels {
            if label != u32::MAX {
                self.control_stack
                    .push(ControlEntry::ForwardBlock { label });
            }
            // Non-If entries: not supported in the all-If case
        }

        Ok(())
    }

    /// Desugar `LoopRestartIfFalse` markers in a loop body into nested `If` nodes.
    /// Each marker becomes: `If { then_body: [rest...], else_body: Some([PushI32(0)]) }`
    /// so that a false condition produces 0 for the outer UNTIL to restart the loop.
    fn desugar_loop_restarts(body: Vec<IrOp>) -> Vec<IrOp> {
        if let Some(pos) = body
            .iter()
            .position(|op| matches!(op, IrOp::LoopRestartIfFalse))
        {
            let mut prefix: Vec<IrOp> = body[..pos].to_vec();
            let rest: Vec<IrOp> = body[pos + 1..].to_vec();
            let desugared_rest = Self::desugar_loop_restarts(rest);
            prefix.push(IrOp::If {
                then_body: desugared_rest,
                else_body: Some(vec![IrOp::PushI32(0)]),
            });
            prefix
        } else {
            body
        }
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
        self.compiling_word_addr = self.dictionary.latest();
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
        // SEE source capture starts at the `:` token itself (its position
        // was recorded by next_token before dispatch reached us).
        self.compiling_source.clear();
        self.source_capture_from = Some(self.last_token_start);
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
        self.compiling_word_addr = self.dictionary.latest();
        self.compiling_ir.clear();
        self.control_stack.clear();
        self.state = -1;
        self.saw_create_in_def = false;
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);

        Ok(())
    }

    /// `[:` — start a quotation. Saves the current compile frame (if any)
    /// and begins compiling an anonymous inner definition. The inner xt is
    /// produced by `;]`.
    fn start_quotation(&mut self) -> anyhow::Result<()> {
        let frame = CompileFrame {
            compiling_name: self.compiling_name.take(),
            compiling_word_id: self.compiling_word_id.take(),
            compiling_word_addr: self.compiling_word_addr,
            compiling_ir: std::mem::take(&mut self.compiling_ir),
            control_stack: std::mem::take(&mut self.control_stack),
            saw_create_in_def: self.saw_create_in_def,
            compiling_locals: std::mem::take(&mut self.compiling_locals),
            compiling_local_kinds: std::mem::take(&mut self.compiling_local_kinds),
            local_batch_base: self.local_batch_base.take(),
            state: self.state,
        };
        self.compile_frames.push(frame);

        let name = format!("_quot_{}_", self.next_table_index);
        let word_id = self
            .dictionary
            .create(&name, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        self.compiling_word_addr = self.dictionary.latest();
        self.dictionary.reveal();

        self.compiling_name = Some(name);
        self.compiling_word_id = Some(word_id);
        self.compiling_ir.clear();
        self.control_stack.clear();
        self.state = -1;
        self.saw_create_in_def = false;
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);
        Ok(())
    }

    /// `;]` — finish the current quotation. Compiles its body as an anonymous
    /// word, pops the saved outer frame, and either pushes the new xt on the
    /// data stack (interpret mode) or emits a literal push into the outer IR
    /// (compile mode).
    fn finish_quotation(&mut self) -> anyhow::Result<()> {
        if self.compile_frames.is_empty() {
            anyhow::bail!(";]: no matching [:");
        }
        let inner_xt = self
            .compiling_word_id
            .ok_or_else(|| anyhow::anyhow!(";]: no active quotation"))?
            .0;
        self.finish_colon_def()?;

        let frame = self.compile_frames.pop().unwrap();
        self.compiling_name = frame.compiling_name;
        self.compiling_word_id = frame.compiling_word_id;
        self.compiling_word_addr = frame.compiling_word_addr;
        self.compiling_ir = frame.compiling_ir;
        self.control_stack = frame.control_stack;
        self.saw_create_in_def = frame.saw_create_in_def;
        self.compiling_locals = frame.compiling_locals;
        self.compiling_local_kinds = frame.compiling_local_kinds;
        self.local_batch_base = frame.local_batch_base;
        self.state = frame.state;

        if self.state != 0 {
            self.push_ir(IrOp::PushI32(inner_xt as i32));
        } else {
            self.push_data_stack(inner_xt as i32)?;
        }
        Ok(())
    }

    /// Run all enabled optimization passes on an IR sequence.
    fn optimize_ir(&self, ir: Vec<IrOp>, bodies: &HashMap<WordId, Vec<IrOp>>) -> Vec<IrOp> {
        optimize(ir, &self.config.opt, bodies)
    }

    /// Parse a `{: args | locals -- comment :}` block and compile local
    /// initializations. Supports `F:` prefix (gforth/SwiftForth-style) to
    /// mark the next local as float-typed. Int locals pop from the data
    /// stack via `ForthLocalSet`; float locals pop from the float stack
    /// via `ForthFLocalSet`.
    fn compile_locals_block(&mut self) -> anyhow::Result<()> {
        let mut args: Vec<(String, LocalKind)> = Vec::new();
        let mut uninits: Vec<(String, LocalKind)> = Vec::new();
        let mut in_comment = false;
        let mut in_uninit = false;
        let mut next_is_float = false;

        loop {
            let tok = self
                .next_token()
                .ok_or_else(|| anyhow::anyhow!("{{: missing :}}"))?;
            let tok_upper = tok.to_ascii_uppercase();
            match tok_upper.as_str() {
                ":}" => break,
                "--" => in_comment = true,
                "|" => in_uninit = true,
                "F:" => next_is_float = true,
                _ => {
                    if in_comment {
                        continue;
                    }
                    let kind = if next_is_float {
                        LocalKind::Float
                    } else {
                        LocalKind::Int
                    };
                    next_is_float = false;
                    if in_uninit {
                        uninits.push((tok_upper, kind));
                    } else {
                        args.push((tok_upper, kind));
                    }
                }
            }
        }

        let base = self.compiling_locals.len();
        let n_args = args.len();

        // Args first (assigned stack→local), then uninits (no init pop).
        for (name, kind) in args.iter().chain(uninits.iter()) {
            self.compiling_locals.push(name.clone());
            self.compiling_local_kinds.push(*kind);
        }

        // Emit init: pop in reverse declaration order. Rightmost arg is on
        // the top of its stack, so it's assigned first.
        for i in (0..n_args).rev() {
            let slot = base + i;
            let kind = self.compiling_local_kinds[slot];
            let kind_idx = self.compiling_local_kinds[0..slot]
                .iter()
                .filter(|k| **k == kind)
                .count() as u32;
            match kind {
                LocalKind::Int => self.push_ir(IrOp::ForthLocalSet(kind_idx)),
                LocalKind::Float => self.push_ir(IrOp::ForthFLocalSet(kind_idx)),
            }
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
        self.compiling_local_kinds.clear();
        self.local_batch_base = None;

        let name = self
            .compiling_name
            .take()
            .ok_or_else(|| anyhow::anyhow!("no word being compiled"))?;
        let word_id = self
            .compiling_word_id
            .take()
            .ok_or_else(|| anyhow::anyhow!("no word being compiled"))?;
        // SEE: bank the tail of the current buffer through the `;` token.
        // Capture is only armed by `:` — :NONAME and quotations never store.
        if let Some(from) = self.source_capture_from.take() {
            let end = self.input_pos.min(self.input_buffer.len());
            let mut src = std::mem::take(&mut self.compiling_source);
            src.push_str(&self.input_buffer[from.min(end)..end]);
            self.word_sources
                .insert(word_id, src.trim_end().to_string());
        }

        let ir = std::mem::take(&mut self.compiling_ir);
        let bodies = self.ir_bodies.clone();
        let ir = self.optimize_ir(ir, &bodies);
        self.ir_bodies.insert(word_id, ir.clone());

        // Compile to WASM
        let config = self.codegen_config(word_id.0);
        let compiled =
            compile_word(&name, &ir, &config).map_err(|e| anyhow::anyhow!("codegen error: {e}"))?;

        // Instantiate and install in the table
        self.instantiate_and_install(&compiled, word_id)?;

        // Reveal the word by its saved address (not LATEST, which may have
        // moved due to intermediate dict entries — quotations, DOES> helpers).
        if self.compiling_word_addr != 0 {
            self.dictionary.reveal_at(self.compiling_word_addr);
        } else {
            self.dictionary.reveal();
        }
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
        let module_bytes = compile_consolidated_module(
            &words,
            &local_fn_map,
            table_size,
            self.stack_guard_param(),
        )
        .map_err(|e| anyhow::anyhow!("consolidation codegen error: {e}"))?;

        // Instantiate: the element section in the module handles table placement
        // We use fn_index=0 since the element section has the correct offsets
        self.rt.instantiate_and_install(&module_bytes, 0)?;

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

        let module_bytes = compile_consolidated_module(
            &words,
            &local_fn_map,
            table_size,
            self.stack_guard_param(),
        )
        .map_err(|e| anyhow::anyhow!("batch compile error: {e}"))?;

        self.total_module_bytes += module_bytes.len() as u64;
        // Instantiate: the element section in the module handles table placement
        self.rt.instantiate_and_install(&module_bytes, 0)?;

        Ok(())
    }

    // -----------------------------------------------------------------------
    // WASM instantiation
    // -----------------------------------------------------------------------

    /// Get the current table size.
    fn table_size(&mut self) -> u32 {
        self.rt.table_size()
    }

    /// Ensure the table is large enough for the given index.
    fn ensure_table_size(&mut self, needed: u32) -> anyhow::Result<()> {
        self.rt.ensure_table_size(needed)
    }

    /// Instantiate a compiled WASM module and install its function in the table.
    fn instantiate_and_install(
        &mut self,
        compiled: &CompiledModule,
        word_id: WordId,
    ) -> anyhow::Result<()> {
        self.rt.ensure_table_size(word_id.0)?;
        self.total_module_bytes += compiled.bytes.len() as u64;
        self.rt.instantiate_and_install(&compiled.bytes, word_id.0)
    }

    // -----------------------------------------------------------------------
    // Word execution
    // -----------------------------------------------------------------------

    /// Execute a word by its `WordId` (calls through the function table).
    fn execute_word(&mut self, word_id: WordId) -> anyhow::Result<()> {
        // Rebuild word lookup so inline FIND host function has latest data
        self.rebuild_word_lookup();

        self.rt.call_func(word_id.0)?;
        // Check if the word changed BASE via WASM memory
        self.sync_base_from_wasm();
        // Handle pending defining actions (CONSTANT, VARIABLE, CREATE called at runtime)
        self.handle_pending_define()?;
        // Handle pending DOES> patch (runtime DOES> from double-DOES> words)
        self.handle_pending_does_patch()?;
        // Handle pending COMPILE, operations (used by [ ... ] sequences)
        self.handle_pending_actions()?;
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
        let sp = self.rt.get_dsp();
        let mem_len = self.rt.mem_len() as u32;
        if sp < CELL_SIZE + crate::memory::DATA_STACK_BASE || sp > mem_len {
            anyhow::bail!("data stack overflow");
        }
        let new_sp = sp - CELL_SIZE;
        self.rt.mem_write_i32(new_sp, value);
        self.rt.set_dsp(new_sp);
        Ok(())
    }

    /// Pop a value from the data stack.
    fn pop_data_stack(&mut self) -> anyhow::Result<i32> {
        let sp = self.rt.get_dsp();
        let mem_len = self.rt.mem_len() as u32;
        if sp >= DATA_STACK_TOP || sp > mem_len {
            anyhow::bail!("stack underflow");
        }
        let value = self.rt.mem_read_i32(sp);
        self.rt.set_dsp(sp + CELL_SIZE);
        Ok(value)
    }

    // -----------------------------------------------------------------------
    // Float stack operations
    // -----------------------------------------------------------------------

    /// Push a value onto the float stack.
    fn fpush(&mut self, val: f64) -> anyhow::Result<()> {
        let sp = self.rt.get_fsp();
        let new_sp = sp - FLOAT_SIZE;
        if new_sp < FLOAT_STACK_BASE {
            anyhow::bail!("float stack overflow");
        }
        self.rt.set_fsp(new_sp);
        self.rt.mem_write_slice(new_sp, &val.to_le_bytes());
        Ok(())
    }

    /// Pop a value from the float stack.
    fn fpop(&mut self) -> anyhow::Result<f64> {
        let sp = self.rt.get_fsp();
        if sp >= FLOAT_STACK_TOP {
            anyhow::bail!("float stack underflow");
        }
        let bytes = self.rt.mem_read_slice(sp, 8);
        self.rt.set_fsp(sp + 8);
        Ok(f64::from_le_bytes(bytes.try_into().unwrap()))
    }

    /// Read the current float stack contents (top-first).
    #[cfg(test)]
    fn float_stack(&mut self) -> Vec<f64> {
        let sp = self.rt.get_fsp();
        let mut stack = Vec::new();
        let mut addr = sp;
        while addr < FLOAT_STACK_TOP {
            let bytes = self.rt.mem_read_slice(addr, 8);
            stack.push(f64::from_le_bytes(bytes.try_into().unwrap()));
            addr += FLOAT_SIZE;
        }
        stack
    }

    // -----------------------------------------------------------------------
    // Number parsing
    // -----------------------------------------------------------------------

    /// Try to convert a token to a number, following `SwiftForth`'s input
    /// number conversion rules.
    ///
    /// Punctuation (`,` `.` `+` `-` `/` `:`) forces double-cell conversion, so
    /// `12.34`, `1,234`, `12:30:45` and `2026-08-06` all convert as doubles.
    /// Only a leading `-` escapes this and binds as a sign, which keeps `-1`
    /// single-cell; a leading `+` stays punctuation, so `+7` is the double 7.
    ///
    /// `DPL` counts up once per digit from [`DPL_INIT`] and resets to zero at
    /// every punctuation character, so it ends up holding the digit count right
    /// of the rightmost punctuation, and stays negative for unpunctuated tokens.
    fn parse_numeric_literal(&self, token: &str) -> Option<NumberLiteral> {
        let token = token.trim();
        if token.is_empty() {
            return None;
        }

        // A leading sign binds to the number; it is not double punctuation.
        let (neg_outer, rest) = strip_sign(token);

        if rest.is_empty() {
            return None;
        }

        // Character literal: 'x' → ASCII value of x. No digits, so DPL stays
        // at its seed and the result is always single-cell.
        if rest.len() == 3 && rest.as_bytes()[0] == b'\'' && rest.as_bytes()[2] == b'\'' {
            let value = i64::from(rest.as_bytes()[1]);
            return Some(NumberLiteral {
                value: if neg_outer { -value } else { value },
                dpl: DPL_INIT,
                is_double: false,
            });
        }

        // A base-override prefix sits before the leftmost digit.
        let (radix, after_prefix) = match rest.as_bytes()[0] {
            b'$' => (16, &rest[1..]),
            b'#' => (10, &rest[1..]),
            b'%' => (2, &rest[1..]),
            _ => (self.base, rest),
        };

        // Forth 2012 spells a signed based number `#-1289`, so the sign can
        // also follow the prefix. Either way it precedes the leftmost digit
        // and so is a sign rather than double punctuation.
        let (neg_inner, digits) = strip_sign(after_prefix);
        let negative = neg_outer ^ neg_inner;

        if digits.is_empty() {
            return None;
        }

        // Walk the digit string, stripping punctuation and tracking DPL.
        let mut buf = String::with_capacity(digits.len());
        let mut dpl = DPL_INIT;
        let mut is_double = false;
        for &b in digits.as_bytes() {
            if DOUBLE_PUNCTUATION.contains(&b) {
                is_double = true;
                dpl = 0;
            } else {
                buf.push(char::from(b));
                dpl += 1;
            }
        }

        if buf.is_empty() {
            return None;
        }

        // i128 accumulation so the full u64 range survives conversion.
        let magnitude = i128::from_str_radix(&buf, radix).ok()?;
        let value = if negative {
            -(magnitude as i64)
        } else {
            magnitude as i64
        };

        Some(NumberLiteral {
            value,
            dpl,
            is_double,
        })
    }

    /// Publish the outcome of a conversion in `DPL` and `NH`.
    ///
    /// `NH` only carries meaning after a single-cell conversion, where it holds
    /// the high-order cell that the stack result dropped.
    fn record_number_conversion(&mut self, lit: NumberLiteral) {
        self.rt.mem_write_i32(SYSVAR_DPL, lit.dpl);
        if !lit.is_double {
            self.rt.mem_write_i32(SYSVAR_NH, lit.hi());
        }
    }

    /// Try to parse a token as a single-cell number, ignoring `DPL`/`NH`.
    /// Used where only a plain cell value is meaningful.
    fn parse_number(&self, token: &str) -> Option<i32> {
        self.parse_numeric_literal(token)
            .filter(|lit| !lit.is_double)
            .map(NumberLiteral::lo)
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

    /// `_STACK_FAULT_` table index when stack guards are enabled.
    pub(crate) fn stack_guard_param(&self) -> Option<u32> {
        self.config
            .codegen
            .stack_guards
            .then_some(self.stack_fault_id)
    }

    /// Codegen configuration for compiling one word.
    fn codegen_config(&mut self, base_fn_index: u32) -> CodegenConfig {
        CodegenConfig {
            base_fn_index,
            table_size: self.table_size(),
            stack_to_local_promotion: self.config.codegen.stack_to_local_promotion,
            stack_guards: self.stack_guard_param(),
        }
    }

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
            let config = self.codegen_config(word_id.0);
            let compiled = compile_word(name, &ir_body, &config)
                .map_err(|e| anyhow::anyhow!("codegen error for {name}: {e}"))?;
            self.instantiate_and_install(&compiled, word_id)?;
        }

        Ok(word_id)
    }

    /// Register a primitive whose implementation is a host function (not IR-compiled).
    ///
    /// Public so downstream crates (like `kelvar-cli`) can extend the VM with
    /// their own I/O host words without forking WAFER.
    pub fn register_host_primitive(
        &mut self,
        name: &str,
        immediate: bool,
        func: HostFn,
    ) -> anyhow::Result<WordId> {
        let word_id = self
            .dictionary
            .create(name, immediate)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        self.rt.ensure_table_size(word_id.0)?;
        self.rt.register_host_func(word_id.0, func)?;
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

        // _STACK_FAULT_ must exist before any guarded word is compiled:
        // compiled guards write a throw code to SYSVAR_FAULT_CODE and
        // call this word, which converts it into a Forth THROW.
        let throw_code = Arc::clone(&self.throw_code);
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let code = ctx.mem_read_i32(SYSVAR_FAULT_CODE);
            *throw_code.lock().unwrap() = Some(code);
            Err(anyhow::anyhow!("forth-throw"))
        });
        self.stack_fault_id = self
            .register_host_primitive("_STACK_FAULT_", false, func)?
            .0;

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
        self.register_primitive("PAGE", false, vec![IrOp::PushI32(0x0C), IrOp::Emit])?;

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
        self.register_primitive("RP@", false, vec![IrOp::RpFetch])?;
        // RDEPTH, .RS: defined in boot.fth (use RP@ IR op)
        self.register_immediate_word()?;
        self.register_decimal()?;
        self.register_hex()?;
        // TYPE, SPACES: defined in boot.fth
        self.register_tick()?;
        self.register_to_body()?;
        self.register_environment_q()?;
        // SOURCE: defined in boot.fth
        self.register_abort()?;
        self.register_tools()?;

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
        self.register_number_conversion_vars()?;

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

        // CS-PICK, CS-ROLL, __CTRL__ for Programming-Tools / POSTPONE of control words
        self.register_cs_pick_roll()?;

        // (LOCAL) for Forth 2012 §13.6.1.0086 lower-level locals primitive
        self.register_local_paren()?;

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

        // RANDOM ( -- u ), RND-SEED ( u -- )
        self.register_random()?;

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
        self.register_words()?;

        // -- Search-Order word set --
        self.register_search_order()?;

        // -- Floating-Point word set --
        self.register_float_words()?;

        // -- Crypto: SHA1, SHA256, SHA512 (gated) --
        #[cfg(feature = "crypto")]
        self.register_crypto_words()?;

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
        let output = Arc::clone(&self.output);

        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = ctx.get_dsp();
            let mut out = output.lock().unwrap();
            if sp >= DATA_STACK_TOP {
                out.push_str("<0> ");
                return Ok(());
            }
            let depth = (DATA_STACK_TOP - sp) / CELL_SIZE;
            let base = ctx.mem_read_i32(SYSVAR_BASE_VAR) as u32;
            out.push_str(&format!("<{depth}> "));
            // Print from bottom to top, in the current BASE
            let mut addr = DATA_STACK_TOP - CELL_SIZE;
            while addr >= sp {
                let v = ctx.mem_read_i32(addr as u32);
                out.push_str(&fmt_in_base(v, base));
                out.push(' ');
                if addr < CELL_SIZE {
                    break;
                }
                addr -= CELL_SIZE;
            }
            Ok(())
        });

        self.register_host_primitive(".S", false, func)?;
        Ok(())
    }

    /// Register interactive inspection tools: DUMP and F.S.
    fn register_tools(&mut self) -> anyhow::Result<()> {
        // DUMP ( addr u -- ) hex+ASCII dump, 16 bytes per line.
        let output = Arc::clone(&self.output);
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            const MAX: i32 = 4096;
            let len = host_pop(ctx)?;
            let addr = host_pop(ctx)? as u32;
            let end = ctx.mem_len() as u32;
            let clipped = (len.clamp(0, MAX) as u32).min(end.saturating_sub(addr.min(end)));
            let bytes = ctx.mem_read_slice(addr, clipped as usize);
            let mut out = output.lock().unwrap();
            for (i, row) in bytes.chunks(16).enumerate() {
                let hex: Vec<String> = row.iter().map(|b| format!("{b:02X}")).collect();
                let ascii: String = row
                    .iter()
                    .map(|&b| {
                        if (0x20..0x7F).contains(&b) {
                            b as char
                        } else {
                            '.'
                        }
                    })
                    .collect();
                let row_addr = addr + (i as u32) * 16;
                out.push_str(&format!(
                    "{row_addr:08X}: {:<47} |{ascii}|\n",
                    hex.join(" ")
                ));
            }
            if len as u32 > clipped {
                out.push_str("... (truncated)\n");
            }
            Ok(())
        });
        self.register_host_primitive("DUMP", false, func)?;

        // F.S ( -- ) print the float stack without consuming.
        let output = Arc::clone(&self.output);
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = ctx.get_fsp();
            let depth = (FLOAT_STACK_TOP.saturating_sub(sp)) / FLOAT_SIZE;
            let mut out = output.lock().unwrap();
            out.push_str(&format!("F:<{depth}> "));
            for i in (0..depth).rev() {
                let bytes: [u8; 8] = ctx
                    .mem_read_slice(sp + i * FLOAT_SIZE, 8)
                    .try_into()
                    .unwrap();
                out.push_str(&format!("{} ", f64::from_le_bytes(bytes)));
            }
            Ok(())
        });
        self.register_host_primitive("F.S", false, func)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Crypto: SHA1 / SHA256 / SHA512 (and any algos in `crypto::ALGOS`)
    // -----------------------------------------------------------------------

    /// Register one Forth word per entry in [`crate::crypto::ALGOS`].
    ///
    /// Each word has stack effect `( c-addr u -- c-addr2 u2 )`: it hashes
    /// the `u` bytes at `c-addr` and writes the digest into the shared
    /// scratch region at [`crate::memory::HASH_SCRATCH_BASE`]. The output
    /// is overwritten by every subsequent hash call.
    #[cfg(feature = "crypto")]
    fn register_crypto_words(&mut self) -> anyhow::Result<()> {
        for algo in crate::crypto::ALGOS {
            let hash_fn = algo.hash;
            let digest_len = algo.digest_len as i32;
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                // Pop ( c-addr u )
                let dsp = host_need(ctx, 2)?;
                let u = ctx.mem_read_i32(dsp) as u32;
                let c_addr = ctx.mem_read_i32(dsp + CELL_SIZE) as u32;

                // Read input bytes and hash.
                let bytes = ctx.mem_read_slice(c_addr, u as usize);
                let digest = hash_fn(&bytes);

                // Write digest to scratch.
                ctx.mem_write_slice(HASH_SCRATCH_BASE, &digest);

                // Push ( scratch-addr digest-len ) — same dsp position, two
                // cells overwritten in place.
                ctx.mem_write_i32(dsp + CELL_SIZE, HASH_SCRATCH_BASE as i32);
                ctx.mem_write_i32(dsp, digest_len);
                Ok(())
            });
            self.register_host_primitive(algo.name, false, func)?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Priority 1: Loop support host functions
    // -----------------------------------------------------------------------

    /// Register LEAVE as a host function.
    /// Sets the loop index equal to the limit and sets the leave flag
    /// so the loop exits on the next +LOOP/LOOP check.
    fn register_leave(&mut self) -> anyhow::Result<()> {
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let rsp_val = ctx.get_rsp();
            // rsp points to index, rsp+4 = limit
            let limit = ctx.mem_read_i32(rsp_val + 4);
            // Set index = limit
            ctx.mem_write_i32(rsp_val, limit);
            // Set leave flag so +LOOP exits even with step=0
            ctx.mem_write_i32(SYSVAR_LEAVE_FLAG, 1);
            Ok(())
        });

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
        self.rt.mem_write_i32(var_addr as u32, 0i32 as i32);

        // Compile a tiny word that pushes the variable's address
        let ir_body = vec![IrOp::PushI32(var_addr as i32)];
        self.ir_bodies.insert(word_id, ir_body.clone());
        self.word_sources
            .insert(word_id, format!("VARIABLE {name}"));
        let config = self.codegen_config(word_id.0);
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
        self.word_sources
            .insert(word_id, format!("{value} CONSTANT {name}"));
        let config = self.codegen_config(word_id.0);
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
        self.word_sources.insert(word_id, format!("CREATE {name}"));
        let config = self.codegen_config(word_id.0);
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
        self.rt.mem_write_i32(val_addr as u32, value as i32);

        // Compile a word that fetches from the value's address
        let ir_body = vec![IrOp::PushI32(val_addr as i32), IrOp::Fetch];
        self.ir_bodies.insert(word_id, ir_body.clone());
        let config = self.codegen_config(word_id.0);
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
        self.rt.mem_write_i32(defer_addr as u32, default_xt as i32);

        // Compile a word that fetches the xt and executes it
        let ir_body = vec![IrOp::PushI32(defer_addr as i32), IrOp::Fetch, IrOp::Execute];
        self.ir_bodies.insert(word_id, ir_body.clone());
        let config = self.codegen_config(word_id.0);
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
            self.word_sources
                .insert(new_word_id, format!("SYNONYM {new_name} {old_name}"));
            let config = self.codegen_config(new_word_id.0);
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
    /// Called via `pending_define` when IMMEDIATE is executed from compiled code.
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
        self.word_sources
            .insert(word_id, format!("{size} BUFFER: {name}"));
        let config = self.codegen_config(word_id.0);
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
    /// Snapshot everything a marker rollback restores.
    fn snapshot_marker_state(&self, keep: bool) -> MarkerState {
        MarkerState {
            dict_state: self.dictionary.save_state(),
            user_here: self.user_here,
            next_table_index: self.next_table_index,
            word_pfa_map: self.word_pfa_map.clone(),
            ir_bodies: self.ir_bodies.clone(),
            does_definitions: self.does_definitions.clone(),
            host_word_names: self.host_word_names.clone(),
            word_sources: self.word_sources.clone(),
            two_value_words: self.two_value_words.clone(),
            fvalue_words: self.fvalue_words.clone(),
            search_order: self.search_order.lock().unwrap().clone(),
            next_wid: *self.next_wid.lock().unwrap(),
            current_wid: self.dictionary.current_wid(),
            substitutions: self.substitutions.lock().unwrap().clone(),
            abort_messages_len: self.abort_messages.lock().unwrap().len(),
            keep,
        }
    }

    /// Roll the VM back to a marker snapshot. Also discards marker
    /// entries created after the snapshot (their words no longer exist).
    fn apply_marker_state(&mut self, state: MarkerState) {
        self.dictionary.restore_state(state.dict_state);
        self.dictionary.set_current_wid(state.current_wid);
        self.user_here = state.user_here;
        self.next_table_index = state.next_table_index;
        self.word_pfa_map = state.word_pfa_map;
        self.ir_bodies = state.ir_bodies;
        self.does_definitions = state.does_definitions;
        self.host_word_names = state.host_word_names;
        self.word_sources = state.word_sources;
        self.two_value_words = state.two_value_words;
        self.fvalue_words = state.fvalue_words;
        *self.search_order.lock().unwrap() = state.search_order;
        *self.next_wid.lock().unwrap() = state.next_wid;
        *self.substitutions.lock().unwrap() = state.substitutions;
        self.abort_messages
            .lock()
            .unwrap()
            .truncate(state.abort_messages_len);
        self.marker_states
            .retain(|&k, _| k < state.next_table_index);
        self.sync_here_cell();
        self.rebuild_word_lookup();
    }

    /// MARKER <name> (keep = false): rollback to just BEFORE the marker
    /// was defined; the marker removes itself.
    /// REMEMBER <name> (keep = true): rollback to just AFTER the marker
    /// was defined; the marker survives and can be re-run.
    fn define_marker(&mut self, keep: bool) -> anyhow::Result<()> {
        let name = self
            .next_token()
            .ok_or_else(|| anyhow::anyhow!("MARKER: expected name"))?;

        // MARKER saves state before its own word exists
        let before = (!keep).then(|| self.snapshot_marker_state(false));

        let word_id = self
            .dictionary
            .create(&name, false)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        // Compile the marker word: push marker_id, call _MARKER_RESTORE_
        let restore_id = self
            .dictionary
            .find("_MARKER_RESTORE_")
            .map(|(_, id, _)| id)
            .ok_or_else(|| anyhow::anyhow!("_MARKER_RESTORE_ not found"))?;
        let ir_body = vec![IrOp::PushI32(word_id.0 as i32), IrOp::Call(restore_id)];
        self.ir_bodies.insert(word_id, ir_body.clone());
        let config = self.codegen_config(word_id.0);
        let compiled = compile_word(&name, &ir_body, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for MARKER {name}: {e}"))?;

        self.instantiate_and_install(&compiled, word_id)?;
        self.dictionary.reveal();
        self.next_table_index = self.next_table_index.max(word_id.0 + 1);

        // REMEMBER saves state after its own word is fully in place
        let saved = before.unwrap_or_else(|| self.snapshot_marker_state(true));
        self.marker_states.insert(word_id.0, saved);

        Ok(())
    }

    /// Register `_MARKER_RESTORE_` host function.
    /// ( `marker_id` -- ) Signals the outer interpreter to restore state.
    fn register_marker_restore(&mut self) -> anyhow::Result<()> {
        let pending = Arc::clone(&self.pending_marker_restore);

        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // Pop marker_id from data stack
            let sp = ctx.get_dsp();
            let marker_id = ctx.mem_read_i32(sp as u32) as u32;
            let new_sp = sp + 4;
            ctx.set_dsp((new_sp as i32) as u32);
            *pending.lock().unwrap() = Some(marker_id);
            Ok(())
        });

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
                    self.rt.mem_write_slice(pfa as u32, &value.to_le_bytes());
                } else if self.two_value_words.contains(&word_id.0) {
                    // 2VALUE: pop two cells
                    let hi = self.pop_data_stack()?;
                    let lo = self.pop_data_stack()?;
                    self.rt.mem_write_i32(pfa as u32, lo as i32);
                    self.rt.mem_write_slice(pfa as u32 + 4, &hi.to_le_bytes());
                } else {
                    let value = self.pop_data_stack()?;
                    self.rt.mem_write_i32(pfa as u32, value as i32);
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
                self.rt.mem_write_i32(pfa as u32, xt as i32);
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
                let xt = self.rt.mem_read_i32(pfa as u32);
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
            let kind = self.compiling_local_kinds[idx];
            let kind_idx = self.compiling_local_kinds[0..idx]
                .iter()
                .filter(|k| **k == kind)
                .count() as u32;
            match kind {
                LocalKind::Int => self.push_ir(IrOp::ForthLocalSet(kind_idx)),
                LocalKind::Float => self.push_ir(IrOp::ForthFLocalSet(kind_idx)),
            }
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
        let mem_len = self.rt.mem_len() as u32;
        let mem_here = self.rt.mem_read_i32(SYSVAR_HERE) as u32;
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
        self.rt.mem_write_i32(SYSVAR_HERE, self.user_here as i32);
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
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // Pop u from stack
            let sp = host_need(ctx, 1)?;
            let u = ctx.mem_read_i32(sp as u32) as u32;
            host_need(ctx, u.saturating_add(2))?;
            let sp = sp + CELL_SIZE; // pop u

            if u == 0 {
                // 0 ROLL is a no-op
                ctx.set_dsp((sp as i32) as u32);
                return Ok(());
            }

            // Save xu (the deep item to bring to top)
            let xu_addr = sp + u * CELL_SIZE;
            let saved_val = ctx.mem_read_i32(xu_addr as u32);
            // Shift items from sp to sp+(u-1)*4 toward higher addresses by one cell
            // (i.e., move each item one position deeper)
            let src_start = sp as usize;
            let count = (u * CELL_SIZE) as usize;
            // Copy backward to handle overlap correctly
            for i in (0..count).rev() {
                let byte = ctx.mem_read_u8((src_start + i) as u32);
                ctx.mem_write_u8((src_start + CELL_SIZE as usize + i) as u32, byte);
            }

            // Write saved xu at new TOS
            ctx.mem_write_i32(sp, saved_val);

            ctx.set_dsp((sp as i32) as u32);
            Ok(())
        });

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
        let func = Box::new(move |_ctx: &mut dyn HostAccess| {
            pending.lock().unwrap().push(12);
            Ok(())
        });

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
        let func = Box::new(move |_ctx: &mut dyn HostAccess| Ok(()));

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
        // Share the PFA map with the host function via Arc<Mutex<>>
        let pfa_map = Arc::new(Mutex::new(self.word_pfa_map.clone()));
        // Store the Arc for later updates
        self.word_pfa_map_shared = Some(Arc::clone(&pfa_map));

        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // Pop xt from data stack
            let sp = host_need(ctx, 1)?;
            let xt = ctx.mem_read_i32(sp as u32) as u32;

            // Look up PFA for this xt
            let map = pfa_map.lock().unwrap();
            let pfa = map.get(&xt).copied().unwrap_or(0);
            drop(map);

            // Replace TOS with PFA
            ctx.mem_write_i32(sp as u32, (pfa as i32) as i32);
            Ok(())
        });

        self.register_host_primitive(">BODY", false, func)?;
        Ok(())
    }

    /// ENVIRONMENT? -- ( c-addr u -- false | value true ) query system parameters.
    fn register_environment_q(&mut self) -> anyhow::Result<()> {
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = host_need(ctx, 2)?;
            let u = ctx.mem_read_i32(sp as u32) as u32;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 4) as u32).to_le_bytes();
            let addr = u32::from_le_bytes(b);
            let query = String::from_utf8_lossy(&ctx.mem_read_slice(addr as u32, u as usize))
                .to_ascii_uppercase();

            match query.as_str() {
                "#LOCALS" => {
                    // Return (16 TRUE) — support at least 16 locals
                    ctx.mem_write_i32((sp + 4) as u32, 16i32 as i32);
                    ctx.mem_write_i32(sp as u32, (-1i32) as i32); // TRUE
                    ctx.set_dsp((sp as i32) as u32);
                }
                _ => {
                    // Unknown: pop 2, push FALSE
                    let new_sp = sp + 4;
                    ctx.mem_write_i32(new_sp as u32, 0i32 as i32);
                    ctx.set_dsp((new_sp as i32) as u32);
                }
            }
            Ok(())
        });

        self.register_host_primitive("ENVIRONMENT?", false, func)?;
        Ok(())
    }

    /// ABORT -- clear stacks and throw error.
    fn register_abort(&mut self) -> anyhow::Result<()> {
        let throw_code = Arc::clone(&self.throw_code);
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // Reset stack pointers and throw -1 (ABORT)
            ctx.set_dsp((DATA_STACK_TOP as i32) as u32);
            ctx.set_rsp((RETURN_STACK_TOP as i32) as u32);
            *throw_code.lock().unwrap() = Some(-1);
            Err(anyhow::anyhow!("forth-throw"))
        });
        self.register_host_primitive("ABORT", false, func)?;

        // _ABORT_Q_ ( idx -- ) runtime of ABORT": record text, throw -2.
        let msgs = Arc::clone(&self.abort_messages);
        let pending = Arc::clone(&self.abort_message);
        let throw_code = Arc::clone(&self.throw_code);
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let idx = host_pop(ctx)? as usize;
            *pending.lock().unwrap() = msgs.lock().unwrap().get(idx).cloned();
            *throw_code.lock().unwrap() = Some(-2);
            Err(anyhow::anyhow!("forth-throw"))
        });
        self.register_host_primitive("_ABORT_Q_", false, func)?;

        // QUIT ( -- ) ( R: i*x -- ) empty the return stack and return to the
        // interpreter. The data stack is deliberately untouched -- that is the
        // whole difference to ABORT, which is specified as "empty the data
        // stack, then QUIT". Unwinding rides the throw plumbing so nested
        // EVALUATE / INCLUDE frames are abandoned on the way out; the standard
        // code -56 tells `evaluate` and CATCH what this is.
        let throw_code = Arc::clone(&self.throw_code);
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            ctx.set_rsp((RETURN_STACK_TOP as i32) as u32);
            *throw_code.lock().unwrap() = Some(QUIT_THROW);
            Err(anyhow::anyhow!("forth-throw"))
        });
        self.register_host_primitive("QUIT", false, func)?;

        // BYE ( -- ) request REPL/driver exit.
        let bye = Arc::clone(&self.bye);
        let func: HostFn = Box::new(move |_ctx: &mut dyn HostAccess| {
            bye.store(true, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        });
        self.register_host_primitive("BYE", false, func)?;
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

        // THROW ( exception# -- )
        let throw_code_for_throw = Arc::clone(&throw_code);
        let throw_func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // Pop throw code from data stack
            let sp = ctx.get_dsp();
            if sp >= DATA_STACK_TOP {
                return Err(anyhow::anyhow!("THROW: stack underflow"));
            }
            let code = ctx.mem_read_i32(sp as u32);
            // Pop TOS
            ctx.set_dsp(((sp + CELL_SIZE) as i32) as u32);

            if code == 0 {
                return Ok(());
            }

            // Store the throw code and trigger a trap to unwind back to CATCH
            *throw_code_for_throw.lock().unwrap() = Some(code);
            Err(anyhow::anyhow!("forth-throw"))
        });
        self.register_host_primitive("THROW", false, throw_func)?;

        // CATCH ( xt -- exception# | 0 )
        let throw_code_for_catch = Arc::clone(&throw_code);
        let catch_func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // Pop xt from data stack
            let sp = ctx.get_dsp();
            if sp >= DATA_STACK_TOP {
                return Err(anyhow::anyhow!("CATCH: stack underflow"));
            }
            let xt = ctx.mem_read_i32(sp as u32) as u32;
            // Pop TOS (remove xt)
            let sp_after_pop = sp + CELL_SIZE;
            ctx.set_dsp((sp_after_pop as i32) as u32);

            // Save stack depths for restoration on THROW
            let saved_dsp = sp_after_pop;
            let saved_rsp = ctx.get_rsp();

            // Call the word -- if THROW is invoked, call_func returns Err
            match ctx.call_func(xt) {
                Ok(()) => {
                    // Normal completion: push 0
                    let current_sp = ctx.get_dsp();
                    let mem_len = ctx.mem_len() as u32;
                    let new_sp = current_sp.wrapping_sub(CELL_SIZE);
                    if new_sp >= mem_len {
                        return Err(anyhow::anyhow!("stack overflow in CATCH"));
                    }
                    ctx.mem_write_i32(new_sp as u32, 0_i32 as i32);
                    ctx.set_dsp((new_sp as i32) as u32);
                    Ok(())
                }
                Err(e) => {
                    // Check if this was a THROW (vs some other trap)
                    let mut tc = throw_code_for_catch.lock().unwrap();
                    // QUIT is not an exception: it unwinds past CATCH straight
                    // to the interpreter, leaving both stacks as it found them
                    // (verified against gforth and SwiftForth).
                    if *tc == Some(QUIT_THROW) {
                        drop(tc);
                        return Err(e);
                    }
                    let code = tc.take().unwrap_or(-1);
                    drop(tc);

                    // Restore stack pointers to saved depths
                    ctx.set_dsp((saved_dsp as i32) as u32);
                    ctx.set_rsp((saved_rsp as i32) as u32);

                    // Push the throw code onto the restored stack
                    let mem_len = ctx.mem_len() as u32;
                    let new_sp = saved_dsp.wrapping_sub(CELL_SIZE);
                    if new_sp >= mem_len {
                        return Err(anyhow::anyhow!("stack overflow in CATCH"));
                    }
                    ctx.mem_write_i32(new_sp as u32, code as i32);
                    ctx.set_dsp((new_sp as i32) as u32);
                    Ok(())
                }
            }
        });
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
        let mem_len = self.rt.mem_len() as u32;
        if addr > mem_len || addr.wrapping_add(len) > mem_len {
            anyhow::bail!("EVALUATE: invalid address/length");
        }

        // Read the string from WASM memory
        let s =
            String::from_utf8_lossy(&self.rt.mem_read_slice(addr as u32, len as usize)).to_string();

        // Save current input state and SOURCE-ID
        let saved_buffer = std::mem::take(&mut self.input_buffer);
        let saved_pos = self.input_pos;
        let saved_source_id = self.rt.mem_read_i32(crate::memory::SYSVAR_SOURCE_ID);

        // Set new input and SOURCE-ID = -1 (string source)
        self.input_buffer = s;
        self.input_pos = 0;
        self.rt.mem_write_i32(crate::memory::SYSVAR_SOURCE_ID, -1);

        // Sync input buffer, >IN, and #TIB to WASM (for SOURCE and WORD)
        self.sync_full_input_to_wasm();

        // Interpret with >IN sync (supports >IN manipulation)
        while let Some(token) = self.next_token() {
            {
                self.rt
                    .mem_write_i32(SYSVAR_TO_IN as u32, (self.input_pos as u32) as i32);
            }
            let wasm_to_in_before = self.input_pos;
            self.interpret_token(&token)?;
            let wasm_to_in = self.rt.mem_read_i32(SYSVAR_TO_IN) as u32 as usize;
            if wasm_to_in != wasm_to_in_before {
                self.input_pos = wasm_to_in;
            }
            if self.input_pos >= self.input_buffer.len() {
                break;
            }
        }

        // A definition left open by the EVALUATEd string: bank its tail and
        // re-anchor capture at the resume point of the restored buffer.
        let capture_open = self.state != 0 && self.source_capture_from.is_some();
        if let Some(from) = self.source_capture_from.filter(|_| self.state != 0) {
            let from = from.min(self.input_buffer.len());
            self.compiling_source.push_str(&self.input_buffer[from..]);
            self.compiling_source.push('\n');
        }

        // Restore input state, SOURCE-ID, and sync back to WASM
        self.input_buffer = saved_buffer;
        self.input_pos = saved_pos;
        if capture_open {
            self.source_capture_from = Some(self.input_pos);
        }
        self.sync_full_input_to_wasm();
        self.rt
            .mem_write_i32(crate::memory::SYSVAR_SOURCE_ID, saved_source_id);

        Ok(())
    }

    /// Write the current input buffer, >IN, and #TIB to WASM memory
    /// (for SOURCE, WORD, and >IN manipulation from Forth code).
    fn sync_full_input_to_wasm(&mut self) {
        let bytes = self.input_buffer.as_bytes();
        let len = bytes.len().min(INPUT_BUFFER_SIZE as usize);
        self.rt.mem_write_slice(INPUT_BUFFER_BASE, &bytes[..len]);
        self.rt.mem_write_i32(SYSVAR_TO_IN, self.input_pos as i32);
        self.rt.mem_write_i32(SYSVAR_NUM_TIB, len as i32);
    }

    /// Register the loader that INCLUDE/INCLUDED use to read source files.
    /// The CLI installs a filesystem reader; the web REPL leaves it unset,
    /// which makes INCLUDE a defined error.
    pub fn set_source_loader(&mut self, loader: SourceLoader) {
        self.source_loader = Some(loader);
    }

    /// Run a source file through the include machinery (also used by the
    /// CLI file mode, so `wafer prog.fth` gets `file:line` error context and
    /// a base directory for nested INCLUDEs).
    pub fn include(&mut self, path: &str) -> anyhow::Result<()> {
        self.include_file(path)
    }

    /// INCLUDED's engine: resolve the path, load, and feed the file
    /// line-by-line through `evaluate` with the parent input saved around
    /// it. Compile state and SEE source capture already span `evaluate`
    /// calls, so multi-line definitions inside files just work.
    fn include_file(&mut self, path: &str) -> anyhow::Result<()> {
        const MAX_INCLUDE_DEPTH: usize = 16;
        // Relative paths resolve against the including file's directory.
        let resolved = match self.include_frames.last() {
            Some((parent, _)) if std::path::Path::new(path).is_relative() => {
                match std::path::Path::new(parent).parent() {
                    Some(dir) if dir != std::path::Path::new("") => {
                        dir.join(path).to_string_lossy().into_owned()
                    }
                    _ => path.to_string(),
                }
            }
            _ => path.to_string(),
        };
        if self.include_frames.iter().any(|(p, _)| *p == resolved) {
            anyhow::bail!("INCLUDE: cycle: {resolved}");
        }
        if self.include_frames.len() >= MAX_INCLUDE_DEPTH {
            anyhow::bail!("INCLUDE: nesting deeper than {MAX_INCLUDE_DEPTH}");
        }
        let Some(loader) = self.source_loader.as_ref() else {
            anyhow::bail!("INCLUDE: no source loader registered");
        };
        let text = loader(&resolved).map_err(|e| anyhow::anyhow!("INCLUDE: {resolved}: {e}"))?;

        // Save the parent input source; SOURCE-ID becomes a synthetic
        // positive id per nesting level (0 = terminal, -1 = string).
        let saved_buffer = std::mem::take(&mut self.input_buffer);
        let saved_pos = self.input_pos;
        let saved_source_id = self.rt.mem_read_i32(crate::memory::SYSVAR_SOURCE_ID);
        self.rt.mem_write_i32(
            crate::memory::SYSVAR_SOURCE_ID,
            self.include_frames.len() as i32 + 1,
        );
        self.include_frames.push((resolved, 0));

        let mut result = Ok(());
        for (i, line) in text.lines().enumerate() {
            if let Some(frame) = self.include_frames.last_mut() {
                frame.1 = i + 1;
            }
            if let Err(e) = self.evaluate(line) {
                let (p, n) = self.include_frames.last().cloned().unwrap_or_default();
                result = Err(e.context(format!("{p}:{n}")));
                break;
            }
            if self.bye.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
        }

        // Restore the parent input on every path (success, error, BYE).
        self.include_frames.pop();
        self.input_buffer = saved_buffer;
        self.input_pos = saved_pos;
        // Re-anchor an open definition's capture at the parent resume point.
        if self.state != 0 && self.source_capture_from.is_some() {
            self.source_capture_from = Some(self.input_pos);
        }
        self.sync_full_input_to_wasm();
        self.rt
            .mem_write_i32(crate::memory::SYSVAR_SOURCE_ID, saved_source_id);
        result
    }

    /// INCLUDED ( c-addr u -- ) -- include the named source file.
    fn do_included(&mut self) -> anyhow::Result<()> {
        let len = self.pop_data_stack()? as u32;
        let addr = self.pop_data_stack()? as u32;
        let path = String::from_utf8_lossy(&self.rt.mem_read_slice(addr, len as usize)).to_string();
        self.include_file(&path)
    }

    /// INCLUDE <name> -- parsing form of INCLUDED.
    fn do_include(&mut self) -> anyhow::Result<()> {
        let path = self.parse_name_arg("INCLUDE")?;
        self.include_file(&path)
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
        self.rt.mem_write_u8(buf_addr as u32, (word_len) as u8);
        self.rt.mem_write_slice(buf_addr as u32 + 1, word_bytes);

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
        let config = self.codegen_config(word_id.0);
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
            let config = self.codegen_config(second_word_id.0);
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
        let saved_local_kinds = std::mem::take(&mut self.compiling_local_kinds);
        let saved_local_batch_base = self.local_batch_base.take();

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
        let config = self.codegen_config(does_word_id.0);
        let compiled = compile_word("_does_action_", &does_ir, &config)
            .map_err(|e| anyhow::anyhow!("codegen error for DOES> body: {e}"))?;
        self.instantiate_and_install(&compiled, does_word_id)?;

        // Restore compilation state
        self.compiling_name = saved_name;
        self.compiling_word_id = saved_word_id;
        self.control_stack = saved_control;
        self.compiling_locals = saved_locals;
        self.compiling_local_kinds = saved_local_kinds;
        self.local_batch_base = saved_local_batch_base;

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
        let config = self.codegen_config(defining_word_id.0);
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
            let config = self.codegen_config(new_word_id.0);
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

            let config = self.codegen_config(tmp_word_id.0);
            let compiled = compile_word("_create_part_", &create_ir, &config)
                .map_err(|e| anyhow::anyhow!("codegen: {e}"))?;
            self.instantiate_and_install(&compiled, tmp_word_id)?;
            self.execute_word(tmp_word_id)?;

            // Step 4: Patch the new word to push PFA and call does-action
            self.refresh_user_here();
            let patched_ir = vec![IrOp::PushI32(pfa as i32), IrOp::Call(does_action_id)];
            let config = self.codegen_config(new_word_id.0);
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
            let config = self.codegen_config(target_word_id.0);
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
        let word_lookup = Arc::clone(&self.word_lookup);

        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = ctx.get_dsp();
            let mem_len = ctx.mem_len() as u32;

            // Stack pointer sanity check
            if sp < CELL_SIZE || sp > mem_len {
                return Err(anyhow::anyhow!("stack error in FIND"));
            }

            let c_addr = ctx.mem_read_i32(sp as u32) as u32;

            // Bounds check
            if c_addr >= mem_len {
                // Push c-addr and 0 (not found)
                let new_sp = sp - CELL_SIZE;
                ctx.mem_write_i32(new_sp as u32, 0i32 as i32);
                ctx.set_dsp((new_sp as i32) as u32);
                return Ok(());
            }

            let count = ctx.mem_read_u8(c_addr as u32) as usize;
            let name_start = (c_addr + 1) as usize;
            if name_start + count > mem_len as usize {
                let new_sp = sp - CELL_SIZE;
                ctx.mem_write_i32(new_sp as u32, 0i32 as i32);
                ctx.set_dsp((new_sp as i32) as u32);
                return Ok(());
            }

            let name_bytes = &ctx.mem_read_slice(name_start as u32, count as usize);
            let name = String::from_utf8_lossy(name_bytes).to_ascii_uppercase();

            let lookup = word_lookup.lock().unwrap();
            if let Some(&(xt, is_imm)) = lookup.get(&name) {
                // Found: replace c-addr with xt, push flag
                let new_sp = sp - CELL_SIZE;
                let flag: i32 = if is_imm { 1 } else { -1 };
                // Replace c-addr with xt
                ctx.mem_write_i32(new_sp + 4, xt as i32);
                // Push flag
                ctx.mem_write_i32(new_sp, flag);
                ctx.set_dsp((new_sp as i32) as u32);
            } else {
                // Not found: push c-addr and 0
                let new_sp = sp - CELL_SIZE;
                ctx.mem_write_i32(new_sp as u32, 0i32 as i32);
                ctx.set_dsp((new_sp as i32) as u32);
            }

            Ok(())
        });

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
        self.rt.mem_write_i32(SYSVAR_BASE_VAR as u32, 10u32 as i32);

        self.register_primitive("BASE", false, vec![IrOp::PushI32(SYSVAR_BASE_VAR as i32)])?;
        Ok(())
    }

    /// DPL ( -- addr ) and NH ( -- addr ): input number conversion results.
    ///
    /// `DPL` holds the digit count right of the rightmost punctuation
    /// character in the last converted number, or a negative value when the
    /// token carried none. `NH` holds the high-order cell dropped by a
    /// single-cell conversion, so an out-of-range token can be recovered as a
    /// double.
    fn register_number_conversion_vars(&mut self) -> anyhow::Result<()> {
        self.rt.mem_write_i32(SYSVAR_DPL, DPL_INIT);
        self.rt.mem_write_i32(SYSVAR_NH, 0);

        self.register_primitive("DPL", false, vec![IrOp::PushI32(SYSVAR_DPL as i32)])?;
        self.register_primitive("NH", false, vec![IrOp::PushI32(SYSVAR_NH as i32)])?;
        Ok(())
    }

    /// M* ( n1 n2 -- d ) signed multiply producing double-cell result.
    fn register_m_star(&mut self) -> anyhow::Result<()> {
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = host_need(ctx, 2)?;
            let n2 = ctx.mem_read_i32(sp as u32) as i64;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 4) as u32).to_le_bytes();
            let n1 = i32::from_le_bytes(b) as i64;
            let result = n1 * n2;
            // Store as double-cell: low cell deeper, high cell on top
            let lo = result as i32;
            let hi = (result >> 32) as i32;
            // Overwrite the two stack slots (net: pop 2, push 2 = same sp)
            ctx.mem_write_i32((sp + 4) as u32, lo as i32);
            ctx.mem_write_i32(sp as u32, hi as i32);
            Ok(())
        });

        self.register_host_primitive("M*", false, func)?;
        Ok(())
    }

    /// UM* ( u1 u2 -- ud ) unsigned multiply producing double-cell result.
    fn register_um_star(&mut self) -> anyhow::Result<()> {
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = host_need(ctx, 2)?;
            let u2 = ctx.mem_read_i32(sp as u32) as u32 as u64;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 4) as u32).to_le_bytes();
            let u1 = u32::from_le_bytes(b) as u64;
            let result = u1 * u2;
            let lo = result as u32;
            let hi = (result >> 32) as u32;
            ctx.mem_write_i32((sp + 4) as u32, lo as i32);
            ctx.mem_write_i32(sp as u32, hi as i32);
            Ok(())
        });

        self.register_host_primitive("UM*", false, func)?;
        Ok(())
    }

    /// UM/MOD ( ud u -- rem quot ) unsigned double-cell divide.
    fn register_um_div_mod(&mut self) -> anyhow::Result<()> {
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = host_need(ctx, 3)?;
            // Pop u (divisor)
            let divisor = ctx.mem_read_i32(sp as u32) as u32 as u64;
            // Pop ud (double-cell): high at sp+4, low at sp+8
            let b: [u8; 4] = ctx.mem_read_i32((sp + 4) as u32).to_le_bytes();
            let hi = u32::from_le_bytes(b) as u64;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 8) as u32).to_le_bytes();
            let lo = u32::from_le_bytes(b) as u64;
            let dividend = (hi << 32) | lo;

            if divisor == 0 {
                return Err(anyhow::anyhow!("division by zero"));
            }

            let quot = (dividend / divisor) as u32;
            let rem = (dividend % divisor) as u32;

            // Pop 3, push 2: net sp + 4
            let new_sp = sp + 4;
            // rem deeper, quot on top
            ctx.mem_write_i32(new_sp + 4, rem as i32);
            ctx.mem_write_i32(new_sp, quot as i32);
            ctx.set_dsp(new_sp);
            Ok(())
        });

        self.register_host_primitive("UM/MOD", false, func)?;
        Ok(())
    }

    /// >NUMBER ( ud1 c-addr1 u1 -- ud2 c-addr2 u2 ) convert string to number.
    fn register_to_number(&mut self) -> anyhow::Result<()> {
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = ctx.get_dsp();
            let mem_len = ctx.mem_len() as u32;
            if sp.wrapping_add(16) > mem_len || sp > mem_len {
                return Err(anyhow::anyhow!("stack underflow in >NUMBER"));
            }
            // Stack: u1 at sp, c-addr1 at sp+4, ud1-hi at sp+8, ud1-lo at sp+12
            let mut u1 = ctx.mem_read_i32(sp as u32) as u32;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 4) as u32).to_le_bytes();
            let mut c_addr = u32::from_le_bytes(b);
            let b: [u8; 4] = ctx.mem_read_i32((sp + 8) as u32).to_le_bytes();
            let ud_hi = u32::from_le_bytes(b) as u64;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 12) as u32).to_le_bytes();
            let ud_lo = u32::from_le_bytes(b) as u64;
            let mut ud = (ud_hi << 32) | ud_lo;

            // Read BASE from WASM memory (not base_cell)
            let b: [u8; 4] = ctx.mem_read_i32(SYSVAR_BASE_VAR as u32).to_le_bytes();
            let base = u32::from_le_bytes(b) as u64;

            while u1 > 0 {
                let ch = ctx.mem_read_u8(c_addr as u32) as char;
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
            ctx.mem_write_i32(sp, u1 as i32);
            ctx.mem_write_i32(sp + 4, c_addr as i32);
            ctx.mem_write_i32(sp + 8, ud_hi_new as i32);
            ctx.mem_write_i32(sp + 12, ud_lo_new as i32);
            Ok(())
        });

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
        let pending = Arc::clone(&self.pending_actions);

        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // Pop xt from data stack
            let sp = host_need(ctx, 1)?;
            let xt = ctx.mem_read_i32(sp as u32) as u32;
            // Drop top of stack
            let new_sp = sp + 4;
            ctx.set_dsp((new_sp as i32) as u32);
            // Signal the outer interpreter to compile a call to this xt
            pending.lock().unwrap().push(PendingAction::CompileCall(xt));
            Ok(())
        });

        self.register_host_primitive("COMPILE,", false, func)?;
        Ok(())
    }

    /// Register CS-PICK, CS-ROLL, and __CTRL__ host functions.
    /// CS-PICK ( n -- ) copies the n-th control-flow stack entry (compile-time).
    /// CS-ROLL ( n -- ) rotates the top n+1 control-flow stack entries (compile-time).
    /// __CTRL__ ( code -- ) triggers a compile-time control-flow operation (for POSTPONE).
    fn register_cs_pick_roll(&mut self) -> anyhow::Result<()> {
        // Helper: pop one cell from data stack
        fn pop_cell(ctx: &mut dyn HostAccess) -> i32 {
            let sp = ctx.get_dsp();
            let val = ctx.mem_read_i32(sp);
            ctx.set_dsp(sp + CELL_SIZE);
            val
        }

        // CS-PICK
        let pending = Arc::clone(&self.pending_actions);
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let n = pop_cell(ctx);
            pending
                .lock()
                .unwrap()
                .push(PendingAction::CsPick(n as u32));
            Ok(())
        });
        self.register_host_primitive("CS-PICK", false, func)?;

        // CS-ROLL
        let pending = Arc::clone(&self.pending_actions);
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let n = pop_cell(ctx);
            pending
                .lock()
                .unwrap()
                .push(PendingAction::CsRoll(n as u32));
            Ok(())
        });
        self.register_host_primitive("CS-ROLL", false, func)?;

        // __CTRL__ (used by POSTPONE of control-flow keywords)
        let pending = Arc::clone(&self.pending_actions);
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let code = pop_cell(ctx);
            pending
                .lock()
                .unwrap()
                .push(PendingAction::CompileControl(code));
            Ok(())
        });
        self.register_host_primitive("__CTRL__", false, func)?;

        Ok(())
    }

    /// Register `(LOCAL)` per Forth 2012 §13.6.1.0086.
    ///
    /// Compile-time `( c-addr u -- )`. When `u > 0`, declare a local named by
    /// the byte slice at `c-addr`/`u`. When `u = 0`, emit the initialization
    /// code for all locals declared since the last sentinel (the runtime
    /// `ForthLocalSet`s that pop args from the data stack in reverse
    /// declaration order).
    ///
    /// The word is non-immediate: it runs when its containing immediate word
    /// (typically user-defined `LOCAL` or `END-LOCALS`) executes during the
    /// outer compilation loop. Because `HostAccess` cannot reach into the
    /// outer-interpreter compile state directly, the actual mutation is
    /// deferred via `PendingAction::DeclareLocal` / `DeclareLocalEnd` and
    /// processed in `handle_pending_actions` once the immediate word returns.
    fn register_local_paren(&mut self) -> anyhow::Result<()> {
        let pending = Arc::clone(&self.pending_actions);

        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // ( c-addr u -- ) — pop both cells.
            let sp = host_need(ctx, 2)?;
            let u = ctx.mem_read_i32(sp) as u32;
            let addr = ctx.mem_read_i32(sp + CELL_SIZE) as u32;
            ctx.set_dsp(sp + 2 * CELL_SIZE);

            let action = if u == 0 {
                PendingAction::DeclareLocalEnd
            } else {
                let bytes = ctx.mem_read_slice(addr, u as usize);
                let name = String::from_utf8_lossy(&bytes).to_ascii_uppercase();
                PendingAction::DeclareLocal(name)
            };
            pending.lock().unwrap().push(action);
            Ok(())
        });

        self.register_host_primitive("(LOCAL)", false, func)?;
        Ok(())
    }

    /// Register `_does_patch_` as a host function for runtime DOES> patching.
    /// ( `does_action_id` -- ) Signals the outer interpreter to patch the most
    /// recently `CREATEd` word with a new DOES> action.
    fn register_does_patch(&mut self) -> anyhow::Result<()> {
        let pending_does_patch = Arc::clone(&self.pending_does_patch);

        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // Pop does_action_id from data stack
            let sp = ctx.get_dsp();
            let does_action_id = ctx.mem_read_i32(sp as u32) as u32;
            let new_sp = sp + 4;
            ctx.set_dsp((new_sp as i32) as u32);
            *pending_does_patch.lock().unwrap() = Some(does_action_id);
            Ok(())
        });

        self.register_host_primitive("_DOES_PATCH_", false, func)?;
        Ok(())
    }

    /// Register CONSTANT, VARIABLE, CREATE as host functions so they can
    /// be compiled into colon definitions (e.g., `: EQU CONSTANT ;`).
    fn register_defining_words(&mut self) -> anyhow::Result<()> {
        // CONSTANT: sets pending_define to 1
        {
            let pending = Arc::clone(&self.pending_define);
            let func = Box::new(move |_ctx: &mut dyn HostAccess| {
                pending.lock().unwrap().push(1);
                Ok(())
            });
            self.register_host_primitive("CONSTANT", false, func)?;
        }

        // VARIABLE: sets pending_define to 2
        {
            let pending = Arc::clone(&self.pending_define);
            let func = Box::new(move |_ctx: &mut dyn HostAccess| {
                pending.lock().unwrap().push(2);
                Ok(())
            });
            self.register_host_primitive("VARIABLE", false, func)?;
        }

        // CREATE: sets pending_define to 3
        {
            let pending = Arc::clone(&self.pending_define);
            let func = Box::new(move |_ctx: &mut dyn HostAccess| {
                pending.lock().unwrap().push(3);
                Ok(())
            });
            self.register_host_primitive("CREATE", false, func)?;
        }

        // 2CONSTANT: sets pending_define to 9
        {
            let pending = Arc::clone(&self.pending_define);
            let func = Box::new(move |_ctx: &mut dyn HostAccess| {
                pending.lock().unwrap().push(9);
                Ok(())
            });
            self.register_host_primitive("2CONSTANT", false, func)?;
        }

        // 2VARIABLE: sets pending_define to 10
        {
            let pending = Arc::clone(&self.pending_define);
            let func = Box::new(move |_ctx: &mut dyn HostAccess| {
                pending.lock().unwrap().push(10);
                Ok(())
            });
            self.register_host_primitive("2VARIABLE", false, func)?;
        }

        // DEFER: sets pending_define to 11
        {
            let pending = Arc::clone(&self.pending_define);
            let func = Box::new(move |_ctx: &mut dyn HostAccess| {
                pending.lock().unwrap().push(11);
                Ok(())
            });
            self.register_host_primitive("DEFER", false, func)?;
        }

        Ok(())
    }

    /// Register EVALUATE as a host function callable from compiled code.
    fn register_evaluate_word(&mut self) -> anyhow::Result<()> {
        let pending = Arc::clone(&self.pending_define);
        let func = Box::new(move |_ctx: &mut dyn HostAccess| {
            pending.lock().unwrap().push(4);
            Ok(())
        });
        self.register_host_primitive("EVALUATE", false, func)?;
        Ok(())
    }

    /// Register WORD as a host function callable from compiled code.
    /// WORD ( char -- c-addr ) reads from the WASM input buffer and updates >IN.
    fn register_word_word(&mut self) -> anyhow::Result<()> {
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let delim = host_pop(ctx)? as u8;

            // Read >IN and #TIB from WASM memory
            let b: [u8; 4] = ctx.mem_read_i32(SYSVAR_TO_IN as u32).to_le_bytes();
            let mut to_in = u32::from_le_bytes(b);
            let b: [u8; 4] = ctx.mem_read_i32(SYSVAR_NUM_TIB as u32).to_le_bytes();
            let num_tib = u32::from_le_bytes(b);

            // Skip leading delimiters (also skip spaces when delimiter != space)
            while to_in < num_tib {
                let ch = ctx.mem_read_u8((INPUT_BUFFER_BASE + to_in) as u32);
                if ch == delim || (delim != b' ' && ch == b' ') {
                    to_in += 1;
                } else {
                    break;
                }
            }

            // Collect word
            let start = to_in;
            while to_in < num_tib {
                if ctx.mem_read_u8((INPUT_BUFFER_BASE + to_in) as u32) == delim {
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
            ctx.mem_write_i32(SYSVAR_TO_IN as u32, to_in as i32);

            // Store counted string at dedicated WORD buffer
            let buf_addr = crate::memory::WORD_BUF_BASE;
            ctx.mem_write_u8(buf_addr as u32, (word_len) as u8);
            let src_start = (INPUT_BUFFER_BASE + start) as usize;
            let dst_start = buf_addr as usize + 1;
            for i in 0..word_len as usize {
                let byte = ctx.mem_read_u8((src_start + i) as u32);
                ctx.mem_write_u8((dst_start + i) as u32, byte);
            }

            // Push c-addr onto data stack (reuse the popped delim's slot)
            let new_sp = ctx.get_dsp() - CELL_SIZE;
            ctx.mem_write_i32(new_sp, buf_addr as i32);
            ctx.set_dsp(new_sp);

            Ok(())
        });
        self.register_host_primitive("WORD", false, func)?;
        Ok(())
    }

    /// FIND ( c-addr -- c-addr 0 | xt 1 | xt -1 ) Look up counted string in dictionary.
    fn interpret_find(&mut self) -> anyhow::Result<()> {
        // Pop counted string address
        let c_addr = self.pop_data_stack()? as u32;

        // Bounds check: c_addr must be within WASM memory
        let mem_len = self.rt.mem_len() as u32;
        if c_addr >= mem_len {
            // Invalid address -- push original address and 0 (not found)
            self.push_data_stack(c_addr as i32)?;
            self.push_data_stack(0)?;
            return Ok(());
        }

        // Read counted string from WASM memory
        let count = self.rt.mem_read_u8(c_addr as u32) as usize;
        let name_start = (c_addr + 1) as usize;
        if name_start + count > mem_len as usize {
            // String extends past memory -- push original address and 0
            self.push_data_stack(c_addr as i32)?;
            self.push_data_stack(0)?;
            return Ok(());
        }
        let name =
            String::from_utf8_lossy(&self.rt.mem_read_slice(name_start as u32, count as usize))
                .to_string();

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
                40 => self.do_words(),
                41 => self.do_see()?,
                42 => self.do_see_ir()?,
                43 => self.do_help()?,
                44 => self.do_included()?,
                45 => self.do_include()?,
                _ => {}
            }
        }
        Ok(())
    }

    /// Drain `pending_compile` and push `IrOp::Call` for each entry into `compiling_ir`.
    /// Called after executing an immediate word during compilation.
    /// Process all pending actions from host functions (COMPILE,, CS-PICK, CS-ROLL, etc.).
    fn handle_pending_actions(&mut self) -> anyhow::Result<()> {
        let actions: Vec<PendingAction> = {
            let mut v = self.pending_actions.lock().unwrap();
            std::mem::take(&mut *v)
        };
        for action in actions {
            match action {
                PendingAction::CompileCall(xt) => {
                    self.push_ir(IrOp::Call(WordId(xt)));
                }
                PendingAction::CsPick(n) => {
                    self.cs_pick(n)?;
                }
                PendingAction::CsRoll(n) => {
                    self.cs_roll(n)?;
                }
                PendingAction::CompileControl(code) => match code {
                    CTRL_IF => self.compile_if()?,
                    CTRL_ELSE => self.compile_else()?,
                    CTRL_THEN => self.compile_then()?,
                    CTRL_BEGIN => self.compile_begin()?,
                    CTRL_UNTIL => self.compile_until()?,
                    CTRL_WHILE => self.compile_while()?,
                    CTRL_REPEAT => self.compile_repeat()?,
                    CTRL_AGAIN => self.compile_again()?,
                    CTRL_DO => self.compile_do()?,
                    CTRL_LOOP => self.compile_loop(false)?,
                    CTRL_PLUS_LOOP => self.compile_loop(true)?,
                    CTRL_AHEAD => self.compile_ahead()?,
                    _ => anyhow::bail!("unknown control code: {code}"),
                },
                // Forth 2012 §13.6.1.0086 `(LOCAL)`: append the named local
                // to the current compile context. Locals declared via
                // `(LOCAL)` are int-only per spec (float locals are not
                // covered by this word).
                PendingAction::DeclareLocal(name) => {
                    if self.state == 0 {
                        anyhow::bail!("(LOCAL): only valid during compilation");
                    }
                    if self.local_batch_base.is_none() {
                        self.local_batch_base = Some(self.compiling_locals.len());
                    }
                    self.compiling_locals.push(name);
                    self.compiling_local_kinds.push(LocalKind::Int);
                }
                // Forth 2012 §13.6.1.0086 `(LOCAL)` sentinel: emit init
                // code for the batch of locals just declared. Pop the
                // runtime args from the data stack in reverse declaration
                // order — consistent with `compile_locals_block` at the
                // `{: ... :}` flow.
                PendingAction::DeclareLocalEnd => {
                    if let Some(base) = self.local_batch_base.take() {
                        for slot in (base..self.compiling_locals.len()).rev() {
                            let kind_idx = self.compiling_local_kinds[0..slot]
                                .iter()
                                .filter(|k| **k == LocalKind::Int)
                                .count() as u32;
                            self.push_ir(IrOp::ForthLocalSet(kind_idx));
                        }
                    }
                    // No-op if no batch is pending — spec-permissible for
                    // a user that calls `0 0 (LOCAL)` at the top of a
                    // definition before declaring anything.
                }
            }
        }
        Ok(())
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
            let config = self.codegen_config(target_word_id.0);
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
            // REMEMBER-style entries survive their own execution; MARKER
            // entries remove themselves. apply_marker_state discards all
            // entries newer than the snapshot either way.
            let state = match self.marker_states.get(&id) {
                Some(s) if s.keep => Some(s.clone()),
                Some(_) => self.marker_states.remove(&id),
                None => None,
            };
            if let Some(state) = state {
                self.apply_marker_state(state);
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Backslash comment as a compilable immediate word
    // -----------------------------------------------------------------------

    /// Register `\` as an immediate host function that sets >IN to end of input.
    fn register_backslash(&mut self) -> anyhow::Result<()> {
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // Forth 2012 §6.2.2535 `\`: "Parse and discard the remainder of
            // the parse area." The parse area extends to the end of the
            // current **line**, not the end of the input buffer. Scan from
            // the current `>IN` forward for the first `\n`, and set `>IN`
            // to the position after it. If there's no newline, stop at
            // `#TIB` (end of buffer), matching the single-line case.
            let b: [u8; 4] = ctx.mem_read_i32(SYSVAR_NUM_TIB as u32).to_le_bytes();
            let num_tib = u32::from_le_bytes(b);
            let b: [u8; 4] = ctx.mem_read_i32(SYSVAR_TO_IN as u32).to_le_bytes();
            let mut to_in = u32::from_le_bytes(b);
            while to_in < num_tib {
                let ch = ctx.mem_read_u8(INPUT_BUFFER_BASE + to_in);
                to_in += 1;
                if ch == b'\n' {
                    break;
                }
            }
            ctx.mem_write_i32(SYSVAR_TO_IN as u32, to_in as i32);
            Ok(())
        });

        self.register_host_primitive("\\", true, func)?;

        // .( is an immediate word that prints until closing paren.
        // Register as no-op in dictionary so FIND can discover it as immediate.
        // The actual parsing is handled by interpret_token_immediate/compile_token.
        let func = Box::new(|_ctx: &mut dyn HostAccess| Ok(()));
        self.register_host_primitive(".(", true, func)?;

        // ( is an immediate word (comment). Register in dictionary for FIND.
        let func = Box::new(|_ctx: &mut dyn HostAccess| Ok(()));
        self.register_host_primitive("(", true, func)?;

        // Register [IF], [ELSE], [THEN], [DEFINED], [UNDEFINED] as immediate no-ops
        // so they are findable by WORD+FIND. Actual logic is in interpret_token.
        for name in &["[IF]", "[ELSE]", "[THEN]", "[DEFINED]", "[UNDEFINED]"] {
            let func = Box::new(|_ctx: &mut dyn HostAccess| Ok(()));
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
        self.rt.mem_write_slice(INPUT_BUFFER_BASE, &bytes[..len]);
        // Write >IN
        self.rt.mem_write_i32(SYSVAR_TO_IN, self.input_pos as i32);
        // Write STATE
        self.rt.mem_write_i32(SYSVAR_STATE, self.state);
        // Write BASE
        self.rt.mem_write_i32(SYSVAR_BASE_VAR, self.base as i32);
        // Write #TIB (input buffer length)
        self.rt.mem_write_i32(SYSVAR_NUM_TIB, len as i32);
    }

    /// Sync BASE from WASM memory back to Rust after executing a word.
    fn sync_base_from_wasm(&mut self) {
        // Check if BASE was changed via WASM memory write (e.g., `10 BASE !`)
        let wasm_base = self.rt.mem_read_i32(SYSVAR_BASE_VAR) as u32;
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
        self.rt.mem_write_i32(0x30, word_id.0 as i32);
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
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let rsp_val = ctx.get_rsp();
            if rsp_val + 2 * CELL_SIZE > RETURN_STACK_TOP {
                anyhow::bail!("Return stack underflow");
            }
            let sp = ctx.get_dsp();
            // Return stack: x2 at rsp, x1 at rsp+4
            let b: [u8; 4] = ctx.mem_read_i32(rsp_val as u32).to_le_bytes();
            let x2 = i32::from_le_bytes(b);
            let b: [u8; 4] = ctx.mem_read_i32((rsp_val + 4) as u32).to_le_bytes();
            let x1 = i32::from_le_bytes(b);
            // Push x1 then x2 onto data stack
            let mem_len = ctx.mem_len() as u32;
            if sp < 8 || sp > mem_len {
                return Err(anyhow::anyhow!("data stack overflow in 2R@"));
            }
            let new_sp = sp - 8;
            ctx.mem_write_i32((new_sp + 4) as u32, x1 as i32);
            ctx.mem_write_i32(new_sp as u32, x2 as i32);
            ctx.set_dsp((new_sp as i32) as u32);
            Ok(())
        });

        self.register_host_primitive("2R@", false, func)?;
        Ok(())
    }

    /// UNUSED ( -- u ) return available dictionary space.
    fn register_unused(&mut self) -> anyhow::Result<()> {
        let here_cell = self.here_cell.clone();

        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let mut here_val = here_cell.as_ref().map_or(0, |c| *c.lock().unwrap());
            let mem_size = ctx.mem_len() as u32;
            // Also read SYSVAR_HERE from WASM (Forth ALLOT/,/C, update it directly)
            let mem_here = ctx.mem_read_i32(SYSVAR_HERE) as u32;
            if mem_here > here_val && mem_here < mem_size {
                here_val = mem_here;
            }
            let unused = mem_size.saturating_sub(here_val);
            let sp = ctx.get_dsp();
            if sp < CELL_SIZE || sp > mem_size {
                return Err(anyhow::anyhow!("data stack overflow in UNUSED"));
            }
            let new_sp = sp - CELL_SIZE;
            ctx.mem_write_i32(new_sp as u32, (unused as i32) as i32);
            ctx.set_dsp((new_sp as i32) as u32);
            Ok(())
        });

        self.register_host_primitive("UNUSED", false, func)?;
        Ok(())
    }

    /// UTIME ( -- ud ) push microseconds since epoch as a double-cell value.
    fn register_utime(&mut self) -> anyhow::Result<()> {
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // SystemTime::now() panics on wasm32-unknown-unknown; fail as a
            // catchable Forth error instead of poisoning the VM with a trap.
            #[cfg(target_arch = "wasm32")]
            {
                let _ = ctx;
                Err(anyhow::anyhow!("UTIME: no time source on this platform"))
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                use std::time::{SystemTime, UNIX_EPOCH};
                let us = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                let lo = us as i32;
                let hi = (us >> 32) as i32;
                // Push double: lo first (deeper), then hi on top
                let sp = ctx.get_dsp();
                let new_sp = sp - 2 * CELL_SIZE;
                ctx.mem_write_i32(new_sp as u32, hi as i32);
                ctx.mem_write_slice(new_sp as u32 + 4, &lo.to_le_bytes());
                ctx.set_dsp((new_sp as i32) as u32);
                Ok(())
            }
        });

        self.register_host_primitive("UTIME", false, func)?;
        Ok(())
    }

    /// RANDOM ( -- u ) return a 32-bit pseudo-random cell (xorshift64).
    /// RND-SEED ( u -- ) reseed the PRNG; seed=0 is forced to a nonzero constant.
    fn register_random(&mut self) -> anyhow::Result<()> {
        let state = Arc::clone(&self.rng_state);
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let mut s = state.lock().unwrap();
            let mut x = *s;
            if x == 0 {
                x = 0xDEAD_BEEF_CAFE_BABE;
            }
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            *s = x;
            drop(s);
            let sp = ctx.get_dsp();
            let new_sp = sp - CELL_SIZE;
            ctx.mem_write_i32(new_sp as u32, x as i32);
            ctx.set_dsp(new_sp);
            Ok(())
        });
        self.register_host_primitive("RANDOM", false, func)?;

        let state = Arc::clone(&self.rng_state);
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let seed = host_pop(ctx)? as u32 as u64;
            let mut s = state.lock().unwrap();
            *s = if seed == 0 {
                0xDEAD_BEEF_CAFE_BABE
            } else {
                seed
            };
            Ok(())
        });
        self.register_host_primitive("RND-SEED", false, func)?;
        Ok(())
    }

    /// PARSE ( char "ccc<char>" -- c-addr u ) as inline host function.
    fn register_parse_host(&mut self) -> anyhow::Result<()> {
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // Pop delimiter from data stack
            let sp = host_need(ctx, 1)?;
            let delim = ctx.mem_read_i32(sp as u32) as u8;
            let sp = sp + CELL_SIZE; // pop delimiter

            // Read >IN and #TIB from WASM memory
            let b: [u8; 4] = ctx.mem_read_i32(SYSVAR_TO_IN as u32).to_le_bytes();
            let mut to_in = u32::from_le_bytes(b);
            let b: [u8; 4] = ctx.mem_read_i32(SYSVAR_NUM_TIB as u32).to_le_bytes();
            let num_tib = u32::from_le_bytes(b);

            // Skip one leading space (outer interpreter's trailing delimiter)
            if to_in < num_tib && ctx.mem_read_u8((INPUT_BUFFER_BASE + to_in) as u32) == b' ' {
                to_in += 1;
            }

            // Parse until delimiter
            let start = to_in;
            while to_in < num_tib {
                if ctx.mem_read_u8((INPUT_BUFFER_BASE + to_in) as u32) == delim {
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
            ctx.mem_write_i32(SYSVAR_TO_IN as u32, to_in as i32);

            // Push (c-addr u) to data stack
            let c_addr = INPUT_BUFFER_BASE + start;
            let new_sp = sp - 2 * CELL_SIZE;
            ctx.mem_write_i32(new_sp, parsed_len as i32);
            ctx.mem_write_i32(new_sp + CELL_SIZE, c_addr as i32);
            ctx.set_dsp(new_sp);

            Ok(())
        });

        self.register_host_primitive("PARSE", false, func)?;
        Ok(())
    }

    /// PARSE-NAME ( "<spaces>name<space>" -- c-addr u ) as inline host function.
    fn register_parse_name_host(&mut self) -> anyhow::Result<()> {
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // Read >IN and #TIB from WASM memory
            let b: [u8; 4] = ctx.mem_read_i32(SYSVAR_TO_IN as u32).to_le_bytes();
            let mut to_in = u32::from_le_bytes(b);
            let b: [u8; 4] = ctx.mem_read_i32(SYSVAR_NUM_TIB as u32).to_le_bytes();
            let num_tib = u32::from_le_bytes(b);

            // Skip leading whitespace
            while to_in < num_tib {
                if !ctx
                    .mem_read_u8((INPUT_BUFFER_BASE + to_in) as u32)
                    .is_ascii_whitespace()
                {
                    break;
                }
                to_in += 1;
            }
            let start = to_in;

            // Parse until whitespace
            while to_in < num_tib {
                if ctx
                    .mem_read_u8((INPUT_BUFFER_BASE + to_in) as u32)
                    .is_ascii_whitespace()
                {
                    break;
                }
                to_in += 1;
            }
            let parsed_len = to_in - start;

            // Update >IN
            ctx.mem_write_i32(SYSVAR_TO_IN as u32, to_in as i32);

            // Push (c-addr u) to data stack
            let c_addr = INPUT_BUFFER_BASE + start;
            let sp = ctx.get_dsp();
            let new_sp = sp - 2 * CELL_SIZE;
            ctx.mem_write_i32(new_sp, parsed_len as i32);
            ctx.mem_write_i32(new_sp + CELL_SIZE, c_addr as i32);
            ctx.set_dsp(new_sp);

            Ok(())
        });

        self.register_host_primitive("PARSE-NAME", false, func)?;
        Ok(())
    }

    /// REFILL ( -- flag ) in piped/string mode, always returns FALSE.
    fn register_refill(&mut self) -> anyhow::Result<()> {
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = ctx.get_dsp();
            let mem_len = ctx.mem_len() as u32;
            if sp < CELL_SIZE || sp > mem_len {
                return Err(anyhow::anyhow!("data stack overflow in REFILL"));
            }
            let new_sp = sp - CELL_SIZE;
            ctx.mem_write_i32(new_sp as u32, 0i32 as i32);
            ctx.set_dsp((new_sp as i32) as u32);
            Ok(())
        });

        self.register_host_primitive("REFILL", false, func)?;

        // ACCEPT ( c-addr +n1 -- +n2 )  receive up to +n1 characters.
        // In non-interactive mode, return 0 (no input).
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // Pop +n1 (max count) and c-addr from stack
            let sp = host_need(ctx, 2)?;
            let new_sp = sp + CELL_SIZE; // pop +n1
            let new_sp = new_sp + CELL_SIZE; // pop c-addr
            // Push 0 (no characters received)
            let result_sp = new_sp - CELL_SIZE;
            ctx.mem_write_i32(result_sp as u32, 0i32 as i32);
            ctx.set_dsp((result_sp as i32) as u32);
            Ok(())
        });
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
        // ALLOCATE ( u -- a-addr ior )
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = host_need(ctx, 1)?;
            let size = ctx.mem_read_i32(sp as u32) as u32;

            let mem_len = ctx.mem_len() as u32;

            // Reject obviously impossible sizes (> available memory)
            if size > mem_len / 2 {
                ctx.mem_write_i32(sp as u32, 0i32 as i32);
                let new_sp = sp - CELL_SIZE;
                ctx.mem_write_i32(new_sp as u32, (-1i32) as i32);
                ctx.set_dsp((new_sp as i32) as u32);
                return Ok(());
            }

            // Allocate from top of memory, growing downward
            // Use last 4 bytes of memory as the allocation pointer
            let alloc_ptr_addr = mem_len - 4;
            let mut alloc_top = ctx.mem_read_i32(alloc_ptr_addr) as u32;
            if alloc_top == 0 {
                alloc_top = mem_len - 8; // Initialize: leave room for pointer
            }

            // Block: [size(4)] [data(size)] — aligned to 4 bytes
            let aligned_size = (size + 3) & !3;
            let block_size = 4 + aligned_size;

            if alloc_top < block_size + 0x20000 {
                // Not enough memory (leave some space for dictionary growth)
                // Replace u with a-addr=0, push ior=-1
                ctx.mem_write_i32(sp as u32, 0i32 as i32);
                let new_sp = sp - CELL_SIZE;
                ctx.mem_write_i32(new_sp as u32, (-1i32) as i32);
                ctx.set_dsp((new_sp as i32) as u32);
                return Ok(());
            }

            let block_start = alloc_top - block_size;
            let data_addr = block_start + 4; // skip size header
            // Write size header
            ctx.mem_write_i32(block_start as u32, size as i32);
            // Zero the allocated area
            for i in 0..aligned_size as usize {
                ctx.mem_write_u8(data_addr + i as u32, 0);
            }
            // Update allocation pointer
            ctx.mem_write_i32(alloc_ptr_addr, block_start as i32);

            // Replace u with a-addr, push ior=0
            ctx.mem_write_i32(sp as u32, (data_addr as i32) as i32);
            let new_sp = sp - CELL_SIZE;
            ctx.mem_write_i32(new_sp as u32, 0i32 as i32);
            ctx.set_dsp((new_sp as i32) as u32);
            Ok(())
        });
        self.register_host_primitive("ALLOCATE", false, func)?;

        // FREE ( a-addr -- ior )
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            // Simple allocator: FREE is a no-op (arena style), return ior=0
            let sp = host_need(ctx, 1)?;
            // Replace a-addr with ior=0
            ctx.mem_write_i32(sp as u32, 0i32 as i32);
            Ok(())
        });
        self.register_host_primitive("FREE", false, func)?;

        // RESIZE ( a-addr u -- a-addr2 ior )
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = host_need(ctx, 2)?;
            let new_size = ctx.mem_read_i32(sp as u32) as u32;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 4) as u32).to_le_bytes();
            let old_addr = u32::from_le_bytes(b);

            let mem_len = ctx.mem_len() as u32;

            // Reject obviously impossible sizes
            if new_size > mem_len / 2 {
                ctx.mem_write_i32(sp + 4, old_addr as i32);
                ctx.mem_write_i32(sp, -1);
                return Ok(());
            }

            // Read old size from header (4 bytes before old_addr)
            let old_size = if old_addr >= 4 {
                let b: [u8; 4] = ctx.mem_read_i32((old_addr - 4) as u32).to_le_bytes();
                u32::from_le_bytes(b)
            } else {
                0
            };

            let alloc_ptr_addr = mem_len - 4;
            let mut alloc_top = ctx.mem_read_i32(alloc_ptr_addr) as u32;
            if alloc_top == 0 {
                alloc_top = mem_len - 8;
            }

            let aligned_size = (new_size + 3) & !3;
            let block_size = 4 + aligned_size;

            if alloc_top < block_size + 0x20000 {
                // Allocation failure
                // Keep old a-addr, push ior=-1
                let new_sp = sp + CELL_SIZE; // pop new_size
                ctx.mem_write_i32(new_sp, old_addr as i32);
                let new_sp = new_sp - CELL_SIZE;
                ctx.mem_write_i32(new_sp, -1);
                ctx.set_dsp(new_sp);
                return Ok(());
            }

            let block_start = alloc_top - block_size;
            let new_addr = block_start + 4;

            // Copy old data to new location
            let copy_len = old_size.min(new_size) as usize;
            for i in 0..copy_len {
                let byte = ctx.mem_read_u8(old_addr + i as u32);
                ctx.mem_write_u8(new_addr + i as u32, byte);
            }
            // Zero any extra space
            for i in copy_len..aligned_size as usize {
                ctx.mem_write_u8(new_addr + i as u32, 0);
            }
            // Write size header
            ctx.mem_write_i32(block_start as u32, new_size as i32);
            // Update allocation pointer
            ctx.mem_write_i32(alloc_ptr_addr, block_start as i32);

            // Replace (a-addr u) with (a-addr2 ior)
            ctx.mem_write_i32(sp + 4, new_addr as i32);
            ctx.mem_write_i32(sp, 0);
            Ok(())
        });
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
        let name =
            String::from_utf8_lossy(&self.rt.mem_read_slice(addr as u32, u as usize)).to_string();

        if let Some((_word_addr, word_id, is_imm)) = self.dictionary.find_in_wid(&name, wid) {
            self.push_data_stack(word_id.0 as i32)?;
            self.push_data_stack(if is_imm { 1 } else { -1 })?;
        } else {
            self.push_data_stack(0)?;
        }
        Ok(())
    }

    /// WORDS ( -- ) list visible words; `WORDS <substring>` filters them
    /// (case-insensitive). Internal (underscore-prefixed) words are
    /// skipped. Output wraps at 78 columns and ends with a count.
    fn do_words(&mut self) {
        // In interpret mode an optional token on the same line is a filter;
        // the special filter ALL switches to the grouped full view.
        let filter = if self.state == 0 {
            self.next_token().map(|t| t.to_ascii_uppercase())
        } else {
            None
        };
        if filter.as_deref() == Some("ALL") {
            self.do_words_all();
            return;
        }
        let names = self.dictionary.visible_words(false);
        let shown: Vec<&str> = names
            .iter()
            .filter(|n| {
                filter
                    .as_ref()
                    .is_none_or(|f| n.to_ascii_uppercase().contains(f))
            })
            .map(String::as_str)
            .collect();
        let mut out = self.output.lock().unwrap();
        push_wrapped(&mut out, &shown);
        out.push_str(&format!("{} words\n", shown.len()));
    }

    /// `WORDS ALL` -- grouped full view: one section per wordlist (search
    /// order first, then any other populated wids), then internal words.
    fn do_words_all(&mut self) {
        let entries = self.dictionary.visible_entries();
        let mut wids: Vec<u32> = self.search_order.lock().unwrap().clone();
        for (_, wid, _) in &entries {
            if !wids.contains(wid) {
                wids.push(*wid);
            }
        }
        let wid_name = |wid: u32| {
            if wid == 1 {
                "FORTH".to_string()
            } else {
                format!("wid#{wid}")
            }
        };
        let mut out = self.output.lock().unwrap();
        for wid in wids {
            let names: Vec<&str> = entries
                .iter()
                .filter(|(_, w, internal)| *w == wid && !internal)
                .map(|(n, _, _)| n.as_str())
                .collect();
            if names.is_empty() {
                continue;
            }
            out.push_str(&format!("-- {} ({} words)\n", wid_name(wid), names.len()));
            push_wrapped(&mut out, &names);
        }
        let internals: Vec<&str> = entries
            .iter()
            .filter(|(_, _, internal)| *internal)
            .map(|(n, _, _)| n.as_str())
            .collect();
        if !internals.is_empty() {
            out.push_str(&format!("-- internal ({} words)\n", internals.len()));
            push_wrapped(&mut out, &internals);
        }
    }

    /// Map function-table index -> word name via a dictionary walk.
    /// Newest-first, so redefinitions resolve to the visible name.
    fn word_id_names(&self) -> HashMap<u32, String> {
        let mut map = HashMap::new();
        let mut addr = self.dictionary.latest();
        while addr != 0 {
            if let (Ok(name), Ok(code)) = (
                self.dictionary.word_name(addr),
                self.dictionary.code_field(addr),
            ) {
                map.entry(code).or_insert(name);
            }
            let link = self.dictionary.read_link(addr);
            if link == addr {
                break;
            }
            addr = link;
        }
        map
    }

    /// Parse the mandatory word-name argument of SEE/SEE-IR/HELP.
    fn parse_name_arg(&mut self, who: &str) -> anyhow::Result<String> {
        self.next_token()
            .ok_or_else(|| anyhow::anyhow!("{who}: expected word name"))
    }

    /// `HELP [name]` — stack effect + description from the doc table; user
    /// words echo their leading `( ... -- ... )` comment from the captured
    /// source. Bare HELP prints usage.
    fn do_help(&mut self) -> anyhow::Result<()> {
        // Like WORDS' filter, the name is read from the same line (optional).
        let name = if self.state == 0 {
            self.next_token()
        } else {
            None
        };
        let Some(name) = name else {
            self.output.lock().unwrap().push_str(
                "HELP <word>  -- stack effect + description. \
                 Also try: WORDS, SEE <word>, SEE-IR <word>\n",
            );
            return Ok(());
        };
        let upper = name.to_ascii_uppercase();
        let found = self.dictionary.find(&upper);
        let mut line = if let Some((effect, desc)) = crate::wordhelp::lookup(&upper) {
            format!("{upper}  {effect}  {desc}")
        } else if let Some((_, word_id, _)) = found {
            match self
                .word_sources
                .get(&word_id)
                .and_then(|s| crate::wordhelp::stack_comment(s))
            {
                Some(effect) => {
                    format!("{upper}  {effect}  user word; SEE {upper} shows the source")
                }
                None => format!("no help for {upper}; try SEE {upper}"),
            }
        } else if INTERPRETER_TOKENS.contains(&upper.as_str()) {
            format!("{upper} is handled by the outer interpreter")
        } else {
            anyhow::bail!("HELP: unknown word: {name}");
        };
        if found.is_some_and(|(_, _, imm)| imm) {
            line.push_str("  immediate");
        }
        line.push('\n');
        self.output.lock().unwrap().push_str(&line);
        Ok(())
    }

    /// `SEE name` — print captured source, a synthesized definition for
    /// data words, or an IR/stub fallback. Never dead-ends on a defined word.
    fn do_see(&mut self) -> anyhow::Result<()> {
        let name = self.parse_name_arg("SEE")?;
        let upper = name.to_ascii_uppercase();
        let Some((addr, word_id, is_immediate)) = self.dictionary.find(&upper) else {
            if INTERPRETER_TOKENS.contains(&upper.as_str()) {
                self.output.lock().unwrap().push_str(&format!(
                    "SEE: {upper} is handled by the outer interpreter (compiler word)\n"
                ));
                return Ok(());
            }
            anyhow::bail!("SEE: unknown word: {name}");
        };
        let stored_name = self.dictionary.word_name(addr).unwrap_or(upper);
        // Built-in words carry their HELP line as a leading comment.
        let help = crate::wordhelp::lookup(&stored_name)
            .map(|(effect, desc)| format!("\\ {stored_name}  {effect}  {desc}\n"))
            .unwrap_or_default();
        let mut text = if let Some(src) = self.word_sources.get(&word_id) {
            src.clone()
        } else if let Some(synth) = self.synthesize_data_word(&stored_name, word_id) {
            synth
        } else if let Some(body) = self.ir_bodies.get(&word_id) {
            let names = self.word_id_names();
            let ir = crate::see::format_ir_with(body, &|id| names.get(&id.0).cloned());
            format!("\\ {stored_name} is a primitive; IR:\n{}", ir.trim_end())
        } else if self.host_word_names.contains_key(&word_id) {
            format!("\\ {stored_name} is a built-in host word")
        } else {
            format!("\\ {stored_name}: no source available")
        };
        if is_immediate {
            text.push_str("\nimmediate");
        }
        text.push('\n');
        self.output.lock().unwrap().push_str(&(help + &text));
        Ok(())
    }

    /// Synthesize `SEE` output for mutable data words (VALUE family, DEFER)
    /// whose current value lives in WASM memory. Gated on `word_pfa_map` so
    /// address-pushing primitives (BASE, ...) never masquerade as data
    /// words. A DOES>-product whose body happens to match a VALUE shape
    /// prints as one — behaviorally equivalent, provenance lost.
    fn synthesize_data_word(&mut self, name: &str, word_id: WordId) -> Option<String> {
        let &pfa = self.word_pfa_map.get(&word_id.0)?;
        if self.two_value_words.contains(&word_id.0) {
            let lo = self.rt.mem_read_i32(pfa);
            let hi = self.rt.mem_read_i32(pfa + CELL_SIZE);
            return Some(format!("{lo} {hi} 2VALUE {name}"));
        }
        if self.fvalue_words.contains(&word_id.0) {
            let bytes: [u8; 8] = self.rt.mem_read_slice(pfa, 8).try_into().ok()?;
            let r = f64::from_le_bytes(bytes);
            return Some(format!("{r:e} FVALUE {name}"));
        }
        match self.ir_bodies.get(&word_id)?.as_slice() {
            [IrOp::PushI32(addr), IrOp::Fetch] => {
                let cur = self.rt.mem_read_i32(*addr as u32);
                Some(format!("{cur} VALUE {name}"))
            }
            [IrOp::PushI32(addr), IrOp::Fetch, IrOp::Execute] => {
                let xt = self.rt.mem_read_i32(*addr as u32) as u32;
                Some(match self.word_id_names().get(&xt) {
                    Some(t) => format!("DEFER {name}  ( IS {t} )"),
                    None => format!("DEFER {name}"),
                })
            }
            _ => None,
        }
    }

    /// `SEE-IR name` — print the stored post-optimization IR of a word.
    fn do_see_ir(&mut self) -> anyhow::Result<()> {
        let name = self.parse_name_arg("SEE-IR")?;
        let upper = name.to_ascii_uppercase();
        let help = crate::wordhelp::lookup(&upper)
            .map(|(effect, desc)| format!("\\ {upper}  {effect}  {desc}\n"))
            .unwrap_or_default();
        let text = if let Some((_addr, word_id, is_immediate)) = self.dictionary.find(&upper) {
            if let Some(body) = self.ir_bodies.get(&word_id) {
                let mut header = format!("\\ {upper} -- {} ops (optimized IR)", body.len());
                if is_immediate {
                    header.push_str(" immediate");
                }
                if self.does_definitions.contains_key(&word_id) {
                    header.push_str(" does>");
                }
                header.push('\n');
                let names = self.word_id_names();
                header + &crate::see::format_ir_with(body, &|id| names.get(&id.0).cloned())
            } else if self.host_word_names.contains_key(&word_id) {
                format!("SEE-IR: {upper} is a built-in host word\n")
            } else {
                format!("SEE-IR: {upper} has no IR body\n")
            }
        } else if INTERPRETER_TOKENS.contains(&upper.as_str()) {
            format!("SEE-IR: {upper} is handled directly by the outer interpreter\n")
        } else {
            anyhow::bail!("SEE-IR: unknown word: {name}");
        };
        self.output.lock().unwrap().push_str(&(help + &text));
        Ok(())
    }

    /// Register Search-Order word set words.
    fn register_search_order(&mut self) -> anyhow::Result<()> {
        // FORTH-WORDLIST ( -- wid )
        self.register_primitive("FORTH-WORDLIST", false, vec![IrOp::PushI32(1)])?;

        // GET-CURRENT ( -- wid )
        // Returns the current compilation wordlist from pending mechanism
        let pending = Arc::clone(&self.pending_define);
        let func = Box::new(move |_ctx: &mut dyn HostAccess| {
            pending.lock().unwrap().push(20); // GET-CURRENT action
            Ok(())
        });
        self.register_host_primitive("GET-CURRENT", false, func)?;

        // SET-CURRENT ( wid -- )
        let pending = Arc::clone(&self.pending_define);
        let func = Box::new(move |_ctx: &mut dyn HostAccess| {
            pending.lock().unwrap().push(21); // SET-CURRENT action
            Ok(())
        });
        self.register_host_primitive("SET-CURRENT", false, func)?;

        // WORDLIST ( -- wid ) — directly allocates and pushes
        {
            let nw = Arc::clone(&self.next_wid);
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let mut nw_val = nw.lock().unwrap();
                let wid = *nw_val;
                *nw_val += 1;
                drop(nw_val);
                let sp = ctx.get_dsp();
                let new_sp = sp - CELL_SIZE;
                ctx.mem_write_i32(new_sp as u32, (wid as i32) as i32);
                ctx.set_dsp((new_sp as i32) as u32);
                Ok(())
            });
            self.register_host_primitive("WORDLIST", false, func)?;
        }

        // GET-ORDER ( -- widn ... wid1 n ) — directly pushes search order
        {
            let so = Arc::clone(&self.search_order);
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let order = so.lock().unwrap().clone();
                let n = order.len() as u32;
                let sp = ctx.get_dsp();
                let new_sp = sp - (n + 1) * CELL_SIZE;
                // wid1 (top of search order) = closest to n on stack
                // widn (bottom of search order) = deepest on stack
                for (i, &wid) in order.iter().enumerate() {
                    ctx.mem_write_i32(new_sp + CELL_SIZE + i as u32 * CELL_SIZE, wid as i32);
                }
                ctx.mem_write_i32(new_sp, n as i32);
                ctx.set_dsp(new_sp);
                Ok(())
            });
            self.register_host_primitive("GET-ORDER", false, func)?;
        }

        // SET-ORDER ( widn ... wid1 n -- ) — directly pops and sets search order
        {
            let so = Arc::clone(&self.search_order);
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let sp = host_need(ctx, 1)?;
                let n = ctx.mem_read_i32(sp as u32);
                if !(-1..=64).contains(&n) {
                    anyhow::bail!("SET-ORDER: bad wordlist count: {n}");
                }
                if n != -1 {
                    host_need(ctx, 1 + n as u32)?;
                }

                if n == -1 {
                    *so.lock().unwrap() = vec![1];
                    ctx.set_dsp(((sp + CELL_SIZE) as i32) as u32);
                } else {
                    let n = n as u32;
                    let mut order = Vec::new();
                    // wid1 is just above n on stack, widn is deepest
                    for i in 0..n {
                        let wid = ctx.mem_read_i32(sp + CELL_SIZE + i * CELL_SIZE) as u32;
                        order.push(wid);
                    }
                    *so.lock().unwrap() = order;
                    ctx.set_dsp(sp + (1 + n) * CELL_SIZE);
                }
                Ok(())
            });
            self.register_host_primitive("SET-ORDER", false, func)?;
        }

        // ONLY ( -- ) set minimum search order
        {
            let so = Arc::clone(&self.search_order);
            let func = Box::new(move |_ctx: &mut dyn HostAccess| {
                *so.lock().unwrap() = vec![1];
                Ok(())
            });
            self.register_host_primitive("ONLY", false, func)?;
        }

        // ALSO ( -- ) duplicate top of search order
        {
            let so = Arc::clone(&self.search_order);
            let func = Box::new(move |_ctx: &mut dyn HostAccess| {
                let mut order = so.lock().unwrap();
                if let Some(&top) = order.first() {
                    order.insert(0, top);
                }
                Ok(())
            });
            self.register_host_primitive("ALSO", false, func)?;
        }

        // PREVIOUS ( -- ) remove top of search order
        {
            let so = Arc::clone(&self.search_order);
            let func = Box::new(move |_ctx: &mut dyn HostAccess| {
                let mut order = so.lock().unwrap();
                if !order.is_empty() {
                    order.remove(0);
                }
                Ok(())
            });
            self.register_host_primitive("PREVIOUS", false, func)?;
        }

        // DEFINITIONS ( -- ) set compilation wordlist to top of search order
        {
            let so = Arc::clone(&self.search_order);
            let pending = Arc::clone(&self.pending_define);
            let func = Box::new(move |_ctx: &mut dyn HostAccess| {
                let order = so.lock().unwrap();
                if !order.is_empty() {
                    // Use pending to set current_wid (needs dictionary access)
                    drop(order);
                    pending.lock().unwrap().push(33);
                }
                Ok(())
            });
            self.register_host_primitive("DEFINITIONS", false, func)?;
        }

        // FORTH ( -- ) replace top of search order with FORTH wordlist
        {
            let so = Arc::clone(&self.search_order);
            let func = Box::new(move |_ctx: &mut dyn HostAccess| {
                let mut order = so.lock().unwrap();
                if !order.is_empty() {
                    order[0] = 1;
                } else {
                    order.push(1);
                }
                Ok(())
            });
            self.register_host_primitive("FORTH", false, func)?;
        }

        // SEARCH-WORDLIST ( c-addr u wid -- 0 | xt 1 | xt -1 )
        let pending = Arc::clone(&self.pending_define);
        let func = Box::new(move |_ctx: &mut dyn HostAccess| {
            pending.lock().unwrap().push(25); // SEARCH-WORDLIST action
            Ok(())
        });
        self.register_host_primitive("SEARCH-WORDLIST", false, func)?;

        Ok(())
    }

    /// Register N>R and NR> for the Programming-Tools word set.
    fn register_n_to_r(&mut self) -> anyhow::Result<()> {
        // N>R ( xn..x1 n -- ; R: -- x1..xn n )
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = host_need(ctx, 1)?;
            let n = ctx.mem_read_i32(sp as u32) as u32;
            host_need(ctx, n.saturating_add(1))?;

            let mut rsp_val = ctx.get_rsp();

            // Move n items from data stack to return stack, plus n itself
            // Data stack: x1(deepest)..xn(just below n), n(top)
            // Need to push x1 first (deepest on R), then x2, ..., xn, then n
            let items_base = sp + 4; // past n
            for i in (0..n).rev() {
                let val = ctx.mem_read_i32(items_base + i * 4);
                rsp_val -= 4;
                ctx.mem_write_i32(rsp_val, val);
            }
            // Push n to return stack
            rsp_val -= 4;
            ctx.mem_write_i32(rsp_val as u32, (n as i32) as i32);
            ctx.set_rsp((rsp_val as i32) as u32);

            // Pop n+1 items from data stack
            let new_sp = sp + (n + 1) * 4;
            ctx.set_dsp((new_sp as i32) as u32);
            Ok(())
        });
        self.register_host_primitive("N>R", false, func)?;

        // NR> ( -- xn..x1 n ; R: x1..xn n -- )
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let mut rsp_val = ctx.get_rsp();
            // Pop n from return stack
            let b: [u8; 4] = ctx.mem_read_i32(rsp_val as u32).to_le_bytes();
            let n = i32::from_le_bytes(b) as u32;
            rsp_val += 4;

            let sp = ctx.get_dsp();
            // Make space for n+1 items on data stack
            let new_sp = sp - (n + 1) * 4;

            // Pop n items from return stack to data stack
            // R-stack has x1(deepest)..xn(top after n)
            // Data stack needs xn..x1 n (with n on top)
            for i in 0..n {
                let val = ctx.mem_read_i32(rsp_val);
                rsp_val += 4;
                ctx.mem_write_i32(new_sp + 4 + i * 4, val);
            }
            ctx.set_rsp((rsp_val as i32) as u32);

            // Push n on top of data stack
            ctx.mem_write_i32(new_sp as u32, (n as i32) as i32);
            ctx.set_dsp((new_sp as i32) as u32);
            Ok(())
        });
        self.register_host_primitive("NR>", false, func)?;

        Ok(())
    }

    /// Register WORDS / SEE-IR for the Programming-Tools word set.
    /// Each runs Rust-side via `pending_define` so it can parse arguments
    /// with `next_token()` and write to `self.output`.
    fn register_words(&mut self) -> anyhow::Result<()> {
        for (name, code) in [
            ("WORDS", 40),
            ("SEE", 41),
            ("SEE-IR", 42),
            ("HELP", 43),
            ("INCLUDED", 44),
            ("INCLUDE", 45),
        ] {
            let pending = Arc::clone(&self.pending_define);
            let func: HostFn = Box::new(move |_ctx: &mut dyn HostAccess| {
                pending.lock().unwrap().push(code);
                Ok(())
            });
            self.register_host_primitive(name, false, func)?;
        }
        Ok(())
    }

    /// Register UNESCAPE, SUBSTITUTE, REPLACES for the String word set.
    fn register_string_substitution(&mut self) -> anyhow::Result<()> {
        // UNESCAPE ( c-addr1 u1 c-addr2 -- c-addr2 u2 )
        // Copy string escaping each % as %%
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = host_need(ctx, 3)?;
            let dest = ctx.mem_read_i32(sp as u32) as u32;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 4) as u32).to_le_bytes();
            let u1 = u32::from_le_bytes(b);
            let b: [u8; 4] = ctx.mem_read_i32((sp + 8) as u32).to_le_bytes();
            let src = u32::from_le_bytes(b);

            // Read source
            let src_bytes: Vec<u8> = ctx.mem_read_slice(src as u32, u1 as usize);

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
            ctx.mem_write_slice(dest as u32, &result[..u2 as usize]);

            // Pop 3, push 2: net sp + 4
            let new_sp = sp + 4;
            ctx.mem_write_i32(new_sp + 4, dest as i32);
            ctx.mem_write_slice(new_sp as u32, &(u2 as i32).to_le_bytes());
            ctx.set_dsp((new_sp as i32) as u32);
            Ok(())
        });
        self.register_host_primitive("UNESCAPE", false, func)?;

        // REPLACES ( c-addr1 u1 c-addr2 u2 -- )
        // Define substitution: name (c-addr2 u2) → replacement (c-addr1 u1)
        let subs = Arc::clone(&self.substitutions);
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = host_need(ctx, 4)?;
            // Stack: u2(sp), c-addr2(sp+4), u1(sp+8), c-addr1(sp+12)
            let u2 = ctx.mem_read_i32(sp as u32) as u32;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 4) as u32).to_le_bytes();
            let name_addr = u32::from_le_bytes(b);
            let b: [u8; 4] = ctx.mem_read_i32((sp + 8) as u32).to_le_bytes();
            let u1 = u32::from_le_bytes(b);
            let b: [u8; 4] = ctx.mem_read_i32((sp + 12) as u32).to_le_bytes();
            let repl_addr = u32::from_le_bytes(b);

            let name = String::from_utf8_lossy(&ctx.mem_read_slice(name_addr as u32, u2 as usize))
                .to_ascii_uppercase();

            // Copy replacement string to Rust-side storage (WASM addresses are transient)
            let repl_bytes = ctx.mem_read_slice(repl_addr as u32, u1 as usize);
            subs.lock().unwrap().insert(name, repl_bytes);

            // Pop 4 items
            let new_sp = sp + 16;
            ctx.set_dsp((new_sp as i32) as u32);
            Ok(())
        });
        self.register_host_primitive("REPLACES", false, func)?;

        // SUBSTITUTE ( c-addr1 u1 c-addr2 u2 -- c-addr2 u2 n )
        // Replace %name% patterns, %% → %
        let subs = Arc::clone(&self.substitutions);
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = host_need(ctx, 4)?;
            // Stack: u2/capacity(sp), c-addr2/dest(sp+4), u1(sp+8), c-addr1(sp+12)
            let capacity = ctx.mem_read_i32(sp as u32) as u32 as usize;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 4) as u32).to_le_bytes();
            let dest = u32::from_le_bytes(b);
            let b: [u8; 4] = ctx.mem_read_i32((sp + 8) as u32).to_le_bytes();
            let u1 = u32::from_le_bytes(b);
            let b: [u8; 4] = ctx.mem_read_i32((sp + 12) as u32).to_le_bytes();
            let src = u32::from_le_bytes(b);

            let src_bytes: Vec<u8> = ctx.mem_read_slice(src as u32, u1 as usize);

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
            ctx.mem_write_slice(dest as u32, &result[..u2 as usize]);

            // Pop 4, push 3: net sp + 4
            let new_sp = sp + 4;
            ctx.mem_write_i32(new_sp + 8, dest as i32);
            ctx.mem_write_i32(new_sp + 4, u2 as i32);
            ctx.mem_write_slice(new_sp as u32, &sub_count.to_le_bytes());
            ctx.set_dsp((new_sp as i32) as u32);
            Ok(())
        });
        self.register_host_primitive("SUBSTITUTE", false, func)?;

        Ok(())
    }

    /// M*/ ( d n1 n2 -- d ) multiply d by n1, divide by n2.
    fn register_m_star_slash(&mut self) -> anyhow::Result<()> {
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = host_need(ctx, 4)?;
            // Stack: n2(sp), n1(sp+4), d-hi(sp+8), d-lo(sp+12)
            let n2 = ctx.mem_read_i32(sp as u32) as i128;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 4) as u32).to_le_bytes();
            let n1 = i32::from_le_bytes(b) as i128;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 8) as u32).to_le_bytes();
            let d_hi = i32::from_le_bytes(b) as i64;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 12) as u32).to_le_bytes();
            let d_lo = u32::from_le_bytes(b) as i64;
            let d = ((d_hi << 32) | (d_lo & 0xFFFF_FFFF)) as i128;

            if n2 == 0 {
                return Err(anyhow::anyhow!("M*/: division by zero"));
            }

            // Symmetric (truncating) division to match WAFER's / behavior
            let product = d * n1;
            let quot = product / n2;

            let result = quot as i64;
            let lo = result as i32;
            let hi = (result >> 32) as i32;
            // Pop 4, push 2: net sp + 8
            let new_sp = sp + 8;
            ctx.mem_write_i32((new_sp + 4) as u32, lo as i32);
            ctx.mem_write_i32(new_sp as u32, hi as i32);
            ctx.set_dsp((new_sp as i32) as u32);
            Ok(())
        });

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
        self.word_sources
            .insert(word_id, format!("{lo} {hi} 2CONSTANT {name}"));
        let config = self.codegen_config(word_id.0);
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
        self.rt.mem_write_slice(addr as u32, &[0u8; 8]);
        self.user_here += 8;
        self.sync_here_cell();

        let word_id = self.dictionary.create(&name, false)?;
        self.dictionary.reveal();

        let ir = vec![IrOp::PushI32(addr as i32)];
        self.ir_bodies.insert(word_id, ir.clone());
        self.word_sources
            .insert(word_id, format!("2VARIABLE {name}"));
        let config = self.codegen_config(word_id.0);
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
        self.rt.mem_write_i32(addr as u32, lo as i32);
        self.rt.mem_write_slice(addr as u32 + 4, &hi.to_le_bytes());
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
        let config = self.codegen_config(word_id.0);
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
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = host_need(ctx, 4)?;
            // Stack: u2(sp), c-addr2(sp+4), u1(sp+8), c-addr1(sp+12)
            let u2 = ctx.mem_read_i32(sp as u32) as usize;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 4) as u32).to_le_bytes();
            let addr2 = u32::from_le_bytes(b) as usize;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 8) as u32).to_le_bytes();
            let u1 = i32::from_le_bytes(b) as usize;
            let b: [u8; 4] = ctx.mem_read_i32((sp + 12) as u32).to_le_bytes();
            let addr1 = u32::from_le_bytes(b) as usize;

            let mem_len = ctx.mem_len();

            // If needle is empty, always found at start
            if u2 == 0 {
                // Return (c-addr1 u1 true)
                // Pop 4, push 3: net sp + 4
                let new_sp = sp + 4;
                ctx.mem_write_i32(new_sp + 8, addr1 as i32);
                ctx.mem_write_i32(new_sp + 4, u1 as i32);
                ctx.mem_write_i32(new_sp as u32, (-1i32) as i32);
                ctx.set_dsp((new_sp as i32) as u32);
                return Ok(());
            }

            if u2 > u1 {
                // Can't find, return (c-addr1 u1 false)
                let new_sp = sp + 4;
                ctx.mem_write_i32(new_sp + 8, addr1 as i32);
                ctx.mem_write_i32(new_sp + 4, u1 as i32);
                ctx.mem_write_i32(new_sp as u32, 0i32 as i32);
                ctx.set_dsp((new_sp as i32) as u32);
                return Ok(());
            }

            // Search for needle in haystack
            let mut found = false;
            let mut found_offset = 0usize;
            for i in 0..=(u1 - u2) {
                let mut matched = true;
                for j in 0..u2 {
                    let h = if addr1 + i + j < mem_len {
                        ctx.mem_read_u8((addr1 + i + j) as u32)
                    } else {
                        0
                    };
                    let n = if addr2 + j < mem_len {
                        ctx.mem_read_u8((addr2 + j) as u32)
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
            if found {
                let new_addr = (addr1 + found_offset) as i32;
                let new_len = (u1 - found_offset) as i32;
                ctx.mem_write_i32((new_sp + 8) as u32, new_addr as i32);
                ctx.mem_write_i32((new_sp + 4) as u32, new_len as i32);
                ctx.mem_write_i32(new_sp as u32, (-1i32) as i32);
            } else {
                ctx.mem_write_i32(new_sp + 8, addr1 as i32);
                ctx.mem_write_i32(new_sp + 4, u1 as i32);
                ctx.mem_write_i32(new_sp as u32, 0i32 as i32);
            }
            ctx.set_dsp((new_sp as i32) as u32);
            Ok(())
        });

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
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = ctx.get_fsp();
            if sp >= FLOAT_STACK_TOP {
                return Err(anyhow::anyhow!("float stack underflow"));
            }
            let bytes: [u8; 8] = ctx.mem_read_slice(sp as u32, 8).try_into().unwrap();
            let a = f64::from_le_bytes(bytes);
            let result = op(a);
            ctx.mem_write_slice(sp as u32, &result.to_le_bytes());
            Ok(())
        });
        self.register_host_primitive(name, false, func)?;
        Ok(())
    }

    /// Pattern for binary float ops: pop two floats (b then a), compute, push result.
    fn register_float_binary(&mut self, name: &str, op: fn(f64, f64) -> f64) -> anyhow::Result<()> {
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = ctx.get_fsp();
            if sp + 8 >= FLOAT_STACK_TOP {
                return Err(anyhow::anyhow!("float stack underflow"));
            }
            let b_bytes: [u8; 8] = ctx.mem_read_slice(sp, 8).try_into().unwrap();
            let a_bytes: [u8; 8] = ctx.mem_read_slice(sp + 8, 8).try_into().unwrap();
            let b = f64::from_le_bytes(b_bytes);
            let a = f64::from_le_bytes(a_bytes);
            let result = op(a, b);
            let new_sp = sp + 8;
            ctx.set_fsp(new_sp);
            ctx.mem_write_slice(new_sp, &result.to_le_bytes());
            Ok(())
        });
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
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let sp = host_fneed(ctx, 3)?;
                let c: [u8; 8] = ctx.mem_read_slice(sp, 8).try_into().unwrap();
                let b: [u8; 8] = ctx.mem_read_slice(sp + 8, 8).try_into().unwrap();
                let a: [u8; 8] = ctx.mem_read_slice(sp + 16, 8).try_into().unwrap();
                ctx.mem_write_slice(sp, &a);
                ctx.mem_write_slice(sp + 8, &c);
                ctx.mem_write_slice(sp + 16, &b);
                Ok(())
            });
            self.register_host_primitive("FROT", false, func)?;
        }

        // FDEPTH ( -- +n ) number of floats on the float stack, pushed onto DATA stack
        {
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let fsp_val = ctx.get_fsp();
                let depth = if fsp_val <= FLOAT_STACK_TOP {
                    ((FLOAT_STACK_TOP - fsp_val) / FLOAT_SIZE) as i32
                } else {
                    0
                };
                // Push onto data stack
                let sp = ctx.get_dsp();
                let new_sp = sp - CELL_SIZE;
                ctx.set_dsp((new_sp as i32) as u32);
                ctx.mem_write_i32(new_sp as u32, depth as i32);
                Ok(())
            });
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
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let r3 = host_fpop(ctx)?;
                let r2 = host_fpop(ctx)?;
                let r1 = host_fpop(ctx)?;

                let result = if r3 > 0.0 {
                    (r1 - r2).abs() < r3
                } else if r3 == 0.0 {
                    r1.to_bits() == r2.to_bits()
                } else {
                    // r3 < 0: relative comparison
                    (r1 - r2).abs() < r3.abs() * (r1.abs() + r2.abs())
                };

                let flag: i32 = if result { -1 } else { 0 };
                let dsp_val = ctx.get_dsp();
                let new_dsp = dsp_val
                    .checked_sub(CELL_SIZE)
                    .ok_or_else(|| anyhow::anyhow!("data stack overflow in F~"))?;
                ctx.set_dsp((new_dsp as i32) as u32);
                ctx.mem_write_i32(new_dsp, flag);
                Ok(())
            });
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
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let sp = host_need(ctx, 1)?;
                let addr = ctx.mem_read_i32(sp as u32) as u32;
                let aligned = (addr + 7) & !7;
                ctx.mem_write_i32(sp as u32, aligned as i32);
                Ok(())
            });
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
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let sp = host_need(ctx, 2)?;
                // Double-cell: hi on top, lo below
                let hi_bytes: [u8; 4] = ctx.mem_read_slice(sp, 4).try_into().unwrap();
                let lo_bytes: [u8; 4] = ctx.mem_read_slice(sp + 4, 4).try_into().unwrap();
                let hi = i32::from_le_bytes(hi_bytes);
                let lo = i32::from_le_bytes(lo_bytes);
                let d = ((hi as i64) << 32) | (lo as u32 as i64);
                let f = d as f64;
                // Pop two cells from data stack
                ctx.set_dsp(((sp + 8) as i32) as u32);
                // Push onto float stack
                let fsp_val = ctx.get_fsp();
                let new_fsp = fsp_val - FLOAT_SIZE;
                ctx.set_fsp((new_fsp as i32) as u32);
                ctx.mem_write_slice(new_fsp as u32, &f.to_le_bytes());
                Ok(())
            });
            self.register_host_primitive("D>F", false, func)?;
        }

        // F>D ( -- d ) ( F: r -- ) convert float to double-cell integer
        {
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let f = host_fpop(ctx)?;
                // Convert to i64
                let d = f as i64;
                let lo = d as i32;
                let hi = (d >> 32) as i32;
                // Push lo then hi onto data stack
                let sp = ctx.get_dsp();
                let new_sp = sp - 8; // two cells
                ctx.set_dsp(new_sp);
                ctx.mem_write_i32(new_sp + 4, lo);
                ctx.mem_write_i32(new_sp, hi);
                Ok(())
            });
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
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let sp = ctx.get_fsp();
                let bytes: [u8; 8] = ctx.mem_read_slice(sp as u32, 8).try_into().unwrap();
                let val = f64::from_le_bytes(bytes);
                let sin_val = val.sin();
                let cos_val = val.cos();
                // Replace TOS with sin, push cos on top
                // Result: sin deeper, cos on top
                let new_sp = sp - 8; // one more item
                if new_sp < FLOAT_STACK_BASE {
                    return Err(anyhow::anyhow!("float stack overflow"));
                }
                ctx.set_fsp((new_sp as i32) as u32);
                ctx.mem_write_slice(new_sp + 8, &sin_val.to_le_bytes());
                ctx.mem_write_slice(new_sp, &cos_val.to_le_bytes());
                Ok(())
            });
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
            let output = Arc::clone(&self.output);
            let precision = Arc::clone(&self.float_precision);
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let val = host_fpop(ctx)?;
                let prec = *precision.lock().unwrap();
                let s = format!("{val:.prec$} ");
                output.lock().unwrap().push_str(&s);
                Ok(())
            });
            self.register_host_primitive("F.", false, func)?;
        }

        // FE. ( F: r -- ) print float in engineering notation
        {
            let output = Arc::clone(&self.output);
            let precision = Arc::clone(&self.float_precision);
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let val = host_fpop(ctx)?;
                let prec = *precision.lock().unwrap();
                let s = format_engineering(val, prec);
                output.lock().unwrap().push_str(&s);
                Ok(())
            });
            self.register_host_primitive("FE.", false, func)?;
        }

        // FS. ( F: r -- ) print float in scientific notation
        {
            let output = Arc::clone(&self.output);
            let precision = Arc::clone(&self.float_precision);
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let val = host_fpop(ctx)?;
                let prec = *precision.lock().unwrap();
                let s = format!("{val:.prec$E} ");
                output.lock().unwrap().push_str(&s);
                Ok(())
            });
            self.register_host_primitive("FS.", false, func)?;
        }

        // PRECISION ( -- u ) get current float output precision
        {
            let precision = Arc::clone(&self.float_precision);
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let prec = *precision.lock().unwrap() as i32;
                let sp = ctx.get_dsp();
                let new_sp = sp - CELL_SIZE;
                ctx.set_dsp((new_sp as i32) as u32);
                ctx.mem_write_i32(new_sp as u32, prec as i32);
                Ok(())
            });
            self.register_host_primitive("PRECISION", false, func)?;
        }

        // SET-PRECISION ( u -- ) set float output precision
        {
            let precision = Arc::clone(&self.float_precision);
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let n = host_pop(ctx)? as usize;
                *precision.lock().unwrap() = n;
                Ok(())
            });
            self.register_host_primitive("SET-PRECISION", false, func)?;
        }

        // REPRESENT ( c-addr u -- n flag1 flag2 ) ( F: r -- )
        {
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let sp = host_need(ctx, 2)?;
                let u = ctx.mem_read_i32(sp) as usize;
                let c_addr = ctx.mem_read_i32(sp + 4) as u32;
                let val = host_fpop(ctx)?;
                // Pop the 2 data cells
                ctx.set_dsp(sp + 8);

                let (digits, exp, is_negative, is_valid) = represent_float(val, u);

                // Store digits at c-addr, then push results
                let digit_bytes = digits.as_bytes();
                let copy_len = digit_bytes.len().min(u);
                // Push n, flag1 (sign), flag2 (valid) onto data stack
                let cur_sp = ctx.get_dsp();
                let new_sp = cur_sp - 12;
                ctx.set_dsp(new_sp);
                ctx.mem_write_slice(c_addr, &digit_bytes[..copy_len]);
                // Bottom: n (exponent)
                ctx.mem_write_i32(new_sp + 8, exp);
                // Middle: flag1 (is_negative => true flag)
                let sign_flag: i32 = if is_negative { -1 } else { 0 };
                ctx.mem_write_i32(new_sp + 4, sign_flag);
                // Top: flag2 (is_valid => true flag)
                let valid_flag: i32 = if is_valid { -1 } else { 0 };
                ctx.mem_write_i32(new_sp, valid_flag);
                Ok(())
            });
            self.register_host_primitive("REPRESENT", false, func)?;
        }

        // >FLOAT ( c-addr u -- flag ) ( F: -- r | ) parse string as float
        {
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let sp = host_need(ctx, 2)?;
                let u = ctx.mem_read_i32(sp) as usize;
                let c_addr = ctx.mem_read_i32(sp + 4) as u32;
                let s_bytes = ctx.mem_read_slice(c_addr, u);
                let s_owned = std::str::from_utf8(&s_bytes).unwrap_or("").to_string();
                // Pop u and c-addr (2 cells), will push back 1 cell (flag)
                ctx.set_dsp(sp + 4);

                let result = parse_forth_float(&s_owned);

                match result {
                    Some(f) => {
                        // Push float onto float stack
                        let fsp_val = ctx.get_fsp();
                        let new_fsp = fsp_val - FLOAT_SIZE;
                        ctx.set_fsp(new_fsp);
                        let flag_sp = ctx.get_dsp();
                        ctx.mem_write_slice(new_fsp, &f.to_le_bytes());
                        ctx.mem_write_i32(flag_sp, -1);
                    }
                    None => {
                        let flag_sp = ctx.get_dsp();
                        ctx.mem_write_i32(flag_sp, 0);
                    }
                }
                Ok(())
            });
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
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let addr = host_pop(ctx)? as u32;
                let val = host_fpop(ctx)?;
                let f32_bytes = (val as f32).to_le_bytes();
                ctx.mem_write_slice(addr, &f32_bytes);
                Ok(())
            });
            self.register_host_primitive("SF!", false, func)?;
        }

        // SF@ ( sf-addr -- ) ( F: -- r ) fetch single-precision float (f32)
        {
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let addr = host_pop(ctx)? as u32;
                let fsp_val = ctx.get_fsp();
                let f32_bytes: [u8; 4] = ctx.mem_read_slice(addr, 4).try_into().unwrap();
                let val = f32::from_le_bytes(f32_bytes) as f64;
                let new_fsp = fsp_val - FLOAT_SIZE;
                ctx.set_fsp(new_fsp);
                ctx.mem_write_slice(new_fsp, &val.to_le_bytes());
                Ok(())
            });
            self.register_host_primitive("SF@", false, func)?;
        }

        // DF! ( df-addr -- ) ( F: r -- ) same as F! (our floats are already f64)
        {
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let addr = host_pop(ctx)? as u32;
                let float_bytes = host_fpop(ctx)?.to_le_bytes();
                ctx.mem_write_slice(addr, &float_bytes);
                Ok(())
            });
            self.register_host_primitive("DF!", false, func)?;
        }

        // DF@ ( df-addr -- ) ( F: -- r ) same as F@ (our floats are already f64)
        {
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let addr = host_pop(ctx)? as u32;
                let fsp_val = ctx.get_fsp();
                let float_bytes: [u8; 8] = ctx.mem_read_slice(addr, 8).try_into().unwrap();
                let val = f64::from_le_bytes(float_bytes);
                let new_fsp = fsp_val - FLOAT_SIZE;
                ctx.set_fsp(new_fsp);
                ctx.mem_write_slice(new_fsp, &val.to_le_bytes());
                Ok(())
            });
            self.register_host_primitive("DF@", false, func)?;
        }

        // SFALIGNED, DFALIGNED (alignment words for single/double floats)
        {
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let sp = host_need(ctx, 1)?;
                let addr = ctx.mem_read_i32(sp as u32) as u32;
                let aligned = (addr + 3) & !3; // 4-byte alignment for single float
                ctx.mem_write_i32(sp as u32, aligned as i32);
                Ok(())
            });
            self.register_host_primitive("SFALIGNED", false, func)?;
        }

        // DFALIGNED is the same as FALIGNED (8-byte alignment)
        {
            let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
                let sp = host_need(ctx, 1)?;
                let addr = ctx.mem_read_i32(sp as u32) as u32;
                let aligned = (addr + 7) & !7;
                ctx.mem_write_i32(sp as u32, aligned as i32);
                Ok(())
            });
            self.register_host_primitive("DFALIGNED", false, func)?;
        }

        Ok(())
    }

    /// Allocate a function table slot for an anonymous host function.
    /// Returns a `WordId` that can be used in `IrOp::Call`.
    /// Does NOT touch the dictionary, so it's safe during colon compilation.
    fn install_anon_func(&mut self, func: HostFn) -> anyhow::Result<WordId> {
        let fn_idx = self.dictionary.next_fn_index();
        self.dictionary.reserve_fn_index();
        self.rt.ensure_table_size(fn_idx)?;
        self.rt.register_host_func(fn_idx, func)?;
        self.next_table_index = self.next_table_index.max(fn_idx + 1);
        Ok(WordId(fn_idx))
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
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = ctx.get_fsp();
            let bytes: [u8; 8] = ctx.mem_read_slice(sp as u32, 8).try_into().unwrap();
            ctx.set_fsp(((sp + FLOAT_SIZE) as i32) as u32);
            ctx.mem_write_slice(pfa as u32, &bytes);
            Ok(())
        });
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
        self.rt.mem_write_slice(addr as u32, &0.0_f64.to_le_bytes());

        // Compile a word that pushes the address onto the DATA stack
        let ir_body = vec![IrOp::PushI32(addr as i32)];
        self.ir_bodies.insert(word_id, ir_body.clone());
        self.word_sources
            .insert(word_id, format!("FVARIABLE {name}"));
        let config = self.codegen_config(word_id.0);
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
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let sp = ctx.get_fsp();
            let new_sp = sp - FLOAT_SIZE;
            if new_sp < FLOAT_STACK_BASE {
                return Err(anyhow::anyhow!("float stack overflow"));
            }
            ctx.set_fsp((new_sp as i32) as u32);
            ctx.mem_write_slice(new_sp as u32, &val.to_le_bytes());
            Ok(())
        });

        self.rt.ensure_table_size(word_id.0)?;
        self.rt.register_host_func(word_id.0, func)?;
        self.dictionary.reveal();
        self.word_sources
            .insert(word_id, format!("{val:e} FCONSTANT {name}"));
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
        self.rt.mem_write_slice(val_addr, &val.to_le_bytes());

        // Create a host function that fetches from storage and pushes onto float stack
        let func: HostFn = Box::new(move |ctx: &mut dyn HostAccess| {
            let bytes = ctx.mem_read_slice(val_addr, 8);
            let sp = ctx.get_fsp();
            let new_sp = sp - FLOAT_SIZE;
            if new_sp < FLOAT_STACK_BASE {
                return Err(anyhow::anyhow!("float stack overflow"));
            }
            ctx.set_fsp(new_sp);
            ctx.mem_write_slice(new_sp, &bytes);
            Ok(())
        });

        self.rt.ensure_table_size(word_id.0)?;
        self.rt.register_host_func(word_id.0, func)?;
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

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;
    use crate::runtime_native::NativeRuntime;

    fn eval(input: &str) -> (Vec<i32>, String) {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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

    #[test]
    fn test_minus_rot() {
        // ( 1 2 3 -- 3 1 2 )  top-first: [2, 1, 3]
        assert_eq!(eval_stack("1 2 3 -ROT"), vec![2, 1, 3]);
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

    #[test]
    fn test_less_or_equal() {
        assert_eq!(eval_stack("3 5 <="), vec![-1]);
        assert_eq!(eval_stack("5 5 <="), vec![-1]);
        assert_eq!(eval_stack("5 3 <="), vec![0]);
    }

    #[test]
    fn test_greater_or_equal() {
        assert_eq!(eval_stack("5 3 >="), vec![-1]);
        assert_eq!(eval_stack("5 5 >="), vec![-1]);
        assert_eq!(eval_stack("3 5 >="), vec![0]);
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
    // New words: S  (state-smart parse-next-token-as-string)
    // ===================================================================

    #[test]
    fn test_s_interpret_type() {
        assert_eq!(eval_output("S hello TYPE"), "hello");
    }

    #[test]
    fn test_s_interpret_length() {
        // S pushes ( c-addr u ); NIP leaves the length on top.
        assert_eq!(eval_stack("S foo NIP"), vec![3]);
    }

    #[test]
    fn test_s_compile_mode() {
        assert_eq!(eval_output(": GREET S world TYPE ; GREET"), "world");
    }

    #[test]
    fn test_s_compile_stored_literal() {
        // The string compiled into a colon def must still be readable after
        // the enclosing input line is gone.
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": NAME S kelvar ;").unwrap();
        vm.evaluate("NAME TYPE").unwrap();
        assert_eq!(vm.take_output(), "kelvar");
    }

    #[test]
    fn test_s_interpret_survives_refill() {
        // Regression: `S name` in interpret mode used to return an address
        // pointing into TIB, so the next REFILL clobbered the string.
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("S test").unwrap();
        vm.evaluate(".S").unwrap();
        vm.take_output();
        vm.evaluate("TYPE").unwrap();
        assert_eq!(vm.take_output(), "test");
    }

    // ===================================================================
    // Float locals: F: prefix in {: ... :}
    // ===================================================================

    #[test]
    fn test_flocal_hypot() {
        // Classic Pythagorean: sqrt(x*x + y*y).
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": HYPOT {: F: x F: y :} x x F* y y F* F+ FSQRT ;")
            .unwrap();
        vm.evaluate("3E 4E HYPOT F>S").unwrap();
        assert_eq!(vm.data_stack(), vec![5]);
    }

    #[test]
    fn test_flocal_to() {
        // TO on a float local reads from the float stack, not the data stack.
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": SETF {: F: a :} 10E TO a a ;").unwrap();
        vm.evaluate("1E SETF F>S").unwrap();
        assert_eq!(vm.data_stack(), vec![10]);
    }

    #[test]
    fn test_flocal_mixed_int_and_float_args() {
        // Declaration order matters for init: rightmost arg is popped first
        // from its stack. Here `n` is int (from dstack) and `f` is float (from fstack).
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": MIX {: n F: f :} f n S>F F+ ;").unwrap();
        vm.evaluate("3 4E MIX F>S").unwrap();
        assert_eq!(vm.data_stack(), vec![7]);
    }

    #[test]
    fn test_flocal_uninit() {
        // Uninitialized float local (after `|`) starts at 0.0 until assigned.
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": U {: | F: tmp :} 9E TO tmp tmp ;").unwrap();
        vm.evaluate("U F>S").unwrap();
        assert_eq!(vm.data_stack(), vec![9]);
    }

    #[test]
    fn test_local_named_s_not_hijacked_by_s_shortcut() {
        // Forth 2012 §13.3.3.2: locals supersede dictionary names within
        // their scope. Regression — local `s` was previously hijacked by
        // the compile-mode `S` string shortcut in compile_token.
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("VARIABLE V 42 V !").unwrap();
        vm.evaluate(": T {: | s :} V TO s s @ ;").unwrap();
        vm.evaluate("T").unwrap();
        assert_eq!(vm.data_stack(), vec![42]);
    }

    #[test]
    fn test_local_named_s_with_fetch_and_store() {
        // Exercises both ForthLocalGet and ForthLocalSet for a local named `s`.
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("VARIABLE V 0 V !").unwrap();
        vm.evaluate(": STORE-VIA-S {: | s :} V TO s 99 s ! ;")
            .unwrap();
        vm.evaluate("STORE-VIA-S V @").unwrap();
        assert_eq!(vm.data_stack(), vec![99]);
    }

    #[test]
    fn test_int_uninit_local_via_pipe_syntax() {
        // Missing coverage: int uninit locals via `{: | name :}` — only the
        // float variant was covered (test_flocal_uninit).
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": U {: | tmp :} 7 TO tmp tmp ;").unwrap();
        vm.evaluate("U").unwrap();
        assert_eq!(vm.data_stack(), vec![7]);
    }

    #[test]
    fn test_local_primitive_lt32() {
        // Forth 2012 §13.6.1.0086 `(LOCAL)` — replica of LT32 from
        // localstest.fth line 118-120 (the test that was silently skipped
        // before `(LOCAL)` was implemented).
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": LOCAL BL WORD COUNT (LOCAL) ; IMMEDIATE")
            .unwrap();
        vm.evaluate(": END-LOCALS 0 0 (LOCAL) ; IMMEDIATE").unwrap();
        vm.evaluate(": LT32 LOCAL A LOCAL B LOCAL C END-LOCALS A B C ;")
            .unwrap();
        vm.evaluate("61 62 63 LT32").unwrap();
        assert_eq!(vm.data_stack(), vec![63, 62, 61]);
    }

    #[test]
    fn test_multiline_colon_then_variable() {
        // Regression: combined `:` def across newlines must leave state at
        // interpret afterwards. Earlier, WAFER's `\` (backslash comment)
        // consumed to `#TIB` instead of the next `\n`, so multi-line chunks
        // lost the closing `;` inside a comment and left state in compile
        // mode. The symptom was a later `VARIABLE X 0 X !` erroring on
        // `unknown word: X`, because the outer `:` never actually closed.
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": EMPTY-STACK\n   DEPTH ?DUP IF DUP 0< IF NEGATE 0 DO 0 LOOP ELSE 0 DO DROP LOOP THEN THEN ;").unwrap();
        vm.evaluate("VARIABLE #ERRORS 0 #ERRORS !").unwrap();
        vm.evaluate("#ERRORS @").unwrap();
        assert_eq!(vm.data_stack(), vec![0]);
    }

    #[test]
    fn test_backslash_stops_at_newline() {
        // Forth 2012 §6.2.2535 `\`: parse-and-discard ends at end-of-line,
        // not end of input buffer. Multi-line input must survive a `\`
        // comment on a prior line.
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("\\ comment line\n42").unwrap();
        assert_eq!(vm.data_stack(), vec![42]);
    }

    #[test]
    fn test_local_primitive_end_sentinel_only() {
        // `0 0 (LOCAL)` with no prior names must be a harmless no-op.
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": END-LOCALS 0 0 (LOCAL) ; IMMEDIATE").unwrap();
        vm.evaluate(": T END-LOCALS 42 ;").unwrap();
        vm.evaluate("T").unwrap();
        assert_eq!(vm.data_stack(), vec![42]);
    }

    // ===================================================================
    // Quotations: [: ... ;]
    // ===================================================================

    #[test]
    fn test_quotation_interpret() {
        assert_eq!(eval_stack("[: 42 ;] EXECUTE"), vec![42]);
    }

    #[test]
    fn test_quotation_compile_mode() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": APPLY EXECUTE ;").unwrap();
        vm.evaluate("[: 1 2 + ;] APPLY .").unwrap();
        assert_eq!(vm.take_output(), "3 ");
    }

    #[test]
    fn test_quotation_inside_colon_def() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": MYDUP [: DUP ;] EXECUTE ;").unwrap();
        vm.evaluate("5 MYDUP").unwrap();
        assert_eq!(vm.data_stack(), vec![5, 5]);
    }

    #[test]
    fn test_quotation_nested() {
        assert_eq!(eval_stack("[: [: 1 ;] EXECUTE ;] EXECUTE"), vec![1]);
    }

    #[test]
    fn test_quotation_inside_if() {
        // Control stack must travel with the saved frame so the outer IF/ELSE
        // still finds its matching THEN after an inner [: ... ;].
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": CHOOSE IF [: 1 ;] ELSE [: 2 ;] THEN EXECUTE ;")
            .unwrap();
        vm.evaluate("-1 CHOOSE  0 CHOOSE").unwrap();
        assert_eq!(vm.data_stack(), vec![2, 1]);
    }

    // ===================================================================
    // Structures (BEGIN-STRUCTURE / +FIELD / FIELD: / CFIELD: / END-STRUCTURE)
    // ===================================================================

    #[test]
    fn test_struct_basic_point() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("BEGIN-STRUCTURE POINT FIELD: P.X FIELD: P.Y END-STRUCTURE")
            .unwrap();
        vm.evaluate("POINT").unwrap();
        assert_eq!(vm.pop_data_stack().unwrap(), 8);

        vm.evaluate("CREATE ORIGIN POINT ALLOT").unwrap();
        vm.evaluate("1 ORIGIN P.X !  2 ORIGIN P.Y !").unwrap();
        vm.evaluate("ORIGIN P.X @  ORIGIN P.Y @").unwrap();
        assert_eq!(vm.data_stack(), vec![2, 1]);
    }

    #[test]
    fn test_struct_field_offsets() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("BEGIN-STRUCTURE REC FIELD: A FIELD: B FIELD: C END-STRUCTURE")
            .unwrap();
        vm.evaluate("REC 0 A 0 B 0 C").unwrap();
        assert_eq!(vm.data_stack(), vec![8, 4, 0, 12]);
    }

    #[test]
    fn test_struct_mixed_cfield() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("BEGIN-STRUCTURE MIX CFIELD: TAG FIELD: VAL END-STRUCTURE")
            .unwrap();
        vm.evaluate("MIX 0 TAG 0 VAL").unwrap();
        assert_eq!(vm.data_stack(), vec![4, 0, 8]);
    }

    // ===================================================================
    // New words: RANDOM / RND-SEED
    // ===================================================================

    #[test]
    fn test_random_deterministic_after_seed() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("42 RND-SEED RANDOM RANDOM RANDOM").unwrap();
        let first = vm.data_stack().clone();

        let mut vm2 = ForthVM::<NativeRuntime>::new().unwrap();
        vm2.evaluate("42 RND-SEED RANDOM RANDOM RANDOM").unwrap();
        let second = vm2.data_stack().clone();

        assert_eq!(first, second, "same seed must produce same sequence");
        assert_eq!(first.len(), 3);
    }

    #[test]
    fn test_random_distinct_values() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("1 RND-SEED").unwrap();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            vm.evaluate("RANDOM").unwrap();
            let v = vm.pop_data_stack().unwrap();
            seen.insert(v);
        }
        // xorshift64's low-32 sequence repeats after a long period; 1000 pulls
        // should hit at least 900 unique cells.
        assert!(
            seen.len() >= 900,
            "only {} distinct out of 1000",
            seen.len()
        );
    }

    #[test]
    fn test_rnd_seed_zero_forced_nonzero() {
        // xorshift with state 0 is a fixed point; seeding with 0 must avoid that.
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("0 RND-SEED RANDOM RANDOM").unwrap();
        let stack = vm.data_stack();
        assert!(
            stack[0] != 0 || stack[1] != 0,
            "seed-0 must not freeze the stream"
        );
    }

    // ===================================================================
    // New words: COUNT
    // ===================================================================

    #[test]
    fn test_count() {
        // Create a counted string: length byte followed by characters
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        let result = vm.evaluate(": TEST -1 ABORT\" oops\" 42 . ; TEST");
        assert!(result.is_err());
    }

    // ===================================================================
    // QUIT — Forth 2012 6.1.2050. Semantics checked against gforth 0.7.3
    // and SwiftForth sf64: the data stack survives, nothing is printed,
    // the rest of the input is abandoned, and CATCH does not see it.
    // ===================================================================

    #[test]
    fn test_quit_keeps_data_stack_and_abandons_the_rest() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("1 2 QUIT 99 .").unwrap(); // not an error, and 99 never runs
        assert_eq!(vm.take_output(), "");
        assert_eq!(vm.data_stack(), vec![2, 1]);
    }

    #[test]
    fn test_quit_from_inside_a_definition() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": T 7 QUIT 8 . ; 5 T 6 .").unwrap();
        assert_eq!(vm.take_output(), "");
        assert_eq!(vm.data_stack(), vec![7, 5]);
    }

    #[test]
    fn test_quit_empties_the_return_stack() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": T 1 >R 2 >R QUIT ; T").unwrap();
        vm.evaluate("RDEPTH .").unwrap();
        assert_eq!(vm.take_output(), "0 ");
    }

    #[test]
    fn test_quit_is_not_caught_by_catch() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("1 2 ' QUIT CATCH .").unwrap();
        assert_eq!(vm.take_output(), "", "CATCH must not report QUIT");
        assert_eq!(vm.data_stack(), vec![2, 1]);
    }

    #[test]
    fn test_quit_leaves_compile_mode_when_it_executes() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": BOOM QUIT ; IMMEDIATE").unwrap();
        vm.evaluate("9 : FOO 1 2 BOOM").unwrap();
        assert!(!vm.is_compiling(), "QUIT enters interpretation state");
        assert_eq!(vm.data_stack(), vec![9], "the data stack is left alone");
        vm.evaluate(": SQ DUP * ; 9 SQ .").unwrap(); // the VM is usable again
        assert_eq!(vm.take_output(), "81 ");
    }

    #[test]
    fn test_quit_inside_evaluate_restores_user_input_source() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("S\" 5 QUIT 6 .\" EVALUATE 7 .").unwrap();
        assert_eq!(vm.take_output(), "", "both the string and the line stop");
        assert_eq!(vm.data_stack(), vec![5]);
        vm.evaluate("SOURCE-ID .").unwrap();
        assert_eq!(vm.take_output(), "0 ", "back to the user input device");
    }

    // ===================================================================
    // ABORT reporting — gforth and sf64 both print nothing for an uncaught
    // ABORT: it is specified as "empty the data stack and perform the
    // function of QUIT", and QUIT displays no message. Only ABORT" prints.
    // ===================================================================

    #[test]
    fn test_abort_is_silent_and_abandons_the_rest() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("1 2 ABORT 99 .").unwrap();
        assert_eq!(vm.take_output(), "");
        assert!(vm.data_stack().is_empty(), "ABORT empties the data stack");
    }

    #[test]
    fn test_abort_is_still_catchable() {
        // Unlike QUIT: CATCH reports -1 and restores the stack depth.
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("1 2 ' ABORT CATCH .").unwrap();
        assert_eq!(vm.take_output(), "-1 ");
        assert_eq!(vm.data_stack(), vec![2, 1]);
    }

    #[test]
    fn test_abort_quote_still_reports_its_text() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        let err = vm.evaluate(": T -1 ABORT\" oops\" ; T").unwrap_err();
        assert_eq!(err.to_string(), "oops");
    }

    #[test]
    fn test_compile_only_words_say_so_in_interpret_mode() {
        for word in ["ABORT\"", "IF", "THEN", "LOOP", "LITERAL", "RECURSE"] {
            let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
            let err = vm.evaluate(word).unwrap_err().to_string();
            assert!(
                err.contains("compile-only word"),
                "{word}: expected the standard condition, got {err}"
            );
        }
    }

    #[test]
    fn test_compile_only_check_does_not_swallow_typos() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        let err = vm.evaluate("NOSUCHWORD").unwrap_err().to_string();
        assert!(err.contains("unknown word"), "{err}");
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("BL WORD HELLO").unwrap();
        let stack = vm.data_stack();
        assert!(!stack.is_empty());
        // The returned address should be a counted string at PAD
        let addr = stack[0] as u32;
        let len = vm.rt.mem_read_u8(addr);
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": GT6 345 ; IMMEDIATE").unwrap();
        vm.evaluate(": GT7 POSTPONE GT6 ;").unwrap();
        vm.evaluate("GT7").unwrap();
        assert_eq!(vm.data_stack(), vec![345]);
    }

    // ===================================================================
    // CS-PICK, CS-ROLL, AHEAD (Programming-Tools)
    // ===================================================================

    #[test]
    fn test_ahead_simple() {
        // : PT1 AHEAD 1111 2222 THEN 3333 ;
        // PT1 -> 3333
        assert_eq!(
            eval_stack(": PT1 AHEAD 1111 2222 THEN 3333 ; PT1"),
            vec![3333]
        );
    }

    #[test]
    fn test_cs_pick_repeat() {
        // ?REPEAT = 0 CS-PICK POSTPONE UNTIL (immediate)
        // 6 PT5 -> 111 111 222 111 222 333 111 222 333
        assert_eq!(
            eval_stack(
                ": ?REPEAT 0 CS-PICK POSTPONE UNTIL ; IMMEDIATE \
                 VARIABLE PT4 \
                 : PT5 PT4 ! BEGIN -1 PT4 +! PT4 @ 4 > 0= ?REPEAT \
                 111 PT4 @ 3 > 0= ?REPEAT 222 PT4 @ 2 > 0= ?REPEAT \
                 333 PT4 @ 1 = UNTIL ; \
                 6 PT5"
            ),
            vec![333, 222, 111, 333, 222, 111, 222, 111, 111]
        );
    }

    #[test]
    fn test_cs_roll_while_equiv() {
        // ?DONE = POSTPONE IF 1 CS-ROLL (same as WHILE)
        assert_eq!(
            eval_stack(
                ": ?DONE POSTPONE IF 1 CS-ROLL ; IMMEDIATE \
                 : PT6 >R BEGIN R@ ?DONE R@ R> 1- >R REPEAT R> DROP ; \
                 5 PT6"
            ),
            vec![1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn test_cs_roll_mix_up() {
        // MIX_UP = 2 CS-ROLL (CS-ROT)
        let setup = ": MIX_UP 2 CS-ROLL ; IMMEDIATE \
                      : PT7 IF 1111 ROT ROT IF 2222 SWAP IF \
                      3333 MIX_UP THEN 4444 THEN 5555 THEN 6666 ;";
        assert_eq!(
            eval_stack(&format!("{setup} -1 -1 -1 PT7")),
            vec![6666, 5555, 4444, 3333, 2222, 1111]
        );
    }

    // ===================================================================
    // WORDS (Programming-Tools)
    // ===================================================================

    #[test]
    fn test_words_lists_defined_words() {
        let output = eval_output("WORDS");
        // Should contain standard primitives
        assert!(output.contains("DUP"));
        assert!(output.contains("DROP"));
        assert!(output.contains("SWAP"));
        assert!(output.contains("+"));
        assert!(output.contains("WORDS"));
    }

    #[test]
    fn test_words_includes_user_defined() {
        let output = eval_output(": MYTEST 42 ; WORDS");
        assert!(output.contains("MYTEST"));
    }

    #[test]
    fn test_words_filter_and_count() {
        let output = eval_output("WORDS FDEPTH");
        assert!(output.contains("FDEPTH"));
        assert!(!output.contains("SWAP"));
        assert!(output.contains(" words\n"));
    }

    #[test]
    fn test_words_all_grouped() {
        let output = eval_output("WORDS ALL");
        assert!(output.contains("-- FORTH ("), "{output}");
        assert!(output.contains("-- internal ("), "{output}");
        // Internals are visible in the ALL view.
        assert!(output.contains("_STACK_FAULT_"), "{output}");
    }

    #[test]
    fn test_words_all_groups_other_wordlists() {
        let output =
            eval_output("WORDLIST SET-CURRENT : INSIDE 1 ; FORTH-WORDLIST SET-CURRENT WORDS ALL");
        assert!(output.contains("-- wid#2 (1 words)"), "{output}");
        let wid2_section = output.split("-- wid#2").nth(1).unwrap();
        assert!(
            wid2_section.lines().nth(1).unwrap().contains("INSIDE"),
            "{output}"
        );
    }

    #[test]
    fn test_words_hides_internal() {
        let output = eval_output("WORDS");
        assert!(!output.contains("_ABORT_Q_"));
        assert!(!output.contains("__CTRL__"));
    }

    #[test]
    fn test_rnd_seed_underflow_is_clean_error() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        let err = vm.evaluate("RND-SEED").unwrap_err();
        assert!(err.to_string().contains("underflow"), "{err}");
        // The stack pointer must not have drifted above the base.
        vm.evaluate("RANDOM .S").unwrap();
        assert!(vm.take_output().starts_with("<1>"), "dsp drifted");
    }

    #[test]
    fn test_host_words_underflow_cleanly() {
        // Every argument-taking host word must fail cleanly on an empty
        // stack and leave both stack pointers at their bases (host words
        // are outside the compiled-code guards).
        let words = [
            "RND-SEED",
            "WORD",
            "SET-ORDER",
            "SET-PRECISION",
            "REPRESENT",
            ">FLOAT",
            "SF!",
            "SF@",
            "DF!",
            "DF@",
            "D>F",
            "F.",
            "FE.",
            "FS.",
            "F~",
            "ROLL",
            ">BODY",
            "ENVIRONMENT?",
            "M*",
            "UM*",
            "UM/MOD",
            "COMPILE,",
            "ACCEPT",
            "ALLOCATE",
            "FREE",
            "RESIZE",
            "N>R",
            "UNESCAPE",
            "REPLACES",
            "SUBSTITUTE",
            "M*/",
            "SEARCH",
            "FALIGNED",
            "SFALIGNED",
            "DFALIGNED",
            "FROT",
            "F>D",
            "2R@",
            // WORD and PARSE are intercepted by the outer interpreter in
            // interpret mode; exercise their host variants compiled.
            ": T_ WORD ; T_",
            ": T_ PARSE ; T_",
            #[cfg(feature = "crypto")]
            "SHA256",
        ];
        for w in words {
            let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
            let r = vm.evaluate(w);
            assert!(r.is_err(), "{w}: silent underflow accepted");
            vm.evaluate("DEPTH FDEPTH + .").unwrap();
            assert_eq!(vm.take_output(), "0 ", "{w}: stack pointer drifted");
        }
    }

    // -- Search order is authoritative (matches gforth + SwiftForth) --

    #[test]
    fn test_search_order_hides_unlisted_wordlists() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(
            "WORDLIST CONSTANT MY-WL MY-WL SET-CURRENT : SECRET 42 ; FORTH-WORDLIST SET-CURRENT",
        )
        .unwrap();
        // MY-WL was never in the search order: SECRET must not resolve.
        let err = vm.evaluate("SECRET").unwrap_err();
        assert!(err.to_string().contains("unknown word"), "{err}");
        // Push MY-WL onto the order: now it resolves.
        vm.evaluate("GET-ORDER MY-WL SWAP 1+ SET-ORDER SECRET .")
            .unwrap();
        assert_eq!(vm.take_output(), "42 ");
        // Back to the default order: hidden again.
        vm.evaluate("-1 SET-ORDER").unwrap();
        assert!(vm.evaluate("SECRET").is_err());
        // FORTH words stay findable throughout.
        vm.evaluate("1 2 + .").unwrap();
        assert_eq!(vm.take_output(), "3 ");
    }

    // -- Error reporting (WS-008) --

    #[test]
    fn test_uncaught_throw_is_typed() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        let err = vm.evaluate("-4 THROW").unwrap_err();
        assert_eq!(err.to_string(), "Stack underflow (throw -4)");
        match err.downcast_ref::<crate::error::WaferError>() {
            Some(crate::error::WaferError::UncaughtThrow { code, .. }) => assert_eq!(*code, -4),
            other => panic!("expected UncaughtThrow, got {other:?}"),
        }
    }

    #[test]
    fn test_uncaught_abort_quote_is_typed() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        let err = vm.evaluate(": F ABORT\" bad input\" ; -1 F").unwrap_err();
        assert_eq!(err.to_string(), "bad input");
        match err.downcast_ref::<crate::error::WaferError>() {
            Some(crate::error::WaferError::UncaughtThrow { code, .. }) => assert_eq!(*code, -2),
            other => panic!("expected UncaughtThrow, got {other:?}"),
        }
    }

    #[test]
    fn test_trap_names_faulting_word() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        // Out-of-bounds fetch traps inside the compiled word; the name
        // section + backtrace naming must identify CRASHER.
        vm.evaluate(": CRASHER 999999999 @ ;").unwrap();
        let err = vm.evaluate("CRASHER").unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("CRASHER"), "{msg}");
        assert!(msg.contains("out of bounds"), "{msg}");
    }

    // -- INCLUDE / INCLUDED --

    /// VM with a virtual source loader over the given (path, text) pairs.
    fn vm_with_files(files: &[(&str, &str)]) -> ForthVM<NativeRuntime> {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        let map: HashMap<String, String> = files
            .iter()
            .map(|(p, t)| (p.to_string(), t.to_string()))
            .collect();
        vm.set_source_loader(Box::new(move |p| {
            map.get(p)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("file not found"))
        }));
        vm
    }

    #[test]
    fn test_include_defines_words_and_resumes_parent_line() {
        let mut vm = vm_with_files(&[("lib.fth", ": DOUBLE 2 * ;\n: TRIPLE 3 * ;\n")]);
        // The 5 after INCLUDE proves the parent input buffer resumes.
        vm.evaluate("INCLUDE lib.fth 5 DOUBLE .").unwrap();
        assert_eq!(vm.take_output(), "10 ");
    }

    #[test]
    fn test_included_string_form() {
        let mut vm = vm_with_files(&[("lib.fth", ": Q 7 ;")]);
        vm.evaluate("S\" lib.fth\" INCLUDED Q .").unwrap();
        assert_eq!(vm.take_output(), "7 ");
    }

    #[test]
    fn test_include_multiline_definition_and_see() {
        let mut vm = vm_with_files(&[("lib.fth", ": TRI\n  DUP DUP ;\n")]);
        vm.evaluate("INCLUDE lib.fth SEE TRI").unwrap();
        assert_eq!(vm.take_output(), ": TRI\n  DUP DUP ;\n");
    }

    #[test]
    fn test_include_nested_relative_path() {
        let mut vm = vm_with_files(&[
            ("dir/a.fth", "INCLUDE b.fth : A B 1 + ;"),
            ("dir/b.fth", ": B 41 ;"),
        ]);
        vm.evaluate("INCLUDE dir/a.fth A .").unwrap();
        assert_eq!(vm.take_output(), "42 ");
    }

    #[test]
    fn test_include_cycle_detected() {
        let mut vm = vm_with_files(&[("a.fth", "INCLUDE b.fth"), ("b.fth", "INCLUDE a.fth")]);
        let err = vm.evaluate("INCLUDE a.fth").unwrap_err();
        assert!(format!("{err:#}").contains("cycle"), "{err:#}");
    }

    #[test]
    fn test_include_depth_bounded() {
        let files: Vec<(String, String)> = (0..20)
            .map(|i| (format!("f{i}.fth"), format!("INCLUDE f{}.fth", i + 1)))
            .collect();
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(p, t)| (p.as_str(), t.as_str()))
            .collect();
        let mut vm = vm_with_files(&refs);
        let err = vm.evaluate("INCLUDE f0.fth").unwrap_err();
        assert!(format!("{err:#}").contains("nesting deeper"), "{err:#}");
    }

    #[test]
    fn test_include_missing_file_and_no_loader() {
        let mut vm = vm_with_files(&[]);
        let err = vm.evaluate("INCLUDE nosuch.fth").unwrap_err();
        assert!(
            format!("{err:#}").contains("INCLUDE: nosuch.fth"),
            "{err:#}"
        );
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        let err = vm.evaluate("INCLUDE x.fth").unwrap_err();
        assert!(err.to_string().contains("no source loader"), "{err}");
    }

    #[test]
    fn test_include_error_carries_file_and_line() {
        let mut vm = vm_with_files(&[("lib.fth", "1 2 +\nNOSUCHWORD\n3 4 +")]);
        let err = vm.evaluate("INCLUDE lib.fth").unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("lib.fth:2"), "{msg}");
        assert!(msg.contains("NOSUCHWORD"), "{msg}");
        // Parent VM stays usable, compile state clean.
        vm.evaluate(": OK 1 ; OK .").unwrap();
        assert_eq!(vm.take_output(), "1 ");
    }

    #[test]
    fn test_include_nested_error_context_chains() {
        let mut vm = vm_with_files(&[("outer.fth", "INCLUDE inner.fth"), ("inner.fth", "BOOM")]);
        let err = vm.evaluate("INCLUDE outer.fth").unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("outer.fth:1"), "{msg}");
        assert!(msg.contains("inner.fth:1"), "{msg}");
    }

    #[test]
    fn test_include_throw_propagates_vm_usable() {
        let mut vm = vm_with_files(&[("t.fth", ": BAD -4 THROW ;\nBAD")]);
        let err = vm.evaluate("INCLUDE t.fth").unwrap_err();
        assert!(format!("{err:#}").contains("Stack underflow"), "{err:#}");
        vm.evaluate("6 7 * .").unwrap();
        assert_eq!(vm.take_output(), "42 ");
    }

    #[test]
    fn test_include_remember_reload_loop() {
        let mut vm = vm_with_files(&[("app.fth", "REMEMBER -WORK\n: APP 1 ;")]);
        vm.evaluate("INCLUDE app.fth").unwrap();
        vm.evaluate("-WORK INCLUDE app.fth APP .").unwrap();
        assert_eq!(vm.take_output(), "1 ");
        // Only one APP in the dictionary after the reload.
        let names = vm.word_names();
        assert_eq!(names.iter().filter(|n| *n == "APP").count(), 1);
    }

    #[test]
    fn test_include_bye_stops_file() {
        let mut vm = vm_with_files(&[("b.fth", "1 .\nBYE\n2 .")]);
        vm.evaluate("INCLUDE b.fth").unwrap();
        assert_eq!(vm.take_output(), "1 ");
        assert!(vm.bye_requested());
    }

    #[test]
    fn test_include_source_id_nested_and_restored() {
        let mut vm = vm_with_files(&[
            ("a.fth", "SOURCE-ID N1 ! INCLUDE b.fth"),
            ("b.fth", "SOURCE-ID N2 !"),
        ]);
        vm.evaluate("VARIABLE N1 VARIABLE N2").unwrap();
        vm.evaluate("INCLUDE a.fth N1 @ . N2 @ . SOURCE-ID .")
            .unwrap();
        assert_eq!(vm.take_output(), "1 2 0 ");
    }

    // -- HELP --

    #[test]
    fn test_help_documented_word() {
        let output = eval_output("HELP DUP");
        assert_eq!(
            output,
            "DUP  ( x -- x x )  Duplicate the top of the data stack.\n"
        );
        // Case-insensitive.
        assert_eq!(eval_output("HELP dup"), output);
    }

    #[test]
    fn test_help_bare_prints_usage() {
        let output = eval_output("HELP");
        assert!(output.contains("HELP <word>"), "{output}");
        assert!(output.contains("SEE <word>"), "{output}");
    }

    #[test]
    fn test_help_user_word_echoes_stack_comment() {
        let output = eval_output(": SQ ( n -- n^2 ) DUP * ; HELP SQ");
        assert!(output.contains("SQ  ( n -- n^2 )"), "{output}");
        assert!(output.contains("SEE SQ"), "{output}");
    }

    #[test]
    fn test_help_undocumented_user_word_hints_see() {
        let output = eval_output(": MYW 1 ; HELP MYW");
        assert_eq!(output, "no help for MYW; try SEE MYW\n");
    }

    #[test]
    fn test_help_immediate_marker() {
        let output = eval_output(": IMH 1 ; IMMEDIATE HELP IMH");
        assert!(output.contains("immediate"), "{output}");
    }

    #[test]
    fn test_help_unknown_word_errors() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        let err = vm.evaluate("HELP NOSUCHWORD").unwrap_err();
        assert!(err.to_string().contains("HELP: unknown word: NOSUCHWORD"));
    }

    #[test]
    fn test_help_covers_every_word_in_fresh_vm() {
        // Total-coverage gate: every visible dictionary word and every
        // outer-interpreter token must have a WORD_DOCS entry, and every
        // entry must resolve back to a real word or token.
        let vm = ForthVM::<NativeRuntime>::new().unwrap();
        let mut missing: Vec<String> = Vec::new();
        let mut names = vm.word_names();
        names.extend(INTERPRETER_TOKENS.iter().map(ToString::to_string));
        for name in &names {
            if crate::wordhelp::lookup(name).is_none() {
                missing.push(name.clone());
            }
        }
        missing.sort();
        missing.dedup();
        assert!(missing.is_empty(), "words without HELP docs: {missing:?}");

        for (name, effect, desc) in crate::wordhelp::WORD_DOCS {
            let known = vm.dictionary.find(&name.to_ascii_uppercase()).is_some()
                || INTERPRETER_TOKENS
                    .iter()
                    .any(|t| t.eq_ignore_ascii_case(name));
            // SHA words vanish without the crypto feature; keep their docs.
            let feature_gated = !cfg!(feature = "crypto") && name.starts_with("SHA");
            assert!(
                known || feature_gated,
                "WORD_DOCS entry for nonexistent word: {name}"
            );
            assert!(!desc.is_empty(), "empty description for {name}");
            let e = *effect;
            assert!(
                e.starts_with('(') && e.ends_with(')'),
                "malformed stack effect for {name}: {e:?}"
            );
        }
    }

    // -- SEE --

    #[test]
    fn test_see_colon_word_verbatim() {
        let output = eval_output(": SQ DUP * ; SEE SQ");
        assert_eq!(output, ": SQ DUP * ;\n");
    }

    #[test]
    fn test_see_multiline_definition() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": TRI").unwrap();
        vm.evaluate("  DUP DUP ;").unwrap();
        vm.evaluate("SEE TRI").unwrap();
        assert_eq!(vm.take_output(), ": TRI\n  DUP DUP ;\n");
    }

    #[test]
    fn test_see_comment_survives() {
        let output = eval_output(": C ( n -- n ) 1+ ; SEE C");
        assert!(output.contains("( n -- n )"), "{output}");
    }

    #[test]
    fn test_see_data_words() {
        assert_eq!(eval_output("42 CONSTANT A SEE A"), "42 CONSTANT A\n");
        assert_eq!(eval_output("VARIABLE V SEE V"), "VARIABLE V\n");
        assert_eq!(eval_output("CREATE CR8 SEE CR8"), "CREATE CR8\n");
        assert_eq!(eval_output("16 BUFFER: B SEE B"), "16 BUFFER: B\n");
        assert_eq!(eval_output("1 2 2CONSTANT D2 SEE D2"), "1 2 2CONSTANT D2\n");
        assert_eq!(
            eval_output("SYNONYM NEWDUP DUP SEE NEWDUP"),
            "SYNONYM NEWDUP DUP\n"
        );
    }

    #[test]
    fn test_see_value_shows_current() {
        assert_eq!(eval_output("5 VALUE X SEE X"), "5 VALUE X\n");
        assert_eq!(eval_output("5 VALUE X 9 TO X SEE X"), "9 VALUE X\n");
    }

    #[test]
    fn test_see_defer_shows_target() {
        let output = eval_output("DEFER D ' DUP IS D SEE D");
        assert_eq!(output, "DEFER D  ( IS DUP )\n");
    }

    #[test]
    fn test_see_boot_word_shows_source() {
        // WITHIN is defined in boot.fth as a colon word; SEE must show
        // real source (with its HELP header line), not an IR dump.
        let output = eval_output("SEE WITHIN");
        assert!(output.starts_with("\\ WITHIN  ("), "{output}");
        assert!(
            output.ends_with(": WITHIN  OVER - >R - R> U< ;\n"),
            "{output}"
        );
    }

    #[test]
    fn test_see_primitive_ir_fallback() {
        let output = eval_output("SEE DUP");
        assert!(output.contains("DUP is a primitive; IR:"), "{output}");
        assert!(output.contains("dup"), "{output}");
    }

    #[test]
    fn test_see_host_word_and_interpreter_token() {
        let output = eval_output("SEE WORDS");
        assert!(output.contains("WORDS is a built-in host word"), "{output}");
        // `:` has no dictionary entry — outer-interpreter stub.
        let output = eval_output("SEE :");
        assert!(
            output.contains(": is handled by the outer interpreter"),
            "{output}"
        );
    }

    #[test]
    fn test_see_immediate_flag() {
        let output = eval_output(": I2 ; IMMEDIATE SEE I2");
        assert!(output.contains(": I2 ;\nimmediate"), "{output}");
    }

    #[test]
    fn test_see_unknown_word_errors() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        let err = vm.evaluate("SEE NOSUCHWORD").unwrap_err();
        assert!(err.to_string().contains("SEE: unknown word: NOSUCHWORD"));
    }

    #[test]
    fn test_see_error_path_no_capture_debris() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        // Force an unknown-word error mid-definition, then define fresh.
        assert!(vm.evaluate(": BAD NOSUCHWORD ;").is_err());
        vm.evaluate(": GOOD 1 ; SEE GOOD").unwrap();
        assert_eq!(vm.take_output(), ": GOOD 1 ;\n");
    }

    #[test]
    fn test_see_marker_roundtrip_restores_source() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": W 1 ; MARKER MK : W 2 ;").unwrap();
        vm.evaluate("SEE W").unwrap();
        assert_eq!(vm.take_output(), ": W 2 ;\n");
        vm.evaluate("MK SEE W").unwrap();
        assert_eq!(vm.take_output(), ": W 1 ;\n");
    }

    #[test]
    fn test_see_redefinition_shows_newest() {
        let output = eval_output(": R 1 ; : R 2 ; SEE R");
        assert_eq!(output, ": R 2 ;\n");
    }

    // -- SEE-IR --

    #[test]
    fn test_see_ir_colon_word() {
        let output = eval_output(": SQ DUP * ; SEE-IR SQ");
        assert!(output.contains("\\ SQ -- 2 ops (optimized IR)"), "{output}");
        assert!(output.contains("dup"));
        assert!(output.contains("mul"));
    }

    #[test]
    fn test_see_ir_shows_inlined_body() {
        let output = eval_output(": SQ DUP * ; : FOO SQ SQ ; SEE-IR FOO");
        // Inlining threshold covers SQ: FOO's stored IR has both muls inlined.
        assert_eq!(output.matches("mul").count(), 2, "{output}");
        assert!(!output.contains("call"), "{output}");
    }

    #[test]
    fn test_see_ir_resolves_callee_names() {
        // A body over the inlining threshold keeps its calls.
        let output = eval_output(
            ": BIG DUP DUP DUP DUP DUP DUP DUP DUP DUP * * * * * * * * * ; \
             : USER BIG BIG ; SEE-IR USER",
        );
        assert!(
            output.contains("call BIG") || output.contains("tail-call BIG"),
            "{output}"
        );
    }

    #[test]
    fn test_see_ir_primitive_and_host_word() {
        let output = eval_output("SEE-IR DUP");
        assert!(output.contains("(optimized IR)"), "{output}");
        assert!(output.contains("dup"));
        let output = eval_output("SEE-IR WORDS");
        assert!(output.contains("WORDS is a built-in host word"), "{output}");
    }

    #[test]
    fn test_see_ir_control_flow_indented() {
        let output = eval_output(": T IF 1 ELSE 2 THEN ; SEE-IR T");
        assert!(
            output.contains("if\n  push 1\nelse\n  push 2\nthen\n"),
            "{output}"
        );
    }

    #[test]
    fn test_see_ir_immediate_flag() {
        let output = eval_output(": IMM 1 ; IMMEDIATE SEE-IR IMM");
        assert!(output.contains("immediate"), "{output}");
    }

    #[test]
    fn test_see_ir_interpreter_token() {
        let output = eval_output("SEE-IR :");
        assert!(
            output.contains(": is handled directly by the outer interpreter"),
            "{output}"
        );
    }

    #[test]
    fn test_see_ir_errors() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        let err = vm.evaluate("SEE-IR NOSUCHWORD").unwrap_err();
        assert!(err.to_string().contains("SEE-IR: unknown word: NOSUCHWORD"));
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        let err = vm.evaluate("SEE-IR").unwrap_err();
        assert!(err.to_string().contains("SEE-IR: expected word name"));
    }

    #[test]
    fn test_dot_s_honors_base() {
        assert_eq!(eval_output("HEX FF .S"), "<1> FF ");
        assert_eq!(eval_output("HEX -A .S"), "<1> -A ");
        assert_eq!(eval_output("5 2 BASE ! .S"), "<1> 101 ");
    }

    #[test]
    fn test_question_fetches_and_prints() {
        assert_eq!(eval_output("VARIABLE QV 17 QV ! QV ?"), "17 ");
    }

    #[test]
    fn test_dump_smoke() {
        let output = eval_output("VARIABLE DV 65 DV C! DV 4 DUMP");
        assert!(output.contains("41"));
        assert!(output.contains("|A"));
    }

    #[test]
    fn test_f_dot_s() {
        let output = eval_output("1.5E0 2.5E0 F.S");
        assert!(output.starts_with("F:<2> 1.5 2.5"));
    }

    #[test]
    fn test_spaces_negative_prints_nothing() {
        assert_eq!(eval_output("-1 SPACES"), "");
        // width smaller than the number: no padding at all
        assert_eq!(eval_output("123 0 .R"), "123");
    }

    #[test]
    fn test_rdepth_empty() {
        assert_eq!(eval_stack("RDEPTH"), vec![0]);
    }

    #[test]
    fn test_rdepth_counts_r_items() {
        assert_eq!(
            eval_stack(": TRD 10 >R 20 >R RDEPTH 2R> 2DROP ; TRD"),
            vec![2]
        );
    }

    #[test]
    fn test_dot_rs_empty() {
        assert_eq!(eval_output(".RS"), "R:<0> ");
    }

    #[test]
    fn test_dot_rs_bottom_to_top() {
        assert_eq!(
            eval_output(": TRS 10 >R 20 >R .RS 2R> 2DROP ; TRS"),
            "R:<2> 10 20 "
        );
    }

    #[test]
    fn test_dot_rs_honors_base() {
        assert_eq!(
            eval_output("HEX : TRS16 FF >R .RS R> DROP ; TRS16"),
            "R:<1> FF "
        );
    }

    #[test]
    fn test_rdepth_sees_loop_params() {
        // Inside DO/LOOP the return stack holds the loop parameters,
        // so RDEPTH must be > 0 there and 0 again after the loop.
        // eval_stack returns top-first: [after-loop RDEPTH, inside-loop RDEPTH]
        let stack = eval_stack(": TRL 0 3 1 DO DROP RDEPTH LOOP RDEPTH ; TRL");
        assert_eq!(stack.len(), 2);
        assert!(stack[1] > 0, "inside loop: expected non-empty return stack");
        assert_eq!(stack[0], 0, "after loop: return stack must be empty");
    }

    #[test]
    fn test_bye_sets_flag_and_stops_line() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("1 BYE 2").unwrap();
        assert!(vm.bye_requested());
        // BYE stops the rest of the line: only the 1 was pushed
        assert_eq!(vm.data_stack(), vec![1]);
    }

    #[test]
    fn test_stack_guard_underflow_interpreted() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        let e = vm.evaluate("DROP").unwrap_err();
        assert_eq!(e.to_string(), "Stack underflow (throw -4)");
        // Stack machinery intact afterwards
        vm.evaluate("7 .").unwrap();
        assert_eq!(vm.take_output(), "7 ");
    }

    #[test]
    fn test_stack_guard_underflow_in_compiled_word() {
        // Straight-line word (promoted path) and mid-word underflow
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": D2 DROP DROP ;").unwrap();
        let e = vm.evaluate("1 D2").unwrap_err();
        assert_eq!(e.to_string(), "Stack underflow (throw -4)");
        // The guard fires BEFORE corruption: the 1 must still be there
        assert_eq!(vm.data_stack(), vec![1]);
        // Non-promoted (contains a call): same guard. Fresh VM — the
        // failed D2 call above deliberately left its stack untouched.
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": D3 DUP . DROP DROP ;").unwrap();
        let e = vm.evaluate("2 D3").unwrap_err();
        assert_eq!(e.to_string(), "Stack underflow (throw -4)");
    }

    #[test]
    fn test_stack_guard_overflow_compiled_loop() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": FLOOD BEGIN 1 AGAIN ;").unwrap();
        let e = vm.evaluate("FLOOD").unwrap_err();
        assert_eq!(e.to_string(), "Stack overflow (throw -3)");
    }

    #[test]
    fn test_stack_guard_float_underflow() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": FD FDROP ;").unwrap();
        let e = vm.evaluate("FD").unwrap_err();
        assert_eq!(e.to_string(), "Floating-point stack underflow (throw -45)");
    }

    #[test]
    fn test_stack_guard_return_stack_underflow() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": RU R> DROP ;").unwrap();
        // R> with nothing user-pushed underflows the return stack
        let e = vm.evaluate("RU").unwrap_err();
        assert_eq!(e.to_string(), "Return stack underflow (throw -6)");
    }

    #[test]
    fn test_stack_guard_catchable() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": BAD DROP ;").unwrap();
        vm.evaluate("' BAD CATCH").unwrap();
        assert_eq!(vm.data_stack(), vec![-4]);
    }

    #[test]
    fn test_stack_guards_off_config() {
        let mut cfg = WaferConfig::all();
        cfg.codegen.stack_guards = false;
        let mut vm = ForthVM::<NativeRuntime>::new_with_config(cfg).unwrap();
        // Compiled DROP underflows silently (documented unguarded mode)
        vm.evaluate(": D1 DROP ;").unwrap();
        assert!(vm.evaluate("D1").is_ok());
    }

    #[test]
    fn test_uncaught_throw_has_message() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        let e = vm.evaluate("-4 THROW").unwrap_err();
        assert_eq!(e.to_string(), "Stack underflow (throw -4)");
        let e = vm.evaluate("-77 THROW").unwrap_err();
        assert_eq!(e.to_string(), "Catch = -77");
    }

    #[test]
    fn test_uncaught_abort_quote_shows_text() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        let e = vm
            .evaluate(": TST -1 ABORT\" all is lost\" ; TST")
            .unwrap_err();
        assert_eq!(e.to_string(), "all is lost");
        // Caught -2 must NOT print the text, and the payload must not leak
        vm.evaluate(": TC ['] TST CATCH ; TC").unwrap();
        assert_eq!(vm.take_output(), "");
        assert_eq!(vm.data_stack(), vec![-2]);
    }

    #[test]
    fn test_order_names_forth() {
        let output = eval_output("ORDER");
        assert!(output.contains("FORTH"));
        assert!(!output.contains('['));
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("10 VALUE FOO").unwrap();
        vm.evaluate(": SETFOO TO FOO ;").unwrap();
        vm.evaluate("20 SETFOO FOO .").unwrap();
        assert_eq!(vm.take_output(), "20 ");
    }

    #[test]
    fn test_defer_basic() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("DEFER MY-DEFER").unwrap();
        vm.evaluate("' DUP IS MY-DEFER").unwrap();
        vm.evaluate("5 MY-DEFER .S").unwrap();
        assert_eq!(vm.take_output(), "<2> 5 5 ");
    }

    #[test]
    fn test_defer_action_of() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("CHAR ) PARSE hello)").unwrap();
        let stack = vm.data_stack();
        assert_eq!(stack.len(), 2);
        assert_eq!(stack[0], 5); // length of "hello"
    }

    #[test]
    fn test_parse_name() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("PARSE-NAME hello").unwrap();
        let stack = vm.data_stack();
        assert_eq!(stack.len(), 2);
        assert_eq!(stack[0], 5); // length of "hello"
    }

    #[test]
    fn test_buffer_colon() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("MARKER MARK1").unwrap();
        // MARK1 should exist and be callable
        vm.evaluate("MARK1").unwrap();
        // ...and remove itself when executed
        assert!(vm.evaluate("MARK1").is_err());
    }

    #[test]
    fn test_remember_survives_and_reruns() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("REMEMBER RST : AA 1 ;").unwrap();
        vm.evaluate("RST").unwrap();
        assert!(vm.evaluate("AA").is_err(), "AA must be pruned");
        // RST survived its own execution and is re-runnable
        vm.evaluate(": BB 2 ; RST").unwrap();
        assert!(vm.evaluate("BB").is_err(), "BB must be pruned");
        // VM stays fully usable afterwards
        vm.evaluate(": CC 3 ; CC").unwrap();
        assert_eq!(vm.data_stack(), vec![3]);
    }

    #[test]
    fn test_remember_restores_wordlist_allocation() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate("REMEMBER RST WORDLIST DROP RST WORDLIST")
            .unwrap();
        // wid allocation rolled back: the post-RST WORDLIST reuses wid 2
        assert_eq!(vm.data_stack(), vec![2]);
    }

    #[test]
    fn test_empty_returns_to_boot() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": ZZ 9 ;").unwrap();
        vm.evaluate("EMPTY").unwrap();
        assert!(vm.evaluate("ZZ").is_err(), "ZZ must be pruned");
        vm.evaluate("1 2 + .").unwrap();
        assert_eq!(vm.take_output(), "3 ");
    }

    #[test]
    fn test_gild_rebaselines_empty() {
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.evaluate(": KEEP1 1 ; GILD : GONE1 2 ;").unwrap();
        vm.evaluate("EMPTY").unwrap();
        assert!(vm.evaluate("GONE1").is_err(), "GONE1 must be pruned");
        vm.evaluate("KEEP1 .").unwrap();
        assert_eq!(vm.take_output(), "1 ");
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
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
    fn punctuation_anywhere_converts_as_double() {
        // The punctuation is a double-cell marker, not a fractional point:
        // every form below carries the same digits, so the value is the same.
        // eval_stack reports top-first, so a double reads as [hi, lo].
        for token in ["1234.", "123.4", "12.34", "1.234", ".1234"] {
            assert_eq!(eval_stack(token), vec![0, 1234], "token {token}");
        }
        // SwiftForth accepts comma, colon, slash, plus and dash too, which is
        // what makes dates and times convert without a custom parser.
        assert_eq!(eval_stack("1,234"), vec![0, 1234]);
        assert_eq!(eval_stack("12:30:45"), vec![0, 123045]);
        assert_eq!(eval_stack("2026-08-06"), vec![0, 20260806]);
        assert_eq!(eval_stack("12/34"), vec![0, 1234]);
        assert_eq!(eval_stack("1+234"), vec![0, 1234]);
    }

    #[test]
    fn dpl_counts_digits_right_of_last_punctuation() {
        for (token, dpl) in [("1234.", 0), ("123.4", 1), ("12.34", 2), (".1234", 4)] {
            assert_eq!(eval_stack(&format!("{token} 2DROP DPL @")), vec![dpl]);
        }
        // Only the rightmost punctuation counts.
        assert_eq!(eval_stack("12:30:45 2DROP DPL @"), vec![2]);
    }

    #[test]
    fn dpl_stays_negative_for_unpunctuated_numbers() {
        // A leading minus is a sign, not punctuation, so these stay single-cell.
        for token in ["1234", "-1", "$FF"] {
            let dpl = eval_stack(&format!("{token} DROP DPL @"))[0];
            assert!(dpl < 0, "token {token} left DPL = {dpl}");
        }
        assert_eq!(eval_stack("-1"), vec![-1]);
        // DPL counts up from its seed once per digit.
        assert_eq!(eval_stack("1234 DROP DPL @"), vec![DPL_INIT + 4]);
        assert_eq!(eval_stack("-1 DROP DPL @"), vec![DPL_INIT + 1]);
    }

    #[test]
    fn leading_plus_is_punctuation_not_a_sign() {
        // sf64 converts `+7` as the double 7 with DPL = 1: unlike `-`, a
        // leading `+` does not bind to the number.
        assert_eq!(eval_stack("+7"), vec![0, 7]);
        assert_eq!(eval_stack("+7 2DROP DPL @"), vec![1]);
        // Same after a base prefix.
        assert_eq!(eval_stack("#+7"), vec![0, 7]);
        assert_eq!(eval_stack("$+F"), vec![0, 15]);
    }

    #[test]
    fn repeated_punctuation_only_counts_from_the_last_one() {
        // sf64: `12..34` is 1234 with DPL 2, `1-2-3` is 123 with DPL 1.
        assert_eq!(eval_stack("12..34"), vec![0, 1234]);
        assert_eq!(eval_stack("12..34 2DROP DPL @"), vec![2]);
        assert_eq!(eval_stack("1-2-3"), vec![0, 123]);
        assert_eq!(eval_stack("1-2-3 2DROP DPL @"), vec![1]);
    }

    #[test]
    fn nh_recovers_an_out_of_range_single_number() {
        // 4000000000 overflows a signed cell, so the stack value is truncated.
        assert_eq!(eval_stack("4000000000"), vec![-294967296]);
        // NH carries the high cell, making the true value recoverable.
        assert_eq!(eval_stack("4000000000 NH @"), vec![0, -294967296]);
        assert_eq!(eval_output("4000000000 NH @ D."), "4000000000 ");
    }

    #[test]
    fn float_literals_still_win_over_double_punctuation() {
        // `1.5E0` has an embedded dot, but "15E0" is not a decimal number,
        // so conversion falls through to the float parser.
        assert_eq!(eval_output("1.5E0 F."), "1.500000 ");
        assert_eq!(eval_output("-3.25E0 F."), "-3.250000 ");
        assert_eq!(eval_output("1E-3 F."), "0.001000 ");
    }

    #[test]
    fn double_punctuation_respects_base_prefixes() {
        assert_eq!(eval_stack("$FF."), vec![0, 255]);
        assert_eq!(eval_stack("$F.F"), vec![0, 255]);
        assert_eq!(eval_stack("%1010."), vec![0, 10]);
        assert_eq!(eval_stack("#12.34"), vec![0, 1234]);
        assert_eq!(eval_stack("-$FF."), vec![-1, -255]);
    }

    #[test]
    fn sign_after_a_base_prefix_is_a_sign_not_punctuation() {
        // Forth 2012 spells signed based numbers with the sign after the
        // prefix. The dash precedes the leftmost digit, so it must not
        // trigger double-cell conversion.
        assert_eq!(eval_stack("#-1289"), vec![-1289]);
        assert_eq!(eval_stack("$-12eF"), vec![-4847]);
        assert_eq!(eval_stack("%-10010110"), vec![-150]);
        // The sign may also precede the prefix, and both spellings cancel.
        assert_eq!(eval_stack("-$FF"), vec![-255]);
        assert_eq!(eval_stack("-$-FF"), vec![255]);
    }

    #[test]
    fn punctuated_numbers_compile_into_definitions() {
        assert_eq!(eval_stack(": D1 12.34 ; D1"), vec![0, 1234]);
        assert_eq!(eval_output(": STAMP 2026-08-06 D. ; STAMP"), "20260806 ");
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
