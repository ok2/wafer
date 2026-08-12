//! Static per-word documentation for `HELP` (and the `SEE`/`SEE-IR` header
//! line): name, stack effect, one-line description.
//!
//! Coverage is **total by construction**: a unit test in `outer.rs` walks a
//! freshly booted VM and asserts every visible dictionary word and every
//! outer-interpreter token has an entry here, and that every entry resolves
//! back. Adding a word to WAFER without documenting it fails the build's
//! test gate.
//!
//! Stack effects follow Forth 2012 notation; `F:` marks the float stack,
//! `R:` the return stack, quoted names (`"name"`) are parsed from the input.

/// (NAME, stack effect, one-line description)
pub const WORD_DOCS: &[(&str, &str, &str)] = &[
    // -- Core: stack manipulation --
    (
        "DUP",
        "( x -- x x )",
        "Duplicate the top of the data stack.",
    ),
    ("DROP", "( x -- )", "Discard the top of the data stack."),
    (
        "SWAP",
        "( x1 x2 -- x2 x1 )",
        "Exchange the top two stack items.",
    ),
    (
        "OVER",
        "( x1 x2 -- x1 x2 x1 )",
        "Copy the second item to the top.",
    ),
    (
        "ROT",
        "( x1 x2 x3 -- x2 x3 x1 )",
        "Rotate the third item to the top.",
    ),
    (
        "-ROT",
        "( x1 x2 x3 -- x3 x1 x2 )",
        "Rotate the top item to third place.",
    ),
    ("NIP", "( x1 x2 -- x2 )", "Discard the second stack item."),
    (
        "TUCK",
        "( x1 x2 -- x2 x1 x2 )",
        "Copy the top item below the second.",
    ),
    (
        "?DUP",
        "( x -- 0 | x x )",
        "Duplicate the top item only if it is nonzero.",
    ),
    (
        "PICK",
        "( xu..x0 u -- xu..x0 xu )",
        "Copy the u-th stack item to the top.",
    ),
    (
        "ROLL",
        "( xu..x0 u -- xu-1..x0 xu )",
        "Rotate the u-th stack item to the top.",
    ),
    ("DEPTH", "( -- n )", "Number of cells on the data stack."),
    (
        "2DUP",
        "( x1 x2 -- x1 x2 x1 x2 )",
        "Duplicate the top cell pair.",
    ),
    ("2DROP", "( x1 x2 -- )", "Discard the top cell pair."),
    (
        "2SWAP",
        "( x1 x2 x3 x4 -- x3 x4 x1 x2 )",
        "Exchange the top two cell pairs.",
    ),
    (
        "2OVER",
        "( x1 x2 x3 x4 -- x1 x2 x3 x4 x1 x2 )",
        "Copy the second cell pair to the top.",
    ),
    (
        "2ROT",
        "( x1 x2 x3 x4 x5 x6 -- x3 x4 x5 x6 x1 x2 )",
        "Rotate the third cell pair to the top.",
    ),
    // -- Core: return stack --
    (
        ">R",
        "( x -- ) ( R: -- x )",
        "Move the top item to the return stack.",
    ),
    (
        "R>",
        "( -- x ) ( R: x -- )",
        "Move the top return-stack item back.",
    ),
    (
        "R@",
        "( -- x ) ( R: x -- x )",
        "Copy the top return-stack item.",
    ),
    (
        "2>R",
        "( x1 x2 -- ) ( R: -- x1 x2 )",
        "Move the top cell pair to the return stack.",
    ),
    (
        "2R>",
        "( -- x1 x2 ) ( R: x1 x2 -- )",
        "Move a cell pair back from the return stack.",
    ),
    (
        "2R@",
        "( -- x1 x2 ) ( R: x1 x2 -- x1 x2 )",
        "Copy the top return-stack cell pair.",
    ),
    (
        "N>R",
        "( i*n +n -- ) ( R: -- j*x +n )",
        "Move +n items to the return stack.",
    ),
    (
        "NR>",
        "( -- i*x +n ) ( R: j*x +n -- )",
        "Move back items stored by N>R.",
    ),
    ("SP@", "( -- addr )", "Current data-stack pointer."),
    ("RP@", "( -- addr )", "Current return-stack pointer."),
    ("RDEPTH", "( -- n )", "Number of cells on the return stack."),
    // -- Core: arithmetic --
    ("+", "( n1 n2 -- n3 )", "Add: n3 = n1 + n2."),
    ("-", "( n1 n2 -- n3 )", "Subtract: n3 = n1 - n2."),
    ("*", "( n1 n2 -- n3 )", "Multiply: n3 = n1 * n2."),
    ("/", "( n1 n2 -- n3 )", "Divide: n3 = n1 / n2."),
    ("MOD", "( n1 n2 -- n3 )", "Remainder of n1 / n2."),
    (
        "/MOD",
        "( n1 n2 -- rem quot )",
        "Remainder and quotient of n1 / n2.",
    ),
    (
        "*/",
        "( n1 n2 n3 -- n4 )",
        "n1 * n2 / n3 with a double-cell intermediate.",
    ),
    (
        "*/MOD",
        "( n1 n2 n3 -- rem quot )",
        "n1 * n2 / n3, remainder and quotient.",
    ),
    ("1+", "( n -- n+1 )", "Add one."),
    ("1-", "( n -- n-1 )", "Subtract one."),
    (
        "2*",
        "( n -- n*2 )",
        "Shift left one bit (multiply by two).",
    ),
    (
        "2/",
        "( n -- n/2 )",
        "Arithmetic shift right one bit (divide by two).",
    ),
    ("NEGATE", "( n -- -n )", "Two's-complement negation."),
    ("ABS", "( n -- |n| )", "Absolute value."),
    ("MIN", "( n1 n2 -- n3 )", "Smaller of n1 and n2."),
    ("MAX", "( n1 n2 -- n3 )", "Larger of n1 and n2."),
    (
        "M*",
        "( n1 n2 -- d )",
        "Signed multiply to a double-cell product.",
    ),
    (
        "M+",
        "( d1 n -- d2 )",
        "Add a single-cell number to a double.",
    ),
    (
        "M*/",
        "( d1 n1 +n2 -- d2 )",
        "d1 * n1 / n2 with a triple-cell intermediate.",
    ),
    (
        "UM*",
        "( u1 u2 -- ud )",
        "Unsigned multiply to a double-cell product.",
    ),
    (
        "UM/MOD",
        "( ud u1 -- rem quot )",
        "Unsigned double divided by single.",
    ),
    (
        "SM/REM",
        "( d n1 -- rem quot )",
        "Symmetric signed double / single division.",
    ),
    (
        "FM/MOD",
        "( d n1 -- rem quot )",
        "Floored signed double / single division.",
    ),
    // -- Core: comparison --
    ("=", "( x1 x2 -- flag )", "True if equal."),
    ("<>", "( x1 x2 -- flag )", "True if not equal."),
    ("<", "( n1 n2 -- flag )", "True if n1 < n2 (signed)."),
    (">", "( n1 n2 -- flag )", "True if n1 > n2 (signed)."),
    ("<=", "( n1 n2 -- flag )", "True if n1 <= n2 (signed)."),
    (">=", "( n1 n2 -- flag )", "True if n1 >= n2 (signed)."),
    ("U<", "( u1 u2 -- flag )", "True if u1 < u2 (unsigned)."),
    (
        "WITHIN",
        "( x lo hi -- flag )",
        "True if lo <= x < hi (circular compare).",
    ),
    ("U>", "( u1 u2 -- flag )", "True if u1 > u2 (unsigned)."),
    ("0=", "( x -- flag )", "True if zero."),
    ("0<", "( n -- flag )", "True if negative."),
    ("0>", "( n -- flag )", "True if positive."),
    ("0<>", "( x -- flag )", "True if nonzero."),
    // -- Core: logic --
    ("AND", "( x1 x2 -- x3 )", "Bitwise AND."),
    ("OR", "( x1 x2 -- x3 )", "Bitwise OR."),
    ("XOR", "( x1 x2 -- x3 )", "Bitwise exclusive OR."),
    ("INVERT", "( x -- ~x )", "Bitwise complement."),
    ("LSHIFT", "( x u -- x' )", "Shift left by u bits."),
    ("RSHIFT", "( x u -- x' )", "Logical shift right by u bits."),
    ("TRUE", "( -- -1 )", "All-bits-set true flag."),
    ("FALSE", "( -- 0 )", "Zero false flag."),
    // -- Core: memory --
    ("@", "( addr -- x )", "Fetch the cell at addr."),
    ("!", "( x addr -- )", "Store x into the cell at addr."),
    ("C@", "( addr -- char )", "Fetch the byte at addr."),
    ("C!", "( char addr -- )", "Store a byte at addr."),
    ("+!", "( n addr -- )", "Add n to the cell at addr."),
    ("2@", "( addr -- x1 x2 )", "Fetch the cell pair at addr."),
    ("2!", "( x1 x2 addr -- )", "Store a cell pair at addr."),
    ("HERE", "( -- addr )", "Next free data-space address."),
    ("ALLOT", "( n -- )", "Reserve n bytes of data space."),
    (",", "( x -- )", "Compile a cell into data space."),
    ("C,", "( char -- )", "Compile a byte into data space."),
    (
        "ALIGN",
        "( -- )",
        "Align the data-space pointer to a cell boundary.",
    ),
    (
        "ALIGNED",
        "( addr -- a-addr )",
        "Round addr up to a cell boundary.",
    ),
    ("CELLS", "( n1 -- n2 )", "Size in bytes of n1 cells."),
    ("CELL+", "( addr1 -- addr2 )", "Advance addr by one cell."),
    ("CHARS", "( n1 -- n2 )", "Size in bytes of n1 characters."),
    (
        "CHAR+",
        "( addr1 -- addr2 )",
        "Advance addr by one character.",
    ),
    (
        "MOVE",
        "( addr1 addr2 u -- )",
        "Copy u bytes, overlap-safe.",
    ),
    ("CMOVE", "( addr1 addr2 u -- )", "Copy u bytes low-to-high."),
    (
        "CMOVE>",
        "( addr1 addr2 u -- )",
        "Copy u bytes high-to-low.",
    ),
    ("FILL", "( addr u char -- )", "Fill u bytes with char."),
    ("ERASE", "( addr u -- )", "Fill u bytes with zero."),
    ("BLANK", "( addr u -- )", "Fill u bytes with spaces."),
    ("PAD", "( -- addr )", "Scratch buffer address."),
    ("UNUSED", "( -- u )", "Bytes of data space remaining."),
    (
        "ALLOCATE",
        "( u -- addr ior )",
        "Allocate u bytes from the heap.",
    ),
    ("FREE", "( addr -- ior )", "Release an ALLOCATEd region."),
    (
        "RESIZE",
        "( addr1 u -- addr2 ior )",
        "Resize an ALLOCATEd region.",
    ),
    // -- Core: I/O and numeric output --
    (
        ".",
        "( n -- )",
        "Print n in the current BASE, followed by a space.",
    ),
    ("U.", "( u -- )", "Print u as an unsigned number."),
    (
        ".R",
        "( n u -- )",
        "Print n right-aligned in a u-wide field.",
    ),
    (
        "U.R",
        "( u1 u2 -- )",
        "Print u1 unsigned, right-aligned in u2 columns.",
    ),
    ("D.", "( d -- )", "Print a double-cell number."),
    (
        "D.R",
        "( d u -- )",
        "Print a double right-aligned in u columns.",
    ),
    ("EMIT", "( char -- )", "Output one character."),
    (
        "TYPE",
        "( c-addr u -- )",
        "Output u characters from c-addr.",
    ),
    ("CR", "( -- )", "Output a newline."),
    ("SPACE", "( -- )", "Output one space."),
    ("SPACES", "( n -- )", "Output n spaces."),
    ("PAGE", "( -- )", "Output a form feed (clear screen)."),
    ("BL", "( -- 32 )", "The space character code."),
    (
        "COUNT",
        "( c-addr1 -- c-addr2 u )",
        "Unpack a counted string.",
    ),
    ("<#", "( -- )", "Begin pictured numeric output."),
    (
        "#",
        "( ud1 -- ud2 )",
        "Convert one digit into the pictured buffer.",
    ),
    ("#S", "( ud -- 0 0 )", "Convert all remaining digits."),
    (
        "#>",
        "( ud -- c-addr u )",
        "End pictured output; yield the string.",
    ),
    (
        "HOLD",
        "( char -- )",
        "Insert char into the pictured buffer.",
    ),
    (
        "HOLDS",
        "( c-addr u -- )",
        "Insert a string into the pictured buffer.",
    ),
    ("SIGN", "( n -- )", "Insert a minus sign if n is negative."),
    (
        ">NUMBER",
        "( ud1 c-addr1 u1 -- ud2 c-addr2 u2 )",
        "Convert digits, accumulating into ud.",
    ),
    ("BASE", "( -- addr )", "Variable holding the number base."),
    (
        "DPL",
        "( -- addr )",
        "Variable: digits right of the last punctuation; negative if none.",
    ),
    (
        "NH",
        "( -- addr )",
        "Variable: high cell dropped by the last single-cell conversion.",
    ),
    ("HEX", "( -- )", "Set BASE to sixteen."),
    ("DECIMAL", "( -- )", "Set BASE to ten."),
    // -- Core: strings --
    (
        "S\"",
        "( \"ccc<quote>\" -- c-addr u )",
        "String literal (interpret or compile).",
    ),
    (
        "S\\\"",
        "( \"ccc<quote>\" -- c-addr u )",
        "String literal with escape sequences.",
    ),
    (
        "C\"",
        "( \"ccc<quote>\" -- c-addr )",
        "Counted-string literal.",
    ),
    (
        "S",
        "( \"name\" -- c-addr u )",
        "WAFER: next token as a string literal.",
    ),
    (".\"", "( \"ccc<quote>\" -- )", "Print a string literal."),
    (
        ".(",
        "( \"ccc<paren>\" -- )",
        "Print immediately while parsing.",
    ),
    (
        "COMPARE",
        "( c-addr1 u1 c-addr2 u2 -- n )",
        "Lexicographic string comparison.",
    ),
    (
        "SEARCH",
        "( c-addr1 u1 c-addr2 u2 -- c-addr3 u3 flag )",
        "Find a substring.",
    ),
    (
        "-TRAILING",
        "( c-addr u1 -- c-addr u2 )",
        "Drop trailing spaces from a string.",
    ),
    (
        "/STRING",
        "( c-addr1 u1 n -- c-addr2 u2 )",
        "Advance a string by n characters.",
    ),
    (
        "UNESCAPE",
        "( c-addr1 u1 c-addr2 -- c-addr2 u2 )",
        "Double each % for SUBSTITUTE.",
    ),
    (
        "SUBSTITUTE",
        "( c1 u1 c2 u2 -- c2 u3 n )",
        "Expand %name% substitutions.",
    ),
    (
        "REPLACES",
        "( c1 u1 c2 u2 -- )",
        "Define a SUBSTITUTE replacement.",
    ),
    // -- Core: definitions and execution --
    (":", "( \"name\" -- )", "Begin a colon definition."),
    (";", "( -- )", "End a colon definition."),
    (
        ":NONAME",
        "( -- xt )",
        "Begin an anonymous definition; leaves its xt.",
    ),
    ("[:", "( -- )", "Begin a quotation (nestable anonymous xt)."),
    (";]", "( -- xt )", "End a quotation; yields its xt."),
    ("[", "( -- )", "Switch to interpret state."),
    ("]", "( -- )", "Switch to compile state."),
    (
        "{:",
        "( arg*i \"locals :}\" -- )",
        "Declare named locals for this definition.",
    ),
    (
        "(LOCAL)",
        "( c-addr u -- )",
        "Declare one local (Forth 2012 batch protocol).",
    ),
    (
        "CREATE",
        "( \"name\" -- )",
        "Define a word that pushes its data address.",
    ),
    ("VARIABLE", "( \"name\" -- )", "Define a one-cell variable."),
    (
        "2VARIABLE",
        "( \"name\" -- )",
        "Define a two-cell variable.",
    ),
    ("FVARIABLE", "( \"name\" -- )", "Define a float variable."),
    (
        "CONSTANT",
        "( x \"name\" -- )",
        "Define a constant pushing x.",
    ),
    (
        "2CONSTANT",
        "( x1 x2 \"name\" -- )",
        "Define a double-cell constant.",
    ),
    (
        "FCONSTANT",
        "( F: r -- ) ( \"name\" -- )",
        "Define a float constant.",
    ),
    (
        "VALUE",
        "( x \"name\" -- )",
        "Define a value; read by name, set with TO.",
    ),
    (
        "2VALUE",
        "( x1 x2 \"name\" -- )",
        "Define a double-cell value.",
    ),
    (
        "FVALUE",
        "( F: r -- ) ( \"name\" -- )",
        "Define a float value.",
    ),
    ("TO", "( x \"name\" -- )", "Store into a VALUE or local."),
    (
        "BUFFER:",
        "( u \"name\" -- )",
        "Define a word pushing a u-byte buffer.",
    ),
    (
        "DEFER",
        "( \"name\" -- )",
        "Define a deferred word (vector).",
    ),
    (
        "DEFER@",
        "( xt1 -- xt2 )",
        "Fetch a deferred word's action.",
    ),
    (
        "DEFER!",
        "( xt2 xt1 -- )",
        "Store a deferred word's action.",
    ),
    ("IS", "( xt \"name\" -- )", "Set a deferred word's action."),
    (
        "ACTION-OF",
        "( \"name\" -- xt )",
        "Fetch a deferred word's action by name.",
    ),
    (
        "DOES>",
        "( -- )",
        "Give the latest CREATEd word a runtime action.",
    ),
    (
        "IMMEDIATE",
        "( -- )",
        "Mark the latest definition immediate.",
    ),
    (
        "POSTPONE",
        "( \"name\" -- )",
        "Compile the compilation semantics of name.",
    ),
    (
        "LITERAL",
        "( x -- )",
        "Compile x as a literal (compile-only).",
    ),
    ("2LITERAL", "( x1 x2 -- )", "Compile a double-cell literal."),
    ("FLITERAL", "( F: r -- )", "Compile a float literal."),
    ("SLITERAL", "( c-addr u -- )", "Compile a string literal."),
    ("COMPILE,", "( xt -- )", "Compile a call to xt."),
    (
        "[']",
        "( \"name\" -- )",
        "Compile the xt of name as a literal.",
    ),
    (
        "'",
        "( \"name\" -- xt )",
        "Find name; push its execution token.",
    ),
    (
        "CHAR",
        "( \"name\" -- char )",
        "First character of the next word.",
    ),
    (
        "[CHAR]",
        "( \"name\" -- )",
        "Compile the first character as a literal.",
    ),
    (
        "EXECUTE",
        "( xt -- )",
        "Execute the word with execution token xt.",
    ),
    (
        "EXIT",
        "( -- )",
        "Return from the current word (compile-only).",
    ),
    (
        "RECURSE",
        "( -- )",
        "Call the word currently being defined.",
    ),
    (
        "SYNONYM",
        "( \"new\" \"old\" -- )",
        "Define new as an alias of old.",
    ),
    (
        "FIND",
        "( c-addr -- c-addr 0 | xt 1 | xt -1 )",
        "Look up a counted-string name.",
    ),
    (
        ">BODY",
        "( xt -- a-addr )",
        "Data-field address of a CREATEd word.",
    ),
    // -- Core: control flow --
    (
        "IF",
        "( flag -- )",
        "Execute the following if flag is nonzero.",
    ),
    ("ELSE", "( -- )", "Alternative branch of IF."),
    ("THEN", "( -- )", "End of IF."),
    ("BEGIN", "( -- )", "Start an indefinite loop."),
    (
        "UNTIL",
        "( flag -- )",
        "Loop back to BEGIN while flag is zero.",
    ),
    ("AGAIN", "( -- )", "Loop back to BEGIN forever."),
    (
        "WHILE",
        "( flag -- )",
        "Continue the loop while flag is nonzero.",
    ),
    ("REPEAT", "( -- )", "Loop back to BEGIN (after WHILE)."),
    ("DO", "( limit start -- )", "Start a counted loop."),
    (
        "?DO",
        "( limit start -- )",
        "Counted loop; skip entirely if limit = start.",
    ),
    (
        "LOOP",
        "( -- )",
        "Increment the index; repeat until it meets the limit.",
    ),
    (
        "+LOOP",
        "( n -- )",
        "Add n to the index; repeat conditionally.",
    ),
    ("I", "( -- n )", "Innermost loop index."),
    ("J", "( -- n )", "Next-outer loop index."),
    ("LEAVE", "( -- )", "Exit the current counted loop."),
    ("UNLOOP", "( -- )", "Discard loop parameters before EXIT."),
    (
        "AHEAD",
        "( -- )",
        "Unconditional forward branch (resolved by THEN).",
    ),
    ("CASE", "( -- )", "Start a CASE structure."),
    ("OF", "( x1 x2 -- | x1 )", "Match one CASE selector."),
    ("ENDOF", "( -- )", "End one OF clause."),
    ("ENDCASE", "( x -- )", "End the CASE structure."),
    (
        "CS-PICK",
        "( u -- )",
        "Copy a control-flow stack entry (compilation).",
    ),
    (
        "CS-ROLL",
        "( u -- )",
        "Rotate control-flow stack entries (compilation).",
    ),
    // -- Core: interpreter, input, exceptions --
    (
        "(",
        "( \"ccc<paren>\" -- )",
        "Comment until the closing paren.",
    ),
    ("\\", "( \"ccc<eol>\" -- )", "Comment until end of line."),
    (
        "[IF]",
        "( flag -- )",
        "Conditional compilation: take branch if nonzero.",
    ),
    ("[ELSE]", "( -- )", "Conditional-compilation alternative."),
    ("[THEN]", "( -- )", "End conditional compilation."),
    (
        "[DEFINED]",
        "( \"name\" -- flag )",
        "True if name is defined.",
    ),
    (
        "[UNDEFINED]",
        "( \"name\" -- flag )",
        "True if name is not defined.",
    ),
    ("STATE", "( -- addr )", "Variable: nonzero while compiling."),
    ("SOURCE", "( -- c-addr u )", "Current input buffer."),
    (
        "SOURCE-ID",
        "( -- n )",
        "Input source: 0 terminal, -1 string.",
    ),
    (
        ">IN",
        "( -- addr )",
        "Variable: offset into the input buffer.",
    ),
    (
        "WORD",
        "( char \"ccc<char>\" -- c-addr )",
        "Parse delimited by char; counted string.",
    ),
    (
        "PARSE",
        "( char \"ccc<char>\" -- c-addr u )",
        "Parse delimited by char.",
    ),
    (
        "PARSE-NAME",
        "( \"name\" -- c-addr u )",
        "Parse a whitespace-delimited name.",
    ),
    (
        "EVALUATE",
        "( c-addr u -- )",
        "Interpret the string as Forth input.",
    ),
    (
        "REFILL",
        "( -- flag )",
        "Refill the input buffer (false when piped).",
    ),
    (
        "ACCEPT",
        "( c-addr +n1 -- +n2 )",
        "Read a line of input (unsupported here).",
    ),
    ("ABORT", "( i*x -- )", "Empty the stacks and abort."),
    (
        "QUIT",
        "( -- ) ( R: i*x -- )",
        "Empty the return stack, return to the interpreter; data stack kept.",
    ),
    (
        "ABORT\"",
        "( flag -- )",
        "If flag is nonzero, abort with a message.",
    ),
    (
        "CATCH",
        "( xt -- 0 | code )",
        "Execute xt, catching any THROW.",
    ),
    (
        "THROW",
        "( code -- )",
        "Raise exception code (0 is a no-op).",
    ),
    (
        "ENVIRONMENT?",
        "( c-addr u -- false | val true )",
        "Query an environment property.",
    ),
    ("BYE", "( -- )", "Leave the REPL / end the session."),
    ("S>D", "( n -- d )", "Sign-extend a single to a double."),
    ("D>S", "( d -- n )", "Narrow a double to a single."),
    // -- Double-cell words --
    ("D+", "( d1 d2 -- d3 )", "Double-cell add."),
    ("D-", "( d1 d2 -- d3 )", "Double-cell subtract."),
    ("DNEGATE", "( d -- -d )", "Double-cell negate."),
    ("DABS", "( d -- |d| )", "Double-cell absolute value."),
    ("D0=", "( d -- flag )", "True if the double is zero."),
    ("D0<", "( d -- flag )", "True if the double is negative."),
    ("D=", "( d1 d2 -- flag )", "True if doubles are equal."),
    ("D<", "( d1 d2 -- flag )", "True if d1 < d2 (signed)."),
    (
        "DU<",
        "( ud1 ud2 -- flag )",
        "True if ud1 < ud2 (unsigned).",
    ),
    ("D2*", "( d1 -- d2 )", "Double-cell shift left."),
    ("D2/", "( d1 -- d2 )", "Double-cell arithmetic shift right."),
    ("DMIN", "( d1 d2 -- d3 )", "Smaller double."),
    ("DMAX", "( d1 d2 -- d3 )", "Larger double."),
    // -- Float words --
    ("F+", "( F: r1 r2 -- r3 )", "Float add."),
    ("F-", "( F: r1 r2 -- r3 )", "Float subtract."),
    ("F*", "( F: r1 r2 -- r3 )", "Float multiply."),
    ("F/", "( F: r1 r2 -- r3 )", "Float divide."),
    ("F**", "( F: r1 r2 -- r3 )", "Raise r1 to the power r2."),
    ("FNEGATE", "( F: r -- -r )", "Float negate."),
    ("FABS", "( F: r -- |r| )", "Float absolute value."),
    ("FSQRT", "( F: r -- r' )", "Float square root."),
    ("FMIN", "( F: r1 r2 -- r3 )", "Smaller float."),
    ("FMAX", "( F: r1 r2 -- r3 )", "Larger float."),
    ("FLOOR", "( F: r -- r' )", "Round toward negative infinity."),
    ("FROUND", "( F: r -- r' )", "Round to nearest."),
    ("FDUP", "( F: r -- r r )", "Duplicate the float top."),
    ("FDROP", "( F: r -- )", "Discard the float top."),
    (
        "FSWAP",
        "( F: r1 r2 -- r2 r1 )",
        "Exchange the top two floats.",
    ),
    (
        "FOVER",
        "( F: r1 r2 -- r1 r2 r1 )",
        "Copy the second float to the top.",
    ),
    (
        "FROT",
        "( F: r1 r2 r3 -- r2 r3 r1 )",
        "Rotate the third float to the top.",
    ),
    ("FNIP", "( F: r1 r2 -- r2 )", "Discard the second float."),
    (
        "FTUCK",
        "( F: r1 r2 -- r2 r1 r2 )",
        "Copy the float top below the second.",
    ),
    ("FDEPTH", "( -- n )", "Number of floats on the float stack."),
    (
        "F=",
        "( F: r1 r2 -- ) ( -- flag )",
        "True if floats are equal.",
    ),
    ("F<", "( F: r1 r2 -- ) ( -- flag )", "True if r1 < r2."),
    (
        "F0=",
        "( F: r -- ) ( -- flag )",
        "True if the float is zero.",
    ),
    (
        "F0<",
        "( F: r -- ) ( -- flag )",
        "True if the float is negative.",
    ),
    (
        "F~",
        "( F: r1 r2 r3 -- ) ( -- flag )",
        "Approximate float equality test.",
    ),
    ("F@", "( addr -- ) ( F: -- r )", "Fetch a float from addr."),
    ("F!", "( addr -- ) ( F: r -- )", "Store a float at addr."),
    ("SF@", "( addr -- ) ( F: -- r )", "Fetch a 32-bit float."),
    ("SF!", "( addr -- ) ( F: r -- )", "Store a 32-bit float."),
    ("DF@", "( addr -- ) ( F: -- r )", "Fetch a 64-bit float."),
    ("DF!", "( addr -- ) ( F: r -- )", "Store a 64-bit float."),
    ("S>F", "( n -- ) ( F: -- r )", "Convert single to float."),
    (
        "F>S",
        "( F: r -- ) ( -- n )",
        "Convert float to single (truncate).",
    ),
    ("D>F", "( d -- ) ( F: -- r )", "Convert double to float."),
    (
        "F>D",
        "( F: r -- ) ( -- d )",
        "Convert float to double (truncate).",
    ),
    ("FLOATS", "( n1 -- n2 )", "Size in bytes of n1 floats."),
    ("FLOAT+", "( addr1 -- addr2 )", "Advance addr by one float."),
    (
        "SFLOATS",
        "( n1 -- n2 )",
        "Size in bytes of n1 32-bit floats.",
    ),
    (
        "SFLOAT+",
        "( addr1 -- addr2 )",
        "Advance addr by one 32-bit float.",
    ),
    (
        "DFLOATS",
        "( n1 -- n2 )",
        "Size in bytes of n1 64-bit floats.",
    ),
    (
        "DFLOAT+",
        "( addr1 -- addr2 )",
        "Advance addr by one 64-bit float.",
    ),
    ("FALIGN", "( -- )", "Align data space for a float."),
    (
        "FALIGNED",
        "( addr -- f-addr )",
        "Round addr up to float alignment.",
    ),
    ("SFALIGN", "( -- )", "Align data space for a 32-bit float."),
    (
        "SFALIGNED",
        "( addr -- sf-addr )",
        "Round addr up to 32-bit float alignment.",
    ),
    ("DFALIGN", "( -- )", "Align data space for a 64-bit float."),
    (
        "DFALIGNED",
        "( addr -- df-addr )",
        "Round addr up to 64-bit float alignment.",
    ),
    (
        "F.",
        "( F: r -- )",
        "Print a float in fixed-point notation.",
    ),
    (
        "FE.",
        "( F: r -- )",
        "Print a float in engineering notation.",
    ),
    (
        "FS.",
        "( F: r -- )",
        "Print a float in scientific notation.",
    ),
    ("PRECISION", "( -- u )", "Digits used by float output."),
    ("SET-PRECISION", "( u -- )", "Set float output digits."),
    (
        "REPRESENT",
        "( c-addr u -- n flag1 flag2 ) ( F: r -- )",
        "Convert a float to digit text.",
    ),
    (
        ">FLOAT",
        "( c-addr u -- flag ) ( F: -- r | )",
        "Parse a string as a float.",
    ),
    ("FSIN", "( F: r -- r' )", "Sine (radians)."),
    ("FCOS", "( F: r -- r' )", "Cosine (radians)."),
    ("FTAN", "( F: r -- r' )", "Tangent (radians)."),
    ("FASIN", "( F: r -- r' )", "Arc sine."),
    ("FACOS", "( F: r -- r' )", "Arc cosine."),
    ("FATAN", "( F: r -- r' )", "Arc tangent."),
    (
        "FATAN2",
        "( F: ry rx -- r )",
        "Arc tangent of ry/rx, quadrant-correct.",
    ),
    (
        "FSINCOS",
        "( F: r -- rsin rcos )",
        "Sine and cosine together.",
    ),
    ("FSINH", "( F: r -- r' )", "Hyperbolic sine."),
    ("FCOSH", "( F: r -- r' )", "Hyperbolic cosine."),
    ("FTANH", "( F: r -- r' )", "Hyperbolic tangent."),
    ("FASINH", "( F: r -- r' )", "Inverse hyperbolic sine."),
    ("FACOSH", "( F: r -- r' )", "Inverse hyperbolic cosine."),
    ("FATANH", "( F: r -- r' )", "Inverse hyperbolic tangent."),
    ("FEXP", "( F: r -- r' )", "e to the power r."),
    ("FEXPM1", "( F: r -- r' )", "e**r - 1, accurate near zero."),
    ("FLN", "( F: r -- r' )", "Natural logarithm."),
    ("FLNP1", "( F: r -- r' )", "ln(1+r), accurate near zero."),
    ("FLOG", "( F: r -- r' )", "Base-10 logarithm."),
    ("FALOG", "( F: r -- r' )", "10 to the power r."),
    // -- Structures --
    (
        "BEGIN-STRUCTURE",
        "( \"name\" -- addr 0 )",
        "Start a structure definition.",
    ),
    (
        "END-STRUCTURE",
        "( addr +n -- )",
        "Finish a structure definition.",
    ),
    (
        "+FIELD",
        "( offset size \"name\" -- offset' )",
        "Define a field of the given size.",
    ),
    (
        "FIELD:",
        "( offset \"name\" -- offset' )",
        "Define an aligned cell field.",
    ),
    (
        "CFIELD:",
        "( offset \"name\" -- offset' )",
        "Define a one-byte field.",
    ),
    (
        "FFIELD:",
        "( offset \"name\" -- offset' )",
        "Define an aligned float field.",
    ),
    (
        "SFFIELD:",
        "( offset \"name\" -- offset' )",
        "Define a 32-bit float field.",
    ),
    (
        "DFFIELD:",
        "( offset \"name\" -- offset' )",
        "Define a 64-bit float field.",
    ),
    // -- Search order --
    (
        "FORTH-WORDLIST",
        "( -- wid )",
        "The main wordlist identifier.",
    ),
    ("WORDLIST", "( -- wid )", "Create a new wordlist."),
    (
        "GET-CURRENT",
        "( -- wid )",
        "Wordlist receiving new definitions.",
    ),
    ("SET-CURRENT", "( wid -- )", "Set the compilation wordlist."),
    ("GET-ORDER", "( -- widn..wid1 n )", "Current search order."),
    (
        "SET-ORDER",
        "( widn..wid1 n -- )",
        "Set the search order (-1 = default).",
    ),
    (
        "SEARCH-WORDLIST",
        "( c-addr u wid -- 0 | xt 1 | xt -1 )",
        "Look up a name in one wordlist.",
    ),
    (
        "DEFINITIONS",
        "( -- )",
        "New definitions go to the top wordlist.",
    ),
    ("ALSO", "( -- )", "Duplicate the top of the search order."),
    (
        ">ORDER",
        "( wid -- )",
        "Push wid on top of the search order (gforth extension).",
    ),
    (
        "-ORDER",
        "( wid -- )",
        "Remove wid from the search order (VFX extension).",
    ),
    (
        "VOCABULARY",
        "( \"name\" -- )",
        "Create a named wordlist; executing name replaces the top of the search order.",
    ),
    ("ONLY", "( -- )", "Reset the search order to the minimum."),
    ("PREVIOUS", "( -- )", "Drop the top of the search order."),
    (
        "FORTH",
        "( -- )",
        "Replace the search-order top with FORTH-WORDLIST.",
    ),
    (
        "ORDER",
        "( -- )",
        "Print the search order and compilation wordlist.",
    ),
    // -- Programming tools --
    (
        "WORDS",
        "( \"filter\"? -- )",
        "List words; optional substring filter; ALL = grouped view.",
    ),
    (
        "SEE",
        "( \"name\" -- )",
        "Show a word's source (or best fallback).",
    ),
    (
        "SEE-IR",
        "( \"name\" -- )",
        "Show a word's post-optimization IR.",
    ),
    (
        "HELP",
        "( \"name\"? -- )",
        "Show stack effect and description for a word.",
    ),
    (".S", "( -- )", "Print the data stack, respecting BASE."),
    ("F.S", "( -- )", "Print the float stack."),
    (".RS", "( -- )", "Print the return stack, respecting BASE."),
    ("?", "( addr -- )", "Fetch and print the cell at addr."),
    (
        "DUMP",
        "( addr u -- )",
        "Hex + ASCII dump of u bytes at addr.",
    ),
    (
        "MARKER",
        "( \"name\" -- )",
        "Word that rolls the dictionary back to before itself.",
    ),
    (
        "REMEMBER",
        "( \"name\" -- )",
        "Re-runnable marker: rolls back to just after itself.",
    ),
    (
        "EMPTY",
        "( -- )",
        "Roll back to the boot (or GILDed) state.",
    ),
    (
        "GILD",
        "( -- )",
        "Make the current state the EMPTY baseline.",
    ),
    (
        "INCLUDED",
        "( c-addr u -- )",
        "Interpret the named source file (nestable).",
    ),
    (
        "INCLUDE",
        "( \"name\" -- )",
        "Interpret the source file named in the input.",
    ),
    // -- WAFER-specific --
    (
        "CONSOLIDATE",
        "( -- )",
        "Recompile all IR words into one optimized module.",
    ),
    (
        "SHA1",
        "( c-addr u -- c-addr2 20 )",
        "SHA-1 digest into the hash scratch area.",
    ),
    (
        "SHA256",
        "( c-addr u -- c-addr2 32 )",
        "SHA-256 digest into the hash scratch area.",
    ),
    (
        "SHA512",
        "( c-addr u -- c-addr2 64 )",
        "SHA-512 digest into the hash scratch area.",
    ),
    (
        "RANDOM",
        "( -- u )",
        "32-bit pseudo-random number (xorshift64).",
    ),
    (
        "RND-SEED",
        "( u -- )",
        "Reseed the PRNG (0 forced nonzero).",
    ),
    (
        "UTIME",
        "( -- d )",
        "Microseconds since the Unix epoch, as a double.",
    ),
];

/// Extract a leading stack-effect comment from source text, e.g.
/// `: SQ ( n -- n^2 ) DUP * ;` yields `( n -- n^2 )`. Returns the first
/// parenthesized comment that contains `--`.
pub fn stack_comment(source: &str) -> Option<String> {
    let open = source.find('(')?;
    let close = open + source[open..].find(')')?;
    let inner = &source[open..=close];
    inner.contains("--").then(|| inner.to_string())
}

/// Case-insensitive lookup: returns (stack effect, description).
pub fn lookup(name: &str) -> Option<(&'static str, &'static str)> {
    WORD_DOCS
        .iter()
        .find(|(n, _, _)| n.eq_ignore_ascii_case(name))
        .map(|(_, effect, desc)| (*effect, *desc))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_is_case_insensitive() {
        assert_eq!(lookup("dup"), lookup("DUP"));
        assert!(lookup("DUP").is_some());
        assert!(lookup("NO-SUCH-WORD-EVER").is_none());
    }

    #[test]
    fn no_duplicate_names() {
        let mut seen = std::collections::HashSet::new();
        for (name, _, _) in WORD_DOCS {
            assert!(
                seen.insert(name.to_ascii_uppercase()),
                "duplicate WORD_DOCS entry: {name}"
            );
        }
    }
}
