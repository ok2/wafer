//! Linear memory layout and stack operations for WAFER.
//!
//! WAFER uses WASM linear memory for the dictionary, return stack,
//! and as a fallback for the data and float stacks when types are unknown.
//! When type inference succeeds, values stay in WASM locals/operand stack instead.

/// Size of one memory page in WASM (64 KiB).
pub const PAGE_SIZE: u32 = 65536;

/// Initial number of memory pages.
pub const INITIAL_PAGES: u32 = 16; // 1 MiB

/// Maximum number of memory pages.
pub const MAX_PAGES: u32 = 256; // 16 MiB

// Memory region layout
// All offsets are byte addresses in linear memory.

/// System variables region (STATE, BASE, >IN, HLD, etc.)
pub const SYSVAR_BASE: u32 = 0x0000;
/// Size of system variables region.
pub const SYSVAR_SIZE: u32 = 64;

/// Input buffer for source parsing.
pub const INPUT_BUFFER_BASE: u32 = SYSVAR_BASE + SYSVAR_SIZE; // 0x0040
/// Size of input buffer.
pub const INPUT_BUFFER_SIZE: u32 = 1024;

/// PAD - scratch area for string formatting.
pub const PAD_BASE: u32 = INPUT_BUFFER_BASE + INPUT_BUFFER_SIZE; // 0x0440
/// Size of PAD.
pub const PAD_SIZE: u32 = 256;

/// Pictured numeric output buffer (<# ... #>). Grows downward from top.
pub const PICT_BUF_BASE: u32 = PAD_BASE + PAD_SIZE; // 0x0540
/// Size of pictured output buffer (2*`BITS_PER_CELL` + 2 = 66 min, 128 for margin).
pub const PICT_BUF_SIZE: u32 = 128;
/// Top of pictured output buffer (HLD starts here, grows down).
pub const PICT_BUF_TOP: u32 = PICT_BUF_BASE + PICT_BUF_SIZE;

/// WORD buffer — transient counted-string area for WORD output.
pub const WORD_BUF_BASE: u32 = PICT_BUF_TOP; // 0x05C0
/// Size of WORD buffer (max name 31 + 1 length + padding).
pub const WORD_BUF_SIZE: u32 = 64;

/// Data stack region (fallback when types are unknown).
/// Grows downward from the top of this region.
pub const DATA_STACK_BASE: u32 = WORD_BUF_BASE + WORD_BUF_SIZE; // 0x0600
/// Size of data stack region.
pub const DATA_STACK_SIZE: u32 = 4096; // 1024 cells

/// Return stack region. Grows downward.
pub const RETURN_STACK_BASE: u32 = DATA_STACK_BASE + DATA_STACK_SIZE; // 0x1600
/// Size of return stack region.
pub const RETURN_STACK_SIZE: u32 = 4096;

/// Floating-point stack region (fallback). Grows downward.
pub const FLOAT_STACK_BASE: u32 = RETURN_STACK_BASE + RETURN_STACK_SIZE; // 0x2600
/// Size of float stack region.
pub const FLOAT_STACK_SIZE: u32 = 2048; // 256 doubles

/// Hash scratch region — output buffer for `SHA1`/`SHA256`/`SHA512` and
/// other hash host words. Sized for the largest supported digest (SHA512 = 64 B).
pub const HASH_SCRATCH_BASE: u32 = FLOAT_STACK_BASE + FLOAT_STACK_SIZE; // 0x2E00
/// Size of hash scratch region.
pub const HASH_SCRATCH_SIZE: u32 = 128;

/// Dictionary region start. Grows upward.
pub const DICTIONARY_BASE: u32 = HASH_SCRATCH_BASE + HASH_SCRATCH_SIZE; // 0x2E80

/// Initial top of data stack (grows down from here).
pub const DATA_STACK_TOP: u32 = DATA_STACK_BASE + DATA_STACK_SIZE;

/// Initial top of return stack (grows down from here).
pub const RETURN_STACK_TOP: u32 = RETURN_STACK_BASE + RETURN_STACK_SIZE;

/// Initial top of float stack (grows down from here).
pub const FLOAT_STACK_TOP: u32 = FLOAT_STACK_BASE + FLOAT_STACK_SIZE;

/// Size of one cell (4 bytes for i32).
pub const CELL_SIZE: u32 = 4;

/// Size of one double-cell (8 bytes).
pub const DOUBLE_CELL_SIZE: u32 = 8;

/// Size of one float (8 bytes for f64).
pub const FLOAT_SIZE: u32 = 8;

// System variable offsets within SYSVAR region

/// STATE: 0 = interpreting, -1 (0xFFFFFFFF) = compiling.
pub const SYSVAR_STATE: u32 = SYSVAR_BASE;
/// BASE: current number base (default 10).
pub const SYSVAR_BASE_VAR: u32 = SYSVAR_BASE + 4;
/// >IN: offset into the input buffer.
pub const SYSVAR_TO_IN: u32 = SYSVAR_BASE + 8;
/// HERE: next free dictionary address.
pub const SYSVAR_HERE: u32 = SYSVAR_BASE + 12;
/// LATEST: pointer to the most recent dictionary entry.
pub const SYSVAR_LATEST: u32 = SYSVAR_BASE + 16;
/// SOURCE-ID: current input source (0 = user input, -1 = string).
pub const SYSVAR_SOURCE_ID: u32 = SYSVAR_BASE + 20;
/// #TIB: length of current input.
pub const SYSVAR_NUM_TIB: u32 = SYSVAR_BASE + 24;
/// HLD: pointer for pictured numeric output.
pub const SYSVAR_HLD: u32 = SYSVAR_BASE + 28;
/// LEAVE flag: nonzero when LEAVE has been called inside a DO loop.
pub const SYSVAR_LEAVE_FLAG: u32 = SYSVAR_BASE + 32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_regions_dont_overlap() {
        // Each region should start after the previous one ends
        const { assert!(INPUT_BUFFER_BASE >= SYSVAR_BASE + SYSVAR_SIZE) };
        const { assert!(PAD_BASE >= INPUT_BUFFER_BASE + INPUT_BUFFER_SIZE) };
        const { assert!(DATA_STACK_BASE >= PAD_BASE + PAD_SIZE) };
        const { assert!(RETURN_STACK_BASE >= DATA_STACK_BASE + DATA_STACK_SIZE) };
        const { assert!(FLOAT_STACK_BASE >= RETURN_STACK_BASE + RETURN_STACK_SIZE) };
        const { assert!(HASH_SCRATCH_BASE >= FLOAT_STACK_BASE + FLOAT_STACK_SIZE) };
        const { assert!(DICTIONARY_BASE >= HASH_SCRATCH_BASE + HASH_SCRATCH_SIZE) };
    }

    #[test]
    fn dictionary_starts_within_first_page() {
        const { assert!(DICTIONARY_BASE < PAGE_SIZE) };
    }

    #[test]
    fn stack_tops_are_correct() {
        assert_eq!(DATA_STACK_TOP, DATA_STACK_BASE + DATA_STACK_SIZE);
        assert_eq!(RETURN_STACK_TOP, RETURN_STACK_BASE + RETURN_STACK_SIZE);
        assert_eq!(FLOAT_STACK_TOP, FLOAT_STACK_BASE + FLOAT_STACK_SIZE);
    }

    #[test]
    fn sysvar_offsets_are_within_region() {
        let all_offsets = [
            SYSVAR_STATE,
            SYSVAR_BASE_VAR,
            SYSVAR_TO_IN,
            SYSVAR_HERE,
            SYSVAR_LATEST,
            SYSVAR_SOURCE_ID,
            SYSVAR_NUM_TIB,
            SYSVAR_HLD,
            SYSVAR_LEAVE_FLAG,
        ];
        for offset in all_offsets {
            assert!(offset + CELL_SIZE <= SYSVAR_BASE + SYSVAR_SIZE);
        }
    }
}
