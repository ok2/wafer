//! Forth dictionary: word headers, lookup, and creation.
//!
//! The dictionary is a linked list in linear memory. Each entry contains:
//! - Link to previous entry (4 bytes)
//! - Flags + name length (1 byte)
//! - Name string (N bytes, padded to cell alignment)
//! - Code field: function table index (4 bytes)
//! - Parameter field: data for CREATEd words, DOES> action, etc.

use crate::error::{WaferError, WaferResult};
use crate::memory::{DICTIONARY_BASE, INITIAL_PAGES, PAGE_SIZE};

/// Flags stored in the dictionary entry header.
pub mod flags {
    /// Word executes during compilation.
    pub const IMMEDIATE: u8 = 0x80;
    /// Word is hidden (being compiled, not yet findable).
    pub const HIDDEN: u8 = 0x40;
    /// Mask for the name length (lower 5 bits).
    pub const LENGTH_MASK: u8 = 0x1F;
    /// Maximum word name length.
    pub const MAX_NAME_LEN: usize = 31;
}

/// Unique identifier for a word in the dictionary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WordId(pub u32);

/// The dictionary manages word entries in a simulated linear memory buffer.
pub struct Dictionary {
    /// The memory buffer (simulates WASM linear memory).
    memory: Vec<u8>,
    /// Address of the most recently defined word (LATEST).
    latest: u32,
    /// Next free address in the dictionary (HERE).
    here: u32,
    /// Next available function table index.
    next_fn_index: u32,
}

/// Align an address upward to a 4-byte boundary.
fn align4(addr: u32) -> u32 {
    (addr + 3) & !3
}

impl Dictionary {
    /// Create a new dictionary with the given memory buffer.
    /// `here` is initialized to `DICTIONARY_BASE` from memory.rs.
    pub fn new() -> Self {
        let size = (INITIAL_PAGES * PAGE_SIZE) as usize;
        Self {
            memory: vec![0u8; size],
            latest: 0,
            here: DICTIONARY_BASE,
            next_fn_index: 0,
        }
    }

    /// Create a new dictionary entry (like Forth's CREATE).
    /// Returns the WordId (function table index) assigned to this word.
    /// The word starts HIDDEN (will be revealed when compilation completes).
    pub fn create(&mut self, name: &str, immediate: bool) -> WaferResult<WordId> {
        let name_upper = name.to_ascii_uppercase();
        let name_bytes = name_upper.as_bytes();
        let name_len = name_bytes.len();

        if name_len == 0 || name_len > flags::MAX_NAME_LEN {
            return Err(WaferError::NameTooLong(name.to_string()));
        }

        // Calculate the total space needed:
        //   4 (link) + 1 (flags) + name_len + padding + 4 (code field)
        let entry_start = self.here;
        let name_end = entry_start + 4 + 1 + name_len as u32;
        let code_field_addr = align4(name_end);
        let after_code = code_field_addr + 4;

        // Check bounds
        if after_code as usize > self.memory.len() {
            return Err(WaferError::DictionaryOverflow);
        }

        // Write link field (points to previous LATEST)
        self.write_u32_unchecked(entry_start, self.latest);

        // Write flags byte: HIDDEN | length, optionally IMMEDIATE
        let mut flag_byte = flags::HIDDEN | (name_len as u8 & flags::LENGTH_MASK);
        if immediate {
            flag_byte |= flags::IMMEDIATE;
        }
        self.memory[(entry_start + 4) as usize] = flag_byte;

        // Write name bytes
        let name_start = (entry_start + 5) as usize;
        self.memory[name_start..name_start + name_len].copy_from_slice(name_bytes);

        // Zero padding bytes between name end and code field
        for i in (name_end as usize)..(code_field_addr as usize) {
            self.memory[i] = 0;
        }

        // Write code field (function table index)
        let fn_index = self.next_fn_index;
        self.write_u32_unchecked(code_field_addr, fn_index);
        self.next_fn_index += 1;

        // Update LATEST and HERE
        self.latest = entry_start;
        self.here = after_code;

        Ok(WordId(fn_index))
    }

    /// Reveal the most recent word (remove HIDDEN flag).
    /// Called after `: ... ;` completes compilation.
    pub fn reveal(&mut self) {
        if self.latest == 0 && self.here == DICTIONARY_BASE {
            // No words defined yet
            return;
        }
        let flags_addr = (self.latest + 4) as usize;
        if flags_addr < self.memory.len() {
            self.memory[flags_addr] &= !flags::HIDDEN;
        }
    }

    /// Reveal a word at a specific address (remove HIDDEN flag).
    pub fn reveal_at(&mut self, word_addr: u32) {
        let flags_addr = (word_addr + 4) as usize;
        if flags_addr < self.memory.len() {
            self.memory[flags_addr] &= !flags::HIDDEN;
        }
    }

    /// Set the code field of the most recent word.
    pub fn set_code_field(&mut self, word_addr: u32, fn_index: u32) {
        if let Ok(code_addr) = self.code_field_addr(word_addr) {
            self.write_u32_unchecked(code_addr, fn_index);
        }
    }

    /// Look up a word by name. Returns (word_address, word_id, is_immediate).
    /// Searches from LATEST backward through the linked list.
    /// Skips HIDDEN words.
    pub fn find(&self, name: &str) -> Option<(u32, WordId, bool)> {
        let search_name = name.to_ascii_uppercase();
        let search_bytes = search_name.as_bytes();
        let search_len = search_bytes.len();

        let mut addr = self.latest;
        while addr != 0 || (addr == self.latest && self.latest != 0) {
            let flags_byte = self.memory[(addr + 4) as usize];

            // Skip hidden words
            if flags_byte & flags::HIDDEN == 0 {
                let entry_len = (flags_byte & flags::LENGTH_MASK) as usize;

                if entry_len == search_len {
                    let name_start = (addr + 5) as usize;
                    let entry_name = &self.memory[name_start..name_start + entry_len];

                    if entry_name == search_bytes {
                        let is_immediate = flags_byte & flags::IMMEDIATE != 0;
                        let code_addr = align4(addr + 5 + entry_len as u32);
                        let fn_index = self.read_u32_unchecked(code_addr);
                        return Some((addr, WordId(fn_index), is_immediate));
                    }
                }
            }

            // Follow link to previous entry
            let link = self.read_u32_unchecked(addr);
            if link == addr {
                // Safety: prevent infinite loops
                break;
            }
            addr = link;
            if addr == 0 {
                break;
            }
        }

        None
    }

    /// Get the current HERE pointer.
    pub fn here(&self) -> u32 {
        self.here
    }

    /// Get the current LATEST pointer.
    pub fn latest(&self) -> u32 {
        self.latest
    }

    /// Allocate n bytes at HERE (like Forth's ALLOT).
    pub fn allot(&mut self, n: u32) -> WaferResult<u32> {
        let new_here = self
            .here
            .checked_add(n)
            .ok_or(WaferError::DictionaryOverflow)?;
        if new_here as usize > self.memory.len() {
            return Err(WaferError::DictionaryOverflow);
        }
        let old_here = self.here;
        self.here = new_here;
        Ok(old_here)
    }

    /// Store a cell (u32) at HERE and advance HERE by 4 (like Forth's `,`).
    pub fn comma(&mut self, value: u32) -> WaferResult<()> {
        let addr = self.here;
        if (addr + 4) as usize > self.memory.len() {
            return Err(WaferError::DictionaryOverflow);
        }
        self.write_u32_unchecked(addr, value);
        self.here += 4;
        Ok(())
    }

    /// Store a byte at HERE and advance HERE by 1 (like Forth's `C,`).
    pub fn c_comma(&mut self, value: u8) -> WaferResult<()> {
        let addr = self.here as usize;
        if addr >= self.memory.len() {
            return Err(WaferError::DictionaryOverflow);
        }
        self.memory[addr] = value;
        self.here += 1;
        Ok(())
    }

    /// Read a cell (u32) from the given address.
    pub fn read_u32(&self, addr: u32) -> WaferResult<u32> {
        let a = addr as usize;
        if a + 4 > self.memory.len() {
            return Err(WaferError::InvalidAddress(addr));
        }
        Ok(u32::from_le_bytes([
            self.memory[a],
            self.memory[a + 1],
            self.memory[a + 2],
            self.memory[a + 3],
        ]))
    }

    /// Write a cell (u32) to the given address.
    pub fn write_u32(&mut self, addr: u32, value: u32) -> WaferResult<()> {
        let a = addr as usize;
        if a + 4 > self.memory.len() {
            return Err(WaferError::InvalidAddress(addr));
        }
        let bytes = value.to_le_bytes();
        self.memory[a..a + 4].copy_from_slice(&bytes);
        Ok(())
    }

    /// Read a byte from the given address.
    pub fn read_u8(&self, addr: u32) -> WaferResult<u8> {
        let a = addr as usize;
        if a >= self.memory.len() {
            return Err(WaferError::InvalidAddress(addr));
        }
        Ok(self.memory[a])
    }

    /// Write a byte to the given address.
    pub fn write_u8(&mut self, addr: u32, value: u8) -> WaferResult<()> {
        let a = addr as usize;
        if a >= self.memory.len() {
            return Err(WaferError::InvalidAddress(addr));
        }
        self.memory[a] = value;
        Ok(())
    }

    /// Get the name of the word at the given address.
    pub fn word_name(&self, word_addr: u32) -> WaferResult<String> {
        let flags_addr = (word_addr + 4) as usize;
        if flags_addr >= self.memory.len() {
            return Err(WaferError::InvalidAddress(word_addr));
        }
        let flags_byte = self.memory[flags_addr];
        let name_len = (flags_byte & flags::LENGTH_MASK) as usize;
        let name_start = (word_addr + 5) as usize;
        let name_end = name_start + name_len;
        if name_end > self.memory.len() {
            return Err(WaferError::InvalidAddress(word_addr));
        }
        let name_bytes = &self.memory[name_start..name_end];
        Ok(String::from_utf8_lossy(name_bytes).to_string())
    }

    /// Get the code field (function index) of the word at the given address.
    pub fn code_field(&self, word_addr: u32) -> WaferResult<u32> {
        let code_addr = self.code_field_addr(word_addr)?;
        self.read_u32(code_addr)
    }

    /// Get the parameter field address of the word at the given address.
    pub fn param_field_addr(&self, word_addr: u32) -> WaferResult<u32> {
        let code_addr = self.code_field_addr(word_addr)?;
        Ok(code_addr + 4)
    }

    /// Toggle the IMMEDIATE flag on the most recent word.
    pub fn toggle_immediate(&mut self) -> WaferResult<()> {
        if self.latest == 0 && self.here == DICTIONARY_BASE {
            return Err(WaferError::CompileError("no word defined yet".to_string()));
        }
        let flags_addr = (self.latest + 4) as usize;
        if flags_addr >= self.memory.len() {
            return Err(WaferError::InvalidAddress(self.latest + 4));
        }
        self.memory[flags_addr] ^= flags::IMMEDIATE;
        Ok(())
    }

    /// Get a reference to the raw memory buffer.
    pub fn memory(&self) -> &[u8] {
        &self.memory
    }

    /// Get a mutable reference to the raw memory buffer.
    pub fn memory_mut(&mut self) -> &mut Vec<u8> {
        &mut self.memory
    }

    // -- Private helpers --

    /// Compute the address of the code field for the word at `word_addr`.
    fn code_field_addr(&self, word_addr: u32) -> WaferResult<u32> {
        let flags_addr = (word_addr + 4) as usize;
        if flags_addr >= self.memory.len() {
            return Err(WaferError::InvalidAddress(word_addr));
        }
        let flags_byte = self.memory[flags_addr];
        let name_len = (flags_byte & flags::LENGTH_MASK) as u32;
        Ok(align4(word_addr + 5 + name_len))
    }

    /// Write a u32 in little-endian without bounds checking.
    /// Caller must ensure addr + 4 <= memory.len().
    fn write_u32_unchecked(&mut self, addr: u32, value: u32) {
        let a = addr as usize;
        let bytes = value.to_le_bytes();
        self.memory[a..a + 4].copy_from_slice(&bytes);
    }

    /// Read a u32 in little-endian without bounds checking.
    /// Caller must ensure addr + 4 <= memory.len().
    fn read_u32_unchecked(&self, addr: u32) -> u32 {
        let a = addr as usize;
        u32::from_le_bytes([
            self.memory[a],
            self.memory[a + 1],
            self.memory[a + 2],
            self.memory[a + 3],
        ])
    }
}

impl Default for Dictionary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::DICTIONARY_BASE;

    #[test]
    fn flag_constants() {
        // Flags should not overlap with name length
        assert_eq!(flags::IMMEDIATE & flags::LENGTH_MASK, 0);
        assert_eq!(flags::HIDDEN & flags::LENGTH_MASK, 0);
        // Max name length fits in the length mask
        assert!(flags::MAX_NAME_LEN <= flags::LENGTH_MASK as usize);
    }

    #[test]
    fn create_and_find_word() {
        let mut dict = Dictionary::new();
        let word_id = dict.create("dup", false).unwrap();
        dict.reveal();

        let result = dict.find("DUP");
        assert!(result.is_some());
        let (addr, found_id, is_imm) = result.unwrap();
        assert_eq!(found_id, word_id);
        assert!(!is_imm);
        assert_eq!(addr, DICTIONARY_BASE);
    }

    #[test]
    fn create_multiple_words_and_find_each() {
        let mut dict = Dictionary::new();

        let id_a = dict.create("ALPHA", false).unwrap();
        dict.reveal();
        let id_b = dict.create("BETA", false).unwrap();
        dict.reveal();
        let id_c = dict.create("GAMMA", false).unwrap();
        dict.reveal();

        let (_, fid_a, _) = dict.find("ALPHA").unwrap();
        let (_, fid_b, _) = dict.find("BETA").unwrap();
        let (_, fid_c, _) = dict.find("GAMMA").unwrap();

        assert_eq!(fid_a, id_a);
        assert_eq!(fid_b, id_b);
        assert_eq!(fid_c, id_c);
    }

    #[test]
    fn case_insensitive_lookup() {
        let mut dict = Dictionary::new();
        dict.create("Hello", false).unwrap();
        dict.reveal();

        // All case variants should find the same word
        assert!(dict.find("HELLO").is_some());
        assert!(dict.find("hello").is_some());
        assert!(dict.find("hElLo").is_some());
    }

    #[test]
    fn hidden_words_not_found() {
        let mut dict = Dictionary::new();
        dict.create("SECRET", false).unwrap();
        // Don't reveal

        assert!(dict.find("SECRET").is_none());
    }

    #[test]
    fn reveal_makes_hidden_word_findable() {
        let mut dict = Dictionary::new();
        dict.create("HIDDEN", false).unwrap();
        assert!(dict.find("HIDDEN").is_none());

        dict.reveal();
        assert!(dict.find("HIDDEN").is_some());
    }

    #[test]
    fn immediate_flag_works() {
        let mut dict = Dictionary::new();
        let word_id = dict.create("IF", true).unwrap();
        dict.reveal();

        let (_, found_id, is_imm) = dict.find("IF").unwrap();
        assert_eq!(found_id, word_id);
        assert!(is_imm);
    }

    #[test]
    fn toggle_immediate() {
        let mut dict = Dictionary::new();
        dict.create("MYWORD", false).unwrap();
        dict.reveal();

        // Initially not immediate
        let (_, _, is_imm) = dict.find("MYWORD").unwrap();
        assert!(!is_imm);

        // Toggle to immediate
        dict.toggle_immediate().unwrap();
        let (_, _, is_imm) = dict.find("MYWORD").unwrap();
        assert!(is_imm);

        // Toggle back
        dict.toggle_immediate().unwrap();
        let (_, _, is_imm) = dict.find("MYWORD").unwrap();
        assert!(!is_imm);
    }

    #[test]
    fn comma_advances_here() {
        let mut dict = Dictionary::new();
        let h0 = dict.here();
        dict.comma(42).unwrap();
        assert_eq!(dict.here(), h0 + 4);

        // Verify the value was stored
        let val = dict.read_u32(h0).unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn c_comma_advances_here() {
        let mut dict = Dictionary::new();
        let h0 = dict.here();
        dict.c_comma(0xAB).unwrap();
        assert_eq!(dict.here(), h0 + 1);

        // Verify the value was stored
        let val = dict.read_u8(h0).unwrap();
        assert_eq!(val, 0xAB);
    }

    #[test]
    fn allot_advances_here() {
        let mut dict = Dictionary::new();
        let h0 = dict.here();
        let old = dict.allot(100).unwrap();
        assert_eq!(old, h0);
        assert_eq!(dict.here(), h0 + 100);
    }

    #[test]
    fn memory_read_write_u32() {
        let mut dict = Dictionary::new();
        let addr = DICTIONARY_BASE;
        dict.write_u32(addr, 0xDEADBEEF).unwrap();
        let val = dict.read_u32(addr).unwrap();
        assert_eq!(val, 0xDEADBEEF);
    }

    #[test]
    fn memory_read_write_u8() {
        let mut dict = Dictionary::new();
        let addr = DICTIONARY_BASE;
        dict.write_u8(addr, 0x42).unwrap();
        let val = dict.read_u8(addr).unwrap();
        assert_eq!(val, 0x42);
    }

    #[test]
    fn max_name_length() {
        let mut dict = Dictionary::new();
        let name = "A".repeat(31); // MAX_NAME_LEN = 31
        let result = dict.create(&name, false);
        assert!(result.is_ok());
        dict.reveal();

        let found = dict.find(&name);
        assert!(found.is_some());
        let (_, _, _) = found.unwrap();

        // Verify the name stored correctly
        let word_name = dict.word_name(dict.latest()).unwrap();
        assert_eq!(word_name, name);
    }

    #[test]
    fn name_too_long_rejected() {
        let mut dict = Dictionary::new();
        let name = "A".repeat(32); // Exceeds MAX_NAME_LEN
        let result = dict.create(&name, false);
        assert!(result.is_err());
    }

    #[test]
    fn empty_name_rejected() {
        let mut dict = Dictionary::new();
        let result = dict.create("", false);
        assert!(result.is_err());
    }

    #[test]
    fn unknown_word_returns_none() {
        let mut dict = Dictionary::new();
        dict.create("EXISTS", false).unwrap();
        dict.reveal();

        assert!(dict.find("DOESNOTEXIST").is_none());
    }

    #[test]
    fn param_field_addr_calculation() {
        let mut dict = Dictionary::new();
        dict.create("VAR", false).unwrap();
        dict.reveal();

        let word_addr = dict.latest();
        let pfa = dict.param_field_addr(word_addr).unwrap();
        let cfa_addr = align4(word_addr + 5 + 3); // "VAR" is 3 bytes
        assert_eq!(pfa, cfa_addr + 4);

        // HERE should equal the parameter field address right after create
        assert_eq!(dict.here(), pfa);
    }

    #[test]
    fn dictionary_overflow_detection() {
        let mut dict = Dictionary::new();
        let mem_size = dict.memory().len() as u32;

        // Try to allot beyond memory
        let result = dict.allot(mem_size + 1);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_address_read() {
        let dict = Dictionary::new();
        let mem_size = dict.memory().len() as u32;

        // Reading beyond the end should fail
        assert!(dict.read_u32(mem_size).is_err());
        assert!(dict.read_u8(mem_size).is_err());
    }

    #[test]
    fn invalid_address_write() {
        let mut dict = Dictionary::new();
        let mem_size = dict.memory().len() as u32;

        // Writing beyond the end should fail
        assert!(dict.write_u32(mem_size, 0).is_err());
        assert!(dict.write_u8(mem_size, 0).is_err());
    }

    #[test]
    fn set_code_field_updates_function_index() {
        let mut dict = Dictionary::new();
        dict.create("TEST", false).unwrap();
        dict.reveal();

        let word_addr = dict.latest();
        dict.set_code_field(word_addr, 999);

        let code = dict.code_field(word_addr).unwrap();
        assert_eq!(code, 999);
    }

    #[test]
    fn word_name_retrieval() {
        let mut dict = Dictionary::new();
        dict.create("HELLO", false).unwrap();
        dict.reveal();

        let name = dict.word_name(dict.latest()).unwrap();
        assert_eq!(name, "HELLO");
    }

    #[test]
    fn linked_list_traversal() {
        // Verify that the linked list structure is correct
        let mut dict = Dictionary::new();

        let addr0 = dict.here();
        dict.create("FIRST", false).unwrap();
        dict.reveal();
        assert_eq!(dict.latest(), addr0);

        let addr1 = dict.here();
        dict.create("SECOND", false).unwrap();
        dict.reveal();
        assert_eq!(dict.latest(), addr1);

        // Second word's link should point to first word
        let link = dict.read_u32(addr1).unwrap();
        assert_eq!(link, addr0);

        // First word's link should be 0 (end of list)
        let link = dict.read_u32(addr0).unwrap();
        assert_eq!(link, 0);
    }

    #[test]
    fn later_definition_shadows_earlier() {
        let mut dict = Dictionary::new();

        let id1 = dict.create("DUP", false).unwrap();
        dict.reveal();
        let id2 = dict.create("DUP", false).unwrap();
        dict.reveal();

        // find should return the later (most recent) definition
        let (_, found_id, _) = dict.find("DUP").unwrap();
        assert_eq!(found_id, id2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn alignment_padding() {
        let mut dict = Dictionary::new();

        // "AB" is 2 bytes at offset 5 => name_end = base + 4 + 1 + 2 = base + 7
        // align4(base + 7) should round up properly
        dict.create("AB", false).unwrap();
        dict.reveal();

        let word_addr = dict.latest();
        let pfa = dict.param_field_addr(word_addr).unwrap();
        // code field should be at align4(word_addr + 5 + 2) = align4(word_addr + 7)
        let expected_code = align4(word_addr + 7);
        assert_eq!(pfa, expected_code + 4);
        // HERE should be 4-byte aligned
        assert_eq!(dict.here() % 4, 0);
    }

    #[test]
    fn memory_access() {
        let mut dict = Dictionary::new();

        // Test raw memory access
        let mem = dict.memory();
        assert_eq!(mem.len(), (INITIAL_PAGES * PAGE_SIZE) as usize);

        // Test mutable access
        let mem = dict.memory_mut();
        mem[0] = 0xFF;
        assert_eq!(dict.memory()[0], 0xFF);
    }

    #[test]
    fn default_trait() {
        let dict = Dictionary::default();
        assert_eq!(dict.here(), DICTIONARY_BASE);
        assert_eq!(dict.latest(), 0);
    }

    #[test]
    fn comma_overflow() {
        let mut dict = Dictionary::new();
        // Move HERE to near the end of memory
        let mem_size = dict.memory().len() as u32;
        dict.here = mem_size - 2; // Only 2 bytes left
        let result = dict.comma(42);
        assert!(result.is_err());
    }

    #[test]
    fn c_comma_overflow() {
        let mut dict = Dictionary::new();
        let mem_size = dict.memory().len() as u32;
        dict.here = mem_size; // No space left
        let result = dict.c_comma(42);
        assert!(result.is_err());
    }

    #[test]
    fn word_ids_are_sequential() {
        let mut dict = Dictionary::new();
        let id0 = dict.create("A", false).unwrap();
        dict.reveal();
        let id1 = dict.create("B", false).unwrap();
        dict.reveal();
        let id2 = dict.create("C", false).unwrap();
        dict.reveal();

        assert_eq!(id0, WordId(0));
        assert_eq!(id1, WordId(1));
        assert_eq!(id2, WordId(2));
    }

    #[test]
    fn toggle_immediate_no_word_errors() {
        let mut dict = Dictionary::new();
        let result = dict.toggle_immediate();
        assert!(result.is_err());
    }
}
