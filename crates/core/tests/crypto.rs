//! End-to-end tests for the `SHA1` / `SHA256` / `SHA512` Forth host words.
//!
//! These run inside a real WAFER VM (NativeRuntime). The Forth program writes
//! a counted string into `PAD`, calls the hash word, then the test reads the
//! digest out of WAFER linear memory and compares it to the RFC-3174 / FIPS-180
//! reference vectors.

use wafer_core::memory::{HASH_SCRATCH_BASE, PAD_BASE};
use wafer_core::outer::ForthVM;
use wafer_core::runtime::Runtime;
use wafer_core::runtime_native::NativeRuntime;

/// Hash `input` using the named Forth word and return the digest bytes.
fn hash_via_forth(word: &str, input: &[u8]) -> Vec<u8> {
    let mut vm = ForthVM::<NativeRuntime>::new().expect("vm");

    // Place input bytes in PAD via a sequence of `c C!` operations.
    // Then push (PAD u) and call the hash word.
    let mut prog = String::new();
    for (i, b) in input.iter().enumerate() {
        prog.push_str(&format!("{} {} C! ", b, PAD_BASE as usize + i));
    }
    prog.push_str(&format!("{} {} {} ", PAD_BASE, input.len(), word));

    vm.evaluate(&prog).expect("eval");

    // Stack now: ( c-addr2 u2 ). Read u2 then c-addr2 from data stack.
    let stack = vm.data_stack();
    assert!(stack.len() >= 2, "expected (addr len) on stack, got {stack:?}");
    let u2 = stack[0] as usize;
    let addr2 = stack[1] as u32;
    assert_eq!(addr2, HASH_SCRATCH_BASE, "digest should land in HASH_SCRATCH");

    // Read the digest out of WAFER linear memory.
    let mut bytes = Vec::with_capacity(u2);
    let rt = vm.runtime_mut();
    for i in 0..u2 {
        bytes.push(rt.mem_read_u8(addr2 + i as u32));
    }
    bytes
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

#[test]
fn sha1_abc_vector() {
    assert_eq!(
        hex(&hash_via_forth("SHA1", b"abc")),
        "a9993e364706816aba3e25717850c26c9cd0d89d"
    );
}

#[test]
fn sha256_abc_vector() {
    assert_eq!(
        hex(&hash_via_forth("SHA256", b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn sha512_abc_vector() {
    assert_eq!(
        hex(&hash_via_forth("SHA512", b"abc")),
        concat!(
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a",
            "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
        )
    );
}

#[test]
fn sha1_empty_string() {
    assert_eq!(
        hex(&hash_via_forth("SHA1", b"")),
        "da39a3ee5e6b4b0d3255bfef95601890afd80709"
    );
}

#[test]
fn sha1_skey_seed_matches_hel_first_round() {
    // Reference: hel/src/skey.rs::tests::encoding_test, first SHA1 of
    // "test1" || "my secret" yields [141, 231, 26, 167,  ... 5 ints, ...]
    // == raw SHA1 of those concatenated bytes.
    let input = b"test1my secret";
    let digest = hash_via_forth("SHA1", input);
    assert_eq!(digest.len(), 20);
    // Folded 20->8 yields ints[0]^ints[2]^ints[4] and ints[1]^ints[3].
    // Just check the raw SHA1 here; the fold is implemented in kelvar Forth.
    let expected = "8af6385b5053c32db569166a615739479885c9bc";
    assert_eq!(hex(&digest), expected);
}
