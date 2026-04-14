//! Cryptographic hash primitives for WAFER.
//!
//! Provides a small registry of hash algorithms ([`ALGOS`]) and a Forth
//! word per algorithm. Each word has stack effect:
//!
//! ```text
//! ( c-addr u -- c-addr2 u2 )
//! ```
//!
//! It hashes the `u` bytes starting at `c-addr` and writes the digest into
//! the [`crate::memory::HASH_SCRATCH_BASE`] region. The output buffer is
//! shared and overwritten by every subsequent hash call — copy the bytes
//! out before the next invocation if you need to keep them.
//!
//! The registry is designed to grow: add a new entry to [`ALGOS`] and the
//! word is registered automatically by
//! [`crate::outer::ForthVM::register_crypto_words`].

use sha1::{Digest as Sha1Digest, Sha1};
use sha2::{Sha256, Sha512};

/// One hash algorithm registered as a Forth word.
pub struct HashAlgo {
    /// Forth word name (uppercase by convention).
    pub name: &'static str,
    /// Digest length in bytes.
    pub digest_len: usize,
    /// Hash function.
    pub hash: fn(&[u8]) -> Vec<u8>,
}

fn sha1_hash(input: &[u8]) -> Vec<u8> {
    let mut h = Sha1::new();
    h.update(input);
    h.finalize().to_vec()
}

fn sha256_hash(input: &[u8]) -> Vec<u8> {
    let mut h = Sha256::new();
    h.update(input);
    h.finalize().to_vec()
}

fn sha512_hash(input: &[u8]) -> Vec<u8> {
    let mut h = Sha512::new();
    h.update(input);
    h.finalize().to_vec()
}

/// All hash algorithms registered as Forth words.
pub const ALGOS: &[HashAlgo] = &[
    HashAlgo {
        name: "SHA1",
        digest_len: 20,
        hash: sha1_hash,
    },
    HashAlgo {
        name: "SHA256",
        digest_len: 32,
        hash: sha256_hash,
    },
    HashAlgo {
        name: "SHA512",
        digest_len: 64,
        hash: sha512_hash,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    #[test]
    fn sha1_rfc3174_abc() {
        assert_eq!(hex(&sha1_hash(b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
    }

    #[test]
    fn sha256_fips180_abc() {
        assert_eq!(
            hex(&sha256_hash(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha512_fips180_abc() {
        assert_eq!(
            hex(&sha512_hash(b"abc")),
            concat!(
                "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a",
                "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
            )
        );
    }

    #[test]
    fn registry_lengths_match() {
        assert_eq!(ALGOS[0].digest_len, sha1_hash(b"").len());
        assert_eq!(ALGOS[1].digest_len, sha256_hash(b"").len());
        assert_eq!(ALGOS[2].digest_len, sha512_hash(b"").len());
    }
}
