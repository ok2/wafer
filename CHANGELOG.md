# Changelog

All notable changes to WAFER are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-06

The usability release: introspection, source files, honest errors, and a
safety net under every compiled word.

### Added

- **Stack guards in compiled code**: under/overflow checks at the
  stack-pointer choke points of generated WASM. Faults THROW standard codes
  (`-3`..`-6`, `-44`, `-45`), are CATCHable, and print standard messages
  instead of silently corrupting memory. Default on; `wafer build` output
  stays unguarded; `WAFER_STACK_GUARDS=0|1` overrides.
- **`SEE`**: source-level decompiler. Colon words (including everything in
  `boot.fth`) show their captured verbatim source; data words show
  synthesized definitions with current values (`9 VALUE X`,
  `DEFER D ( IS DUP )`); primitives fall back to a readable IR dump —
  `SEE` never dead-ends on a defined word.
- **`SEE-IR`**: post-optimization IR view with resolved callee names and
  indented control flow — shows what the optimizer actually did.
- **`HELP`**: stack effect + one-line description for **every** word in a
  fresh VM (dictionary words and outer-interpreter tokens alike); coverage
  is enforced by a unit test, so an undocumented new word fails the build.
  User words echo their leading `( n -- n )` comment.
- **`INCLUDE` / `INCLUDED`**: nestable source-file loading with cycle
  detection, depth bound, paths relative to the including file, and
  per-level `SOURCE-ID`. The loader is injected (CLI: filesystem; web:
  defined error), so the core stays IO-free. `wafer prog.fth` now runs
  through the same machinery.
- **`MARKER` extensions**: `REMEMBER` (re-runnable marker), `EMPTY` and
  `GILD` (boot-state rollback and re-baselining). Marker rollback now also
  restores search order, wordlists, `REPLACES` substitutions, `ABORT"`
  texts, and captured word sources — enabling the `REMEMBER` + `INCLUDE`
  edit-reload loop.
- **`WORDS`**: optional substring filter (`WORDS FLOAT`), word count, and
  `WORDS ALL` — a grouped full view by wordlist plus internal words.
- **Return-stack introspection**: `.RS`, `RDEPTH`, `RP@`.
- **Tools**: `.S` honors `BASE`, `F.S`, `?`, bounds-checked `DUMP`, real
  `BYE`, named `ORDER` output.
- **CLI REPL**: persistent history (XDG state dir, `0600`), dictionary-backed
  tab completion, prefix history search on Up/Down, Ctrl-C clears the line.
- **Web REPL**: history persisted to localStorage, User Words palette,
  `BASE` indicator in the stack bar.
- **Error reporting**: uncaught `THROW` codes map to standard messages;
  `ABORT"` text prints only when uncaught; errors inside included files
  carry `file.fth:line:` context; uncaught throws are typed
  (`WaferError::UncaughtThrow`) for embedding consumers; compiled words
  carry WASM name sections, so genuine traps name the faulting word
  (`in CRASHER: wasm trap: out of bounds memory access`).
- **SwiftForth correctness lane**: the cross-engine program corpus can run
  against sf64 as an oracle (`just compare-correctness`), alongside the
  existing gforth lane and the sf64 performance lane.

### Fixed

- Multi-line command output in the CLI REPL starts on its own line
  (inline ` ok` echo only for single-line output).
- `.S` printed in decimal regardless of `BASE`.
- A bare interpreted `R>` underflowed silently (exposed by the new stack
  guards; compliance baseline updated).
- `SPACES` with a negative count now outputs nothing, per Forth 2012
  6.1.2230.

### Changed

- `wafer prog.fth` reports errors with `file:line` context and resolves
  nested `INCLUDE`s relative to the file.
- Internal words (`_`-prefixed) are flagged in the dictionary and hidden
  from `WORDS` and completion (`WORDS ALL` shows them).
- Dependencies upgraded across the board: wasmtime 43 → 47,
  wasm-encoder/wasmparser 0.246 → 0.255, plus all semver-compatible
  updates.

## [0.1.0] - 2026-08-04

Initial development line (untagged): Forth 2012 core with IR optimizer and
WASM codegen via wasm-encoder/wasmtime, ~300 words across Core, Double,
Float, String, Search-Order, Exception, and Tools word sets, Forth 2012
compliance suite, `CONSOLIDATE` whole-program recompilation, `wafer build`
AOT export (WASM / native / JS loader), browser REPL, SHA-1/256/512 words,
and cross-engine benchmark lanes against gforth and SwiftForth.

[0.2.0]: https://github.com/ok2/wafer/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ok2/wafer/releases/tag/v0.1.0
