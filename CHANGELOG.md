# Changelog

All notable changes to WAFER are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **A typed calling convention for words with a known stack effect.** Such a
  word now compiles to two entry points: a fast one whose signature is
  `(i32 x p) -> (i32 x q)`, carrying its stack items as WASM values, and the
  usual `( -- )` wrapper that moves those items on and off the memory data
  stack. The wrapper keeps the function-table slot, so `EXECUTE`, the outer
  interpreter, host words and `CATCH` see exactly the ABI they saw before;
  only direct calls inside a module take the fast entry.

  This is what the SwiftForth gap was made of. sf64 keeps TOS in `RBX` and
  the stack pointer in `RBP`, and both survive a `CALL` untouched, so its
  `FIB` is 16 instructions and ~7 memory touches per node. WAFER kept the
  whole stack in linear memory and flushed its cached `$dsp` to an imported
  global before every call: ~36 memory touches per node. The stack simulator
  that already promoted loop and `IF` bodies into WASM locals refused any
  body containing a call or an `EXIT` -- exactly the words where the
  convention cost the most. It now handles both.

  Fibonacci(25) goes from 1035 to 366 µs, 4.3x slower than `sf64` to 1.2x.
  Loop-heavy benchmarks are unchanged (they were already promoted, and
  already beat `sf64`). Words that keep the memory convention: anything
  using `SP@`, `DEPTH`, `EXECUTE`, `>R`/`R>`, floats or locals; anything
  calling a word that is itself untyped, which in the JIT path means every
  call except `RECURSE`; mutually recursive words; and words whose effect is
  not static -- branches that disagree on depth, `EXIT` at the wrong depth,
  a non-neutral loop body, or a recursion that grows the stack per level.

  `CONSOLIDATE` extends this across words, since it puts them all in one
  module: the effects are solved to a fixpoint from the leaves outward, and
  105 of 187 words in a booted dictionary end up typed.

  Stack guards get cheap as a side effect -- they hang off the memory-stack
  push/pop choke points, and a typed word barely has any. The default
  guards-on configuration that the REPL and the web build use went from 1631
  to 365 µs on the same benchmark.

  `WAFER_TYPED_CALLS=0` falls back to the memory-stack convention.

### Fixed

- The Forth 2012 Core suite now also runs against consolidated code
  (`compliance_core_after_consolidate`). `CONSOLIDATE` had no correctness
  test at all before -- only benchmarks.

## [0.2.6] - 2026-08-07

### Fixed

- **An uncaught `ABORT` no longer prints anything.** It used to report
  `ABORT (throw -1)`, but the standard defines `ABORT` as "empty the data
  stack and perform the function of `QUIT`", and `QUIT` displays no
  message. gforth and SwiftForth are both silent here. `CATCH` still
  reports -1 as before, and `ABORT"` still prints its text — that is a
  different word with a different code (-2).
- **Compile-only words used in interpretation state name the condition.**
  `ABORT"`, `IF`, `THEN`, `LOOP`, `LITERAL`, `RECURSE` and the rest of
  the compile-time constructs claimed to be an `unknown word`, which is
  actively misleading for a word the system obviously knows. They now
  report `interpreting a compile-only word: <name> (throw -14)`, the
  standard condition both reference engines give. A genuine typo still
  reports `unknown word`.

## [0.2.5] - 2026-08-06

### Added

- **`QUIT`** ( -- ) ( R: i\*x -- ), the CORE word that was missing: empty
  the return stack, enter interpretation state, hand the input source
  back to the user input device and return to the interpreter without a
  message. The data stack is deliberately left alone — that is the whole
  difference to `ABORT`, which the standard defines as "empty the data
  stack, then `QUIT`". It unwinds through nested `EVALUATE` and
  `INCLUDE`, abandoning them, and `SOURCE-ID` is restored to 0.

  `CATCH` does **not** report it: `QUIT` rides throw code -56, which the
  interpreter treats as a return to the prompt rather than an exception.
  Both behaviours were checked against gforth 0.7.3 and SwiftForth
  `sf64`, which agree — `1 2 ' QUIT CATCH .` prints nothing and leaves
  `1 2` on the stack in all three engines.

  The gap had gone unnoticed because the Forth 2012 test suite skips it
  by its own admission ("I HAVEN'T FIGURED OUT HOW TO TEST KEY, QUIT,
  ABORT, OR ABORT\""), and because `HELP`'s coverage lint compares the
  dictionary against the docs — a word absent from both looks complete.
  `docs/wafer-anki.txt` had been documenting `QUIT` as if it existed.

  Note that `ABORT` was already correct: executing it while a definition
  is open does clear both stacks and return to interpretation state.
  Typing `ABORT` (or `QUIT`) into an unfinished definition compiles it
  rather than running it, exactly as in every other Forth; `[` is the
  word that gets you out.

## [0.2.4] - 2026-08-06

### Fixed

- **Errors from host words in the browser build read like Forth errors
  again.** A host word signals failure by throwing across the JS
  boundary, and the browser runtime reported the exception with its
  `Debug` form, so an empty-stack `RESIZE` came back as
  `call_func(134) failed: JsValue(Error: Stack underflow ...)` trailed by
  an engine stack trace. The thrown message is the Forth message, so it
  is now surfaced verbatim — `Stack underflow`, exactly what the native
  CLI prints. Exceptions that carry no message keep the call context,
  since those are genuine runtime faults rather than Forth throws.
  `CATCH` was never affected: it reads the throw code from its own
  channel, not from the message.

## [0.2.3] - 2026-08-06

### Fixed

- **Release builds of `wafer-web` no longer fail on proc-macro loading.**
  Cargo strips debuginfo from release artifacts by default, and on macOS
  that also strips the metadata proc-macro dylibs need to be loadable, so
  `wasm-pack build --release` died with `can't find crate` for
  `rustversion`, `thiserror_impl` and every other proc-macro. Build
  scripts and proc-macros gain nothing from stripping, so
  `[profile.release.build-override]` now exempts them; release binaries
  stay stripped. Debug builds were never affected, which is why the test
  suite stayed green while the browser REPL could not be built for
  production.
- `wafer-web` and `wafer-cli` requested `wafer-core` version `0.2.1`
  while the workspace had moved to `0.2.2`. The caret requirement still
  resolved, so nothing broke, but the pin is now kept in step.

## [0.2.2] - 2026-08-06

### Added

- **SwiftForth-style input number conversion.** Punctuation (`,` `.` `+`
  `/` `:` and an embedded `-`) anywhere after the leftmost digit now forces
  double-cell conversion, so `12.34`, `1,234`, `12:30:45` and `2026-08-06`
  all convert as doubles without a custom parser. Previously only a
  trailing `.` worked and `1.5` was an "unknown word" error. The
  punctuation is a double-cell marker, not a fractional point: every
  spelling of `1234` (`1234.`, `123.4`, `.1234`) yields the same value.
- **`DPL`** ( -- addr ): digits to the right of the rightmost punctuation
  character in the last converted number, negative when the token carried
  none. Seeded at -1024 and bumped once per digit, matching `sf64`.
  Together with `<# #>` this is how fixed-point input is scaled.
- **`NH`** ( -- addr ): the high-order cell dropped by a single-cell
  conversion, so a token that overflows a cell can be recovered as a
  double (`4000000000 NH @ D.`).

Verified token-for-token against SwiftForth `sf64`: DPL values, double
promotion and sign handling agree on every probed form. One deliberate
divergence — WAFER also accepts a sign before a base prefix (`-$FF`), which
`sf64` rejects; the Forth 2012 spelling `$-FF` works in both. A leading `+`
is punctuation rather than a sign in both engines, so `+7` is the double 7
with `DPL` = 1.

## [0.2.1] - 2026-08-06

### Fixed

- **The search order is now authoritative** (Forth 2012 §16.3.3): a word
  whose wordlist is not in the search order is no longer findable.
  Previously lookup fell back to the newest entry across all wordlists,
  making word hiding impossible. Verified against gforth and SwiftForth,
  and guarded by a cross-engine corpus program.
- **Host words validate their stack arguments.** Around 40 host-implemented
  words (`RND-SEED`, `ACCEPT`, `RESIZE`, `ALLOCATE`, `FREE`, `SEARCH`,
  `SUBSTITUTE`, `ROLL`, `M*`, `UM/MOD`, `SF@ SF! DF@ DF!`, `F. FE. FS. F~`,
  `2R@`, and friends) performed raw stack-pointer arithmetic with no
  underflow check — calling them on an empty stack silently corrupted the
  stack pointer (the compiled-code guards from 0.2.0 do not cover host
  words). All argument-taking host words now fail with a clean, CATCHable
  underflow error, enforced by a class-wide regression test.

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
  (inline `ok` echo only for single-line output).
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

[0.2.1]: https://github.com/ok2/wafer/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/ok2/wafer/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ok2/wafer/releases/tag/v0.1.0
