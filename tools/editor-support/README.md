# Editor support for WAFER

Syntax highlighting assets for editors and pagers.

## bat (and other Sublime-Text-compatible tools)

`bat/WAFER.sublime-syntax` is a Sublime Text grammar covering Forth 2012 plus
WAFER-specific words (`CONSOLIDATE`, `RANDOM`, `RND-SEED`, `UTIME`).

### Install

```
just install-syntax
```

or manually:

```
mkdir -p ~/.config/bat/syntaxes
cp tools/editor-support/bat/WAFER.sublime-syntax ~/.config/bat/syntaxes/
bat cache --build
```

### Verify

```
bat --list-languages | grep -i forth        # should list Forth
bat --language forth crates/core/boot.fth   # should render with colour
```

### Use with `oked`

`oked` auto-detects `.fth` / `.4th` / `.forth` files and invokes `bat` with
`--language forth`. After the install step above, opening any WAFER source in
`oked` and toggling highlight (`H` command, or `oked -S forth`) will use this
syntax.

### Updating the keyword list

Primitives live in `crates/core/src/outer.rs` (`register_primitive` and
`register_host_primitive` calls). When a new **user-facing, non-standard** word
is added, append it to the `wafer_extras` context in
`bat/WAFER.sublime-syntax`. Standard Forth 2012 words are already covered by
the main contexts.

Internal symbols (names that start with `_`) should not be added — they are
implementation details that user code never types.
