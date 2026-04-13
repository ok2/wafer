#!/usr/bin/env python3
"""
WAFER Anki Deck Generator

Generates an Anki .apkg deck from:
1. Hand-crafted cards in anki_data.yaml
2. Auto-parsed IrOp variants from ir.rs
3. Auto-parsed memory constants from memory.rs
4. Auto-parsed error variants from error.rs
5. Auto-extracted peephole patterns from optimizer.rs

Usage:
    pip install genanki pyyaml
    python tools/anki_gen.py

Output: tools/wafer_anki.apkg
"""

import hashlib
import re
import sys
from pathlib import Path

try:
    import genanki
    import yaml
except ImportError:
    print("Required: pip install genanki pyyaml")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TOOLS_DIR = Path(__file__).parent
PROJECT_ROOT = TOOLS_DIR.parent
CORE_SRC = PROJECT_ROOT / "crates" / "core" / "src"
BOOT_FTH = PROJECT_ROOT / "crates" / "core" / "boot.fth"
YAML_FILE = TOOLS_DIR / "anki_data.yaml"
OUTPUT_FILE = TOOLS_DIR / "wafer_anki.apkg"

# ---------------------------------------------------------------------------
# Stable IDs (genanki needs deterministic model/deck IDs)
# ---------------------------------------------------------------------------


def stable_id(name: str) -> int:
    """Generate a stable integer ID from a name."""
    h = hashlib.md5(name.encode()).hexdigest()
    return int(h[:8], 16)


DECK_ID = stable_id("wafer-learning-deck")
MODEL_ID = stable_id("wafer-card-model")

# ---------------------------------------------------------------------------
# Anki model with code styling
# ---------------------------------------------------------------------------

CSS = """\
.card {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 16px;
  line-height: 1.5;
  color: #1a1a1a;
  background: #fafafa;
  padding: 20px;
  max-width: 700px;
  margin: 0 auto;
}
.card.nightMode {
  color: #e0e0e0;
  background: #1e1e1e;
}
.front { font-size: 18px; font-weight: 600; }
pre, code {
  font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
  font-size: 14px;
  background: #f0f0f0;
  border-radius: 4px;
}
.nightMode pre, .nightMode code {
  background: #2d2d2d;
}
pre {
  padding: 12px;
  overflow-x: auto;
  border: 1px solid #ddd;
}
.nightMode pre { border-color: #444; }
code { padding: 2px 5px; }
.source {
  margin-top: 12px;
  font-size: 12px;
  color: #888;
  font-style: italic;
}
.tags-line {
  margin-top: 8px;
  font-size: 11px;
  color: #aaa;
}
.tags-line span {
  background: #e8e8e8;
  padding: 1px 6px;
  border-radius: 3px;
  margin-right: 4px;
}
.nightMode .tags-line span { background: #3a3a3a; }
"""

FRONT_TEMPLATE = """\
<div class="front">{{Front}}</div>
"""

BACK_TEMPLATE = """\
<div class="front">{{Front}}</div>
<hr>
<div class="back">{{Back}}</div>
{{#Code}}
<pre>{{Code}}</pre>
{{/Code}}
{{#Source}}
<div class="source">{{Source}}</div>
{{/Source}}
"""

wafer_model = genanki.Model(
    MODEL_ID,
    "WAFER Card",
    fields=[
        {"name": "Front"},
        {"name": "Back"},
        {"name": "Code"},
        {"name": "Source"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": FRONT_TEMPLATE,
            "afmt": BACK_TEMPLATE,
        },
    ],
    css=CSS,
)

# ---------------------------------------------------------------------------
# Card generation helpers
# ---------------------------------------------------------------------------


def make_note(front: str, back: str, code: str = "", source: str = "", tags: list | None = None) -> genanki.Note:
    """Create a genanki Note with stable GUID."""
    guid = genanki.guid_for(front)
    note = genanki.Note(
        model=wafer_model,
        fields=[front, back, code, source],
        tags=tags or [],
        guid=guid,
    )
    return note


def html_escape(text: str) -> str:
    """Minimal HTML escaping for card content."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_back(text: str) -> str:
    """Convert back text (may have newlines) to HTML."""
    lines = text.strip().split("\n")
    return "<br>".join(html_escape(line) for line in lines)


def format_code(text: str) -> str:
    """Format code block content."""
    return html_escape(text.strip())


# ---------------------------------------------------------------------------
# YAML card loader
# ---------------------------------------------------------------------------


def load_yaml_cards(deck: genanki.Deck) -> int:
    """Load hand-crafted cards from anki_data.yaml."""
    if not YAML_FILE.exists():
        print(f"Warning: {YAML_FILE} not found, skipping hand-crafted cards")
        return 0

    with open(YAML_FILE) as f:
        data = yaml.safe_load(f)

    count = 0
    for category, cards in data.items():
        if not isinstance(cards, list):
            continue
        for card in cards:
            front = card.get("front", "")
            back = card.get("back", "")
            code = card.get("code", "")
            source = card.get("source", "")
            tags = card.get("tags", [])

            # Add category as tag
            all_tags = [f"wafer::{category}"] + [f"wafer::{t}" for t in tags]

            note = make_note(
                front=html_escape(front),
                back=format_back(back),
                code=format_code(code) if code else "",
                source=html_escape(source),
                tags=all_tags,
            )
            deck.add_note(note)
            count += 1

    return count


# ---------------------------------------------------------------------------
# Auto-parse IrOp variants from ir.rs
# ---------------------------------------------------------------------------


def parse_ir_ops(deck: genanki.Deck) -> int:
    """Parse IrOp enum from ir.rs and generate cards."""
    ir_file = CORE_SRC / "ir.rs"
    if not ir_file.exists():
        return 0

    content = ir_file.read_text()
    count = 0

    # Match doc comments + variant lines
    # Pattern: /// comment\n    VariantName or VariantName(type) or VariantName { ... }
    lines = content.split("\n")
    i = 0
    current_category = ""

    while i < len(lines):
        line = lines[i].strip()

        # Track categories from // -- Category -- comments
        cat_match = re.match(r"//\s*--\s*(.+?)\s*--", line)
        if cat_match:
            current_category = cat_match.group(1).strip()
            i += 1
            continue

        # Collect doc comments
        doc_lines = []
        while i < len(lines) and lines[i].strip().startswith("///"):
            doc_lines.append(lines[i].strip().lstrip("/ ").strip())
            i += 1

        if i >= len(lines):
            break

        line = lines[i].strip()

        # Match variant definition
        variant_match = re.match(
            r"^((?:[A-Z][a-zA-Z0-9]+)(?:\([^)]*\))?)\s*[,{]", line
        )
        if variant_match and doc_lines:
            variant = variant_match.group(1)
            # Clean up: remove trailing comma
            variant = variant.rstrip(",")
            doc = " ".join(doc_lines)

            # Extract stack effect if present: ( ... -- ... )
            stack_match = re.search(r"\(\s*(.+?)\s*\)", doc)
            stack_effect = stack_match.group(0) if stack_match else ""

            front = f"IrOp::{variant} — what does it do?"
            back_parts = [doc]
            if stack_effect:
                back_parts.insert(0, f"Stack: {stack_effect}")
            back = "\n".join(back_parts)

            tags = ["wafer::ir", f"wafer::ir_{current_category.lower().replace(' ', '_')}"]

            note = make_note(
                front=html_escape(front),
                back=format_back(back),
                code=format_code(f"IrOp::{variant}"),
                source=f"crates/core/src/ir.rs",
                tags=tags,
            )
            deck.add_note(note)
            count += 1

        i += 1

    return count


# ---------------------------------------------------------------------------
# Auto-parse memory constants from memory.rs
# ---------------------------------------------------------------------------


def parse_memory_constants(deck: genanki.Deck) -> int:
    """Parse constants from memory.rs and generate cards."""
    mem_file = CORE_SRC / "memory.rs"
    if not mem_file.exists():
        return 0

    content = mem_file.read_text()
    count = 0

    # Match: /// doc comment\n pub const NAME: type = value;
    lines = content.split("\n")
    i = 0

    while i < len(lines):
        # Collect doc comments
        doc_lines = []
        while i < len(lines) and lines[i].strip().startswith("///"):
            doc_lines.append(lines[i].strip().lstrip("/ ").strip())
            i += 1

        if i >= len(lines):
            break

        line = lines[i].strip()
        const_match = re.match(
            r"pub const (\w+):\s*\w+\s*=\s*(.+?);", line
        )
        if const_match and doc_lines:
            name = const_match.group(1)
            value_expr = const_match.group(2).strip()
            doc = " ".join(doc_lines)

            # Try to evaluate simple expressions for the card
            # (won't work for all, but catches most)
            front = f"memory.rs: What is {name}?"
            back = f"{doc}\nValue: {value_expr}"

            note = make_note(
                front=html_escape(front),
                back=format_back(back),
                code=format_code(f"pub const {name}: u32 = {value_expr};"),
                source="crates/core/src/memory.rs",
                tags=["wafer::memory", "wafer::constants"],
            )
            deck.add_note(note)
            count += 1

            # Also generate reverse card for address-based constants
            if name.endswith("_BASE") or name.endswith("_TOP"):
                # Try to find hex value
                try:
                    val = eval(value_expr.replace("SYSVAR_BASE + ", "0 + ").replace("SYSVAR_BASE", "0"))
                except Exception:
                    val = None
                if isinstance(val, int):
                    rev_front = f"memory.rs: What region starts at 0x{val:04X}?"
                    rev_back = f"{name}: {doc}"
                    rev_note = make_note(
                        front=html_escape(rev_front),
                        back=format_back(rev_back),
                        source="crates/core/src/memory.rs",
                        tags=["wafer::memory", "wafer::constants", "wafer::reverse"],
                    )
                    deck.add_note(rev_note)
                    count += 1

        i += 1

    return count


# ---------------------------------------------------------------------------
# Auto-parse error variants from error.rs
# ---------------------------------------------------------------------------


def parse_errors(deck: genanki.Deck) -> int:
    """Parse WaferError enum from error.rs and generate cards."""
    err_file = CORE_SRC / "error.rs"
    if not err_file.exists():
        return 0

    content = err_file.read_text()
    count = 0

    # Match #[error("...")] followed by variant
    pattern = re.compile(r'#\[error\("(.+?)"\)\]\s*\n\s*(\w+)(?:\((.+?)\))?', re.MULTILINE)
    for m in pattern.finditer(content):
        msg = m.group(1)
        variant = m.group(2)
        inner = m.group(3) or ""

        front = f"WaferError::{variant} — when is this error raised?"
        back = f'Error message: "{msg}"'
        if inner:
            back += f"\nContains: {inner}"

        note = make_note(
            front=html_escape(front),
            back=format_back(back),
            code=format_code(f"WaferError::{variant}"),
            source="crates/core/src/error.rs",
            tags=["wafer::error"],
        )
        deck.add_note(note)
        count += 1

    return count


# ---------------------------------------------------------------------------
# Auto-extract peephole patterns from optimizer.rs
# ---------------------------------------------------------------------------


def parse_peephole_patterns(deck: genanki.Deck) -> int:
    """Extract peephole optimization patterns from optimizer.rs."""
    opt_file = CORE_SRC / "optimizer.rs"
    if not opt_file.exists():
        return 0

    content = opt_file.read_text()
    count = 0

    # Match comment + pattern in peephole_one_pass
    # Pattern: // Comment\n  (IrOp::X, IrOp::Y) => { ... }
    lines = content.split("\n")
    in_peephole = False
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if "fn peephole_one_pass" in line:
            in_peephole = True
        elif in_peephole and line.startswith("fn "):
            in_peephole = False

        if in_peephole:
            # Match pattern comments like: // PushI32(n), Drop => remove both
            comment_match = re.match(r"//\s*(.+?)\s*=>\s*(.+)", line)
            if comment_match:
                pattern = comment_match.group(1).strip()
                result = comment_match.group(2).strip()

                front = f"Peephole: {pattern} → ?"
                back = result

                note = make_note(
                    front=html_escape(front),
                    back=format_back(back),
                    source="crates/core/src/optimizer.rs",
                    tags=["wafer::optimizer", "wafer::peephole"],
                )
                deck.add_note(note)
                count += 1

        i += 1

    return count


# ---------------------------------------------------------------------------
# Auto-generate primitive registration cards
# ---------------------------------------------------------------------------


def parse_primitives(deck: genanki.Deck) -> int:
    """Extract IR primitive registrations from outer.rs."""
    outer_file = CORE_SRC / "outer.rs"
    if not outer_file.exists():
        return 0

    content = outer_file.read_text()
    count = 0

    # Match: self.register_primitive("NAME", false, vec![IrOp::X, IrOp::Y])?;
    pattern = re.compile(
        r'self\.register_primitive\("(.+?)",\s*(true|false),\s*vec!\[(.+?)\]\)',
        re.DOTALL,
    )

    for m in pattern.finditer(content):
        name = m.group(1)
        immediate = m.group(2) == "true"
        ir_body = m.group(3).strip()
        # Clean up multiline
        ir_body = " ".join(ir_body.split())

        front = f"Forth word {name} — what is its IR body?"
        back = f"IR: [{ir_body}]"
        if immediate:
            back += "\n(IMMEDIATE word)"

        note = make_note(
            front=html_escape(front),
            back=format_back(back),
            code=format_code(f'register_primitive("{name}", {immediate}, vec![{ir_body}])'),
            source="crates/core/src/outer.rs",
            tags=["wafer::primitives", "wafer::ir"],
        )
        deck.add_note(note)
        count += 1

    return count


# ---------------------------------------------------------------------------
# Auto-generate boot.fth definition cards
# ---------------------------------------------------------------------------


def parse_boot_fth(deck: genanki.Deck) -> int:
    """Extract Forth definitions from boot.fth."""
    if not BOOT_FTH.exists():
        return 0

    content = BOOT_FTH.read_text()
    count = 0

    lines = content.split("\n")
    i = 0
    current_comment = ""

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Track section comments
        if stripped.startswith("\\") and not stripped.startswith("\\ -------"):
            comment = stripped.lstrip("\\ ").strip()
            if comment:
                current_comment = comment

        # Match colon definitions
        if stripped.startswith(": "):
            # Collect full definition (may span multiple lines)
            defn = stripped
            while not defn.rstrip().endswith(";") and i + 1 < len(lines):
                i += 1
                defn += " " + lines[i].strip()

            # Extract name
            name_match = re.match(r":\s+(\S+)", defn)
            if name_match:
                name = name_match.group(1)

                front = f"boot.fth: How is {name} defined?"
                back = current_comment if current_comment else f"Forth definition of {name}"

                note = make_note(
                    front=html_escape(front),
                    back=format_back(back),
                    code=format_code(defn),
                    source="crates/core/boot.fth",
                    tags=["wafer::boot_fth"],
                )
                deck.add_note(note)
                count += 1

        i += 1

    return count


# ---------------------------------------------------------------------------
# Auto-parse Runtime trait methods from runtime.rs
# ---------------------------------------------------------------------------


def parse_runtime_trait(deck: genanki.Deck) -> int:
    """Parse Runtime and HostAccess trait methods from runtime.rs."""
    rt_file = CORE_SRC / "runtime.rs"
    if not rt_file.exists():
        return 0

    content = rt_file.read_text()
    count = 0

    # Match trait method signatures with doc comments
    lines = content.split("\n")
    i = 0
    current_trait = ""

    while i < len(lines):
        line = lines[i].strip()

        # Track which trait we're in
        trait_match = re.match(r"(?:pub\s+)?trait (\w+)", line)
        if trait_match:
            current_trait = trait_match.group(1)
            i += 1
            continue

        # Collect doc comments
        doc_lines = []
        while i < len(lines) and lines[i].strip().startswith("///"):
            doc_lines.append(lines[i].strip().lstrip("/ ").strip())
            i += 1

        if i >= len(lines):
            break

        line = lines[i].strip()

        # Check if this is a trait definition (may follow doc comments)
        trait_match = re.match(r"(?:pub\s+)?trait (\w+)", line)
        if trait_match:
            current_trait = trait_match.group(1)
            i += 1
            continue

        # Match fn signatures
        fn_match = re.match(r"fn (\w+)\(", line)
        if fn_match and doc_lines and current_trait:
            fn_name = fn_match.group(1)
            doc = " ".join(doc_lines)

            front = f"{current_trait}::{fn_name}() — what does it do?"
            back = doc

            note = make_note(
                front=html_escape(front),
                back=format_back(back),
                code=format_code(line.rstrip(";")),
                source="crates/core/src/runtime.rs",
                tags=["wafer::runtime", f"wafer::{current_trait.lower()}"],
            )
            deck.add_note(note)
            count += 1

        i += 1

    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate the Anki deck."""
    deck = genanki.Deck(DECK_ID, "WAFER - WebAssembly Forth Engine in Rust")

    print("Generating WAFER Anki deck...")
    print()

    # Load hand-crafted cards
    n = load_yaml_cards(deck)
    print(f"  Hand-crafted cards (YAML):     {n}")

    # Auto-generate from source
    n = parse_ir_ops(deck)
    print(f"  IrOp variant cards:            {n}")

    n = parse_memory_constants(deck)
    print(f"  Memory constant cards:         {n}")

    n = parse_errors(deck)
    print(f"  Error variant cards:           {n}")

    n = parse_peephole_patterns(deck)
    print(f"  Peephole pattern cards:        {n}")

    n = parse_primitives(deck)
    print(f"  Primitive registration cards:  {n}")

    n = parse_boot_fth(deck)
    print(f"  boot.fth definition cards:     {n}")

    n = parse_runtime_trait(deck)
    print(f"  Runtime trait method cards:     {n}")

    total = len(deck.notes)
    print(f"\n  TOTAL:                         {total} cards")

    # Write .apkg
    genanki.Package(deck).write_to_file(str(OUTPUT_FILE))
    print(f"\nWrote {OUTPUT_FILE}")
    print(f"Import into Anki: File > Import > select {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
