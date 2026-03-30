# Forth

## What is Forth?

Forth is a programming language that you build while you use it.

Created by Charles H. Moore in the late 1960s, Forth starts from almost nothing -- whitespace-delimited tokens, two stacks, and a dictionary -- and lets the programmer construct everything else. First developed in 1968, it was put to work controlling radio telescopes at the National Radio Astronomy Observatory by 1971, proving that a language could be both minimal and powerful enough for real-time scientific instrumentation. There is no fixed syntax, no grammar to speak of. There are only _words_: named operations that the programmer defines in terms of other words, all the way down to the machine itself. A Forth program does not describe computation within a language; it extends the language until the language describes the computation.

```forth
: square  ( n -- n*n )  dup * ;
5 square .   \ prints 25
```

This is not merely extensibility in the way other languages mean it. In most languages, you write code _within_ the language. In Forth, you write the language and then the code is the language. The compiler, the interpreter, the control flow, the data structures -- all are words in the same dictionary, redefinable by the same mechanism. The distinction between "built-in" and "user-defined" does not exist.

The essentials:

- **Postfix notation.** Operands come before operators: `3 4 +` leaves `7` on the stack.
- **Two stacks.** The _data stack_ holds working values; the _return stack_ holds control flow addresses and temporary storage.
- **Words.** Every operation is a named dictionary entry. `: square dup * ;` defines a new word from existing ones.
- **Interactive.** The outer interpreter reads tokens, looks them up, and either executes or compiles them. Development is incremental and live.
- **Dual mode.** Interpret mode executes immediately; compile mode appends to the current definition. Immediate words execute even during compilation, enabling compile-time metaprogramming.
- **Minimal syntax.** Tokens separated by whitespace. That is the entire grammar.

## A Different Axis

Most programmers think of languages on a single spectrum: low-level on one end, high-level on the other. Assembly gives you the machine, but you do everything yourself. Python gives you abstractions, but you give up the machine. C sits somewhere in the middle -- close enough to the hardware to write operating systems, high enough to express algorithms without counting registers.

Forth does not sit on this spectrum. It occupies a different axis entirely.

Like assembly language, Forth has no compiler-enforced type system. There are no hidden allocations, no garbage collector, no implicit function calls, no virtual machine mediating between your code and the hardware. Every stack operation is visible. When you write `@ ! + DUP`, you are one step from the machine. On an actual Forth processor -- and people have built them, from the Novix NC4000 in the 1980s to the GreenArrays GA144 today -- primitives like these _are_ the instruction set.

And yet.

Like a high-level domain-specific language, Forth lets you reshape the language itself until it speaks the vocabulary of your problem. A Forth program that controls a robot does not call `moveServo(angle, speed)` through seven layers of abstraction. It says `30 DEGREES LEFT-ARM MOVE` -- and those words compile directly to the motor control instructions, because the programmer built the language that way. The abstraction _is_ the compilation. There is no gap between what you say and what the machine does.

This is the fundamental difference between Forth and everything that came after it.

In C and C++, abstraction costs indirection: function calls, pointer dereferences, virtual dispatch. Each layer you add puts distance between your intent and the machine. In Python or Java, abstraction is the whole point, but you pay for it with a runtime, a virtual machine, garbage collection pauses -- machinery whose behavior you cannot see or control.

In Forth, abstraction is cheap and uniform. Defining a new word does not add a new kind of overhead. It _compiles_. The word `square` becomes a subroutine call to the instructions for `dup` and `*` -- or, in an optimizing Forth, the instructions are inlined directly. Either way, there is no hidden machinery between what you write and what the machine executes. You can build castles of abstraction, and when you look at the generated code, there are no castles -- just the stones.

This is why Forth fits so uncomfortably in taxonomies. It is not a high-level language with low-level escape hatches. It is not a low-level language with macro facilities. It is a system where the distance between the problem domain and the machine is always zero, because the programmer constructs the path, and the path is the program.

As FORTH, Inc. puts it: Forth was designed for "a programmer who was intelligent, highly skilled, and professional; it was intended to empower, not constrain." Forth does not protect you from yourself. It empowers. And it demands that the programmer be worth empowering.

## Where Forth Lives Today

Forth is not mainstream. It never tried to be. But it persists in places where its particular nature -- tiny, deterministic, interactive, self-contained -- is exactly what the situation demands.

**Space.** The Philae lander, which touched down on comet 67P/Churyumov-Gerasimenko in 2014 as part of ESA's Rosetta mission, ran its central command and data management system in Forth-83 on radiation-hardened RTX2010 stack processors. When you are 500 million kilometers from the nearest debugger and your computer has kilobytes of memory, you want a language where you can see every instruction and test every word interactively before committing it to flight.

**Firmware.** Open Firmware, the boot ROM standard used by Apple, Sun, IBM, and the OLPC XO-1, is a Forth environment. Before the operating system loads -- before there are drivers, before there is a filesystem -- there is a Forth interpreter probing hardware, initializing buses, and running platform-independent device drivers encoded as FCode (a compiled Forth bytecode format). Forth is the language you reach for when there is nothing else yet.

**Embedded systems.** Industrial controllers, scientific instruments, real-time signal processing. Anywhere the constraints are tight and the programmer needs to know what every microsecond is doing.

**Why people still choose it.** Forth's entire compiler can fit in a few kilobytes. It needs no operating system. It is interactive from the first instruction -- you can test words on live hardware as you write them. It compiles in microseconds. It is deterministic: no garbage collection pauses, no background threads, no surprises. And it runs on anything, from a chip with 2 KB of ROM to a modern desktop, because a minimal Forth kernel is so small that you can port it in a weekend.

The question is not "Why would anyone still use Forth?" The question is: "What other language gives you all of that?"

## Forth and WebAssembly

WebAssembly is a stack machine: instructions take their operands from the stack, not from registers or named variables. It has structured control flow and modules as the unit of compilation.

Forth is a stack language with a strikingly similar model: operands live on the stack, control flow is structured, and words are the unit of compilation.

The correspondence is not a coincidence, but it goes deeper than shared ancestry. It is structural. The things that make Forth what it is -- stack-based evaluation, compilation to small independent units, subroutine threading -- map directly onto the things WebAssembly provides.

Consider how Forth compiles a word. The compiler reads tokens, resolves them to dictionary entries, and emits a sequence of calls to those entries -- subroutine threading. In a traditional Forth on native hardware, these are `CALL` instructions to machine code. On WebAssembly, they are `call` and `call_indirect` instructions referencing entries in a function table. The mechanism is identical; only the target changes.

Consider how Forth modules compose. Each word is self-contained: it takes values from the stack, does work, and leaves results on the stack. There are no registers, no calling conventions, no ABI to negotiate. WebAssembly modules compose the same way: shared linear memory, a stack-based instruction set, imports and exports with simple signatures. Linking a new Forth word into the system is linking a new WASM module into the runtime.

Even the limitations align. WebAssembly forbids unstructured jumps -- no `goto`, no computed branches into the middle of functions. Forth's standard control flow (`IF/ELSE/THEN`, `BEGIN/UNTIL`, `DO/LOOP`) is already structured. Where languages with `goto` need a "relooper" pass to restructure their control flow graphs for WASM, Forth's structured constructs map naturally.

Forth may be the most natural source language for WebAssembly -- not because it was designed for it, but because both were designed around the same idea: a stack machine that compiles small, composable units of computation.

## Forth 2012

**Forth 2012** is the current standard for the Forth language, ratified in 2012. It supersedes ANS Forth 1994 (ANSI X3.215-1994) and organizes words into word sets:

| Word Set          | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| Core              | The essential ~130 words every compliant system must provide |
| Core Extensions   | Optional but commonly implemented additions to Core          |
| Block             | Mass storage using 1024-byte blocks                          |
| Double            | Double-cell (64-bit) integer arithmetic                      |
| Exception         | `CATCH` and `THROW` for structured error handling            |
| Facility          | Terminal/system interaction (`KEY?`, `EKEY`, etc.)           |
| File              | File I/O words                                               |
| Floating          | Floating-point arithmetic                                    |
| Locals            | Named local variables                                        |
| Memory            | Memory allocation (`ALLOCATE`, `FREE`, `RESIZE`)             |
| Search-Order      | Multiple wordlists and vocabulary control                    |
| String            | String manipulation utilities                                |
| Programming-Tools | Debugging and introspection (`SEE`, `WORDS`, `.S`)           |

Key improvements over ANS Forth 1994: `PARSE` and `PARSE-NAME` for cleaner input parsing, `S\"` with escape sequences, `DEFER`/`IS` for deferred execution, `BUFFER:` for data buffers, and clarified semantics throughout.

- **Online:** <https://forth-standard.org/>
- **PDF:** <http://www.forth200x.org/documents/forth-2012.pdf>

## WAFER -- This Implementation

**WAFER** (WebAssembly Forth Engine in Rust) is an optimizing Forth 2012 compiler that JIT-compiles each Forth word to its own WebAssembly module and executes it via [wasmtime](https://wasmtime.dev/).

### Key Parameters

| Parameter          | Value                                   |
| ------------------ | --------------------------------------- |
| Cell size          | 4 bytes (32-bit, WASM `i32`)            |
| Double-cell size   | 8 bytes (two cells)                     |
| Float size         | 8 bytes (`f64`)                         |
| Address unit       | 1 byte (byte-addressed)                 |
| Data stack depth   | 1024 cells                              |
| Return stack depth | 1024 cells                              |
| Float stack depth  | 256 entries                             |
| Linear memory      | 1 MiB initial, 16 MiB max               |
| Input buffer       | 1024 bytes                              |
| Number base        | Configurable (`DECIMAL`, `HEX`, `BASE`) |
| Character set      | ASCII / UTF-8 input                     |

### Architecture

```
Forth source --> Outer Interpreter --> IR (Vec<IrOp>) --> WASM codegen --> wasmtime
                                                              |
                                                   shared memory + globals
                                                   + function table (call_indirect)
```

- **Subroutine-threaded** via WASM function tables and `call_indirect`
- Each word compiles to a separate WASM module linked to shared memory, globals (data/return stack pointers), and a function table
- IR-based pipeline enables optimization passes before WASM emission
- Dictionary is a linked list in linear memory (simulated in a `Vec<u8>` buffer)
- Primitives are either IR-based (compiled to WASM) or host functions (Rust closures)

### Current Status

- 70+ words implemented, 219 tests passing
- Core word set ~90% complete
- Full control flow: `IF/ELSE/THEN`, `DO/LOOP/+LOOP`, `BEGIN/UNTIL/WHILE/REPEAT`
- Defining words: `:`, `VARIABLE`, `CONSTANT`, `CREATE`, `DOES>`
- Interactive REPL with line editing

See the [README](../README.md) for build instructions and the full word list.

## Recommended Reading

### Starting Forth -- Leo Brodie (1981, 2nd ed. 1987)

The classic introduction. Covers stack manipulation, arithmetic, definitions, conditionals, loops, and I/O with clear illustrations and humor. The best first book for anyone learning Forth.

**Free PDF:** <https://www.forth.com/wp-content/uploads/2018/01/Starting-FORTH.pdf>
**Online:** <https://www.forth.com/starting-forth/>

### Thinking Forth -- Leo Brodie (1984, revised 2004)

Not a language tutorial but a book about _problem-solving_ and _software design_ in Forth. Covers factoring, decomposition, naming conventions, and the philosophy behind writing clean Forth. Valuable for any programmer regardless of language.

**Free PDF (CC BY-NC-SA 2.0):** <https://thinking-forth.sourceforge.net/>
**Color edition:** <https://www.forth.com/wp-content/uploads/2018/11/thinking-forth-color.pdf>

### Programming a Problem-Oriented Language -- Charles H. Moore (1970)

Moore's original manuscript describing the design and implementation of Forth. Dense and technical, but essential reading for understanding why Forth is the way it is -- directly from the mind that created it.
