# The Unreasonable Effectiveness of Stack Machines

_How Forth — and WAFER — can serve as infrastructure for data analytics,
databases, AI inference, AI code generation, and AI agent control._

---

Forth is 55 years old. It has no type system, no garbage collector, no package
manager, no syntax to speak of. By most conventional measures, it shouldn't
still be relevant.

But it keeps showing up at the edges — in firmware, in space probes, in
real-time systems, in places where correctness and determinism matter more than
developer ergonomics. That's worth paying attention to.

The properties that make Forth unusual — concatenative composition, zero-cost
abstraction through word definition, a stack-based execution model that maps
directly to hardware — happen to line up surprisingly well with what five of
the most active areas in modern computing are independently reaching for:

1. **Data analytics** wants composable, streaming pipelines.
2. **Database engines** want stack-based virtual machines for query execution.
3. **AI inference** wants tiny, deterministic, embeddable runtimes.
4. **AI code generation** wants the smallest possible target language.
5. **AI agent systems** want plans that are also executable programs.

Forth won't single-handedly solve any of these. But it offers a useful lens
for understanding what each of them actually needs — and WAFER, a Forth that
compiles to WebAssembly, is in a good position to explore that space.

WAFER (WebAssembly Forth Engine in Rust) JIT-compiles each Forth word to its
own WASM module, linked through shared linear memory, globals, and a function
table. It runs anywhere WASM runs: browsers, edge devices, servers, embedded
systems. It has 160+ words, 100% Forth 2012 compliance on 10 word sets, and
fits in ~50 KB. It has exception handling (`CATCH`/`THROW`), metaprogramming
(`DOES>`), dynamic compilation (`EVALUATE`), and an optimization pipeline
designed for stack-to-local promotion that can achieve 7x speedups.

This document explores what becomes possible when you take these properties
seriously.

---

## 1. Data Analytics: Pipelines Without Plumbing

### The Problem with Pipelines

Every data analytics framework reinvents the same idea: take data, push it
through a sequence of transformations, collect the result. Pandas chains
methods. Spark builds DAGs. dplyr pipes with `%>%`. Unix pipes bytes through
`|`. They all converge on the same shape: **linear composition of operations
on an implicit data flow**.

This is exactly what Forth does. It has done it since 1970. The data stack
_is_ the pipeline. Each word _is_ a transformation. Composition is
juxtaposition — you don't pipe, you don't chain, you don't bind. You just
write the words next to each other.

```forth
\ Pandas: df['amount'].where(df['amount'] > 0).mean()
\ Forth:
: POSITIVE?  ( n -- n flag )  DUP 0> ;
: FILTER-POSITIVE  ( addr n -- addr' n' )
    0 >R  0 >R   \ count and sum accumulators on return stack
    0 DO
        DUP I CELLS + @
        POSITIVE? IF  R> + >R  R> 1+ >R  THEN
    LOOP DROP
    R> R>  \ ( sum count )
;
: MEAN  ( sum count -- avg )  / ;

data 100 FILTER-POSITIVE MEAN .
```

This goes a bit deeper than syntactic sugar. The absence of intermediate
variables is a structural property. In a Pandas chain, every `.method()`
returns a new DataFrame object that must be allocated, tracked, and eventually
collected. In Forth, the data flows through the stack with zero allocation.
The pipeline _is_ the execution.

### Streaming and Incremental Computation

The stack model is inherently streaming. A word consumes its inputs and
produces its outputs in the same motion. There is no "collect all data first,
then process" step unless you explicitly build one. This makes Forth natural
for:

- **Event stream processing**: each event lands on the stack, a word
  processes it, the result is consumed by the next word.
- **Incremental aggregation**: running sums, counts, and statistics
  maintained on the return stack across invocations.
- **Windowed computation**: a circular buffer in linear memory with
  stack-based access patterns.

```forth
\ Running average over a stream of values
VARIABLE running-sum
VARIABLE running-count

: UPDATE-AVG  ( new-value -- running-avg )
    running-sum @ +  DUP running-sum !
    running-count @ 1+  DUP running-count !
    /
;

\ Each incoming value:
42 UPDATE-AVG .    \ prints running average after adding 42
17 UPDATE-AVG .    \ prints updated average after adding 17
```

### Client-Side Analytics via WASM

WAFER compiles to WebAssembly. This means analytics can run _in the browser_
with no server round-trips. A user uploads a CSV, WAFER parses and processes
it entirely client-side, and the results render immediately. No data leaves
the machine. No API calls. No latency.

This isn't just a nice demo. For privacy-sensitive analytics (healthcare,
finance, GDPR-regulated data), client-side processing can be a compliance
requirement, not just a nice-to-have. WAFER's deterministic execution (no GC
pauses, no background threads, fixed memory layout) makes it predictable
enough for real-time dashboards.

### Domain-Specific Languages

Forth's defining feature is that you build the language up to your problem.
An analytics team doesn't write Forth — they write _their DSL_, which
happens to be implemented in Forth:

```forth
\ Define a mini analytics vocabulary
: COLUMN  ( col# -- addr n )  table-base SWAP col-offset + col-length ;
: SUM     ( addr n -- total )  0 ROT ROT 0 DO  OVER I CELLS + @ +  LOOP NIP ;
: COUNT   ( addr n -- n )      NIP ;
: AVG     ( addr n -- avg )    2DUP SUM -ROT COUNT / ;
: WHERE>  ( addr n thresh -- addr' n' )  filter-gt ;

\ The analyst writes:
3 COLUMN  1000 WHERE>  AVG .
\ "Average of column 3 where values exceed 1000"
```

The DSL compiles to WASM through WAFER's IR pipeline. There is no
interpreter overhead at query time. The analyst's vocabulary _is_ the
optimized code.

### A Different Way to Look at It

Most languages treat the absence of named variables as a limitation. But in
data pipelines, it can actually be a **feature**. Named intermediates create
coupling points — places where code can refer to stale state, where
refactoring requires renaming, where parallelization requires dependency
analysis. Point-free composition through a stack sidesteps this whole class
of problems. The data is always _here_, on top of the stack, ready for the
next transformation.

---

## 2. Database Engine: The Query VM You Already Have

### Databases Already Think in Stacks

SQLite — the most deployed database engine in the world — executes queries
through the VDBE (Virtual Database Engine), a stack-based bytecode virtual
machine. When you write `SELECT * FROM users WHERE age > 30`, SQLite's query
planner compiles it into a sequence of stack operations: open cursor, seek,
compare, jump, emit row.

PostgreSQL's executor runs a tree of plan nodes, each of which pushes tuples
upward. MySQL's handler interface is a stack of operations. CockroachDB
compiles SQL to a vectorized execution engine that operates on batches — but
the control flow is still a stack of operators.

There's a pattern here: **query execution engines tend to converge on
stack machines**. Forth just happens to already be one, with no extra
abstraction layers in between.

### Query Plans as Forth Programs

A SQL query plan is a tree. Flattened into execution order, it becomes a
sequence of operations — which is exactly a Forth program:

```sql
SELECT name, salary FROM employees WHERE dept = 'ENG' AND salary > 100000;
```

The query plan, expressed as Forth:

```forth
\ Primitives provided by the storage engine
\ SCAN        ( table -- cursor )
\ NEXT-ROW    ( cursor -- cursor flag )  flag=true if row available
\ COL@        ( cursor col# -- value )
\ EMIT-ROW    ( v1 v2 -- )              send to result set
\ CLOSE       ( cursor -- )

: MATCH-DEPT?   ( cursor -- cursor flag )  DUP 2 COL@ S" ENG" COMPARE 0= ;
: MATCH-SAL?    ( cursor -- cursor flag )  DUP 3 COL@ 100000 > ;
: PROJECT       ( cursor -- )             DUP 0 COL@  OVER 3 COL@  EMIT-ROW ;

: QUERY  ( -- )
    employees SCAN
    BEGIN
        NEXT-ROW
    WHILE
        MATCH-DEPT? IF
            MATCH-SAL? IF
                PROJECT
            THEN
        THEN
    REPEAT
    CLOSE
;
```

This isn't just pseudocode, either. Every word here could be a real WAFER
word backed by storage primitives implemented as host functions. The query
compiles through WAFER's IR pipeline to native WASM, with the same
optimization opportunities as any other Forth word: inlining, constant
folding, dead code elimination.

### EVALUATE as Dynamic Query Compilation

SQL databases accept queries as strings and compile them at runtime. Forth
has `EVALUATE`, which does exactly the same thing — takes a string and
compiles/executes it:

```forth
\ Build a query string dynamically
S" employees SCAN BEGIN NEXT-ROW WHILE MATCH-DEPT? IF PROJECT THEN REPEAT CLOSE"
EVALUATE
```

The difference from SQL: the "query language" and the "implementation
language" are the same. There is no impedance mismatch between the language
the user writes queries in and the language the engine executes them in. A
user-defined function is just another word. An index lookup is just another
word. A join strategy is just another word. They all compose the same way.

### Linear Memory as Storage Pages

WAFER's linear memory model maps directly to how databases manage storage.
A database page is a fixed-size block of bytes at a known offset — exactly
what Forth's `@` and `!` operate on. B-tree nodes are structures in linear
memory traversed by pointer arithmetic:

```forth
\ B-tree node layout:
\   +0: key count (cell)
\   +4: is-leaf flag (cell)
\   +8: keys array (key-count cells)
\   +8+4*key-count: child pointers (key-count+1 cells)

: NODE-KEYS    ( node -- addr )  8 + ;
: NODE-KEY@    ( node i -- key )  CELLS SWAP NODE-KEYS + @ ;
: NODE-CHILD@  ( node i -- child )
    OVER NODE-KEYS
    OVER @ CELLS +    \ skip past keys array
    SWAP CELLS + 4 +  \ index into children
    @
;

: BTREE-SEARCH  ( node target-key -- addr|0 )
    OVER @ 0= IF  2DROP 0  EXIT  THEN  \ empty node
    OVER 4 + @ IF                        \ leaf node
        LEAF-SEARCH
    ELSE
        INTERNAL-SEARCH                  \ recurse into child
    THEN
;
```

### WASM Sandboxing for User-Defined Functions

Safely executing user-defined functions (UDFs) is one of the trickier
problems in database engines. PostgreSQL UDFs in C can crash the server.
JavaScript UDFs require embedding V8. Python UDFs tend to be slow.

WAFER UDFs compile to WASM and execute in a sandbox with bounded memory,
bounded execution time, and no access to anything outside the linear memory
they're given. A malicious UDF can't read other users' data, can't make
network calls, can't crash the host. WAFER gets this for free — it's
inherent to WASM's security model.

```forth
\ User defines a custom scoring function
: SCORE  ( age salary -- score )
    1000 /         \ salary contribution (salary/1000)
    SWAP 50 - ABS  \ age penalty (distance from 50)
    -              \ final score
;

\ Engine uses it in a query
: RANKED-QUERY  ( -- )
    employees SCAN
    BEGIN NEXT-ROW WHILE
        DUP 1 COL@  OVER 3 COL@  SCORE
        50 > IF  PROJECT  THEN
    REPEAT CLOSE
;
```

The `SCORE` function compiles to a WASM module through WAFER's JIT. It runs
at near-native speed, sandboxed, with no FFI overhead.

### A Different Way to Look at It

Database engineers put a lot of effort into building query VMs — designing
bytecode formats, writing interpreters, adding JIT compilation. In a sense,
they're often reinventing something Forth-shaped each time. It's worth asking:
what if you just started with Forth and built the storage layer underneath it?

---

## 3. AI Inference: Neural Networks as Word Composition

### Layers Are Words, Forward Pass Is Composition

A neural network's forward pass is a pipeline: input tensor enters, passes
through a sequence of layers (linear transform, activation, normalization),
and a prediction exits. Each layer takes a tensor and produces a tensor.

In Forth terms: each layer is a word. The tensor sits on the stack. The
forward pass is the composition of those words:

```forth
\ Assuming tensor operations as primitives (host functions):
\ T-MATMUL  ( tensor weights -- tensor )
\ T-ADD     ( tensor bias -- tensor )
\ T-RELU    ( tensor -- tensor )
\ T-SOFTMAX ( tensor -- tensor )

: LINEAR1    ( tensor -- tensor )  w1 T-MATMUL b1 T-ADD ;
: LINEAR2    ( tensor -- tensor )  w2 T-MATMUL b2 T-ADD ;
: LINEAR3    ( tensor -- tensor )  w3 T-MATMUL b3 T-ADD ;

: CLASSIFIER ( tensor -- tensor )
    LINEAR1 T-RELU
    LINEAR2 T-RELU
    LINEAR3 T-SOFTMAX
;

input-data CLASSIFIER  \ forward pass
```

This maps more directly than you might expect. The compositional structure of
neural networks lines up nicely with the compositional structure of Forth
programs. The stack carries the data flow. The words are the layers. The
dictionary holds the model architecture.

### Quantized Inference on the Integer Stack

Most production inference runs quantized — INT8 or INT4 weights, integer
arithmetic, no floating point. Forth's native data type is the integer cell.
WAFER's `i32` stack operations map directly to quantized tensor operations:

```forth
\ INT8 quantized dot product of two vectors
: QDOT  ( addr1 addr2 n -- result )
    0 >R                         \ accumulator on return stack
    0 DO
        OVER I + C@  127 -       \ load and de-bias first element
        OVER I + C@  127 -       \ load and de-bias second element
        *  R> + >R               \ multiply-accumulate
    LOOP
    2DROP R>
;

\ Quantized linear layer
: QLINEAR  ( input-addr weight-addr rows cols -- output-addr )
    \ For each output neuron, compute QDOT with input
    output-buf >R
    0 DO
        2DUP I row-offset + SWAP QDOT
        R@ I CELLS + !
    LOOP
    2DROP R>
;
```

No framework dependency, no Python interpreter, no CUDA runtime — just
integer arithmetic on a stack, compiled to WASM, running on any device.

### Edge AI: The 50 KB Runtime

ML inference frameworks tend to be big. PyTorch is ~500 MB. TensorFlow Lite
is ~1 MB for the runtime alone. ONNX Runtime is ~10 MB.

WAFER is ~50 KB for the full Forth system. The model weights dominate the
binary size, not the runtime. For edge devices — IoT sensors, wearables,
microcontrollers, browser tabs — that size difference can be the difference
between "fits" and "doesn't fit."

WASM's portability means the same inference code runs on an ARM
microcontroller, in a browser, on a server, without recompilation. Write the
model once in Forth, deploy everywhere WASM reaches.

### DOES> for Architecture Generation

Forth's `DOES>` is a metaprogramming facility: it creates words that create
other words, each with custom runtime behavior. This is exactly what neural
architecture construction needs:

```forth
\ LAYER is a defining word that creates layer words
: LAYER  ( weights bias rows cols -- )
    CREATE  , , , ,           \ store dimensions and pointers
    DOES>   ( tensor -- tensor )
        DUP >R                \ save parameter field address
        R@ @  R@ 4 + @       \ get cols, rows
        R@ 8 + @              \ get weights address
        T-MATMUL
        R> 12 + @             \ get bias address
        T-ADD
;

\ Define the network architecture
w1 b1 768 512  LAYER EMBED
w2 b2 512 256  LAYER HIDDEN1
w3 b3 256 10   LAYER OUTPUT

\ The architecture is now executable
: MODEL  ( tensor -- tensor )  EMBED T-RELU HIDDEN1 T-RELU OUTPUT T-SOFTMAX ;
```

Each `LAYER` invocation creates a new word with its own weights and
dimensions baked in. The `MODEL` word composes them. This is the same
pattern as `nn.Sequential` in PyTorch — but it compiles to WASM, has zero
framework overhead, and the "architecture definition" and the "executable
model" are the same thing.

### Automatic Differentiation via Dual Numbers

Backpropagation is reverse-mode automatic differentiation. There is an
elegant formulation using dual numbers (a value paired with its derivative)
that maps to Forth's double-cell operations:

```forth
\ A dual number is a pair ( value derivative ) stored as a double cell
\ WAFER's double-cell words (D+, D-, D*, 2DUP, etc.) operate on these natively

\ Dual addition: (a, a') + (b, b') = (a+b, a'+b')
: D+DUAL  ( a a' b b' -- a+b a'+b' )
    ROT +          \ a' + b'
    >R + R>        \ a + b, then restore derivative
;

\ Dual multiplication: (a, a') * (b, b') = (a*b, a*b' + a'*b)
: D*DUAL  ( a a' b b' -- a*b a*b'+a'*b )
    3 PICK *       \ a * b'
    >R
    ROT *          \ a' * b
    R> +           \ a*b' + a'*b = derivative
    >R
    *              \ a * b = value
    R>
;
```

The chain rule emerges naturally: composing dual-number operations through a
sequence of words automatically computes the derivative of the whole
pipeline. This is the same principle behind JAX's `jvp` — but expressed as
stack operations.

### A Different Way to Look at It

Most of the ML ecosystem's complexity lives in _training_. Inference, by
comparison, is fairly straightforward: load weights, multiply matrices, apply
activations, read output. That's a pipeline of arithmetic operations — which
is pretty much what Forth was designed for. The industry tends to wrap
inference in 500 MB frameworks because training needed those frameworks, and
the two haven't been fully separated. A 50 KB Forth runtime doing quantized
integer operations might be closer to what inference actually needs than we
usually assume.

---

## 4. AI Generating Code: The Smallest Target Language

### The Token Economy

When an LLM generates code, every token costs money and adds latency. A
Python solution to "compute the average of a list" looks like:

```python
def average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
```

That is 25 tokens. The Forth equivalent:

```forth
: AVERAGE  ( addr n -- avg )  2DUP SUM -ROT NIP / ;
```

That is 12 tokens. For the same semantic content, Forth uses roughly half
the tokens. At scale — millions of API calls, each generating hundreds of
lines — this is a meaningful cost reduction. But the token savings are the
least interesting advantage.

### Minimal Syntax, Maximal Verifiability

Forth has essentially no syntax. There are words separated by spaces. There
are numbers. There are a few special constructs (`:` for definitions, `IF`
/`THEN` for conditionals, `DO`/`LOOP` for iteration). That's about it.

An LLM generating Python must get indentation right, match parentheses and
brackets, handle keyword arguments, manage import statements, respect method
resolution order, and navigate a standard library of thousands of functions.
An LLM generating Forth mostly just needs to get the stack effect right.
That's the main failure mode worth worrying about.

And stack effects are **mechanically verifiable**:

```forth
\ Stack effect: ( n1 n2 -- n3 )
\ Verification: start with 2 items on stack, end with 1
: ADD-AND-DOUBLE  ( n1 n2 -- n3 )  + 2* ;

\ Test:
3 4 ADD-AND-DOUBLE   \ stack should contain: 14
```

You don't need a type checker or static analysis. Just run the word with
known inputs and check the stack. If the stack depth and values match the
declared effect, the word is correct. It's hard to think of another practical
language where verification is this straightforward.

### Self-Extending Vocabulary

LLMs struggle with large codebases because context windows are finite. A
Python project with 50 files and 10,000 lines requires the LLM to hold (or
retrieve) vast amounts of context to generate correct code.

Forth's defining characteristic is that you build the language up to your
problem. The LLM doesn't need to generate a 100-line solution. It generates
5-line words, each building on the previous ones:

```forth
\ Step 1: LLM generates basic operations
: CLAMP  ( n lo hi -- n' )  ROT MIN MAX ;
: BETWEEN?  ( n lo hi -- flag )  OVER - >R - R> U< ;

\ Step 2: LLM generates higher-level operations using step 1
: NORMALIZE  ( n -- n' )  0 255 CLAMP ;
: IN-RANGE?  ( n -- flag )  0 100 BETWEEN? ;

\ Step 3: LLM generates application logic using steps 1-2
: PROCESS-SENSOR  ( raw -- calibrated )
    offset @ -          \ remove sensor offset
    NORMALIZE            \ clamp to valid range
    scale @ *  1000 /    \ apply calibration scale
;
```

Each step requires only the _names_ of previously defined words, not their
implementations. The dictionary serves as a compressed representation of the
entire program. An LLM can generate correct code by knowing only the word
names and their stack effects — a few dozen tokens of context instead of
thousands of lines.

### WASM Sandbox: Safe Execution of Untrusted Code

AI-generated code generally needs to be executed to be verified. Running
arbitrary Python is tricky from a security perspective — file system access,
network calls, `import os`, `eval()`. Sandboxing Python typically requires
containerization, seccomp filters, or virtual machines.

WAFER compiles to WASM, which executes in a sandbox by construction. A
WAFER program:

- Cannot access the file system
- Cannot make network calls
- Cannot read memory outside its linear memory
- Cannot execute longer than the host allows (fuel metering)
- Cannot consume more memory than the host allocates

You can run AI-generated Forth with roughly the same confidence as a pure
mathematical function. The sandbox isn't a bolt-on — it's just how WASM
works.

```forth
\ AI generates this code. Is it safe to run? Yes, always.
: FIBONACCI  ( n -- fib )
    DUP 2 < IF EXIT THEN
    DUP 1- RECURSE
    SWAP 2 - RECURSE
    +
;
```

There's nothing this word can do except compute. No side effects, no
escape hatches. The WASM sandbox guarantees that structurally.

### A Different Way to Look at It

The conventional wisdom is that LLMs need expressive, high-level languages
to generate useful code. But there's a good case for the opposite: what LLMs
really benefit from are **verifiable** languages — ones where correctness can
be checked cheaply and deterministically. Expressiveness can actually work
against you here: more syntax means more ways to be wrong, more edge cases
to handle, more context to maintain. Forth's extreme minimalism starts to
look less like a limitation and more like an advantage: generate a few small
words, verify each one by running it, compose them into larger programs with
confidence. The language that's hardest for humans to read might just be the
easiest for machines to write correctly.

---

## 5. AI Agent Control: Plans That Execute Themselves

### The Plan-Program Gap

When an AI agent "plans," it produces a sequence of steps in natural
language:

> 1. Search for files matching "*.config"
> 2. Read each file and extract the "timeout" field
> 3. If timeout > 30, update it to 30
> 4. Write the modified files back

This plan is then "executed" by the agent interpreting each step, calling
tools, handling errors, and managing state — all mediated by the LLM at
every step, consuming tokens and latency for what is fundamentally a
sequential program.

The gap between "plan" and "program" might be more artificial than it looks.
A plan _is_ a program — we just don't usually give agents a good executable
representation for it.

Forth could be that representation.

### Tools as Words

Every agent tool — file read, web search, code execution, API call — maps
to a Forth word. The agent's toolkit becomes a Forth dictionary:

```forth
\ Agent tool vocabulary (host functions)
\ SEARCH-FILES  ( pattern-addr pattern-len -- results-addr count )
\ READ-FILE     ( path-addr path-len -- content-addr content-len )
\ WRITE-FILE    ( content-addr content-len path-addr path-len -- )
\ JSON-GET      ( json-addr key-addr key-len -- value-addr value-len )
\ SHELL         ( cmd-addr cmd-len -- output-addr output-len )
\ ASK-USER      ( question-addr question-len -- answer-addr answer-len )
```

Now the plan from above becomes an executable program:

```forth
: UPDATE-TIMEOUTS  ( -- )
    S" *.config" SEARCH-FILES       \ get matching files
    0 DO                             \ for each file
        DUP I CELLS + @ COUNT        \ get filename
        2DUP READ-FILE               \ read contents
        S" timeout" JSON-GET         \ extract timeout field
        S>NUMBER DROP                \ convert to number
        30 > IF                      \ if timeout > 30
            30 SET-TIMEOUT           \ update to 30
            WRITE-FILE               \ write back
        ELSE
            2DROP                    \ discard unchanged
        THEN
    LOOP
    DROP
;

UPDATE-TIMEOUTS
```

This isn't a description of what to do — it _is_ what to do. The agent
generates it, WAFER compiles it to WASM, and it runs — no LLM in the loop
during execution, no token cost per step, no latency per tool call.

### Error Handling with CATCH/THROW

Of course, agent plans fail. Files don't exist. APIs return errors.
Permissions get denied. Production agent systems need robust error handling,
which typically means calling the LLM at every step to decide what to do
when something goes wrong.

WAFER has `CATCH` and `THROW` — structured exception handling that lets
the plan itself define error recovery:

```forth
: SAFE-READ  ( path-addr path-len -- content-addr content-len | 0 0 )
    ['] READ-FILE CATCH IF
        2DROP  0 0                   \ file not found: return empty
    THEN
;

: SAFE-UPDATE  ( filename-addr filename-len -- )
    2DUP SAFE-READ                   \ try to read
    DUP 0= IF  2DROP 2DROP EXIT THEN \ skip if file missing
    S" timeout" JSON-GET
    S>NUMBER DROP
    30 > IF
        30 SET-TIMEOUT
        WRITE-FILE
    ELSE
        2DROP 2DROP
    THEN
;

: ROBUST-UPDATE-TIMEOUTS  ( -- )
    S" *.config" SEARCH-FILES
    0 DO
        DUP I CELLS + @ COUNT SAFE-UPDATE
    LOOP
    DROP
;
```

The error handling is part of the plan. The agent generates it once, and it
runs to completion without further LLM intervention. Errors are handled at
the speed of WASM, not the speed of an API call to an LLM.

### The Dictionary as Growing Capability

A human Forth programmer builds up vocabulary: small words compose into
larger words, which compose into still larger words. The dictionary grows
with the programmer's understanding of the problem.

An AI agent does the same thing. Each successfully executed plan leaves
behind defined words that can be reused:

```forth
\ First task: agent learns to read configs
: READ-CONFIG  ( path-addr path-len -- json-addr json-len )
    SAFE-READ DUP 0= IF EXIT THEN JSON-PARSE ;

\ Second task: agent learns to update configs
: UPDATE-CONFIG  ( key-addr key-len value path-addr path-len -- )
    2DUP READ-CONFIG JSON-SET WRITE-FILE ;

\ Third task: agent composes previous capabilities
: MIGRATE-CONFIGS  ( -- )
    S" *.config" SEARCH-FILES
    0 DO
        DUP I CELLS + @ COUNT
        S" timeout" 30 ROT ROT UPDATE-CONFIG
    LOOP DROP
;

\ The agent's vocabulary grows with experience.
\ MIGRATE-CONFIGS didn't exist before. Now it does.
\ Next time, the agent can use it as a building block.
```

You could call this _learned tool use_ — not in the machine learning sense,
but in the software engineering sense. The agent defines new capabilities in
terms of old ones, and the dictionary persists across invocations. Over time,
the agent's vocabulary naturally converges on the abstractions that matter
for its operational domain.

### REPL as Test-Before-Commit

Agents that act irreversibly on the first try are risky. WAFER's REPL model
gives agents a natural test-before-commit workflow:

1. **Define**: Generate and compile the plan as Forth words.
2. **Test**: Run the words against sample data on the stack.
3. **Verify**: Check the stack for expected results.
4. **Execute**: Run the plan for real only after verification passes.

```forth
\ Step 1: Define
: CALCULATE-DISCOUNT  ( price tier -- discounted )
    CASE
        1 OF  10 ENDOF   \ tier 1: 10% off
        2 OF  20 ENDOF   \ tier 2: 20% off
        3 OF  35 ENDOF   \ tier 3: 35% off
        0 SWAP
    ENDCASE
    100 SWAP - * 100 /
;

\ Step 2: Test (no side effects, just stack operations)
1000 1 CALCULATE-DISCOUNT .  \ expect 900
1000 2 CALCULATE-DISCOUNT .  \ expect 800
1000 3 CALCULATE-DISCOUNT .  \ expect 650

\ Step 3: Verify output matches expectations
\ Step 4: Apply to real data only after tests pass
```

The agent can generate, test, and iterate without ever touching production
data. The REPL isn't just a debugging convenience here — it's a safety mechanism
for autonomous agents.

### Multi-Agent Coordination

Multiple agents can share a WAFER dictionary through shared linear memory.
One agent defines words. Another agent uses them. A coordinator agent
composes them into higher-level plans:

```forth
\ Agent A defines data retrieval
: FETCH-METRICS  ( -- addr n )  metrics-api QUERY PARSE-JSON ;

\ Agent B defines analysis
: DETECT-ANOMALIES  ( addr n -- anomalies-addr n )
    THRESHOLD @  FILTER-ABOVE ;

\ Agent C defines actions
: ALERT  ( anomalies-addr n -- )
    0 DO  DUP I CELLS + @  SEND-ALERT  LOOP DROP ;

\ Coordinator composes them
: MONITOR  ( -- )
    BEGIN
        FETCH-METRICS DETECT-ANOMALIES
        DUP 0> IF  ALERT  ELSE  DROP  THEN
        60000 DELAY
    AGAIN
;
```

Each agent contributes words to a shared vocabulary. The coordinator doesn't
need to understand the implementation of `FETCH-METRICS` or
`DETECT-ANOMALIES` — it only needs to know their stack effects. This is
composability without coupling, coordination without shared state beyond
the dictionary.

### A Different Way to Look at It

The AI agent community is building increasingly sophisticated "plan
representations" — DAGs, state machines, behavior trees, ReAct loops — all
trying to bridge the gap between the LLM's natural language output and
actual tool execution. But Forth is already a plan representation that
doubles as an execution engine. It has structured control flow (`IF`/`THEN`,
`DO`/`LOOP`, `BEGIN`/`UNTIL`), error handling (`CATCH`/`THROW`),
composability (word definitions), and a test harness (the REPL and stack).
Maybe the gap between "plan" and "program" doesn't need to be bridged so
much as it needs to be _erased_.

---

## Convergence: Five Problems, One Shape

These five domains look different on the surface:

| Domain          | Traditional Tool               | Core Operation       |
| --------------- | ------------------------------ | -------------------- |
| Data analytics  | Pandas, Spark                  | Transform pipeline   |
| Database engine | SQLite VDBE, Postgres executor | Query plan execution |
| AI inference    | PyTorch, TensorFlow            | Layer composition    |
| AI codegen      | Python, JavaScript             | Program synthesis    |
| AI agents       | LangChain, CrewAI              | Plan execution       |

But they share a deep structure: **sequential composition of simple
operations on a data flow**. A data pipeline, a query plan, a forward
pass, a synthesized program, and an agent plan are all the same thing:
a sequence of words applied to a stack.

Forth noticed this in 1970. Charles Moore designed a language around the
observation that most computation is a pipeline of transformations, and
the simplest way to express pipelines is sequential composition on a
stack. The language has no syntax because pipelines don't need syntax.
It has no type system because the data flow _is_ the type. It has no
package manager because each program builds its own vocabulary from
primitives.

WAFER brings these ideas to the modern world by targeting WebAssembly — the
universal runtime that runs in browsers, on servers, on edge devices, in
sandboxes. That combination opens up some interesting possibilities:

- **Analytics in the browser** with no server, no framework, deterministic
  execution.
- **Database VMs** that compile queries to native WASM through an existing
  Forth JIT.
- **Inference engines** that fit in 50 KB and run on any device WASM
  reaches.
- **AI-generated code** in the language with the smallest syntax, cheapest
  verification, and safest sandbox.
- **Agent plans** that are executable programs, testable in a REPL,
  composable through a growing dictionary.

None of this requires Forth to change. Forth has been this shape for 55
years. It's kind of fun that the world's problems seem to be circling back
to it.

---

_WAFER is open source. Start at the [repository root](../README.md)._
_Architecture details: [WAFER.md](WAFER.md). Language introduction:
[FORTH.md](FORTH.md)._
