\ WAFER Bootstrap -- Forth definitions replacing Rust host functions.
\ Loaded at startup after IR primitives are compiled.
\ Compiled WASM with direct calls outperforms host function dispatch.

\ ---------------------------------------------------------------
\ Foundation: stack introspection (needed by 2OVER et al.)
\ ---------------------------------------------------------------

\ DEPTH ( -- n )  number of items on the data stack
\ SP@ must come first so it reads the dsp before DEPTH's own literal push.
\ DATA_STACK_TOP = 5632 (0x1600), uses arithmetic right shift for / 4
: DEPTH  SP@ 5632 SWAP - 2 RSHIFT ;

\ PICK ( xn..x0 n -- xn..x0 xn )  copy nth stack item
: PICK  1+ CELLS SP@ + @ ;

\ ---------------------------------------------------------------
\ Phase 1: Pure stack and memory operations
\ ---------------------------------------------------------------

\ 2OVER ( x1 x2 x3 x4 -- x1 x2 x3 x4 x1 x2 )
: 2OVER  3 PICK 3 PICK ;

\ 2ROT ( x1 x2 x3 x4 x5 x6 -- x3 x4 x5 x6 x1 x2 )
: 2ROT   2>R 2SWAP 2R> 2SWAP ;

\ WITHIN ( n lo hi -- flag )  true if lo <= n < hi (unsigned)
: WITHIN  OVER - >R - R> U< ;

\ 2@ ( addr -- x1 x2 )  fetch double-cell, low addr = deeper stack
: 2@  DUP CELL+ @ SWAP @ ;

\ 2! ( x1 x2 addr -- )  store double-cell
: 2!  SWAP OVER ! CELL+ ! ;

\ 2R@ stays as host function (needs direct return-stack access)

\ FILL ( addr u char -- )  fill u bytes with char
: FILL  ROT ROT  0 ?DO 2DUP I + C! LOOP 2DROP ;

\ CMOVE ( src dst u -- )  forward byte copy
: CMOVE  0 ?DO OVER I + C@ OVER I + C! LOOP 2DROP ;

\ CMOVE> ( src dst u -- )  backward byte copy (for overlap)
: CMOVE>
  DUP 0= IF DROP 2DROP EXIT THEN
  1- >R
  BEGIN R@ 0< INVERT WHILE
    OVER R@ + C@  OVER R@ + C!
    R> 1- >R
  REPEAT
  R> DROP 2DROP ;

\ MOVE ( src dst u -- )  smart copy (handles overlap)
: MOVE
  DUP 0= IF DROP 2DROP EXIT THEN
  >R 2DUP U< IF R> CMOVE> ELSE R> CMOVE THEN ;

\ ERASE ( addr u -- )  zero fill
: ERASE  0 FILL ;

\ BLANK ( addr u -- )  space fill
: BLANK  BL FILL ;

\ /STRING ( addr u n -- addr+n u-n )
: /STRING  ROT OVER + ROT ROT - ;

\ -TRAILING ( addr u -- addr u' )  remove trailing spaces
: -TRAILING
  BEGIN DUP 0> WHILE
    2DUP + 1- C@ BL <> IF EXIT THEN
    1-
  REPEAT ;

\ ---------------------------------------------------------------
\ Common extensions (not in Forth 2012, gforth-compatible)
\ ---------------------------------------------------------------

\ -ROT ( x1 x2 x3 -- x3 x1 x2 )  rotate top item to third place
: -ROT  ROT ROT ;

\ <= ( n1 n2 -- flag )  true if n1 <= n2 (signed)
: <=  > 0= ;

\ >= ( n1 n2 -- flag )  true if n1 >= n2 (signed)
: >=  < 0= ;

\ ---------------------------------------------------------------
\ Phase 2: Double-cell arithmetic
\ ---------------------------------------------------------------

\ D+ ( d1 d2 -- d3 )  double-cell addition with carry
: D+  >R SWAP >R  DUP >R + DUP R> U< IF R> R> + 1+ ELSE R> R> + THEN ;

\ DNEGATE ( d -- -d )  double-cell negate (two's complement)
: DNEGATE  INVERT SWAP INVERT SWAP  1 0 D+ ;

\ D- ( d1 d2 -- d3 )  double-cell subtraction
: D-  DNEGATE D+ ;

\ DABS ( d -- |d| )  double-cell absolute value
: DABS  DUP 0< IF DNEGATE THEN ;

\ D0= ( d -- flag )  true if d is zero
: D0=  OR 0= ;

\ D0< ( d -- flag )  true if d is negative
: D0<  NIP 0< ;

\ D= ( d1 d2 -- flag )  true if d1 = d2
: D=  D- D0= ;

\ D< ( d1 d2 -- flag )  true if d1 < d2 (signed)
\ Cannot use D- D0< because subtraction overflows for extreme values.
\ Compare high cells first (signed); if equal, compare low cells unsigned.
: D<  ROT 2DUP = IF 2DROP U< ELSE 2SWAP 2DROP > THEN ;

\ D2* ( d -- d*2 )  double-cell shift left
: D2*  2DUP D+ ;

\ D2/ ( d -- d/2 )  double-cell arithmetic shift right
: D2/  DUP 1 AND 31 LSHIFT >R  2/ SWAP 1 RSHIFT R> OR SWAP ;

\ DMAX ( d1 d2 -- d-max )  double-cell maximum
: DMAX  2OVER 2OVER D< IF 2SWAP THEN 2DROP ;

\ DMIN ( d1 d2 -- d-min )  double-cell minimum
: DMIN  2OVER 2OVER D< INVERT IF 2SWAP THEN 2DROP ;

\ M+ ( d n -- d+n )  add single to double
: M+  S>D D+ ;

\ DU< ( ud1 ud2 -- flag )  unsigned double-cell less-than
\ Compare high cells (unsigned); if equal, compare low cells unsigned.
: DU<  ROT 2DUP = IF 2DROP U< ELSE SWAP U< NIP NIP THEN ;

\ ---------------------------------------------------------------
\ Phase 3: Mixed arithmetic (built on M* and UM/MOD host primitives)
\ ---------------------------------------------------------------

\ SM/REM ( d n -- rem quot )  symmetric (truncated) division
\ Quotient sign: negative if dividend and divisor signs differ.
\ Remainder sign: same as dividend.
: SM/REM
  OVER >R
  2DUP XOR >R
  ABS >R DABS R>
  UM/MOD
  R> 0< IF NEGATE THEN
  SWAP R> 0< IF NEGATE THEN
  SWAP ;

\ FM/MOD ( d n -- rem quot )  floored division
: FM/MOD
  DUP >R
  SM/REM
  OVER 0<> OVER 0< AND IF
    1- SWAP R> + SWAP
  ELSE
    R> DROP
  THEN ;

\ */ ( n1 n2 n3 -- n4 )  n1*n2/n3 with double intermediate
\ Must use SM/REM (symmetric) to match WAFER's WASM i32.div_s semantics.
: */  >R M* R> SM/REM SWAP DROP ;

\ */MOD ( n1 n2 n3 -- rem quot )
: */MOD  >R M* R> SM/REM ;

\ ---------------------------------------------------------------
\ Phase 4: HERE and ALIGNED
\ ---------------------------------------------------------------

\ HERE reads from SYSVAR_HERE (offset 12 in WASM memory).
\ The Rust side syncs user_here to memory[12] before each evaluate call
\ and whenever host ALLOT/comma modifies it.
: HERE  12 @ ;

\ ALIGNED is already an IR primitive in the compiler.

\ ALLOT ( n -- )  advance HERE by n bytes
: ALLOT  HERE + 12 ! ;

\ , ( x -- )  store cell at HERE, advance by cell
: ,  HERE ! 1 CELLS ALLOT ;

\ C, ( char -- )  store byte at HERE, advance by 1
: C,  HERE C! 1 ALLOT ;

\ ALIGN ( -- )  align HERE to cell boundary
: ALIGN  HERE ALIGNED 12 ! ;

\ ---------------------------------------------------------------
\ Phase 5: I/O, pictured numeric output, formatted output
\ ---------------------------------------------------------------

\ TYPE ( c-addr u -- )  output u characters
: TYPE  0 ?DO DUP C@ EMIT 1+ LOOP DROP ;

\ SPACES ( n -- )  output n spaces (nothing for n <= 0, per 6.1.2230)
: SPACES  0 MAX 0 ?DO SPACE LOOP ;

\ Pictured numeric output constants
\ PICT_BUF_TOP = 0x05C0 = 1472, SYSVAR_HLD = 28

\ <# ( -- )  begin pictured numeric output
: <#  1472 28 ! ;

\ HOLD ( char -- )  add character to pictured output
: HOLD  28 @ 1- DUP 28 ! C! ;

\ HOLDS ( addr u -- )  add string to pictured output
: HOLDS
  BEGIN DUP 0> WHILE
    1- 2DUP + C@ HOLD
  REPEAT 2DROP ;

\ SIGN ( n -- )  if negative, add '-'
: SIGN  0< IF 45 HOLD THEN ;

\ # ( ud -- ud2 )  extract one digit from ud, convert to char, HOLD it.
\ Double-cell division by BASE using two UM/MODs:
\   First UM/MOD divides (0 ud-hi) by BASE -> rem1, quot-hi
\   Second UM/MOD divides (rem1 ud-lo) by BASE -> digit, quot-lo
: #
  BASE @
  >R 0 R@ UM/MOD R> SWAP >R
  UM/MOD
  SWAP DUP 9 > IF 7 + THEN 48 + HOLD
  R> ;

\ #S ( ud -- 0 0 )  convert all digits
: #S  BEGIN # 2DUP OR 0= UNTIL ;

\ #> ( ud -- c-addr u )  end pictured output, return string
: #>  2DROP 28 @ 1472 OVER - ;

\ Formatted output built on pictured numeric output

\ . ( n -- )  print signed number and space
: .  DUP ABS 0 <# #S ROT SIGN #> TYPE SPACE ;

\ U. ( u -- )  print unsigned number and space
: U.  0 <# #S #> TYPE SPACE ;

\ ? ( a-addr -- )  fetch and print
: ?  @ . ;

\ .R ( n width -- )  print right-justified signed number
: .R  >R DUP ABS 0 <# #S ROT SIGN #> R> OVER - SPACES TYPE ;

\ U.R ( u width -- )  print right-justified unsigned number
: U.R  >R 0 <# #S #> R> OVER - SPACES TYPE ;

\ D. ( d -- )  print signed double number and space
: D.  SWAP OVER DABS <# #S ROT SIGN #> TYPE SPACE ;

\ D.R ( d width -- )  print right-justified signed double
: D.R  >R SWAP OVER DABS <# #S ROT SIGN #> R> OVER - SPACES TYPE ;

\ ---------------------------------------------------------------
\ Return-stack introspection (debug aids)
\ ---------------------------------------------------------------

\ RDEPTH ( -- n )  number of cells on the return stack
\ RETURN_STACK_TOP = 9728 (0x2600). Only >R temps and loop params
\ live there; return addresses are on the WASM call stack.
: RDEPTH  9728 RP@ - 2 RSHIFT ;

\ .RS ( -- )  print the return stack bottom-to-top, like .S
\ Walks with BEGIN/WHILE (not DO) so the walk itself never pushes
\ onto the return stack it is printing.
: .RS  ." R:<" RDEPTH 0 .R ." > "
   9728 BEGIN DUP RP@ > WHILE 4 - DUP @ . REPEAT DROP ;

\ ---------------------------------------------------------------
\ Phase 6: DEFER support
\ ---------------------------------------------------------------

\ DEFER! ( xt xt-deferred -- )  store xt into a deferred word
: DEFER!  >BODY ! ;

\ DEFER@ ( xt-deferred -- xt )  retrieve xt from a deferred word
: DEFER@  >BODY @ ;

\ ---------------------------------------------------------------
\ String operations
\ ---------------------------------------------------------------

\ COMPARE ( addr1 u1 addr2 u2 -- n )  compare two strings lexicographically
: COMPARE
  ROT 2DUP SWAP - >R
  MIN 0 ?DO
    OVER I + C@
    OVER I + C@
    2DUP <> IF
      < IF 2DROP R> DROP -1 UNLOOP EXIT
      ELSE  2DROP R> DROP  1 UNLOOP EXIT
      THEN
    THEN 2DROP
  LOOP
  2DROP R>
  DUP 0< IF DROP -1 EXIT THEN
  0> IF 1 EXIT THEN
  0 ;

\ -TRAILING ( c-addr u1 -- c-addr u2 )  remove trailing spaces
: -TRAILING  BEGIN DUP WHILE 2DUP + 1- C@ BL = WHILE 1- REPEAT THEN ;

\ SEARCH stays as a host function (complex multi-line control flow).

\ ---------------------------------------------------------------
\ Phase 7: More easy replacements
\ ---------------------------------------------------------------

\ SOURCE ( -- c-addr u )  input buffer address and length
\ INPUT_BUFFER_BASE = 64, SYSVAR_NUM_TIB = 24
: SOURCE  64 24 @ ;

\ FALIGNED ( addr -- addr )  align to 8-byte float boundary
: FALIGNED  7 + -8 AND ;

\ SFALIGNED ( addr -- addr )  align to 4-byte single-float boundary
: SFALIGNED  3 + -4 AND ;

\ DFALIGNED ( addr -- addr )  align to 8-byte double-float boundary
: DFALIGNED  7 + -8 AND ;

\ FALIGN ( -- )  align HERE to 8-byte float boundary
: FALIGN  HERE FALIGNED 12 ! ;

\ SFALIGN ( -- )  align HERE to 4-byte single-float boundary
: SFALIGN  ALIGN ;

\ DFALIGN ( -- )  align HERE to 8-byte double-float boundary
: DFALIGN  FALIGN ;

\ .S keeps its Rust host function (complex stack introspection).

\ S ( "<spaces>name<space>" -- c-addr u )
\   State-smart string literal for the next whitespace-delimited token.
\   Handled in Rust (outer.rs interpret_token_immediate / compile_token)
\   so the string survives REFILL in interpret mode.

\ ---------------------------------------------------------------
\ Structures (Forth 2012 Facility-ext 10.6.2.0935 family)
\ ---------------------------------------------------------------
\ Usage:
\   BEGIN-STRUCTURE POINT  FIELD: P.X  FIELD: P.Y  END-STRUCTURE
\   CREATE ORIGIN POINT ALLOT
\   1 ORIGIN P.X !   2 ORIGIN P.Y !

\ Each defining word factored inline (CREATE .. DOES>). WAFER dispatches
\ DOES>-defining words only at the outer interpreter, so they can't be
\ factored through other compiled words (FIELD: -> +FIELD would no-op).

: BEGIN-STRUCTURE  ( "name" -- struct-sys 0 )
  CREATE HERE 0 0 , DOES> @ ;

: END-STRUCTURE  ( struct-sys +n -- )
  SWAP ! ;

: +FIELD   ( n1 "name" n2 -- n3 )
  CREATE OVER , + DOES> @ + ;

: FIELD:   ( n1 "name" -- n2 )
  CREATE ALIGNED DUP , 1 CELLS + DOES> @ + ;

: CFIELD:  ( n1 "name" -- n2 )
  CREATE DUP , 1 CHARS + DOES> @ + ;

: FFIELD:  ( n1 "name" -- n2 )
  CREATE FALIGNED DUP , 1 FLOATS + DOES> @ + ;

: SFFIELD: ( n1 "name" -- n2 )
  CREATE SFALIGNED DUP , 1 SFLOATS + DOES> @ + ;

: DFFIELD: ( n1 "name" -- n2 )
  CREATE DFALIGNED DUP , 1 DFLOATS + DOES> @ + ;
