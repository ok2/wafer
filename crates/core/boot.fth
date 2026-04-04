\ WAFER Bootstrap -- Forth definitions replacing Rust host functions.
\ Loaded at startup after IR primitives are compiled.
\ Compiled WASM with direct calls outperforms host function dispatch.

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
: D<  D- D0< ;

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
: DU<  ROT 2DUP = IF 2DROP U< ELSE U< NIP NIP THEN ;

\ ---------------------------------------------------------------
\ Phase 3: Mixed arithmetic (built on M* and UM/MOD host primitives)
\ ---------------------------------------------------------------

\ Phase 3 words (SM/REM, FM/MOD, */, */MOD) kept as Rust host functions
\ for now due to return-stack depth interactions with DABS/DNEGATE.
\ TODO: replace once return-stack nesting issue is resolved.
