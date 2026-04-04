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
