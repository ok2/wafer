\ WAFER Core Word Set - High-level words defined in Forth
\ These implement Forth 2012 Core words that can be expressed
\ in terms of primitives and boot words.

\ TODO: Step 10 - Populate as Core compliance tests are run.
\ Each word here will be tested against forth2012-test-suite/src/core.fr

\ -- Derived stack operations --
\ : NIP   ( x1 x2 -- x2 )           SWAP DROP ;
\ : TUCK  ( x1 x2 -- x2 x1 x2 )    SWAP OVER ;
\ : 2DUP  ( x1 x2 -- x1 x2 x1 x2 ) OVER OVER ;
\ : 2DROP ( x1 x2 -- )              DROP DROP ;
\ : 2SWAP ( x1 x2 x3 x4 -- x3 x4 x1 x2 ) ROT >R ROT R> ;
\ : 2OVER ( x1 x2 x3 x4 -- x1 x2 x3 x4 x1 x2 ) >R >R 2DUP R> R> 2SWAP ;

\ -- Arithmetic --
\ : 1+      1 + ;
\ : 1-      1 - ;
\ : 2*      1 LSHIFT ;
\ : 2/      1 RSHIFT ;
\ : NEGATE  0 SWAP - ;
\ : ABS     DUP 0< IF NEGATE THEN ;
\ : MIN     2DUP > IF SWAP THEN DROP ;
\ : MAX     2DUP < IF SWAP THEN DROP ;
\ : MOD     /MOD DROP ;
\ : /       /MOD NIP ;

\ -- Comparison --
\ : 0=   0 = ;
\ : 0<>  0= 0= ;
\ : 0<   0 < ;
\ : 0>   0 SWAP < ;
\ : <>   = 0= ;

\ -- Memory --
\ : +!     DUP @ ROT + SWAP ! ;
\ : CELLS  4 * ;
\ : CELL+  4 + ;
\ : CHARS  ;
\ : CHAR+  1+ ;

\ -- Defining words --
\ : VARIABLE  CREATE 0 , ;
\ : CONSTANT  CREATE , DOES> @ ;

\ -- I/O --
\ : SPACE   BL EMIT ;
\ : SPACES  0 ?DO SPACE LOOP ;
