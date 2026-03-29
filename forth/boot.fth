\ WAFER Boot - Minimal bootstrap loaded first after primitives
\ This file defines the most fundamental derived words needed
\ before the rest of the standard library can load.

\ These words are defined in terms of the ~35 Rust primitives.
\ They form the foundation for core.fth and all subsequent files.

\ TODO: Step 7/8 - Populate with bootstrap definitions once
\ the compiler and outer interpreter are working.
\ For now this file documents what will go here:

\ Derived stack operations:
\ : NIP    ( x1 x2 -- x2 )       SWAP DROP ;
\ : TUCK   ( x1 x2 -- x2 x1 x2 ) SWAP OVER ;
\ : 2DUP   ( x1 x2 -- x1 x2 x1 x2 ) OVER OVER ;
\ : 2DROP  ( x1 x2 -- )          DROP DROP ;
\ : ?DUP   ( x -- 0 | x x )      DUP IF DUP THEN ;

\ Basic arithmetic derived from primitives:
\ : 1+     ( n -- n+1 )          1 + ;
\ : 1-     ( n -- n-1 )          1 - ;
\ : NEGATE ( n -- -n )           0 SWAP - ;
\ : ABS    ( n -- |n| )          DUP 0< IF NEGATE THEN ;
