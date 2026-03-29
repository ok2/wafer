\ WAFER Core Extensions Word Set
\ Forth 2012 Section 6.2

\ TODO: Step 13 - Implement as compliance tests are enabled
\ : VALUE     CREATE , DOES> @ ;
\ : TO        ' >BODY ! ;
\ : DEFER     CREATE ['] ABORT , DOES> @ EXECUTE ;
\ : DEFER!    >BODY ! ;
\ : DEFER@    >BODY @ ;
\ : IS        STATE @ IF POSTPONE ['] POSTPONE DEFER! ELSE ' DEFER! THEN ; IMMEDIATE
\ : ACTION-OF STATE @ IF POSTPONE ['] POSTPONE DEFER@ ELSE ' DEFER@ THEN ; IMMEDIATE
