//! Consolidation recompiler: merge all JIT-compiled words into a single WASM module.
//!
//! After interactive development, `CONSOLIDATE` recompiles everything:
//! - All `call_indirect` replaced with direct `call`
//! - Single WASM module output for maximum performance
//!
//! The implementation lives across two places:
//! - `codegen::compile_consolidated_module()` generates the multi-function WASM module
//! - `outer::ForthVM::consolidate()` orchestrates collection, compilation, and table update

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::codegen::compile_consolidated_module;
    use crate::dictionary::WordId;
    use crate::ir::IrOp;

    #[test]
    fn consolidated_module_validates_empty() {
        // Empty word list should produce nothing (but we guard against this at call site)
        let words = vec![];
        let map = HashMap::new();
        let result = compile_consolidated_module(&words, &map, 16);
        // Empty is valid -- should produce a valid module with no functions
        assert!(result.is_ok());
    }

    #[test]
    fn consolidated_module_validates_single_word() {
        let words = vec![(WordId(1), vec![IrOp::PushI32(42)])];
        let mut map = HashMap::new();
        map.insert(WordId(1), 1u32); // function index 1 (after emit import)
        let result = compile_consolidated_module(&words, &map, 16);
        assert!(result.is_ok());
    }

    #[test]
    fn consolidated_module_validates_multiple_words() {
        let words = vec![
            (WordId(1), vec![IrOp::PushI32(1)]),
            (WordId(2), vec![IrOp::PushI32(2), IrOp::Add]),
            (
                WordId(3),
                vec![IrOp::Call(WordId(1)), IrOp::Call(WordId(2))],
            ),
        ];
        let mut map = HashMap::new();
        map.insert(WordId(1), 1u32);
        map.insert(WordId(2), 2u32);
        map.insert(WordId(3), 3u32);
        let result = compile_consolidated_module(&words, &map, 16);
        assert!(result.is_ok());
    }

    #[test]
    fn consolidated_module_validates_external_call() {
        // Word 3 calls WordId(99) which is NOT in the module -- should use call_indirect
        let words = vec![(WordId(3), vec![IrOp::Call(WordId(99))])];
        let mut map = HashMap::new();
        map.insert(WordId(3), 1u32);
        let result = compile_consolidated_module(&words, &map, 256);
        assert!(result.is_ok());
    }

    #[test]
    fn consolidated_module_validates_tail_call() {
        let words = vec![
            (WordId(1), vec![IrOp::PushI32(1)]),
            (WordId(2), vec![IrOp::TailCall(WordId(1))]),
        ];
        let mut map = HashMap::new();
        map.insert(WordId(1), 1u32);
        map.insert(WordId(2), 2u32);
        let result = compile_consolidated_module(&words, &map, 16);
        assert!(result.is_ok());
    }

    #[test]
    fn consolidated_module_validates_control_flow_with_calls() {
        // IF body contains a call to a consolidated word
        let words = vec![
            (WordId(1), vec![IrOp::PushI32(1)]),
            (
                WordId(2),
                vec![
                    IrOp::PushI32(1),
                    IrOp::If {
                        then_body: vec![IrOp::Call(WordId(1))],
                        else_body: Some(vec![IrOp::PushI32(0)]),
                    },
                ],
            ),
        ];
        let mut map = HashMap::new();
        map.insert(WordId(1), 1u32);
        map.insert(WordId(2), 2u32);
        let result = compile_consolidated_module(&words, &map, 16);
        assert!(result.is_ok());
    }

    #[test]
    fn consolidated_module_validates_loop_with_calls() {
        // DO LOOP body contains a call to a consolidated word
        let words = vec![
            (WordId(1), vec![IrOp::PushI32(1), IrOp::Add]),
            (
                WordId(2),
                vec![
                    IrOp::PushI32(0),
                    IrOp::PushI32(3),
                    IrOp::PushI32(0),
                    IrOp::DoLoop {
                        body: vec![IrOp::Call(WordId(1))],
                        is_plus_loop: false,
                    },
                ],
            ),
        ];
        let mut map = HashMap::new();
        map.insert(WordId(1), 1u32);
        map.insert(WordId(2), 2u32);
        let result = compile_consolidated_module(&words, &map, 16);
        assert!(result.is_ok());
    }

    #[test]
    fn consolidated_module_validates_begin_until_with_calls() {
        let words = vec![
            (WordId(1), vec![IrOp::PushI32(1), IrOp::Sub]),
            (
                WordId(2),
                vec![
                    IrOp::PushI32(5),
                    IrOp::BeginUntil {
                        body: vec![IrOp::Call(WordId(1)), IrOp::Dup, IrOp::ZeroEq],
                    },
                ],
            ),
        ];
        let mut map = HashMap::new();
        map.insert(WordId(1), 1u32);
        map.insert(WordId(2), 2u32);
        let result = compile_consolidated_module(&words, &map, 16);
        assert!(result.is_ok());
    }

    #[test]
    fn consolidated_module_validates_begin_while_repeat_with_calls() {
        let words = vec![
            (WordId(1), vec![IrOp::PushI32(1), IrOp::Sub]),
            (
                WordId(2),
                vec![
                    IrOp::PushI32(3),
                    IrOp::BeginWhileRepeat {
                        test: vec![IrOp::Dup],
                        body: vec![IrOp::Call(WordId(1))],
                    },
                ],
            ),
        ];
        let mut map = HashMap::new();
        map.insert(WordId(1), 1u32);
        map.insert(WordId(2), 2u32);
        let result = compile_consolidated_module(&words, &map, 16);
        assert!(result.is_ok());
    }
}
