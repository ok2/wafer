//! WASM module export: compile a Forth session to a standalone `.wasm` file.
//!
//! Orchestrates the export pipeline: collect IR words, resolve the entry point,
//! snapshot WASM memory, build metadata, and call the exportable codegen.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::codegen::{ExportSections, compile_exportable_module};
use crate::dictionary::WordId;
use crate::ir::IrOp;
use crate::outer::ForthVM;
use crate::runtime::Runtime;

/// Configuration for `wafer build`.
pub struct ExportConfig {
    /// Explicit entry-point word name (from `--entry` flag).
    pub entry_word: Option<String>,
}

/// Metadata embedded in the "wafer" custom section of exported modules.
pub struct ExportMetadata {
    /// Format version (currently 1).
    pub version: u32,
    /// Table index of the entry-point function, if any.
    pub entry_table_index: Option<u32>,
    /// Host functions referenced by consolidated code: (`table_index`, name).
    pub host_functions: Vec<(u32, String)>,
    /// Number of memory bytes in the data section snapshot.
    pub memory_size: u32,
    /// Initial data-stack pointer.
    pub dsp_init: u32,
    /// Initial return-stack pointer.
    pub rsp_init: u32,
    /// Initial float-stack pointer.
    pub fsp_init: u32,
}

/// Export the current VM state as a standalone WASM module.
///
/// Returns the raw `.wasm` bytes ready to write to a file, plus the metadata.
pub fn export_module(
    vm: &mut ForthVM<impl Runtime>,
    config: &ExportConfig,
) -> anyhow::Result<(Vec<u8>, ExportMetadata)> {
    let mut words = vm.ir_words();

    // Determine the entry point.
    // Priority: --entry flag > MAIN word > recorded top-level execution.
    let toplevel = vm.toplevel_ir().to_vec();
    let entry_word_id = if let Some(ref name) = config.entry_word {
        Some(
            vm.resolve_word(name)
                .ok_or_else(|| anyhow::anyhow!("entry word '{name}' not found"))?,
        )
    } else if let Some(main_id) = vm.resolve_word("MAIN") {
        Some(main_id)
    } else if !toplevel.is_empty() {
        // Synthesize a _start word from recorded top-level execution.
        // Pick a WordId that won't collide (one past the current table size).
        let start_id = WordId(vm.current_table_size());
        words.push((start_id, toplevel.clone()));
        Some(start_id)
    } else {
        None
    };

    if words.is_empty() {
        anyhow::bail!("nothing to export: no compiled words found");
    }

    // Build local_fn_map: WordId -> module-internal function index.
    // Imported functions occupy index 0 (emit), so defined functions start at 1.
    let mut local_fn_map = HashMap::new();
    for (i, (word_id, _)) in words.iter().enumerate() {
        local_fn_map.insert(*word_id, (i as u32) + 1);
    }

    // Resolve entry function index within the module.
    let entry_fn_index = entry_word_id.and_then(|id| local_fn_map.get(&id).copied());

    // Snapshot memory (system variables + user data).
    let memory_snapshot = vm.memory_snapshot();

    // Table size: must accommodate all WordIds including the synthetic _start.
    let max_word_id = words.iter().map(|(id, _)| id.0).max().unwrap_or(0);
    let table_size = (max_word_id + 1).max(vm.current_table_size());

    // Find host functions referenced by any consolidated word.
    let ir_word_ids: HashSet<WordId> = words.iter().map(|(id, _)| *id).collect();
    let mut referenced_host_ids: HashSet<WordId> = HashSet::new();
    for (_, body) in &words {
        collect_external_calls(body, &ir_word_ids, &mut referenced_host_ids);
    }

    let host_names = vm.host_function_names();
    let mut host_functions: Vec<(u32, String)> = referenced_host_ids
        .iter()
        .filter_map(|id| host_names.get(id).map(|name| (id.0, name.clone())))
        .collect();
    host_functions.sort_by_key(|(idx, _)| *idx);

    let (dsp_init, rsp_init, fsp_init) = vm.stack_pointer_inits();

    let metadata = ExportMetadata {
        version: 1,
        entry_table_index: entry_word_id.map(|id| id.0),
        host_functions,
        memory_size: memory_snapshot.len() as u32,
        dsp_init,
        rsp_init,
        fsp_init,
    };

    let metadata_json = serialize_metadata(&metadata);

    let export_sections = ExportSections {
        memory_snapshot: &memory_snapshot,
        entry_fn_index,
        metadata_json: metadata_json.as_bytes(),
    };

    let wasm_bytes = compile_exportable_module(&words, &local_fn_map, table_size, &export_sections)
        .map_err(|e| anyhow::anyhow!("export codegen error: {e}"))?;

    Ok((wasm_bytes, metadata))
}

/// Recursively collect `Call`/`TailCall` targets that are NOT in the IR word set
/// (i.e., they are host functions that the runner must provide).
fn collect_external_calls(ops: &[IrOp], ir_ids: &HashSet<WordId>, host_ids: &mut HashSet<WordId>) {
    for op in ops {
        match op {
            IrOp::Call(id) | IrOp::TailCall(id) if !ir_ids.contains(id) => {
                host_ids.insert(*id);
            }
            IrOp::If {
                then_body,
                else_body,
            } => {
                collect_external_calls(then_body, ir_ids, host_ids);
                if let Some(eb) = else_body {
                    collect_external_calls(eb, ir_ids, host_ids);
                }
            }
            IrOp::DoLoop { body, .. } | IrOp::BeginUntil { body } | IrOp::BeginAgain { body } => {
                collect_external_calls(body, ir_ids, host_ids);
            }
            IrOp::BeginWhileRepeat { test, body } => {
                collect_external_calls(test, ir_ids, host_ids);
                collect_external_calls(body, ir_ids, host_ids);
            }
            IrOp::BeginDoubleWhileRepeat {
                outer_test,
                inner_test,
                body,
                after_repeat,
                else_body,
            } => {
                collect_external_calls(outer_test, ir_ids, host_ids);
                collect_external_calls(inner_test, ir_ids, host_ids);
                collect_external_calls(body, ir_ids, host_ids);
                collect_external_calls(after_repeat, ir_ids, host_ids);
                if let Some(eb) = else_body {
                    collect_external_calls(eb, ir_ids, host_ids);
                }
            }
            _ => {}
        }
    }
}

/// Serialize export metadata to JSON (hand-rolled, no serde dependency).
pub fn serialize_metadata(m: &ExportMetadata) -> String {
    let mut s = String::from("{\n");
    let _ = writeln!(s, "  \"version\": {},", m.version);
    match m.entry_table_index {
        Some(idx) => {
            let _ = writeln!(s, "  \"entry_table_index\": {idx},");
        }
        None => {
            let _ = writeln!(s, "  \"entry_table_index\": null,");
        }
    }
    let _ = writeln!(s, "  \"memory_size\": {},", m.memory_size);
    let _ = writeln!(s, "  \"dsp_init\": {},", m.dsp_init);
    let _ = writeln!(s, "  \"rsp_init\": {},", m.rsp_init);
    let _ = writeln!(s, "  \"fsp_init\": {},", m.fsp_init);
    let _ = write!(s, "  \"host_functions\": [");
    for (i, (idx, name)) in m.host_functions.iter().enumerate() {
        if i > 0 {
            let _ = write!(s, ", ");
        }
        // Escape any quotes in the name (unlikely but safe).
        let escaped: String = name
            .chars()
            .flat_map(|c| if c == '"' { vec!['\\', '"'] } else { vec![c] })
            .collect();
        let _ = write!(s, "{{\"index\": {idx}, \"name\": \"{escaped}\"}}");
    }
    let _ = writeln!(s, "]");
    s.push('}');
    s
}

/// Deserialize export metadata from JSON (minimal parser for our known format).
pub fn deserialize_metadata(json: &str) -> anyhow::Result<ExportMetadata> {
    // Simple extraction by key -- works for our flat JSON structure.
    let get_u32 = |key: &str| -> anyhow::Result<u32> {
        let pat = format!("\"{key}\": ");
        let start = json
            .find(&pat)
            .ok_or_else(|| anyhow::anyhow!("missing key: {key}"))?
            + pat.len();
        let end = json[start..]
            .find([',', '\n', '}'])
            .map_or(json.len(), |i| start + i);
        json[start..end]
            .trim()
            .parse()
            .map_err(|e| anyhow::anyhow!("bad {key}: {e}"))
    };

    let get_optional_u32 = |key: &str| -> anyhow::Result<Option<u32>> {
        let pat = format!("\"{key}\": ");
        let Some(pos) = json.find(&pat) else {
            return Ok(None);
        };
        let start = pos + pat.len();
        let end = json[start..]
            .find([',', '\n', '}'])
            .map_or(json.len(), |i| start + i);
        let val = json[start..end].trim();
        if val == "null" {
            return Ok(None);
        }
        val.parse()
            .map(Some)
            .map_err(|e| anyhow::anyhow!("bad {key}: {e}"))
    };

    // Parse host_functions array
    let mut host_functions = Vec::new();
    if let Some(arr_start) = json.find("\"host_functions\": [") {
        let arr_start = arr_start + "\"host_functions\": [".len();
        let arr_end = json[arr_start..]
            .find(']')
            .map_or(json.len(), |i| arr_start + i);
        let arr = &json[arr_start..arr_end];

        // Parse each {"index": N, "name": "X"} object
        let mut pos = 0;
        while pos < arr.len() {
            if let Some(obj_start) = arr[pos..].find('{') {
                let obj_start = pos + obj_start;
                if let Some(obj_end) = arr[obj_start..].find('}') {
                    let obj = &arr[obj_start..obj_start + obj_end + 1];

                    // Extract index
                    if let Some(idx_start) = obj.find("\"index\": ") {
                        let idx_start = idx_start + "\"index\": ".len();
                        let idx_end = obj[idx_start..]
                            .find([',', '}'])
                            .map_or(obj.len(), |i| idx_start + i);
                        let idx: u32 = obj[idx_start..idx_end].trim().parse().unwrap_or(0);

                        // Extract name
                        if let Some(name_start) = obj.find("\"name\": \"") {
                            let name_start = name_start + "\"name\": \"".len();
                            if let Some(name_end) = obj[name_start..].find('"') {
                                let name = obj[name_start..name_start + name_end].to_string();
                                host_functions.push((idx, name));
                            }
                        }
                    }
                    pos = obj_start + obj_end + 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    Ok(ExportMetadata {
        version: get_u32("version")?,
        entry_table_index: get_optional_u32("entry_table_index")?,
        host_functions,
        memory_size: get_u32("memory_size")?,
        dsp_init: get_u32("dsp_init")?,
        rsp_init: get_u32("rsp_init")?,
        fsp_init: get_u32("fsp_init")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_roundtrip() {
        let m = ExportMetadata {
            version: 1,
            entry_table_index: Some(42),
            host_functions: vec![(5, ".".to_string()), (12, "TYPE".to_string())],
            memory_size: 65536,
            dsp_init: 5440,
            rsp_init: 9536,
            fsp_init: 11584,
        };
        let json = serialize_metadata(&m);
        let m2 = deserialize_metadata(&json).unwrap();
        assert_eq!(m2.version, 1);
        assert_eq!(m2.entry_table_index, Some(42));
        assert_eq!(m2.host_functions.len(), 2);
        assert_eq!(m2.host_functions[0], (5, ".".to_string()));
        assert_eq!(m2.host_functions[1], (12, "TYPE".to_string()));
        assert_eq!(m2.memory_size, 65536);
        assert_eq!(m2.dsp_init, 5440);
    }

    #[test]
    fn metadata_null_entry() {
        let m = ExportMetadata {
            version: 1,
            entry_table_index: None,
            host_functions: vec![],
            memory_size: 1024,
            dsp_init: 5440,
            rsp_init: 9536,
            fsp_init: 11584,
        };
        let json = serialize_metadata(&m);
        assert!(json.contains("\"entry_table_index\": null"));
        let m2 = deserialize_metadata(&json).unwrap();
        assert_eq!(m2.entry_table_index, None);
        assert!(m2.host_functions.is_empty());
    }

    #[test]
    fn collect_calls_finds_host_functions() {
        let ir_ids: HashSet<WordId> = [WordId(1), WordId(2)].iter().copied().collect();
        let body = vec![
            IrOp::Call(WordId(1)),  // IR word, not host
            IrOp::Call(WordId(99)), // host function
            IrOp::If {
                then_body: vec![IrOp::Call(WordId(50))], // host in nested body
                else_body: None,
            },
        ];
        let mut host = HashSet::new();
        collect_external_calls(&body, &ir_ids, &mut host);
        assert!(host.contains(&WordId(99)));
        assert!(host.contains(&WordId(50)));
        assert!(!host.contains(&WordId(1)));
    }

    /// Helper: evaluate Forth code, export to WASM, run, and return the output.
    fn roundtrip(source: &str) -> String {
        use crate::outer::ForthVM;
        use crate::runner::run_wasm_bytes;
        use crate::runtime_native::NativeRuntime;

        let mut vm = ForthVM::<NativeRuntime>::new().unwrap();
        vm.set_recording(true);
        vm.evaluate(source).unwrap();

        let config = ExportConfig { entry_word: None };
        let (wasm_bytes, _metadata) = export_module(&mut vm, &config).unwrap();
        run_wasm_bytes(&wasm_bytes).unwrap()
    }

    #[test]
    fn roundtrip_simple_dot() {
        assert_eq!(roundtrip(": main 42 . ;"), "42 ");
    }

    #[test]
    fn roundtrip_multiple_words() {
        assert_eq!(roundtrip(": double 2 * ; : main 21 double . ;"), "42 ");
    }

    #[test]
    fn roundtrip_variable() {
        assert_eq!(roundtrip("VARIABLE X  99 X !  : main X @ . ;"), "99 ");
    }

    #[test]
    fn roundtrip_emit() {
        assert_eq!(roundtrip(": main 72 EMIT 73 EMIT 10 EMIT ;"), "HI\n");
    }

    #[test]
    fn roundtrip_constant() {
        assert_eq!(roundtrip("42 CONSTANT ANSWER  : main ANSWER . ;"), "42 ");
    }

    #[test]
    fn roundtrip_toplevel_execution() {
        // No MAIN: top-level calls become the entry point.
        assert_eq!(roundtrip(": hello 42 . ; hello"), "42 ");
    }

    #[test]
    fn roundtrip_control_flow() {
        assert_eq!(roundtrip(": main 1 IF 42 ELSE 0 THEN . ;"), "42 ");
    }
}
