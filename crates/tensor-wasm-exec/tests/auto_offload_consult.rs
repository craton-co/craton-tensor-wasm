//! End-to-end check that `tensor_wasm_exec::auto_offload::analyse` runs against
//! the shared `matrix_multiply.wat` fixture and emits at least one
//! verdict. The exact verdict mix is *not* asserted — that's the detector's
//! job; we only confirm the consultation hook is wired and reaches the
//! detector on real Wasm.

use std::path::PathBuf;

#[test]
fn analyse_matrix_multiply_fixture() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("wasm-fixtures")
        .join("matrix_multiply.wat");
    let wat = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("read fixture {}: {e}", path.display());
    });
    let wasm = wat::parse_str(&wat).expect("parse wat");
    let verdicts = tensor_wasm_exec::auto_offload::analyse(&wasm).expect("analyse");
    assert!(
        !verdicts.is_empty(),
        "expected at least one function verdict"
    );
    let total_ops: usize = verdicts.iter().map(|v| v.op_count).sum();
    assert!(total_ops > 0, "expected at least one op walked");
}
