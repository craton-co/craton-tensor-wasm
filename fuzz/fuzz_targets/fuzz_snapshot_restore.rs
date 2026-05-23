#![no_main]

use libfuzzer_sys::fuzz_target;

use bali_snapshot::reader::SnapshotReader;

fuzz_target!(|data: &[u8]| {
    // Cap input length: snapshot validation enforces a max size on real
    // input; we mirror it here so the fuzzer doesn't waste cycles on 100 MiB
    // inputs that would fail immediately anyway.
    if data.len() > 64 * 1024 * 1024 {
        return;
    }
    let reader = SnapshotReader::new();
    let _ = reader.restore(data);
});
