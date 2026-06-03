#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Cap input size to keep iteration rate high.
    if data.len() > 64 * 1024 {
        return;
    }
    let config = wasmtime::Config::new();
    let Ok(engine) = wasmtime::Engine::new(&config) else {
        return;
    };
    let _ = wasmtime::Module::from_binary(&engine, data);
});
