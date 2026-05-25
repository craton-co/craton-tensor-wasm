fn main() {
    let wat = std::fs::read_to_string("bench-out/loop_sum.wat").unwrap();
    let wasm = wat::parse_str(&wat).unwrap();
    std::fs::write("bench-out/loop_sum.wasm", &wasm).unwrap();
    println!("wrote {} bytes", wasm.len());
}
