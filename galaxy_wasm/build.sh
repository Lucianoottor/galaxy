cargo build --target=wasm32-unknown-unknown --release
wasm-opt -O3 -o output.wasm target/wasm32-unknown-unknown/release/galaxy_wasm.wasm