#!/bin/bash
set -e

cd "$(dirname "$0")"

for cmd in cargo wasm-opt; do
    command -v $cmd >/dev/null || { echo "missing: $cmd"; exit 1; }
done

rustup target list --installed | grep -q wasm32-unknown-unknown || {
    echo "missing: rustup target add wasm32-unknown-unknown"; exit 1
}

cargo build --target=wasm32-unknown-unknown --release

wasm-opt -O3 -o output.wasm target/wasm32-unknown-unknown/release/galaxy_wasm.wasm

echo "output.wasm: $(wc -c < output.wasm | tr -d ' ') bytes"
