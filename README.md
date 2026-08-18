# Galaxy WASM

<div align="center">
  <img src="public/galaxy.gif" alt="Galaxy WASM simulation" />
</div>

N-body gravitational simulation in Rust, compiled to WebAssembly with SIMD.

Barnes-Hut quadtree for O(n log n) performance.

## Quick Start

```console
$ git clone <repo>
$ cd galaxy_wasm
$ ./build.sh
$ python3 src/server.py
```

Then open [http://localhost:8000](http://localhost:8000)

## Dependencies

- Rust + `wasm32-unknown-unknown` target
- [binaryen](https://github.com/WebAssembly/binaryen) (`wasm-opt`)

```console
$ rustup target add wasm32-unknown-unknown
$ brew install binaryen
```
