// Manual wasm export example: no wasm_bindgen, no wasm-pack.
// Build: ./build.sh  (cargo build --target wasm32-unknown-unknown --release, then wasm-opt)
// JS reads/writes the exported functions + raw memory directly.

#![cfg_attr(target_arch = "wasm32", no_std, no_main)]
#![allow(static_mut_refs)] // single-threaded JS host calls us one at a time; fine here.

extern crate alloc;
use alloc::vec::Vec;
use core::arch::wasm32::*;

// no_std has no heap by default; `Vec` needs one. dlmalloc is a small
// no_std-friendly allocator that works on wasm32-unknown-unknown.
#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

const G: f32 = 0.000003; // scaled for a -1..1 box; 9.8 (real-world-ish) is way too strong at this scale
const FPS: f32 = 60.0;
const DT: f32 = 1.0 / FPS;
// Plummer softening: added to d^2 before the sqrt/divide. Keeps close
// encounters (and i==j self-interaction, where dx=dy=0) from dividing by
// ~zero, without an `if i == j { continue }` branch breaking the SIMD loop.
const SOFTEN2: f32 = 0.05 * 0.05;
// Barnes-Hut opening angle: a node is treated as one point-mass when
// node.size / distance < THETA. 0 = always recurse to real particles
// (exact, same as brute force). Larger = more approximation, more speed.
const THETA: f32 = 0.6;
const MAX_DEPTH: u32 = 28; // guards against near-infinite recursion on (near-)coincident points
const NONE: u32 = u32::MAX;

// Struct-of-Arrays: each field is its own contiguous Vec<f32>, so 4
// particles' worth of x (or y, or mass, ...) load into a single v128 SIMD
// lane register in one instruction. An array-of-structs (Vec<Particle>)
// can't do this cheaply because x0,y0,x1,y1... are interleaved in memory.
struct Particles {
    x: Vec<f32>,
    y: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    ax: Vec<f32>,
    ay: Vec<f32>,
    mass: Vec<f32>,
}

static mut PARTICLES: Particles = Particles {
    x: Vec::new(),
    y: Vec::new(),
    vx: Vec::new(),
    vy: Vec::new(),
    ax: Vec::new(),
    ay: Vec::new(),
    mass: Vec::new(),
};

// A quadtree node, stored by index in a flat arena (ARENA below) instead of
// via Box/pointers: Vec<Node> can reallocate on push, which would invalidate
// real pointers/references but not indices - indices stay valid regardless.
#[derive(Clone, Copy)]
struct Node {
    x0: f32,
    y0: f32,
    size: f32, // bounding square: [x0, x0+size) x [y0, y0+size)
    com_x: f32,
    com_y: f32,
    mass: f32,          // aggregate mass of every particle under this node
    children: [u32; 4], // arena indices, NONE if absent (bottom-left, bottom-right, top-left, top-right)
    particle: u32,      // NONE unless this node is currently a single-particle leaf
}

const EMPTY_NODE: Node = Node {
    x0: 0.0,
    y0: 0.0,
    size: 0.0,
    com_x: 0.0,
    com_y: 0.0,
    mass: 0.0,
    children: [NONE; 4],
    particle: NONE,
};

// Rebuilt from scratch every tick (particles move every frame anyway); kept
// as a persistent static and `.clear()`-ed rather than a fresh Vec each call
// so the backing heap allocation is reused instead of alloc/free-churned.
static mut ARENA: Vec<Node> = Vec::new();

fn quadrant(node: &Node, x: f32, y: f32) -> usize {
    let mx = node.x0 + node.size * 0.5;
    let my = node.y0 + node.size * 0.5;
    match (x >= mx, y >= my) {
        (false, false) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (true, true) => 3,
    }
}

fn child_bounds(node: &Node, q: usize) -> (f32, f32, f32) {
    let half = node.size * 0.5;
    let x0 = node.x0 + if q == 1 || q == 3 { half } else { 0.0 };
    let y0 = node.y0 + if q == 2 || q == 3 { half } else { 0.0 };
    (x0, y0, half)
}

fn insert(arena: &mut Vec<Node>, node_idx: u32, p: u32, x: &[f32], y: &[f32], mass: &[f32], depth: u32) {
    let (px, py, pm) = (x[p as usize], y[p as usize], mass[p as usize]);

    // Every node on the path to a leaf accumulates the mass-weighted center
    // of mass of everything under it - that running aggregate is what makes
    // "treat this whole node as one point mass" a valid approximation later.
    let was_empty = {
        let node = &mut arena[node_idx as usize];
        let was_empty = node.mass == 0.0 && node.particle == NONE && node.children[0] == NONE;
        let new_mass = node.mass + pm;
        node.com_x = (node.com_x * node.mass + px * pm) / new_mass;
        node.com_y = (node.com_y * node.mass + py * pm) / new_mass;
        node.mass = new_mass;
        was_empty
    };

    if was_empty {
        arena[node_idx as usize].particle = p;
        return;
    }

    if depth >= MAX_DEPTH {
        // (near-)coincident particles that would recurse forever - merge
        // them into this node's aggregate mass/COM (already done above) and
        // stop, rather than subdividing an already-degenerate region.
        return;
    }

    let has_children = arena[node_idx as usize].children[0] != NONE;
    if !has_children {
        // This node currently holds exactly one particle as a leaf - split
        // it into 4 children and push the existing occupant down first.
        let existing = arena[node_idx as usize].particle;
        arena[node_idx as usize].particle = NONE;

        let node_snapshot = arena[node_idx as usize];
        let mut child_idx = [NONE; 4];
        for q in 0..4 {
            let (cx0, cy0, csize) = child_bounds(&node_snapshot, q);
            arena.push(Node { x0: cx0, y0: cy0, size: csize, ..EMPTY_NODE });
            child_idx[q] = (arena.len() - 1) as u32;
        }
        arena[node_idx as usize].children = child_idx;

        let eq = quadrant(&node_snapshot, x[existing as usize], y[existing as usize]);
        insert(arena, child_idx[eq], existing, x, y, mass, depth + 1);
    }

    let node_snapshot = arena[node_idx as usize];
    let q = quadrant(&node_snapshot, px, py);
    insert(arena, node_snapshot.children[q], p, x, y, mass, depth + 1);
}

// Accumulate the force on particle i by walking the tree: distant clusters
// get approximated as one point mass (node.com_x/com_y/mass), only nearby
// regions get opened up and recursed into individual particles.
fn compute_force(arena: &[Node], node_idx: u32, i: usize, x: &[f32], y: &[f32]) -> (f32, f32) {
    let node = arena[node_idx as usize];
    if node.mass == 0.0 {
        return (0.0, 0.0);
    }

    let dx = node.com_x - x[i];
    let dy = node.com_y - y[i];
    let d2 = dx * dx + dy * dy + SOFTEN2;
    let d = libm::sqrtf(d2);

    let is_leaf = node.children[0] == NONE;
    // Self-interaction (node is the leaf holding particle i itself) lands
    // here too: dx = dy = 0, so factor * dx = factor * dy = 0 regardless of
    // how large factor is - no special case required, same trick as before.
    if is_leaf || (node.size / d) < THETA {
        let factor = G * node.mass / (d2 * d);
        return (dx * factor, dy * factor);
    }

    let mut fx = 0.0;
    let mut fy = 0.0;
    for &c in &node.children {
        if c != NONE {
            let (cx, cy) = compute_force(arena, c, i, x, y);
            fx += cx;
            fy += cy;
        }
    }
    (fx, fy)
}

// Tiny LCG PRNG: no OS entropy source is available in a bare no_std wasm module.
fn rand_unit(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (*state >> 8) as f32 / (u32::MAX >> 8) as f32 * 2.0 - 1.0
}

#[unsafe(no_mangle)]
pub extern "C" fn init(n: usize, seed: u32) {
    let mut state = seed | 1;
    unsafe {
        PARTICLES.x = Vec::with_capacity(n);
        PARTICLES.y = Vec::with_capacity(n);
        PARTICLES.vx = Vec::with_capacity(n);
        PARTICLES.vy = Vec::with_capacity(n);
        PARTICLES.ax = Vec::with_capacity(n);
        PARTICLES.ay = Vec::with_capacity(n);
        PARTICLES.mass = Vec::with_capacity(n);
        for _ in 0..n {
            PARTICLES.x.push(rand_unit(&mut state));
            PARTICLES.y.push(rand_unit(&mut state));
            PARTICLES.vx.push(rand_unit(&mut state) * 0.1);
            PARTICLES.vy.push(rand_unit(&mut state) * 0.1);
            PARTICLES.ax.push(0.0);
            PARTICLES.ay.push(0.0);
            PARTICLES.mass.push((rand_unit(&mut state) + 1.0) * 500.0);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tick() {
    unsafe {
        let n = PARTICLES.x.len();
        let chunks = n / 4; // whole groups of 4; any leftover handled scalar below

        // --- Barnes-Hut gravity: O(n log n) instead of O(n^2) ---
        if n > 0 {
            let xs = core::slice::from_raw_parts(PARTICLES.x.as_ptr(), n);
            let ys = core::slice::from_raw_parts(PARTICLES.y.as_ptr(), n);
            let masses = core::slice::from_raw_parts(PARTICLES.mass.as_ptr(), n);

            // Bounding square covering every particle - the tree's root region.
            let (mut min_x, mut max_x) = (f32::MAX, f32::MIN);
            let (mut min_y, mut max_y) = (f32::MAX, f32::MIN);
            for i in 0..n {
                min_x = min_x.min(xs[i]);
                max_x = max_x.max(xs[i]);
                min_y = min_y.min(ys[i]);
                max_y = max_y.max(ys[i]);
            }
            // *1.001 padding: keeps the max-extent particle strictly inside the
            // root square instead of sitting exactly on its edge.
            let span = (max_x - min_x).max(max_y - min_y).max(1e-3) * 1.001;

            ARENA.clear();
            ARENA.push(Node { x0: min_x, y0: min_y, size: span, ..EMPTY_NODE });
            for i in 0..n {
                insert(&mut ARENA, 0, i as u32, xs, ys, masses, 0);
            }

            for i in 0..n {
                let (fx, fy) = compute_force(&ARENA, 0, i, xs, ys);
                PARTICLES.ax[i] = fx;
                PARTICLES.ay[i] = fy;
            }
        }

        // --- integration: elementwise, so also vectorizes cleanly 4-at-a-time ---
        let dt = f32x4_splat(DT);
        for c in 0..chunks {
            let j = c * 4;
            let vx = v128_load(PARTICLES.vx.as_ptr().add(j) as *const v128);
            let vy = v128_load(PARTICLES.vy.as_ptr().add(j) as *const v128);
            let ax = v128_load(PARTICLES.ax.as_ptr().add(j) as *const v128);
            let ay = v128_load(PARTICLES.ay.as_ptr().add(j) as *const v128);
            let x = v128_load(PARTICLES.x.as_ptr().add(j) as *const v128);
            let y = v128_load(PARTICLES.y.as_ptr().add(j) as *const v128);

            let vx2 = f32x4_add(vx, f32x4_mul(ax, dt));
            let vy2 = f32x4_add(vy, f32x4_mul(ay, dt));
            let x2 = f32x4_add(x, f32x4_mul(vx2, dt));
            let y2 = f32x4_add(y, f32x4_mul(vy2, dt));

            v128_store(PARTICLES.vx.as_mut_ptr().add(j) as *mut v128, vx2);
            v128_store(PARTICLES.vy.as_mut_ptr().add(j) as *mut v128, vy2);
            v128_store(PARTICLES.x.as_mut_ptr().add(j) as *mut v128, x2);
            v128_store(PARTICLES.y.as_mut_ptr().add(j) as *mut v128, y2);
        }
        for i in (chunks * 4)..n {
            PARTICLES.vx[i] += PARTICLES.ax[i] * DT;
            PARTICLES.vy[i] += PARTICLES.ay[i] * DT;
            PARTICLES.x[i] += PARTICLES.vx[i] * DT;
            PARTICLES.y[i] += PARTICLES.vy[i] * DT;
        }
    }
}

// JS reads these two buffers directly out of wasm memory as Float32Arrays.
// Only valid until the Vec reallocates - safe here since init() allocates
// once up front and tick() never grows/shrinks it.
#[unsafe(no_mangle)]
pub extern "C" fn x_ptr() -> *const f32 {
    unsafe { PARTICLES.x.as_ptr() }
}

#[unsafe(no_mangle)]
pub extern "C" fn y_ptr() -> *const f32 {
    unsafe { PARTICLES.y.as_ptr() }
}

#[unsafe(no_mangle)]
pub extern "C" fn len() -> usize {
    unsafe { PARTICLES.x.len() }
}

// Lets this file also build/run natively (`cargo run`) for quick sanity checks
// without touching the wasm32 target.
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    init(2000, 42);
    for _ in 0..60 {
        tick();
    }
}
