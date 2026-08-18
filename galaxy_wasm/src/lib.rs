#![cfg_attr(target_arch = "wasm32", no_std, no_main)]
#![allow(static_mut_refs)]

extern crate alloc;
use alloc::vec::Vec;
#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

#[global_allocator]
static ALLOC: dlmalloc::GlobalDlmalloc = dlmalloc::GlobalDlmalloc;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

// --- Constants ---

const G: f32 = 0.000003;
const DT: f32 = 1.0 / 60.0;
const SOFTEN2: f32 = 0.05 * 0.05;
const THETA: f32 = 0.6;
const MAX_DEPTH: u32 = 28;
const NONE: u32 = u32::MAX;
const SUN_MASS: f32 = 10_000.0;
const SUN_SPRING: f32 = 2.0;

// --- Particles (struct-of-arrays for SIMD-friendly layout) ---

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
    x: Vec::new(), y: Vec::new(),
    vx: Vec::new(), vy: Vec::new(),
    ax: Vec::new(), ay: Vec::new(),
    mass: Vec::new(),
};

// --- Quadtree ---

#[derive(Clone, Copy)]
struct Node {
    x0: f32, y0: f32, size: f32,
    com_x: f32, com_y: f32, mass: f32,
    children: [u32; 4],
    particle: u32,
}

const EMPTY_NODE: Node = Node {
    x0: 0.0, y0: 0.0, size: 0.0,
    com_x: 0.0, com_y: 0.0, mass: 0.0,
    children: [NONE; 4], particle: NONE,
};

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
        return;
    }

    if arena[node_idx as usize].children[0] == NONE {
        let existing = arena[node_idx as usize].particle;
        arena[node_idx as usize].particle = NONE;

        let snap = arena[node_idx as usize];
        let mut child_idx = [NONE; 4];
        for q in 0..4 {
            let (cx0, cy0, csize) = child_bounds(&snap, q);
            arena.push(Node { x0: cx0, y0: cy0, size: csize, ..EMPTY_NODE });
            child_idx[q] = (arena.len() - 1) as u32;
        }
        arena[node_idx as usize].children = child_idx;

        let eq = quadrant(&snap, x[existing as usize], y[existing as usize]);
        insert(arena, child_idx[eq], existing, x, y, mass, depth + 1);
    }

    let snap = arena[node_idx as usize];
    let q = quadrant(&snap, px, py);
    insert(arena, snap.children[q], p, x, y, mass, depth + 1);
}

fn compute_force(arena: &[Node], node_idx: u32, i: usize, x: &[f32], y: &[f32]) -> (f32, f32) {
    let node = arena[node_idx as usize];
    if node.mass == 0.0 {
        return (0.0, 0.0);
    }

    let dx = node.com_x - x[i];
    let dy = node.com_y - y[i];
    let d2 = dx * dx + dy * dy + SOFTEN2;
    let d = libm::sqrtf(d2);

    if node.children[0] == NONE || (node.size / d) < THETA {
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

// --- PRNG ---

fn rand_unit(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (*state >> 8) as f32 / (u32::MAX >> 8) as f32 * 2.0 - 1.0
}

// --- Simulation helpers ---

fn bounding_square(xs: &[f32], ys: &[f32]) -> (f32, f32, f32) {
    let (mut min_x, mut max_x) = (f32::MAX, f32::MIN);
    let (mut min_y, mut max_y) = (f32::MAX, f32::MIN);
    for i in 0..xs.len() {
        min_x = min_x.min(xs[i]);
        max_x = max_x.max(xs[i]);
        min_y = min_y.min(ys[i]);
        max_y = max_y.max(ys[i]);
    }
    let span = (max_x - min_x).max(max_y - min_y).max(1e-3) * 1.001;
    (min_x, min_y, span)
}

fn build_tree(arena: &mut Vec<Node>, xs: &[f32], ys: &[f32], masses: &[f32]) {
    let (min_x, min_y, span) = bounding_square(xs, ys);
    arena.clear();
    arena.push(Node { x0: min_x, y0: min_y, size: span, ..EMPTY_NODE });
    for i in 0..xs.len() {
        insert(arena, 0, i as u32, xs, ys, masses, 0);
    }
}

fn compute_accelerations(arena: &[Node], xs: &[f32], ys: &[f32], ax: &mut [f32], ay: &mut [f32]) {
    for i in 0..xs.len() {
        let (fx, fy) = compute_force(arena, 0, i, xs, ys);
        ax[i] = fx;
        ay[i] = fy;
    }
}

fn integrate_scalar(p: &mut Particles, start: usize, end: usize) {
    for i in start..end {
        p.vx[i] += p.ax[i] * DT;
        p.vy[i] += p.ay[i] * DT;
        p.x[i] += p.vx[i] * DT;
        p.y[i] += p.vy[i] * DT;
    }
}

#[cfg(target_arch = "wasm32")]
fn integrate_simd(p: &mut Particles, chunks: usize) {
    unsafe {
        for c in 0..chunks {
            let j = c * 4;
            let dt = f32x4_splat(DT);
            let vx = v128_load(p.vx.as_ptr().add(j) as *const v128);
            let vy = v128_load(p.vy.as_ptr().add(j) as *const v128);
            let ax = v128_load(p.ax.as_ptr().add(j) as *const v128);
            let ay = v128_load(p.ay.as_ptr().add(j) as *const v128);
            let x = v128_load(p.x.as_ptr().add(j) as *const v128);
            let y = v128_load(p.y.as_ptr().add(j) as *const v128);

            let vx2 = f32x4_add(vx, f32x4_mul(ax, dt));
            let vy2 = f32x4_add(vy, f32x4_mul(ay, dt));
            let x2 = f32x4_add(x, f32x4_mul(vx2, dt));
            let y2 = f32x4_add(y, f32x4_mul(vy2, dt));

            v128_store(p.vx.as_mut_ptr().add(j) as *mut v128, vx2);
            v128_store(p.vy.as_mut_ptr().add(j) as *mut v128, vy2);
            v128_store(p.x.as_mut_ptr().add(j) as *mut v128, x2);
            v128_store(p.y.as_mut_ptr().add(j) as *mut v128, y2);
        }
    }
}

fn anchor_sun(p: &mut Particles) {
    p.vx[0] += -p.x[0] * SUN_SPRING * DT;
    p.vy[0] += -p.y[0] * SUN_SPRING * DT;
    p.vx[0] *= 0.98;
    p.vy[0] *= 0.98;
}

// --- Exported API ---

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

        // Particle 0: the sun
        PARTICLES.x.push(0.0);
        PARTICLES.y.push(0.0);
        PARTICLES.vx.push(0.0);
        PARTICLES.vy.push(0.0);
        PARTICLES.ax.push(0.0);
        PARTICLES.ay.push(0.0);
        PARTICLES.mass.push(SUN_MASS);

        for _ in 1..n {
            let angle = (rand_unit(&mut state) + 1.0) * core::f32::consts::PI;
            let r = (rand_unit(&mut state) + 1.0) * 0.45 + 0.05;
            let v = libm::sqrtf(G * SUN_MASS / r);

            PARTICLES.x.push(r * libm::cosf(angle));
            PARTICLES.y.push(r * libm::sinf(angle));
            PARTICLES.vx.push(-v * libm::sinf(angle));
            PARTICLES.vy.push(v * libm::cosf(angle));
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
        if n == 0 { return; }
        let chunks = n / 4;

        let xs = core::slice::from_raw_parts(PARTICLES.x.as_ptr(), n);
        let ys = core::slice::from_raw_parts(PARTICLES.y.as_ptr(), n);
        let masses = core::slice::from_raw_parts(PARTICLES.mass.as_ptr(), n);

        build_tree(&mut ARENA, xs, ys, masses);
        compute_accelerations(&ARENA, xs, ys, &mut PARTICLES.ax, &mut PARTICLES.ay);

        #[cfg(target_arch = "wasm32")]
        integrate_simd(&mut PARTICLES, chunks);
        #[cfg(not(target_arch = "wasm32"))]
        integrate_scalar(&mut PARTICLES, 0, chunks * 4);

        integrate_scalar(&mut PARTICLES, chunks * 4, n);
        anchor_sun(&mut PARTICLES);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn x_ptr() -> *const f32 { unsafe { PARTICLES.x.as_ptr() } }

#[unsafe(no_mangle)]
pub extern "C" fn y_ptr() -> *const f32 { unsafe { PARTICLES.y.as_ptr() } }

#[unsafe(no_mangle)]
pub extern "C" fn len() -> usize { unsafe { PARTICLES.x.len() } }

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    init(2000, 42);
    for _ in 0..60 { tick(); }
}
