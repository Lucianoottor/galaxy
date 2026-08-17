// Manual wasm export example: no wasm_bindgen, no wasm-pack.
// Build: cargo build --bin galaxy_wasm --target wasm32-unknown-unknown --release
// JS reads/writes the exported functions + raw memory directly.

#![cfg_attr(target_arch = "wasm32", no_std, no_main)]
#![allow(static_mut_refs)] // single-threaded JS host calls us one at a time; fine here.

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

const N: usize = 2000;
const G: f32 = 0.000003; 
const FPS: f32 = 60.0;
const DT: f32 = 1.0 / FPS;

// #[repr(C)] gives a fixed, predictable byte layout so JS can read this
// struct straight out of wasm memory as a Float32Array (7 fields * 4 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct Particle {
    x: Vec<f32>,
    y: Vec<f32>,
    vx: Vec<f32>,
    vy: Vec<f32>,
    ax: Vec<f32>,
    ay: Vec<f32>,
    mass: Vec<f32>,
}

static mut PARTICLES: [Particle; N] = [Particle { x: 0.0, y: 0.0, vx: 0.0, vy: 0.0, ax: 0.0, ay: 0.0, mass: 1.0 }; N];

fn distance(x0: f32, x1: f32, y0: f32, y1: f32) -> f32 {
    libm::sqrtf((x1 - x0)*(x1 - x0) + (y1 - y0) * (y1 - y0))
}

fn acceleration(mass: f32, dp: f32, d: f32) -> f32 {
    G * mass * dp / (d * d * d)
}

// Tiny LCG PRNG: no OS entropy source is available in a bare no_std wasm module.
fn rand_unit(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (*state >> 8) as f32 / (u32::MAX >> 8) as f32 * 2.0 - 1.0
}

#[unsafe(no_mangle)]
pub extern "C" fn init(seed: u32) {
    let mut state = seed | 1;
    unsafe {
        for p in PARTICLES.iter_mut() {
            p.x = rand_unit(&mut state);
            p.y = rand_unit(&mut state);
            p.vx = rand_unit(&mut state) * 0.1;
            p.vy = rand_unit(&mut state) * 0.1;
            p.ax = 0.0;
            p.ay = 0.0;
            p.mass = (rand_unit(&mut state) + 1.0) * 500.0;
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tick() {
    unsafe {
        for i in 0..N {
            PARTICLES[i].ax = 0.0;
            PARTICLES[i].ay = 0.0;
        }
        for i in 0..N {
            for j in 0..N {
                if i == j {
                    continue;
                }
                let (pi, pj) = (PARTICLES[i], PARTICLES[j]);
                let d = distance(pi.x, pj.x, pi.y, pj.y).max(0.05);
                PARTICLES[i].ax += acceleration(pj.mass, pj.x - pi.x, d);
                PARTICLES[i].ay += acceleration(pj.mass, pj.y - pi.y, d);
            }
        }
        for p in PARTICLES.iter_mut() {
            p.vx += p.ax * DT;
            p.vy += p.ay * DT;
            p.x += p.vx * DT;
            p.y += p.vy * DT;
        }
    }
}

// JS calls these two to locate the particle buffer inside wasm linear memory,
// then reads it as `new Float32Array(memory.buffer, ptr, len * 7)`.
#[unsafe(no_mangle)]
pub extern "C" fn particles_ptr() -> *const Particle {
    unsafe { PARTICLES.as_ptr() }
}

#[unsafe(no_mangle)]
pub extern "C" fn particles_len() -> usize {
    N
}

// Lets this file also build/run natively (`cargo run`) for quick sanity checks
// without touching the wasm32 target.
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    init(42);
    for _ in 0..60 {
        tick();
    }
}
