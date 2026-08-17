use glam::Vec2;
use rand::{Rng, rng};

#[derive(Debug, Clone, Copy)]
struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    a: Vec2,
    mass: f32,
}

struct Screen {
    width: i32,
    height: i32,
}


// Calculate the influence of 1 particle on another, which means for the universe of N particles,
// for each we calculate N influences
// which in first hand will give O(N^2) complexity
// 
// d = euclidian distance between two points
// Calculus for particles: 
// F = G * ((Mi * Mj)/d^2)
// ai = G * (Mj/d^2)
// ->ai = ai ((dx/d), (dy/d)) will translate below
// ->ai = (G(Mj * dx/d^3), G(Mj * dy/d^3))
// getting ax and ay we calculate velocity
// vx = vx0 + ax * dt, vy = vy0 + ay * dt
// then calculate new x,y position
// x = x0 + vx * dt, y = y0 + vy * dt
// 
// Later, must sum the vector forces to update the particle position

const G: f32 = 9.8;
const FPS: f32 = 60.0;
#[warn(non_upper_case_globals)]
const dt: f32 = 1.0/FPS;

fn distance(x0: f32, x1: f32, y0: f32, y1: f32) -> f32 {
    ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
}

fn acceleration(mass: f32, dp: f32, d: f32) -> f32 {
    G*(mass * dp/d.powi(3))
}

fn velocity(v0: f32, ap: f32) -> f32 {
    v0 + ap * dt
}

fn update_acceleration(p0: &mut Particle, p1: &Particle) {
    // Calculate new p0
    let d = distance(p0.x, p1.x, p0.y, p1.y);
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    
    // ->a0 = (a0x, a0y)
    p0.a.x = acceleration(p1.mass, dx, d);
    p0.a.y = acceleration(p1.mass, dy, d);    

}


fn update_coordinates(p: &mut Particle){
    p.vx = velocity(p.vx, p.a.x);
    p.vy = velocity(p.vy, p.a.y);
    p.x = velocity(p.x, p.vx);
    p.y = velocity(p.y, p.vy);
}

fn frame(particles: &mut Vec<Particle>) {

    for i in 0..particles.len(){
        let p = &mut particles[i];
        p.a = Vec2::new(0.0, 0.0);
    }
    for i in 0..particles.len(){
        for j in 0..particles.len(){
            if i!=j  {
            let p1 = &particles[j].clone();
            update_acceleration(&mut particles[i], p1);
            }
        }
    }

    for p in 0..particles.len() {
        update_coordinates(&mut particles[p]);
    }

    
}

// work with range x,y = -1,..,1
fn resize(screen: Screen, particles: &mut Vec<Particle>) {
    for i in 0..particles.len(){
        let p = &mut particles[i];
        p.x = (1.0+p.x) * screen.width as f32/2.0;
        p.y = 1.0-((1.0+p.y)/2.0) * screen.height as f32;
    }
}

fn rng_particles(n: i32) -> Vec<Particle> {
    let mut particles = Vec::new();
    for i in 0..n{
        let p = Particle {
            x: rand::random_range(-1.0..=1.0),
            y: rand::random_range(-1.0..=1.0),
            vx: rand::random_range(-1.0..=1.0),
            vy: rand::random_range(-1.0..=1.0),
            a:Vec2::new(0.0, 0.0),
            mass: rand::random_range(0.01..1000.0) 
        };
        particles.push(p);
    }

    particles
}


fn main() {
   
    let mut particles = rng_particles(200);

    let screen = Screen {width: 600, height: 600};
    resize(screen, &mut particles);

    loop {
        frame(&mut particles);
    }
    
}
        




