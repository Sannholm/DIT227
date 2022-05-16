mod mesh;

use std::f32::consts::PI;
use std::mem::size_of;
use std::time::Duration;

use glam::{Mat4, Quat, Vec3, vec3, Mat3};
use glium::backend::Facade;
use glium::buffer::Content;
use glium::framebuffer::MultiOutputFrameBuffer;
use glium::index::{PrimitiveType, NoIndices};
use glium::program::ComputeShader;
use glium::texture::{UncompressedFloatFormat, MipmapsOption, DepthFormat, DepthTexture2d};
use glium::texture::texture2d::Texture2d;
use glium::vertex::{EmptyVertexAttributes, EmptyInstanceAttributes};
use glium::{Program, DrawParameters, DepthTest, Depth};
use glium::uniforms::{UniformBuffer};
use glium::{glutin, Surface};
use glium::glutin::dpi::PhysicalSize;
use mesh::Mesh;
use rand;
use ouroboros::self_referencing;

#[macro_use]
extern crate glium;

macro_rules! include_shader {
    ($path:literal) => {
        include_str!(concat!(env!("OUT_DIR"), "/shaders/", $path))
    };
}

struct Camera {
    pos: Vec3,
    rot: Quat,
    fov: f32,
    aspect_ratio: f32,
    z_near: f32,
    z_far: f32
}

struct RenderResources {
    camera_uniforms_buffer: UniformBuffer<CameraUniforms>,
    g_buffer: GBuffer,
    particles: ParticleSystem,
}

#[allow(non_snake_case)]
#[repr(C)]
#[derive(Clone, Copy)]
struct CameraUniforms {
    worldToCameraMatrix: [[f32; 4]; 4],
    cameraToClipMatrix: [[f32; 4]; 4]
}
implement_uniform_block!(CameraUniforms, worldToCameraMatrix, cameraToClipMatrix);

#[self_referencing]
struct GBuffer {
    color: Texture2d,
    normal: Texture2d,
    depth: DepthTexture2d,
    #[borrows(color, normal, depth)]
    #[covariant]
    framebuffer: MultiOutputFrameBuffer<'this>
}

const MAX_NUM_PARTICLES: usize = 3_000_000;

struct ParticleSystem {
    current: Particles,
    prev: Particles
}

type PPos = [f32; 4];
type PVel = [f32; 4];
struct Particles {
    alive_count: UniformBuffer<AliveCount>,
    pos_buffer: UniformBuffer<[PPos]>,
    vel_buffer: UniformBuffer<[PVel]>,
    radius_buffer: UniformBuffer<[f32]>,
    debug: UniformBuffer<[[f32; 4]]>
}

#[allow(non_snake_case)]
#[repr(C)]
#[derive(Clone, Copy)]
struct AliveCount {
    aliveCount: u32
}
implement_uniform_block!(AliveCount, aliveCount);


fn new_particle_buffer<T>(facade: &impl Facade)
                        -> UniformBuffer<[T]> where [T]: Content {
    UniformBuffer::empty_unsized(facade,
        MAX_NUM_PARTICLES * size_of::<T>()).unwrap()
}


fn main() {
    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(1280u32, 720u32));
    let cb = glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();
    
    let (width, height) = display.get_framebuffer_dimensions();
    let mut render_resources = RenderResources {
        g_buffer: GBufferBuilder {
            color: Texture2d::empty_with_format(&display, UncompressedFloatFormat::F32F32F32F32,
                MipmapsOption::NoMipmap, width, height).unwrap(),
            normal: Texture2d::empty_with_format(&display, UncompressedFloatFormat::F32F32F32F32,
                MipmapsOption::NoMipmap, width, height).unwrap(),
            depth: DepthTexture2d::empty_with_format(&display, DepthFormat::F32,
                MipmapsOption::NoMipmap, width, height).unwrap(),
            framebuffer_builder: |color, normal, depth| {
                MultiOutputFrameBuffer::with_depth_buffer(&display, 
                    [
                        ("fragColor", color),
                        ("fragNormal", normal)
                    ].iter().cloned(),
                    depth
                ).unwrap()
            },
        }.build(),

        particles: ParticleSystem {
            current: Particles {
                alive_count: UniformBuffer::new(&display, AliveCount { aliveCount: 0 }).unwrap(),
                pos_buffer: new_particle_buffer(&display),
                vel_buffer: new_particle_buffer(&display),
                radius_buffer: new_particle_buffer(&display),
                debug: new_particle_buffer(&display)
            },
            prev: Particles {
                alive_count: UniformBuffer::new(&display, AliveCount { aliveCount: 0 }).unwrap(),
                pos_buffer: new_particle_buffer(&display),
                vel_buffer: new_particle_buffer(&display),
                radius_buffer: new_particle_buffer(&display),
                debug: new_particle_buffer(&display)
            }
        },

        camera_uniforms_buffer: UniformBuffer::empty(&display).unwrap()
    };

    {
        let particles = &mut render_resources.particles.current;

        let mut mapping = particles.pos_buffer.map();
        for val in mapping.iter_mut() {
            *val = [
                rand::random::<f32>() * 5.0 - 2.5 - 4.0,
                rand::random::<f32>() * 5.0 - 2.5 - 1.0,
                rand::random::<f32>() * 5.0 - 2.5,
                1.0
            ];
        }

        let mut mapping = particles.vel_buffer.map();
        for val in mapping.iter_mut() {
            *val = [
                1.0,
                0.0,
                0.0,
                0.0
            ];
        }

        let mut mapping = particles.radius_buffer.map();
        for val in mapping.iter_mut() {
            *val = rand::random::<f32>() * 0.1 + 0.05;
        }

        particles.alive_count.write(&AliveCount { aliveCount: MAX_NUM_PARTICLES as u32 / 10 });
        //particles.alive_count.write(&AliveCount { aliveCount: 0 });
    }

    let mut camera = Camera {
        pos: vec3(0.0,0.0,10.0),
        rot: Quat::identity(),
        fov: 80.0,
        aspect_ratio: 1.0,
        z_near: 0.01,
        z_far: 10000.0
    };

    let landingpad_mesh = Mesh::load_obj(&display, include_bytes!("../scenes/landingpad.obj"));

    let mut time = Duration::ZERO;

    event_loop.run(move |event, _, control_flow| {

        match event {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                },
                _ => return,
            },
            glutin::event::Event::NewEvents(cause) => match cause {
                glutin::event::StartCause::ResumeTimeReached { .. } => (),
                glutin::event::StartCause::Init => (),
                _ => return,
            },
            _ => return,
        }

        const FRAME_DURATION: Duration = Duration::from_nanos(16_666_667);
        let next_frame_time = std::time::Instant::now() + FRAME_DURATION;
        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);
        
        const RADIANS_PER_SEC: f32 = PI / 2.0;
        let camera_angle = RADIANS_PER_SEC * time.as_secs_f32() * 0.0;
        camera.pos = Mat3::from_rotation_y(camera_angle) * vec3(0.0, 1.0, 7.0);
        camera.rot = Quat::from_rotation_ypr(camera_angle, -30.0*PI/180.0, PI);
        
        let mut target = display.draw();

        camera.aspect_ratio = {
            let (width, height) = target.get_dimensions();
            width as f32 / height as f32
        };

        render(&display, &mut target,
            time.as_secs_f32(), FRAME_DURATION.as_secs_f32(),
            &camera,
            &mut render_resources,
            &landingpad_mesh);

        target.finish().unwrap();

        time += FRAME_DURATION;
    });
}

fn render(display: &glium::Display, target: &mut impl Surface,
            time: f32, delta_time: f32,
            camera: &Camera,
            render_resources: &mut RenderResources,
            landingpad_mesh: &Mesh) {

    {
        let mut mapping = render_resources.camera_uniforms_buffer.map_write();
        mapping.write(CameraUniforms {
            worldToCameraMatrix: (Mat4::from_translation(camera.pos) * Mat4::from_quat(camera.rot)).inverse().to_cols_array_2d(),
            cameraToClipMatrix: Mat4::perspective_rh_gl(camera.fov, camera.aspect_ratio, camera.z_near, camera.z_far).to_cols_array_2d()
        });
    }

    let scene_shader_program = Program::from_source(display,
        include_shader!("static_mesh.vert"),
        include_shader!("static_mesh.frag"),
        None).unwrap();
    render_resources.g_buffer.with_mut(|g_buffer| {
        g_buffer.framebuffer.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);
        render_scene(g_buffer.framebuffer, time,
            &render_resources.camera_uniforms_buffer, landingpad_mesh, &scene_shader_program);
    });

    target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

    let program = Program::from_source(display,
        include_shader!("fullscreen_triangle.vert"),
        include_shader!("blit.frag"),
        None).unwrap();
    target.draw(
        EmptyVertexAttributes { len: 3 },
        NoIndices(PrimitiveType::TrianglesList),
        &program,
        &uniform! {
            blit_source_color: render_resources.g_buffer.borrow_color(),
            blit_source_depth: render_resources.g_buffer.borrow_depth()
        },
        &DrawParameters {
            depth: Depth {
                test: DepthTest::Overwrite, 
                write: true,
                ..Default::default()
            },
            ..Default::default()
        }
    ).unwrap();

    let system = &mut render_resources.particles;

    update_particles(
        display,
        system,
        &render_resources.camera_uniforms_buffer,
        &render_resources.g_buffer,
        time, delta_time
    );

    spawn_particles(
        display,
        &mut system.current,
        &render_resources.camera_uniforms_buffer,
        &render_resources.g_buffer,
        time, delta_time
    );

    render_particles(
        display, target,
        &render_resources.particles.current,
        &render_resources.camera_uniforms_buffer,
        time
    );
}

fn update_particles(display: &glium::Display,
                    system: &mut ParticleSystem,
                    camera_uniforms_buffer: &UniformBuffer<CameraUniforms>,
                    g_buffer: &GBuffer,
                    time: f32, delta_time: f32) {

    std::mem::swap(&mut system.current, &mut system.prev);

    // TODO: Avoid reading back particle count to CPU, use indirect dispatch?
    let alive_count = system.prev.alive_count.read().unwrap().aliveCount;
    let num_workgroups = (alive_count + 1023) / 1024;
    println!("Alive: {}, Workgroups: {}", alive_count, num_workgroups);

    system.current.alive_count.write(&AliveCount { aliveCount: 0 });

    let program = ComputeShader::from_source(display,
        include_shader!("particles/test_particles_update.comp")).unwrap();
    program.execute(
        uniform! {
            time: time,
            deltaTime: delta_time,
            Camera: &*camera_uniforms_buffer,
            
            prevAliveCount: alive_count,
            PrevPositions: &*system.prev.pos_buffer,
            PrevVelocities: &*system.prev.vel_buffer,
            PrevRadiuses: &*system.prev.radius_buffer,
            
            AliveCount: &*system.current.alive_count,
            Positions: &*system.current.pos_buffer,
            Velocities: &*system.current.vel_buffer,
            Radiuses: &*system.current.radius_buffer,
            DebugOutput: &*system.current.debug,

            sceneDepth: g_buffer.borrow_depth(),
            sceneNormal: g_buffer.borrow_normal()
        },
        num_workgroups, 1, 1
    );

    /* {
        let mapping = system.current.pos_buffer.map();
        for val in mapping.iter().take(1) {
            println!("{:?}", glam::vec4(val[0], val[1], val[2], val[3]));
        }
        println!("...");
    } */

    println!("After update: {}", system.current.alive_count.read().unwrap().aliveCount);

    {
        let mapping = system.current.debug.map_read();
        for val in mapping.iter().take(1) {
            println!("{:?}", glam::vec4(val[0], val[1], val[2], val[3]));
        }
        println!("...");
    }

    //panic!();
}

fn spawn_particles(display: &glium::Display,
                    particles: &mut Particles,
                    camera_uniforms_buffer: &UniformBuffer<CameraUniforms>,
                    g_buffer: &GBuffer,
                    time: f32, delta_time: f32) {
    
    let program = ComputeShader::from_source(display,
        include_shader!("particles/test_particles_spawn.comp")).unwrap();
    program.execute(
        uniform! {
            time: time,
            deltaTime: delta_time,
            Camera: &*camera_uniforms_buffer,

            AliveCount: &*particles.alive_count,
            Positions: &*particles.pos_buffer,
            Velocities: &*particles.vel_buffer,
            Radiuses: &*particles.radius_buffer,

            sceneDepth: g_buffer.borrow_depth(),
            sceneNormal: g_buffer.borrow_normal()
        },
        1, 1, 1
    );

    println!("After spawn: {}", particles.alive_count.read().unwrap().aliveCount);
}

fn render_particles(display: &glium::Display, target: &mut impl Surface,
                    particles: &Particles,
                    camera_uniforms_buffer: &UniformBuffer<CameraUniforms>,
                    time: f32) {

    // TODO: Avoid reading back particle count to CPU, use indirect draw?
    let alive_count = particles.alive_count.read().unwrap().aliveCount as usize;

    let program = Program::from_source(display,
        include_shader!("particles/particle_billboard.vert"),
        include_shader!("particles/particle_billboard.frag"),
        None).unwrap();
    target.draw(
        (
            EmptyVertexAttributes { len: 4 },
            EmptyInstanceAttributes { len: alive_count }
        ),
        NoIndices(PrimitiveType::TriangleStrip),
        &program,
        &uniform! {
            time: time,
            Camera: &*camera_uniforms_buffer,
            Positions: &*particles.pos_buffer,
            Radiuses: &*particles.radius_buffer
        },
        &DrawParameters {
            depth: Depth {
                test: DepthTest::IfLess,
                write: true,
                ..Default::default()
            },
            ..Default::default()
        }
    ).unwrap();
}

fn render_scene(target: &mut impl Surface,
    time: f32,
    camera_uniforms_buffer: &UniformBuffer<CameraUniforms>,
    landingpad_mesh: &Mesh,
    scene_shader_program: &Program) {
    
    let draw_parameters = DrawParameters {
        depth: Depth {
            test: DepthTest::IfLess,
            write: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let pad_transform = Mat4::from_scale_rotation_translation(
        vec3(0.1,0.1,0.1),
        Quat::from_rotation_z(45.0 * PI/180.0),
        vec3(0.0, -3.0, 0.0)
    );
    target.draw(&landingpad_mesh.vertices, &landingpad_mesh.indices,
        scene_shader_program,
        &uniform! {
            Camera: &*camera_uniforms_buffer,
            modelToWorldMatrix: pad_transform.to_cols_array_2d()
        },
        &draw_parameters
    ).unwrap();

    let pad_transform = Mat4::from_scale_rotation_translation(
        vec3(1.0,1.0,1.0),
        Quat::from_rotation_z(0.0),
        vec3(0.0, -10.0, 0.0)
    );
    target.draw(&landingpad_mesh.vertices, &landingpad_mesh.indices,
        scene_shader_program,
        &uniform! {
            Camera: &*camera_uniforms_buffer,
            modelToWorldMatrix: pad_transform.to_cols_array_2d()
        },
        &draw_parameters
    ).unwrap();
}