mod mesh;

use std::collections::HashMap;
use std::f32::consts::PI;
use std::mem::size_of;
use std::time::Duration;

use glam::{Mat4, Quat, Vec3, vec3, Mat3};
use glium::backend::Facade;
use glium::buffer::Content;
use glium::framebuffer::MultiOutputFrameBuffer;
use glium::index::{PrimitiveType, NoIndices};
use glium::program::ComputeShader;
use glium::texture::{UncompressedFloatFormat, MipmapsOption, DepthFormat, DepthTexture2d, RawImage2d};
use glium::texture::texture2d::Texture2d;
use glium::vertex::{EmptyVertexAttributes, EmptyInstanceAttributes};
use glium::{Program, DrawParameters, DepthTest, Depth, VertexBuffer, IndexBuffer, draw_parameters};
use glium::uniforms::{UniformBuffer};
use glium::{glutin, Surface};
use glium::glutin::dpi::PhysicalSize;
use gltf::Gltf;
use gltf::image::Format;
use glutin::event::VirtualKeyCode;
use itertools::Itertools;
use mesh::Mesh;
use rand;
use ouroboros::self_referencing;

use crate::mesh::MeshVertex;

#[macro_use]
extern crate glium;

struct ShaderLoader {
    programs: HashMap<String, Program>,
    compute_programs: HashMap<String, ComputeShader>,
}

impl ShaderLoader {
    fn new() -> Self {
        Self {
            programs: HashMap::new(),
            compute_programs: HashMap::new()
        }
    }

    fn clear_cache(&mut self) {
        self.programs.clear();
        self.compute_programs.clear();
    }

    fn get(&mut self, facade: &impl Facade, vert_path: &str, frag_path: &str) -> &Program {
        let key = format!("{}{}", vert_path, frag_path);
        self.programs.entry(key).or_insert_with(|| {
            loop {
                let program = Program::from_source(facade,
                    ShaderLoader::load_program_source(vert_path).as_str(),
                    ShaderLoader::load_program_source(frag_path).as_str(),
                    None);
                match program {
                    Ok(program) => return program,
                    Err(e) => {
                        println!("Failed to load shader program {} or {}:\n{}", vert_path, frag_path, e)
                    }
                }
            }
        })
    }

    fn get_compute(&mut self, facade: &impl Facade, path: &str) -> &ComputeShader {
        self.compute_programs.entry(path.to_string()).or_insert_with(|| {
            loop {
                let src = ShaderLoader::load_program_source(path);
                let program = ComputeShader::from_source(facade,
                    src.as_str());
                match program {
                    Ok(program) => return program,
                    Err(e) => {
                        println!("Failed to load compute program {}:\n{}\n{}", path, e,
                            ShaderLoader::format_source_for_debug(&src))
                    }
                }
            }
        })
    }

    fn load_program_source(path: &str) -> String {
        const SHADER_SRC_PATH: &str = "src/shaders";
    
        let tera = tera::Tera::new(format!("{}/**/*", SHADER_SRC_PATH).as_str()).unwrap();
        tera.render(path, &tera::Context::new()).unwrap()
    }

    fn format_source_for_debug(src: &str) -> String {
        src.lines().zip(std::iter::successors(Some(1), |x| Some(x+1)))
            .map(|(line, num)| format!("{}: {}", num, line))
            .join("\n")
    }
}



#[derive(Copy, Clone)]
struct Camera {
    pos: Vec3,
    rot: Quat,
    fov: f32,
    aspect_ratio: f32,
    z_near: f32,
    z_far: f32
}

struct RenderResources<'a> {
    shaders: ShaderLoader,
    camera_uniforms_buffer: UniformBuffer<CameraUniforms>,
    g_buffer: GBuffer,
    particle_systems: Vec<ParticleSystem<'a>>,
}

#[allow(non_snake_case)]
#[repr(C)]
#[derive(Clone, Copy)]
struct CameraUniforms {
    worldToCameraMatrix: [[f32; 4]; 4],
    cameraToWorldMatrix: [[f32; 4]; 4],
    cameraToClipMatrix: [[f32; 4]; 4],
    clipToCameraMatrix: [[f32; 4]; 4]
}
implement_uniform_block!(CameraUniforms,
    worldToCameraMatrix, cameraToWorldMatrix,
    cameraToClipMatrix, clipToCameraMatrix
);

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

struct ParticleSystem<'a> {
    current_frame_num: u32,
    current: Particles,
    prev: Particles,
    spawn_programs: Vec<&'a str>,
    update_program: &'a str,
    vertex_program: &'a str,
    fragment_program: &'a str,
    draw_parameters: DrawParameters<'a>
}

type PPos = [f32; 4];
type PVel = [f32; 4];
struct Particles {
    alive_count: UniformBuffer<AliveCount>,
    lifetime_buffer: UniformBuffer<[f32]>,
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

fn setup_particle_systems(facade: &impl Facade, systems: &mut Vec<ParticleSystem>) {
    let system =
        |spawns, update, vert, frag, draw_params| {
            ParticleSystem::new(facade, spawns, update,
                vert, frag, draw_params)
        };
    
    let draw_params = DrawParameters {
        depth: Depth {
            test: DepthTest::IfLess,
            write: true,
            ..Default::default()
        },
        ..Default::default()
    };
    
    systems.extend([
        /*system(
            &[
                "particles/test_particles_spawn.comp",
                "particles/rain_splash_spawn.comp"
            ],
            "particles/test_particles_update.comp",
            "particles/particle_billboard.vert",
            "particles/particle_billboard.frag",
            draw_params
        ),
        system(
            &[
                "particles/smoke_spawn.comp"
            ],
            "particles/smoke_update.comp",
            "particles/particle_billboard.vert",
            "particles/particle_billboard.frag",
            draw_params
        )*/
        system(
            &[
                "particles/rainfall_spawn.comp"
            ],
            "particles/rainfall_update.comp",
            "particles/particle_billboard.vert",
            "particles/particle_billboard.frag",
            draw_params
        )
    ]);
}

impl<'a> ParticleSystem<'a> {
    fn new(facade: &impl Facade,
            spawn_programs: &[&'a str],
            update_program: &'a str,
            vertex_program: &'a str,
            fragment_program: &'a str,
            draw_parameters: DrawParameters<'a>) -> Self {
        Self {
            current_frame_num: 0,
            current: Particles {
                alive_count: UniformBuffer::new(facade, AliveCount { aliveCount: 0 }).unwrap(),
                lifetime_buffer: Self::new_particle_buffer(facade),
                pos_buffer: Self::new_particle_buffer(facade),
                vel_buffer: Self::new_particle_buffer(facade),
                radius_buffer: Self::new_particle_buffer(facade),
                debug: Self::new_particle_buffer(facade)
            },
            prev: Particles {
                alive_count: UniformBuffer::new(facade, AliveCount { aliveCount: 0 }).unwrap(),
                lifetime_buffer: Self::new_particle_buffer(facade),
                pos_buffer: Self::new_particle_buffer(facade),
                vel_buffer: Self::new_particle_buffer(facade),
                radius_buffer: Self::new_particle_buffer(facade),
                debug: Self::new_particle_buffer(facade)
            },
            spawn_programs: Vec::from_iter(spawn_programs.iter().copied()),
            update_program,
            vertex_program,
            fragment_program,
            draw_parameters
        }
    }

    fn new_particle_buffer<T>(facade: &impl Facade)
                            -> UniformBuffer<[T]> where [T]: Content {
        UniformBuffer::empty_unsized(facade,
            MAX_NUM_PARTICLES * size_of::<T>()).unwrap()
    }
}

fn main() {
    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(1280u32, 720u32));
    let cb = glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();
    
    let scene = load_scene(&display);

    let (width, height) = display.get_framebuffer_dimensions();
    let mut render_resources = RenderResources {
        shaders: ShaderLoader::new(),
        camera_uniforms_buffer: UniformBuffer::empty(&display).unwrap(),
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

        particle_systems: Vec::new()
    };

    setup_particle_systems(&display, &mut render_resources.particle_systems);

    /* {
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
        particles.alive_count.write(&AliveCount { aliveCount: 0 });
    } */

    //let landingpad_mesh = Mesh::load_obj(&display, include_bytes!("../scenes/landingpad.obj"));
    //let storm_mesh = Mesh::load_obj(&display, include_bytes!("../scenes/storm/Storm.obj"));

    let mut time = Duration::ZERO;

    event_loop.run(move |event, _, control_flow| {

        match event {
            glutin::event::Event::WindowEvent { event, .. } => match event {
                glutin::event::WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                },
                glutin::event::WindowEvent::KeyboardInput { device_id: _, input, is_synthetic: _ } => {
                    if input.state == glutin::event::ElementState::Released {
                        if let Some(VirtualKeyCode::R) = input.virtual_keycode {
                            render_resources.shaders.clear_cache()
                        }
                    }
                    return
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
        
        //const RADIANS_PER_SEC: f32 = PI / 2.0;
        //let camera_angle = RADIANS_PER_SEC * time.as_secs_f32() * 0.0;
        //camera.pos = Mat3::from_rotation_y(camera_angle) * vec3(0.0, 1.0, 7.0);
        //camera.rot = Quat::from_rotation_ypr(camera_angle, -30.0*PI/180.0, PI);
        
        let mut target = display.draw();

        let mut camera = scene.camera;
        camera.aspect_ratio = {
            let (width, height) = target.get_dimensions();
            width as f32 / height as f32
        };

        render(&display, &mut target,
            time.as_secs_f32(), FRAME_DURATION.as_secs_f32(),
            &camera,
            &mut render_resources,
            &scene);

        target.finish().unwrap();

        time += FRAME_DURATION;
    });
}

struct Scene {
    objects: Vec<SceneObject>,
    camera: Camera
}

struct SceneObject {
    transform: Mat4,
    mesh: mesh::Mesh
}

fn load_scene(facade: &impl Facade) -> Scene {
    let (document, buffers, images) = gltf::import("scenes/storm/Storm.gltf").unwrap();

    let textures: Vec<_> = images.iter().map(|data| {
        let dims = (data.width, data.height);
        let image = match data.format {
            Format::R8G8B8 => RawImage2d::from_raw_rgb(data.pixels.clone(), dims),
            Format::R8G8B8A8 => RawImage2d::from_raw_rgba(data.pixels.clone(), dims),
            _ => todo!()
        };
        glium::texture::SrgbTexture2d::new(facade, image)
    }).collect();

    let mut objects = Vec::new();
    let mut cameras = Vec::new();

    fn traverse(facade: &impl Facade, buffers: &Vec<gltf::buffer::Data>,
                objects: &mut Vec<SceneObject>, cameras: &mut Vec<Camera>,
                node: gltf::Node, parent_to_world_matrix: Mat4) {
        let node_to_parent_matrix = Mat4::from_cols_array_2d(&node.transform().matrix());
        let node_to_world_matrix = parent_to_world_matrix * node_to_parent_matrix;

        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer: gltf::Buffer| Some(&buffers[buffer.index()]));
                
                let vertices: Vec<MeshVertex> = itertools::izip!(
                    reader.read_positions().unwrap(),
                    reader.read_normals().unwrap(),
                    reader.read_tex_coords(0).unwrap().into_f32()
                ).map(|(position, normal, texture)| {
                    mesh::MeshVertex { position, normal, texture: [texture[0], texture[1], 0.0] }
                }).collect();
                let indices: Vec<u32> = reader.read_indices().unwrap().into_u32().collect();

                let mesh = mesh::Mesh {
                    vertices: VertexBuffer::immutable(facade, &vertices).unwrap(),
                    indices: IndexBuffer::immutable(facade, PrimitiveType::TrianglesList, &indices).unwrap()
                };

                objects.push(SceneObject {
                    transform: node_to_world_matrix,
                    mesh
                });
            }
        }

        if let Some(camera) = node.camera() {
            if let gltf::camera::Projection::Perspective(persp) = camera.projection() {
                let (_, rot, pos) = node_to_world_matrix.to_scale_rotation_translation();
                cameras.push(Camera {
                    pos,
                    rot,
                    aspect_ratio: 1.0,
                    fov: persp.yfov(),
                    z_near: persp.znear(),
                    z_far: persp.zfar().unwrap()
                });
            }
        }

        for child in node.children() {
            traverse(facade, &buffers, objects, cameras, child, node_to_world_matrix);
        }
    }

    for root in document.default_scene().unwrap().nodes() {
        traverse(facade, &buffers, &mut objects, &mut cameras, root, Mat4::identity());
    }

    Scene {
        objects,
        camera: cameras[0]
    }
}

fn render(display: &glium::Display, target: &mut impl Surface,
            time: f32, delta_time: f32,
            camera: &Camera,
            render_resources: &mut RenderResources,
            scene: &Scene) {

    {
        let camera_to_world_matrix = Mat4::from_translation(camera.pos) * Mat4::from_quat(camera.rot);
        let camera_to_clip_matrix = Mat4::perspective_rh_gl(camera.fov, camera.aspect_ratio, camera.z_near, camera.z_far);

        let value = CameraUniforms {
            worldToCameraMatrix: camera_to_world_matrix.inverse().to_cols_array_2d(),
            cameraToWorldMatrix: camera_to_world_matrix.to_cols_array_2d(),
            cameraToClipMatrix: camera_to_clip_matrix.to_cols_array_2d(),
            clipToCameraMatrix: camera_to_clip_matrix.inverse().to_cols_array_2d()
        };
        render_resources.camera_uniforms_buffer.write(&value);
    }

    let scene_shader_program = render_resources.shaders.get(display,
        "static_mesh.vert",
        "static_mesh.frag");
    render_resources.g_buffer.with_mut(|g_buffer| {
        g_buffer.framebuffer.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);
        //render_landingpad_scene(g_buffer.framebuffer, time,
        //    &render_resources.camera_uniforms_buffer, landingpad_mesh, &scene_shader_program);
        //render_storm_scene(g_buffer.framebuffer, time,
        //    &render_resources.camera_uniforms_buffer, storm_mesh, &scene_shader_program);
        render_scene(scene, g_buffer.framebuffer, time,
                    &render_resources.camera_uniforms_buffer,
                    &scene_shader_program);
    });

    target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

    let program = render_resources.shaders.get(display,
        "fullscreen_triangle.vert",
        "blit.frag");
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

    for system in &mut render_resources.particle_systems {
        let program = render_resources.shaders.get_compute(display, system.update_program);
        update_particles(
            system,
            &program,
            &render_resources.camera_uniforms_buffer,
            &render_resources.g_buffer,
            time, delta_time
        );
    }
    
    for system in &mut render_resources.particle_systems {
        for program_path in &system.spawn_programs {
            let program = render_resources.shaders.get_compute(display, program_path);
            spawn_particles(
                &program,
                &mut system.current,
                &render_resources.camera_uniforms_buffer,
                &render_resources.g_buffer,
                system.current_frame_num, time, delta_time
            );
        }
    }
    
    for system in &render_resources.particle_systems {
        let program = render_resources.shaders.get(display,
            &system.vertex_program,
            &system.fragment_program);
        render_particles(
            target,
            &system.current,
            &render_resources.camera_uniforms_buffer,
            &program,
            &system.draw_parameters,
            time
        );
    }
}

fn update_particles(system: &mut ParticleSystem,
                    program: &ComputeShader,
                    camera_uniforms_buffer: &UniformBuffer<CameraUniforms>,
                    g_buffer: &GBuffer,
                    time: f32, delta_time: f32) {

    std::mem::swap(&mut system.current, &mut system.prev);

    // TODO: Avoid reading back particle count to CPU, use indirect dispatch?
    let alive_count = system.prev.alive_count.read().unwrap().aliveCount;
    let num_workgroups = (alive_count + 1023) / 1024;
    println!("Frame: {}, Alive: {}, Workgroups: {}", system.current_frame_num, alive_count, num_workgroups);
    
    let uniforms = uniform! {
        time: time,
        deltaTime: delta_time,
        Camera: &*camera_uniforms_buffer,
        
        prevAliveCount: alive_count,
        PrevLifetimes: &*system.prev.lifetime_buffer,
        PrevPositions: &*system.prev.pos_buffer,
        PrevVelocities: &*system.prev.vel_buffer,
        PrevRadiuses: &*system.prev.radius_buffer,
        
        AliveCount: &*system.current.alive_count,
        Lifetimes: &*system.current.lifetime_buffer,
        Positions: &*system.current.pos_buffer,
        Velocities: &*system.current.vel_buffer,
        Radiuses: &*system.current.radius_buffer,
        DebugOutput: &*system.current.debug,

        sceneDepth: g_buffer.borrow_depth(),
        sceneNormal: g_buffer.borrow_normal()
    };

    system.current.alive_count.write(&AliveCount { aliveCount: 0 });
    program.execute(uniforms, num_workgroups, 1, 1);
    system.current_frame_num += 1;

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
}

fn spawn_particles(program: &ComputeShader,
                    particles: &mut Particles,
                    camera_uniforms_buffer: &UniformBuffer<CameraUniforms>,
                    g_buffer: &GBuffer,
                    frame_num: u32, time: f32, delta_time: f32) {
    
    let uniforms = uniform! {
        frameNum: frame_num,
        time: time,
        deltaTime: delta_time,
        Camera: &*camera_uniforms_buffer,

        AliveCount: &*particles.alive_count,
        Lifetimes: &*particles.lifetime_buffer,
        Positions: &*particles.pos_buffer,
        Velocities: &*particles.vel_buffer,
        Radiuses: &*particles.radius_buffer,

        sceneDepth: g_buffer.borrow_depth(),
        sceneNormal: g_buffer.borrow_normal()
    };

    program.execute(uniforms, 1, 1, 1);

    println!("After spawn: {}", particles.alive_count.read().unwrap().aliveCount);
}

fn render_particles(target: &mut impl Surface,
                    particles: &Particles,
                    camera_uniforms_buffer: &UniformBuffer<CameraUniforms>,
                    program: &Program,
                    draw_params: &DrawParameters,
                    time: f32) {

    // TODO: Avoid reading back particle count to CPU, use indirect draw?
    let alive_count = particles.alive_count.read().unwrap().aliveCount as usize;

    target.draw(
        (
            EmptyVertexAttributes { len: 4 },
            EmptyInstanceAttributes { len: alive_count }
        ),
        NoIndices(PrimitiveType::TriangleStrip),
        program,
        &uniform! {
            time: time,
            Camera: &*camera_uniforms_buffer,

            Lifetimes: &*particles.lifetime_buffer,
            Positions: &*particles.pos_buffer,
            Velocities: &*particles.vel_buffer,
            Radiuses: &*particles.radius_buffer,
        },
        draw_params
    ).unwrap();
}

fn render_scene(
    scene: &Scene,
    target: &mut impl Surface,
    time: f32,
    camera_uniforms_buffer: &UniformBuffer<CameraUniforms>,
    scene_shader_program: &Program) {
    
    let draw_parameters = DrawParameters {
        depth: Depth {
            test: DepthTest::IfLess,
            write: true,
            ..Default::default()
        },
        ..Default::default()
    };

    for object in &scene.objects {
        target.draw(&object.mesh.vertices, &object.mesh.indices,
            scene_shader_program,
            &uniform! {
                Camera: &*camera_uniforms_buffer,
                modelToWorldMatrix: object.transform.to_cols_array_2d()
            },
            &draw_parameters
        ).unwrap();
    }
}

fn render_storm_scene(target: &mut impl Surface,
    time: f32,
    camera_uniforms_buffer: &UniformBuffer<CameraUniforms>,
    storm_mesh: &Mesh,
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
        vec3(1.0, 1.0, 1.0),
        Quat::identity(),
        vec3(0.0, 0.0, 0.0)
    );
    target.draw(&storm_mesh.vertices, &storm_mesh.indices,
        scene_shader_program,
        &uniform! {
            Camera: &*camera_uniforms_buffer,
            modelToWorldMatrix: pad_transform.to_cols_array_2d()
        },
        &draw_parameters
    ).unwrap();
}

fn render_landingpad_scene(target: &mut impl Surface,
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