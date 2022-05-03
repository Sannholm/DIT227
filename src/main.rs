use std::f32::consts::PI;
use std::mem::size_of;
use std::time::Duration;

use glam::{Mat4, Quat, Vec3, vec3, Mat3};
use glium::backend::Facade;
use glium::buffer::Content;
use glium::index::{PrimitiveType, NoIndices, Index};
use glium::program::ComputeShader;
use glium::vertex::{EmptyVertexAttributes, EmptyInstanceAttributes};
use glium::{Program, Frame, DrawParameters, DepthTest, Depth};
use glium::uniforms::UniformBuffer;
use glium::{glutin, Surface, VertexBuffer, IndexBuffer};
use glium::glutin::dpi::PhysicalSize;
use rand;

#[macro_use]
extern crate glium;

const NUM_PARTICLES: usize = 4096;

type PPos = [f32; 4];
type PVel = [f32; 4];

#[allow(non_snake_case)]
#[repr(C)]
#[derive(Clone, Copy)]
struct CameraUniforms {
    worldToCameraMatrix: [[f32; 4]; 4],
    cameraToClipMatrix: [[f32; 4]; 4]
}
implement_uniform_block!(CameraUniforms, worldToCameraMatrix, cameraToClipMatrix);

struct Camera {
    pos: Vec3,
    rot: Quat,
    fov: f32,
    aspect_ratio: f32,
    z_near: f32,
    z_far: f32
}

const UPDATE_PARTICLES_SRC: &str = r#"
    #version 430

    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

    uniform float time;
    uniform float deltaTime;

    layout(std140) buffer ParticlePositions {
        vec4 positions[];
    };

    layout(std140) buffer ParticleVelocities {
        vec4 velocities[];
    };

    void main() {
        vec3 pos = positions[gl_GlobalInvocationID.x].xyz;
        vec3 vel = velocities[gl_GlobalInvocationID.x].xyz;
        
        vel += vec3(0,-9.82,0);
        pos += vel * deltaTime; // TODO: Euler integration?

        positions[gl_GlobalInvocationID.x] = vec4(pos, 1);
        velocities[gl_GlobalInvocationID.x] = vec4(vel, 1);
    }
"#;

const VERTEX_SHADER_SRC: &str = r#"
    #version 430

    const vec2 OFFSETS[] = vec2[4](
        vec2(-0.5,  0.5),
        vec2(-0.5, -0.5),
        vec2(0.5,   0.5),
        vec2(0.5,  -0.5)
    );

    layout(std140) buffer Camera {
        mat4 worldToCameraMatrix;
        mat4 cameraToClipMatrix;
    };
    const vec3 cameraPos = inverse(worldToCameraMatrix)[3].xyz;

    layout(std140) buffer ParticlePositions {
        vec4 positions[];
    };

    out vec2 uv;

    void main() {
        vec3 particlePos = positions[gl_InstanceID].xyz;
        vec3 normal = normalize(cameraPos - particlePos); // Viewpoint-facing
        //normal = normalize(inverse(mat3(worldToCameraMatrix)) * vec3(0, 0, 1)); // Viewplane-facing
        
        vec3 upBasis = vec3(0, 1, 0);
        vec3 rightBasis = normalize(cross(upBasis, normal));
        upBasis = cross(rightBasis, normal);
        mat3 billboardMatrix = mat3(rightBasis, upBasis, normal);

        vec2 offset = OFFSETS[gl_VertexID];
        vec3 vertexPos = particlePos + billboardMatrix * vec3(offset, 0);

        uv = offset + vec2(0.5);
        gl_Position = cameraToClipMatrix * worldToCameraMatrix * vec4(vertexPos, 1);
    }
"#;

const FRAGMENT_SHADER_SRC: &str = r#"
    #version 430

    in vec2 uv;
    out vec4 color;

    void main() {
        color = vec4(uv, 0.0, 1.0);
    }
"#;

const MESH_VERTEX_SHADER_SRC: &str = r#"
    #version 430

    layout(std140) buffer Camera {
        mat4 worldToCameraMatrix;
        mat4 cameraToClipMatrix;
    };

    uniform mat4 modelToWorldMatrix;

    in vec3 position;
    in vec3 normal;
    in vec3 texture;
    
    out vec3 n;
    out vec2 uv;

    void main() {
        n = normal;
        uv = texture.xy;
        gl_Position = cameraToClipMatrix * worldToCameraMatrix * modelToWorldMatrix * vec4(position, 1);
    }
"#;

const MESH_FRAGMENT_SHADER_SRC: &str = r#"
    #version 430

    in vec3 n;
    in vec2 uv;

    out vec4 color;

    void main() {
        const vec3 diffColor = vec3(0.8);
        const vec3 ambientLight = vec3(0.1,0.1,0.2);
        const vec3 lightDir = vec3(1,1,0);
        const vec3 lightColor = vec3(1.0,1.0,0.5);
        float ndotl = max(0.0, dot(n, lightDir));
        vec3 radiance = (ambientLight + lightColor * ndotl) * diffColor;
        color = vec4(radiance, 1.0);
    }
"#;

fn new_particle_buffer<T>(facade: &impl Facade)
                        -> UniformBuffer<[T]> where [T]: Content {
    UniformBuffer::<[T]>::empty_unsized(facade,
        NUM_PARTICLES * size_of::<T>()).unwrap()
}

struct Mesh {
    pub vertices: VertexBuffer<MeshVertex>,
    pub indices: IndexBuffer<u32>
}

#[derive(Copy, Clone)]
pub struct MeshVertex {
    /// Position vector of a vertex.
    pub position: [f32; 3],
    /// Normal vertor of a vertex.
    pub normal: [f32; 3],
    /// Texture of a vertex.
    pub texture: [f32; 3],
}
implement_vertex!(MeshVertex, position, normal, texture);

impl Mesh {
    fn load_obj<F: Facade>(facade: &F, bytes: &[u8]) -> Self {
        let input = bytes;
        let obj = obj::load_obj(&input[..]).unwrap();

        let vertices = obj.vertices.iter().map(|v: &obj::TexturedVertex| {
            MeshVertex {
                position: v.position,
                normal: v.normal,
                texture: v.texture,
            }
        }).collect::<Vec<_>>();

        Mesh {
            vertices: VertexBuffer::immutable(facade, &vertices).unwrap(),
            indices: IndexBuffer::immutable(facade, PrimitiveType::TrianglesList, &obj.indices).unwrap()
        }
    }
}


fn main() {
    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new()
        .with_inner_size(PhysicalSize::new(1280u32, 720u32));
    let cb = glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    let mut particle_pos_buffer: UniformBuffer<[PPos]> = new_particle_buffer(&display);
    let mut particle_vel_buffer: UniformBuffer<[PVel]> = new_particle_buffer(&display);

    {
        let mut mapping = particle_pos_buffer.map();
        for val in mapping.iter_mut() {
            *val = [
                rand::random::<f32>() * 100.0 - 50.0,
                rand::random::<f32>() * 10.0 - 5.0,
                rand::random::<f32>() * 100.0 - 50.0,
                1.0
            ];
        }
    }

    let mut camera_uniforms_buffer =
              UniformBuffer::<CameraUniforms>::empty(&display).unwrap();

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
        let camera_angle = RADIANS_PER_SEC * time.as_secs_f32();
        camera.pos = Mat3::from_rotation_y(camera_angle) * vec3(0.0, 10.0, 70.0);
        camera.rot = Quat::from_rotation_ypr(camera_angle, -30.0*PI/180.0, PI);
        
        let mut target = display.draw();

        let (width, height) = target.get_dimensions();
        camera.aspect_ratio = width as f32 / height as f32;

        render(&display, &mut target,
            time.as_secs_f32(), FRAME_DURATION.as_secs_f32(),
            &camera,
            &mut camera_uniforms_buffer,
            &mut particle_pos_buffer,
            &mut particle_vel_buffer,
            &landingpad_mesh);

        target.finish().unwrap();

        time += FRAME_DURATION;
    });
}

fn render(display: &glium::Display, target: &mut Frame,
    time: f32, delta_time: f32,
    camera: &Camera,
    camera_uniforms_buffer: &mut UniformBuffer<CameraUniforms>,
    particle_pos_buffer: &mut UniformBuffer<[PPos]>,
    particle_vel_buffer: &mut UniformBuffer<[PVel]>,
    landingpad_mesh: &Mesh) {
    
    {
        let mut mapping = camera_uniforms_buffer.map_write();
        mapping.write(CameraUniforms {
            worldToCameraMatrix: (Mat4::from_translation(camera.pos) * Mat4::from_quat(camera.rot)).inverse().to_cols_array_2d(),
            cameraToClipMatrix: Mat4::perspective_rh_gl(camera.fov, camera.aspect_ratio, camera.z_near, camera.z_far).to_cols_array_2d()
        });
    }
        
    target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

    let program = Program::from_source(display,
        MESH_VERTEX_SHADER_SRC, MESH_FRAGMENT_SHADER_SRC, None).unwrap();
    let pad_transform = Mat4::from_scale_rotation_translation(
        vec3(1.0,1.0,1.0),
        Quat::identity(),
        vec3(0.0, -30.0, 0.0)
    );
    target.draw(&landingpad_mesh.vertices, &landingpad_mesh.indices,
        &program,
        &uniform! {
            Camera: &*camera_uniforms_buffer,
            modelToWorldMatrix: pad_transform.to_cols_array_2d()
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

    let program = ComputeShader::from_source(display, UPDATE_PARTICLES_SRC).unwrap();
    program.execute(
        uniform! {
            time: time,
            deltaTime: delta_time,
            ParticlePositions: &*particle_pos_buffer,
            ParticleVelocities: &*particle_vel_buffer
        },
        NUM_PARTICLES as u32, 1, 1
    );

    /* {
        let mapping = buffer.map();
        for val in mapping.iter().take(3) {
            println!("{:?}", glam::vec4(val[0], val[1], val[2], val[3]));
        }
        println!("...");
    } */

    let program = Program::from_source(display, VERTEX_SHADER_SRC,
                                                FRAGMENT_SHADER_SRC, None).unwrap();
    target.draw(
        (
            EmptyVertexAttributes { len: 4 },
            EmptyInstanceAttributes { len: NUM_PARTICLES }
        ),
        NoIndices(PrimitiveType::TriangleStrip),
        &program,
        &uniform! {
            time: time,
            Camera: &*camera_uniforms_buffer,
            ParticlePositions: &*particle_pos_buffer
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