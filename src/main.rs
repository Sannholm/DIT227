use std::f32::consts::PI;
use std::mem::size_of;
use std::time::Duration;

use glam::{Mat4, Quat, Vec3, vec3, Mat3};
use glium::backend::Facade;
use glium::buffer::Content;
use glium::framebuffer::MultiOutputFrameBuffer;
use glium::index::{PrimitiveType, NoIndices, Index};
use glium::program::ComputeShader;
use glium::texture::{UncompressedFloatFormat, MipmapsOption, DepthFormat, DepthTexture2d};
use glium::texture::texture2d::Texture2d;
use glium::vertex::{EmptyVertexAttributes, EmptyInstanceAttributes};
use glium::{Program, Frame, DrawParameters, DepthTest, Depth, BlitMask, BlitTarget, Rect};
use glium::uniforms::{UniformBuffer, MagnifySamplerFilter};
use glium::{glutin, Surface, VertexBuffer, IndexBuffer};
use glium::glutin::dpi::PhysicalSize;
use rand;
use ouroboros::self_referencing;

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
    
    layout(std140) buffer Camera {
        mat4 worldToCameraMatrix;
        mat4 cameraToClipMatrix;
    };

    layout(std140) buffer ParticlePositions {
        vec4 positions[];
    };

    layout(std140) buffer ParticleVelocities {
        vec4 velocities[];
    };

    layout(std140) buffer ParticleRadiuses {
        float radiuses[];
    };

    layout(std140) buffer DebugOutput {
        vec4 debug[];
    };

    uniform sampler2D sceneDepth;
    uniform sampler2D sceneNormal;

    struct Collision {
        vec3 pos;
        vec3 normal;
    };
    const Collision DUMMY_COLLISION = Collision(vec3(0.0), vec3(0.0));

    struct CollisionQuery {
        bool colliding;
        Collision collision;
    };

    CollisionQuery checkCollision(vec3 pos, float radius) {
        vec4 viewPos = worldToCameraMatrix * vec4(pos, 1.0);
        vec4 clipPos = cameraToClipMatrix * viewPos;
        vec2 ndcPos = clipPos.xy / clipPos.w;

        float geometryNdcDepth = texture(sceneDepth, ndcPos * 0.5 + 0.5).x * 2.0 - 1.0;
        vec4 geometryNdcPos = vec4(ndcPos, geometryNdcDepth, 1.0);
        vec4 geometryViewPos = inverse(cameraToClipMatrix) * geometryNdcPos;
        geometryViewPos /= geometryViewPos.w;

        float particleDepth = -viewPos.z;
        float geometryDepth = -geometryViewPos.z;

        if (abs(particleDepth - geometryDepth) > radius) {
            return CollisionQuery(false, DUMMY_COLLISION);
        }

        vec3 geometryNormal = normalize(texture(sceneNormal, ndcPos * 0.5 + 0.5).xyz);

        Collision coll;
        coll.pos = vec3(0); // TODO
        coll.normal = geometryNormal;
        return CollisionQuery(true, coll);
    }

    void main() {
        //positions[gl_GlobalInvocationID.x] = vec4(0.0, 0.0, 0.0, 1.0);

        vec3 p0 = positions[gl_GlobalInvocationID.x].xyz;
        vec3 v0 = velocities[gl_GlobalInvocationID.x].xyz;
        float radius = radiuses[gl_GlobalInvocationID.x];
        
        // Apply impluses // TODO: Before or after?
        vec3 v1 = v0 + vec3(0.0, -9.82, 0.0) * deltaTime; // TODO: Euler integration?
        vec3 p1 = p0 + v1 * deltaTime; // TODO: Euler integration?

        CollisionQuery query = checkCollision(p1, radius);
        if (query.colliding) {
            // Collision response
            p1 = p0; // TODO: More accurately find intersection point
            const float RESTITUTION = 0.9;
            v1 = reflect(v1, query.collision.normal) * RESTITUTION;
        }

        positions[gl_GlobalInvocationID.x] = vec4(p1, 1.0);
        velocities[gl_GlobalInvocationID.x] = vec4(v1, 1.0);
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

    layout(std140) buffer ParticleRadiuses {
        float radiuses[];
    };

    out vec2 uv;

    void main() {
        vec3 particlePos = positions[gl_InstanceID].xyz;
        float particleRadius = radiuses[gl_InstanceID];

        vec3 normal = normalize(cameraPos - particlePos); // Viewpoint-facing
        //normal = normalize(inverse(mat3(worldToCameraMatrix)) * vec3(0, 0, 1)); // Viewplane-facing
        
        vec3 upBasis = vec3(0, 1, 0);
        vec3 rightBasis = normalize(cross(upBasis, normal));
        upBasis = cross(rightBasis, normal);
        mat3 billboardMatrix = mat3(rightBasis, upBasis, normal);

        vec2 offset = OFFSETS[gl_VertexID];
        vec3 vertexPos = particlePos + billboardMatrix * vec3(offset * particleRadius, 0);

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
        n = (modelToWorldMatrix * vec4(normal, 0.0)).xyz; // TODO: Proper transform for normal
        uv = texture.xy;
        gl_Position = cameraToClipMatrix * worldToCameraMatrix * modelToWorldMatrix * vec4(position, 1);
    }
"#;

const MESH_FRAGMENT_SHADER_SRC: &str = r#"
    #version 430

    in vec3 n;
    in vec2 uv;

    out vec4 fragColor;
    out vec4 fragNormal;

    void main() {
        vec3 normal = normalize(n);

        const vec3 diffColor = vec3(0.8);
        const vec3 ambientLight = vec3(0.1,0.1,0.2);
        const vec3 lightDir = vec3(1,1,0);
        const vec3 lightColor = vec3(1.0,1.0,0.5);
        float ndotl = max(0.0, dot(normal, lightDir));
        vec3 radiance = (ambientLight + lightColor * ndotl) * diffColor;

        fragColor = vec4(radiance, 1.0);
        fragNormal = vec4(normal, 0.0);
    }
"#;

const FULLSCREEN_VERTEX_SHADER_SRC: &str = r#"
    #version 430

    out vec2 uv;

    void main() {
        // Adapted from https://www.slideshare.net/DevCentralAMD/vertex-shader-tricks-bill-bilodeau
        gl_Position = vec4(
            vec2(gl_VertexID / 2, gl_VertexID % 2) * 4.0 - 1.0,
            0.0,
            1.0
        );
        uv = vec2(
            (gl_VertexID / 2) * 2.0,
            (gl_VertexID % 2) * 2.0
        );
    }
"#;

const BLIT_FRAGMENT_SHADER_SRC: &str = r#"
    #version 430

    uniform sampler2D blit_source_color;
    uniform sampler2D blit_source_depth;

    in vec2 uv;
    out vec4 color;

    void main() {
        color = texture(blit_source_color, uv);
        gl_FragDepth = texture(blit_source_depth, uv).x;
    }
"#;


fn new_particle_buffer<T>(facade: &impl Facade)
                        -> UniformBuffer<[T]> where [T]: Content {
    UniformBuffer::empty_unsized(facade,
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


struct RenderResources {
    g_buffer: GBuffer,
    camera_uniforms_buffer: UniformBuffer<CameraUniforms>,
    particles: Particles,
}

#[self_referencing]
struct GBuffer {
    color: Texture2d,
    normal: Texture2d,
    depth: DepthTexture2d,
    #[borrows(color, normal, depth)]
    #[covariant]
    framebuffer: MultiOutputFrameBuffer<'this>
}

struct Particles {
    pos_buffer: UniformBuffer<[PPos]>,
    vel_buffer: UniformBuffer<[PVel]>,
    radius_buffer: UniformBuffer<[f32]>,
    debug: UniformBuffer<[[f32; 4]]>
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
            color: Texture2d::empty_with_format(&display, UncompressedFloatFormat::F32F32F32F32, MipmapsOption::NoMipmap, width, height).unwrap(),
            normal: Texture2d::empty_with_format(&display, UncompressedFloatFormat::F32F32F32F32, MipmapsOption::NoMipmap, width, height).unwrap(),
            depth: DepthTexture2d::empty_with_format(&display, DepthFormat::F32, MipmapsOption::NoMipmap, width, height).unwrap(),
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

        particles: Particles {
            pos_buffer: new_particle_buffer(&display),
            vel_buffer: new_particle_buffer(&display),
            radius_buffer: new_particle_buffer(&display),
            debug: new_particle_buffer(&display)
        },

        camera_uniforms_buffer: UniformBuffer::empty(&display).unwrap()
    };

    {
        let mut mapping = render_resources.particles.pos_buffer.map();
        for val in mapping.iter_mut() {
            *val = [
                rand::random::<f32>() * 5.0 - 2.5 - 4.0,
                rand::random::<f32>() * 5.0 - 2.5 - 1.0,
                rand::random::<f32>() * 5.0 - 2.5,
                1.0
            ];
        }

        let mut mapping = render_resources.particles.vel_buffer.map();
        for val in mapping.iter_mut() {
            *val = [
                10.0,
                0.0,
                0.0,
                1.0
            ];
        }

        let mut mapping = render_resources.particles.radius_buffer.map();
        for val in mapping.iter_mut() {
            *val = rand::random::<f32>() * 0.1 + 0.05;
        }
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
        MESH_VERTEX_SHADER_SRC, MESH_FRAGMENT_SHADER_SRC, None).unwrap();
    render_resources.g_buffer.with_mut(|g_buffer| {
        g_buffer.framebuffer.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);
        render_scene(g_buffer.framebuffer, time,
            &render_resources.camera_uniforms_buffer, landingpad_mesh, &scene_shader_program);
    });

    target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

    let program = Program::from_source(display,
        FULLSCREEN_VERTEX_SHADER_SRC, BLIT_FRAGMENT_SHADER_SRC, None).unwrap();
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

    let program = ComputeShader::from_source(display, UPDATE_PARTICLES_SRC).unwrap();
    program.execute(
        uniform! {
            time: time,
            deltaTime: delta_time,
            Camera: &*render_resources.camera_uniforms_buffer,
            ParticlePositions: &*render_resources.particles.pos_buffer,
            ParticleVelocities: &*render_resources.particles.vel_buffer,
            ParticleRadiuses: &*render_resources.particles.radius_buffer,
            DebugOutput: &*render_resources.particles.debug,
            sceneDepth: render_resources.g_buffer.borrow_depth(),
            sceneNormal: render_resources.g_buffer.borrow_normal()
        },
        NUM_PARTICLES as u32, 1, 1
    );

    /* {
        let mapping = render_resources.particles.particle_pos_buffer.map();
        for val in mapping.iter().take(1) {
            println!("{:?}", glam::vec4(val[0], val[1], val[2], val[3]));
        }
        println!("...");
    } */
    {
        let mapping = render_resources.particles.debug.map();
        for val in mapping.iter().take(1) {
            println!("{:?}", glam::vec4(val[0], val[1], val[2], val[3]));
        }
        println!("...");
    }

    render_particles(display, target, time, render_resources);
}

fn render_particles(display: &glium::Display, target: &mut impl Surface,
                    time: f32,
                    render_resources: &mut RenderResources) {
    
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
            Camera: &*render_resources.camera_uniforms_buffer,
            ParticlePositions: &*render_resources.particles.pos_buffer,
            ParticleRadiuses: &*render_resources.particles.radius_buffer
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