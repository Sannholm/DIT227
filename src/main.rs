use std::mem::size_of;

use glium::VertexBuffer;
use glium::uniforms::UniformBuffer;
use glium::{glutin, Surface};
use rand;

#[macro_use]
extern crate glium;

const NUM_PARTICLES: usize = 4096;


#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}
implement_vertex!(Vertex, position);

type ParticlePos = [f32; 4];

const UPDATE_PARTICLES_SRC: &str = r#"
    #version 430

    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

    layout(std140) buffer MyBlock {
        vec4 values[];
    };

    void main() {
        vec4 val = values[gl_GlobalInvocationID.x];
        values[gl_GlobalInvocationID.x] += 0.001f;
    }
"#;

const VERTEX_SHADER_SRC: &str = r#"
    #version 430

    layout(std140) buffer MyBlock {
        vec4 values[];
    };

    uniform mat4 matrix;

    in vec2 position;
    out vec2 my_attr;      // our new attribute

    void main() {
        my_attr = position;     // we need to set the value of each `out` variable.
        gl_Position = matrix * (vec4(position, 0.0, 1.0) + values[gl_InstanceID]);
    }
"#;

const FRAGMENT_SHADER_SRC: &str = r#"
    #version 430

    in vec2 my_attr;
    out vec4 color;

    void main() {
        color = vec4(my_attr, 0.0, 1.0);   // we build a vec4 from a vec2 and two floats
    }
"#;

fn main() {
    let event_loop = glutin::event_loop::EventLoop::new();
    let wb = glutin::window::WindowBuilder::new();
    let cb = glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &event_loop).unwrap();

    let mut buffer: glium::uniforms::UniformBuffer<[ParticlePos]> =
              glium::uniforms::UniformBuffer::empty_unsized(&display,
                NUM_PARTICLES * size_of::<ParticlePos>()).unwrap();

    {
        let mut mapping = buffer.map();
        for val in mapping.iter_mut() {
            *val = [rand::random::<f32>(),rand::random::<f32>(),rand::random::<f32>(),rand::random::<f32>()];
        }
    }

    let mut t = -0.5;

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

        let next_frame_time = std::time::Instant::now() +
            std::time::Duration::from_nanos(16_666_667);
        *control_flow = glutin::event_loop::ControlFlow::WaitUntil(next_frame_time);
        
        t += 0.02;
        if t > 0.5 {
            t = -0.5;
        }

        let mut target = display.draw();
        render(&display, &mut target, t, &mut buffer);
        target.finish().unwrap();
    });
}

fn render(display: &glium::Display, target: &mut glium::Frame, t: f32, buffer: &mut UniformBuffer<[ParticlePos]>) {
        
    target.clear_color(0.0, 0.0, 1.0, 1.0);

    let program = glium::program::ComputeShader::from_source(display, UPDATE_PARTICLES_SRC).unwrap();
    program.execute(uniform! { MyBlock: &*buffer }, NUM_PARTICLES as u32, 1, 1);

    /* {
        let mapping = buffer.map();
        for val in mapping.iter().take(3) {
            println!("{:?}", glam::vec4(val[0], val[1], val[2], val[3]));
        }
        println!("...");
    } */

    let uniforms = uniform! {
        matrix: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0 , 0.0, 0.0, 1.0f32],
        ],
        MyBlock: &*buffer
    };

    let shape = vec![
        Vertex { position: [-0.5, -0.5] },
        Vertex { position: [ 0.0,  0.5] },
        Vertex { position: [ 0.5, -0.25] }
    ];

    let vertex_buffer = glium::VertexBuffer::new(display, &shape).unwrap();
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

    let program = glium::Program::from_source(display, VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, None).unwrap();
    target.draw(
        (&vertex_buffer, glium::vertex::EmptyInstanceAttributes { len: NUM_PARTICLES }),
        &indices, &program, &uniforms, &Default::default()
    ).unwrap();
}