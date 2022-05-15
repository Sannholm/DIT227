use glium::{VertexBuffer, IndexBuffer, index::PrimitiveType, backend::Facade};

pub struct Mesh {
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
    pub fn load_obj<F: Facade>(facade: &F, bytes: &[u8]) -> Self {
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