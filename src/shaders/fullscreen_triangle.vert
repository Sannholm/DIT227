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