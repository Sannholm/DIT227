#version 430

uniform sampler2D blit_source_color;
uniform sampler2D blit_source_depth;

in vec2 uv;
out vec4 color;

void main() {
    color = texture(blit_source_color, uv);
    gl_FragDepth = texture(blit_source_depth, uv).x;
}