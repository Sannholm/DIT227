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