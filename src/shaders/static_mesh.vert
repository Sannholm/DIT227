#version 430

{% include "common/camera.glsl" %}

uniform mat4 modelToWorldMatrix;

in vec3 position;
in vec3 normal;
in vec3 texture;

out vec3 pos;
out vec3 n;
out vec2 uv;

void main() {
    vec4 worldPos = modelToWorldMatrix * vec4(position, 1);

    pos = worldPos.xyz;
    n = (modelToWorldMatrix * vec4(normal, 0.0)).xyz; // TODO: Proper transform for normal
    uv = texture.xy;

    gl_Position = cameraToClipMatrix * worldToCameraMatrix * worldPos;
}