#version 430

{% include "common/camera.glsl" %}
{% include "particles/common/system_buffers.glsl" %}

const vec2 OFFSETS[] = vec2[4](
    vec2(-0.5,  0.5),
    vec2(-0.5, -0.5),
    vec2(0.5,   0.5),
    vec2(0.5,  -0.5)
);

out vec2 uv;

void main() {
    vec3 particlePos = positions[gl_InstanceID].xyz;
    float particleRadius = radiuses[gl_InstanceID];

    vec3 normal = normalize(cameraPos - particlePos); // Viewpoint-facing
    //normal = normalize(mat3(cameraToWorldMatrix) * vec3(0, 0, 1)); // Viewplane-facing
    
    // TODO: Properly handle case when up and normal are close
    vec3 upBasis = vec3(0, 1, 0);
    vec3 rightBasis = normalize(cross(upBasis, normal));
    upBasis = cross(rightBasis, normal);
    mat3 billboardMatrix = mat3(rightBasis, upBasis, normal);

    vec2 offset = OFFSETS[gl_VertexID];
    vec3 vertexPos = particlePos + billboardMatrix * vec3(offset * particleRadius, 0);

    uv = offset + vec2(0.5);
    gl_Position = cameraToClipMatrix * worldToCameraMatrix * vec4(vertexPos, 1);
}