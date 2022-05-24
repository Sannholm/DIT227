#version 430

{% include "particles/common/system_buffers.glsl" %}

in flat uint particleId;
in vec2 uv;

out vec4 color;

void main() {
    const float MAX_LIFETIME = 20; // Seconds

    float lifetime = lifetimes[particleId];

    const float fadeOutDuration = 1;
    float fadeOutStart = MAX_LIFETIME - fadeOutDuration;
    float fadeOutFactor = smoothstep(0, 1, clamp((lifetime - fadeOutStart) / fadeOutDuration, 0, 1));
    
    float distFromCenter = length((uv - vec2(0.5)) / (1 - fadeOutFactor)) * 2;
    float alpha = smoothstep(1, 0, distFromCenter) * (1 - fadeOutFactor);
    color = vec4(vec3(1), alpha);
}