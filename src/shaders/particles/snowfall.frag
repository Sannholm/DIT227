#version 430

{% include "common/camera.glsl" %}
{% include "particles/common/system_buffers.glsl" %}

in flat uint particleId;
in vec3 pos;
in vec2 uv;

out vec4 color;

vec2 mapping = uv * 2 - 1;
vec3 normal = normalize((cameraToWorldMatrix * vec4(mapping, sqrt(1-dot(mapping,mapping)), 0)).xyz);
vec3 spherePos = pos + normal * radiuses[particleId];

vec3 pointLight(vec3 color, vec3 lightPos) {
    vec3 lightSurfacePosDiff = lightPos - spherePos;

    float distSq = dot(lightSurfacePosDiff, lightSurfacePosDiff);
    float falloff = 1.0 / distSq;

    vec3 lightDir = normalize(lightSurfacePosDiff);
    float ndotl = max(0.0, dot(normal, lightDir) * 0.5 + 0.5);

    return color * ndotl * clamp(falloff, 0, 1);
}

void main() {
    vec3 diffuseColor = vec3(1);

    vec3 totalIncomingLight = vec3(0);

    vec3 ambientLight = vec3(0.1,0.1,0.5) * 0.5;
    totalIncomingLight += ambientLight;

    {
        vec3 lightDir = vec3(1,1,0);
        vec3 lightColor = vec3(0.1,0.1,0.5) * 2;
        float ndotl = max(0.0, dot(normal, lightDir) * 0.5 + 0.5);
        vec3 moon = lightColor * ndotl;
        totalIncomingLight += moon;
    }

    // Interior light
    {
        vec3 color = vec3(255,147,41) / 255 * 5;
        vec3 lightPos = vec3(16.2201, 10.6423, -30.9471);
        totalIncomingLight += pointLight(color, lightPos);
    }


    const float MAX_LIFETIME = 20; // Seconds
    float lifetime = lifetimes[particleId];

    const float fadeOutDuration = 2;
    float fadeOutStart = MAX_LIFETIME - fadeOutDuration;
    float fadeOutFactor = smoothstep(0, 1, clamp((lifetime - fadeOutStart) / fadeOutDuration, 0, 1));
    
    float distFromCenter = length((uv - vec2(0.5)) / (1 - fadeOutFactor)) * 2;
    float alpha = smoothstep(1, 0, distFromCenter) * (1 - fadeOutFactor);
    color = vec4(diffuseColor * totalIncomingLight, alpha);

    vec4 sphereClipPos = cameraToClipMatrix * worldToCameraMatrix * vec4(spherePos, 1);
    float ndcDepth = sphereClipPos.z / sphereClipPos.w;
    gl_FragDepth = (1.0 - 0.0) * 0.5 * ndcDepth + (1.0 + 0.0) * 0.5;

    if (alpha <= 0) {
        discard;
    }
}