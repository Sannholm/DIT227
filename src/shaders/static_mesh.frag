#version 430

{% include "common/camera.glsl" %}

in vec3 pos;
in vec3 n;
in vec2 uv;

out vec4 fragColor;
out vec4 fragNormal;

void main() {
    vec3 normal = normalize(n);

    vec3 diffColor = vec3(0.18);
    vec3 ambientLight = vec3(0.1,0.1,0.5) * 0.5;

    vec3 moon;
    {
        vec3 lightDir = vec3(1,1,0);
        vec3 lightColor = vec3(0.1,0.1,0.5) * 2;
        float ndotl = max(0.0, dot(normal, lightDir));
        moon = lightColor * ndotl;
    }

    vec3 interiorLight;
    {
        vec3 color = vec3(255,147,41) / 255 * 5;

        vec3 lightPos = vec3(16.2201, 10.6423, -30.9471);
        vec3 lightSurfacePosDiff = lightPos - pos;

        float distSq = dot(lightSurfacePosDiff, lightSurfacePosDiff);
        float falloff = 1.0 / distSq;

        vec3 lightDir = normalize(lightSurfacePosDiff);
        float ndotl = max(0.0, dot(normal, lightDir));

        interiorLight = color * ndotl * clamp(falloff, 0, 1);
    }

    vec3 radiance = (ambientLight + moon + interiorLight) * diffColor;

    fragColor = vec4(radiance, 1.0);
    fragNormal = vec4(normal, 0.0);
}