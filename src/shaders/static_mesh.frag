#version 430

{% include "common/camera.glsl" %}

in vec3 pos;
in vec3 n;
in vec2 uv;

out vec4 fragColor;
out vec4 fragNormal;

vec3 normal = normalize(n);

vec3 pointLight(vec3 color, vec3 lightPos) {
    vec3 lightSurfacePosDiff = lightPos - pos;

    float distSq = dot(lightSurfacePosDiff, lightSurfacePosDiff);
    float falloff = 1.0 / distSq;

    vec3 lightDir = normalize(lightSurfacePosDiff);
    float ndotl = max(0.0, dot(normal, lightDir));

    return color * ndotl * clamp(falloff, 0, 1);
}

void main() {
    vec3 diffColor = vec3(0.18);

    vec3 totalIncomingLight = vec3(0);
    
    vec3 ambientLight = vec3(0.1,0.1,0.5) * 0.5;
    totalIncomingLight += ambientLight;

    {
        vec3 lightDir = vec3(1,1,0);
        vec3 lightColor = vec3(0.1,0.1,0.5) * 2;
        float ndotl = max(0.0, dot(normal, lightDir));
        vec3 moon = lightColor * ndotl;
        totalIncomingLight += moon;
    }

    // Interior light
    {
        vec3 color = vec3(255,147,41) / 255 * 5;
        vec3 lightPos = vec3(16.2201, 10.6423, -30.9471);
        totalIncomingLight += pointLight(color, lightPos);
    }

    // Exterior lights
    const vec3 lights[] = vec3[](
        vec3(-2.00865, 5.94997, -67.7607),
        vec3(-5.58865, 5.94997, -67.7607)
    );
    for (int i = 0; i < lights.length(); i++) {
        vec3 color = vec3(255,147,41) / 255 * 5;
        vec3 lightPos = lights[i];
        totalIncomingLight += pointLight(color, lightPos);
    }

    vec3 radiance = totalIncomingLight * diffColor;
    fragColor = vec4(radiance, 1.0);
    fragNormal = vec4(normal, 0.0);
}