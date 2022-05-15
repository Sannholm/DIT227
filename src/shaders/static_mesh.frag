#version 430

in vec3 n;
in vec2 uv;

out vec4 fragColor;
out vec4 fragNormal;

void main() {
    vec3 normal = normalize(n);

    const vec3 diffColor = vec3(0.8);
    const vec3 ambientLight = vec3(0.1,0.1,0.2);
    const vec3 lightDir = vec3(1,1,0);
    const vec3 lightColor = vec3(1.0,1.0,0.5);
    float ndotl = max(0.0, dot(normal, lightDir));
    vec3 radiance = (ambientLight + lightColor * ndotl) * diffColor;

    fragColor = vec4(radiance, 1.0);
    fragNormal = vec4(normal, 0.0);
}