uniform uint frameNum;
uniform float time;

{% include "common/camera.glsl" %}

{% include "particles/common/system_buffers.glsl" %}

uniform sampler2D sceneDepth;
uniform sampler2D sceneNormal;

{% include "particles/common/emit_function.glsl" %}

{% include "common/random.glsl" %}

RngState rngState;

void initRng() {
    // One unique random stream per frame and per invocation of compute program
    uint invocationIndex = gl_LocalInvocationIndex;
    uint numInvocations = gl_WorkGroupSize.x + gl_WorkGroupSize.y + gl_WorkGroupSize.z;
    initRng(rngState, frameNum * numInvocations + invocationIndex);
}

uint randInt() {
	return randInt(rngState);
}

uint randIntRange(const uint bound) {
    return randIntRange(rngState, bound);
}

float rand() {
	return rand(rngState);
}

float randBetween(float p1, float p2) {
    return mix(p1, p2, rand());
}

vec2 randBetween(vec2 p1, vec2 p2) {
    return mix(p1, p2, vec2(rand(), rand()));
}

vec3 randBetween(vec3 p1, vec3 p2) {
    return mix(p1, p2, vec3(rand(), rand(), rand()));
}