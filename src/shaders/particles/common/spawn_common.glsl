uniform uint frameNum;

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

uint randInt()
{
	return randInt(rngState);
}

uint randIntRange(const uint bound)
{
    return randIntRange(rngState, bound);
}

float rand()
{
	return rand(rngState);
}