{% include "common/camera.glsl" %}

{% include "particles/common/system_buffers.glsl" %}

uniform sampler2D sceneDepth;
uniform sampler2D sceneNormal;

{% include "particles/common/emit_function.glsl" %}