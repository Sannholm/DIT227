uniform float time;
uniform float deltaTime;

{% include "common/camera.glsl" %}

uniform uint prevAliveCount;

layout(std140) restrict readonly buffer PrevPositions {
    vec4 prevPositions[];
};

layout(std140) restrict readonly buffer PrevVelocities {
    vec4 prevVelocities[];
};

layout(std140) restrict readonly buffer PrevRadiuses {
    float prevRadiuses[];
};

{% include "particles/common/system_buffers.glsl" %}

uniform sampler2D sceneDepth;
uniform sampler2D sceneNormal;

struct Collision {
    vec3 pos;
    vec3 normal;
};
const Collision DUMMY_COLLISION = Collision(vec3(0.0), vec3(0.0));

struct CollisionQuery {
    bool colliding;
    Collision collision;
};

CollisionQuery checkCollision(vec3 pos, float radius) {
    vec4 viewPos = worldToCameraMatrix * vec4(pos, 1.0);
    vec4 clipPos = cameraToClipMatrix * viewPos;
    vec2 ndcPos = clipPos.xy / clipPos.w;

    float geometryNdcDepth = texture(sceneDepth, ndcPos * 0.5 + 0.5).x * 2.0 - 1.0;
    vec4 geometryNdcPos = vec4(ndcPos, geometryNdcDepth, 1.0);
    vec4 geometryViewPos = clipToCameraMatrix * geometryNdcPos;
    geometryViewPos /= geometryViewPos.w;

    float particleDepth = -viewPos.z;
    float geometryDepth = -geometryViewPos.z;

    if (abs(particleDepth - geometryDepth) > radius) {
        return CollisionQuery(false, DUMMY_COLLISION);
    }

    vec3 geometryNormal = normalize(texture(sceneNormal, ndcPos * 0.5 + 0.5).xyz);

    Collision coll;
    coll.pos = vec3(0); // TODO
    coll.normal = geometryNormal;
    return CollisionQuery(true, coll);
}

{% include "particles/common/emit_function.glsl" %}