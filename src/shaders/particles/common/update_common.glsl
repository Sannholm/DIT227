layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

uniform uint frameNum;
uniform float time;
uniform float deltaTime;

{% include "common/camera.glsl" %}

uniform uint prevAliveCount;

layout(std140) restrict readonly buffer PrevLifetimes {
    float prevLifetimes[];
};

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

void updateParticle(float t0, vec3 p0, vec3 v0, float radius);

void main() {
    // Ignore extra invocations when number of alive particles is not
    // a multiple of work group size
    if (gl_GlobalInvocationID.x >= prevAliveCount)
        return;
    
    float t0 = prevLifetimes[gl_GlobalInvocationID.x];
    vec3 p0 = prevPositions[gl_GlobalInvocationID.x].xyz;
    vec3 v0 = prevVelocities[gl_GlobalInvocationID.x].xyz;
    float radius = prevRadiuses[gl_GlobalInvocationID.x];
    
    updateParticle(t0, p0, v0, radius);
}

{% include "common/random.glsl" %}

RngState rngState;

void initRng() {
    // One unique random stream per frame and per alive particle
    // Note: Potential overlap of streams between frames since alive count changes
    uint particleId = gl_GlobalInvocationID.x;
    initRng(rngState, frameNum * prevAliveCount + particleId);
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