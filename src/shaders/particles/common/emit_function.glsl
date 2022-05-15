void emitParticle(vec3 pos, vec3 vel, float radius) {
    uint nextIndex = atomicAdd(aliveCount, 1);

    positions[nextIndex] = vec4(pos, 1.0);
    velocities[nextIndex] = vec4(vel, 1.0);
    radiuses[nextIndex] = radius;
}