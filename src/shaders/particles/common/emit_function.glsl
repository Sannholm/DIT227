void emitParticle(float lifetime, vec3 pos, vec3 vel, float radius) {
    uint nextIndex = atomicAdd(aliveCount, 1);

    lifetimes[nextIndex] = lifetime;
    positions[nextIndex] = vec4(pos, 1.0);
    velocities[nextIndex] = vec4(vel, 1.0);
    radiuses[nextIndex] = radius;
}

void emitParticle(vec3 pos, vec3 vel, float radius) {
    emitParticle(0.0, pos, vel, radius);
}