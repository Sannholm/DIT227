layout(std140) restrict buffer AliveCount {
    uint aliveCount;
};

layout(std140) restrict buffer Lifetimes {
    float lifetimes[];
};

layout(std140) restrict buffer Positions {
    vec4 positions[];
};

layout(std140) restrict buffer Velocities {
    vec4 velocities[];
};

layout(std140) restrict buffer Radiuses {
    float radiuses[];
};

layout(std140) restrict writeonly buffer DebugOutput {
    vec4 debug[];
};