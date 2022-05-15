layout(std140) restrict readonly buffer Camera {
    mat4 worldToCameraMatrix;
    mat4 cameraToClipMatrix;
};

const vec3 cameraPos = inverse(worldToCameraMatrix)[3].xyz;