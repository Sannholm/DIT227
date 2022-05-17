layout(std140) restrict readonly buffer Camera {
    mat4 worldToCameraMatrix;
    mat4 cameraToWorldMatrix;
    mat4 cameraToClipMatrix;
    mat4 clipToCameraMatrix;
};

const vec3 cameraPos = cameraToWorldMatrix[3].xyz;