layout(std140) restrict readonly buffer Camera {
    mat4 worldToCameraMatrix;
    mat4 cameraToWorldMatrix;
    mat4 cameraToClipMatrix;
    mat4 clipToCameraMatrix;
};

vec3 cameraPos() {
    return cameraToWorldMatrix[3].xyz;
}