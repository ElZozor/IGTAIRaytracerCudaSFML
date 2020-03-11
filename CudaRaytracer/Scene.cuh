#pragma once

#include <glm/gtc/quaternion.hpp>
#include "SceneObjects.cuh"

#define M_PI    3.141592653589793238462643383279502884
#define M_PIf32 float(3.141592653589793238462643383279502884)


namespace SceneParameters
{
	extern size_t s_width;
	extern size_t s_height;
};

using namespace SceneParameters;

struct Camera
{
    glm::vec3 xdir, ydir, zdir;
    glm::vec3 position;
    glm::vec3 center;

    float fov;
    float aspect;

    size_t width, height;
};

struct Scene
{
    color3 skyColor;
    Camera camera;

    Light* lights;
    size_t lightsCount;

	Object* objects;
	size_t objectsCount;
};



//Allocation and transfert functions
__host__ Scene* createSceneOnHost();
__host__ Scene* createSceneOnBoth();
__host__ Scene* transfertSceneToDevice(Scene* scene);



//Free functions
__host__ void freeSceneOnHost(Scene** scene);
__host__ void freeSceneOnDevice(Scene* scene);



inline __host__ void setCamera(Scene* scene, glm::vec3 position, glm::vec3 at, glm::vec3 up, float fov, float aspect) {
    scene->camera.width = s_width;
    scene->camera.height = s_height;
    scene->camera.fov = fov;
    scene->camera.aspect = aspect;
    scene->camera.position = position;
    scene->camera.zdir = glm::normalize(at - position);
    scene->camera.xdir = glm::normalize(glm::cross(up, scene->camera.zdir));
    scene->camera.ydir = glm::normalize(glm::cross(scene->camera.zdir, scene->camera.xdir));
    scene->camera.center = 1.f / tanf((scene->camera.fov * M_PI / 180.f) * 0.5f) * scene->camera.zdir;
}


inline __host__ glm::quat getQuaternion(float rad, glm::vec3& pivot) {
    float x = pivot.x * sin(rad / 2);
    float y = pivot.y * sin(rad / 2);
    float z = pivot.z * sin(rad / 2);
    float w = cos(rad / 2);

    return glm::quat(w, x, y, z);
}


inline __host__ void rotateCamera(Camera* camera, glm::vec3 rotationVector, float angle)
{
    glm::quat pivotRotation = getQuaternion(angle, rotationVector);

    camera->xdir = pivotRotation * camera->xdir;
    camera->ydir = pivotRotation * camera->ydir;
    camera->zdir = pivotRotation * camera->zdir;
    camera->center = 1.f / tanf((camera->fov * M_PI / 180.f) * 0.5f) * camera->zdir;
}
