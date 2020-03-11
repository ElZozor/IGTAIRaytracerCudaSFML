#pragma once

#include "Scene.cuh"
#include "SceneObjects.cuh"


struct CameraParameters
{
	glm::vec3 position;
	glm::vec3 at;
	glm::vec3 up;
	float fov;
	float ratio;
};

Scene* initScene(int sceneNumber);

inline CameraParameters getSceneCameraParameters()
{
	return CameraParameters {
		glm::vec3(4.5, .8, 4.5), 
		glm::vec3(0, 0.3, 0), 
		glm::vec3(0, 1, 0), 
		60.f, 
		float(s_width) / float(s_height)
	};
}