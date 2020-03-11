#pragma once

#include "cuda_headers.h"
#include "SceneObjects.cuh"
#include "Scene.cuh"

#include <SFML/Graphics.hpp>
#include <vector>


struct Intersection
{
	glm::vec3 position;
	glm::vec3 normal;
	Material* material;
};


__host__ sf::Uint8* renderScene(Scene* scene);