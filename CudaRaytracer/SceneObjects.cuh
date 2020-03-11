#pragma once

#include "cuda_headers.h"

typedef glm::vec3 color3;
#define BLACK color3(0)

enum ObjectType
{
	SPHERE,
	PLANE
};

struct Sphere
{
	glm::vec3 center;
	float radius;
};


struct Plane
{
	glm::vec3 normal;
	float dist;
};


struct Material
{
	float IOR;
	float roughness;
	color3 specularColor;
	color3 diffuseColor;
};


struct Object
{
	union 
	{
		Sphere sphere;
		Plane plane;
	} Geometry;

	ObjectType type;
	Material mat;
};


struct Ray
{
	glm::vec3 orig;
	glm::vec3 dir;

	int depth = 0;
	int maxdepth = 10;

	float tmin = 0.f;
	float tmax = 100000.f;
};


struct Light
{
	glm::vec3 position;
	color3 color;
};