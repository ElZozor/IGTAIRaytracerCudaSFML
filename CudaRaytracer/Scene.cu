#include "Scene.cuh"

#include <malloc.h>
#include <assert.h>



size_t SceneParameters::s_width = 800;
size_t SceneParameters::s_height = 600;




__host__ Scene* createSceneOnDevice()
{
	Scene* scene;
	cudaMalloc(&scene, sizeof(Scene));

	return scene;
}

__host__ Object* allocateObjectsMemoryOnDevice(size_t n)
{
	Object* objects;
	cudaMalloc(&objects, sizeof(Object) * n);

	return objects;
}


__host__ Scene* createSceneOnHost()
{
	Scene* scene = (Scene*) malloc(sizeof(Scene));
	assert(scene != nullptr);

	scene->objects = nullptr;
	scene->objectsCount = 0;

	return scene;
}

__host__ Scene* createSceneOnBoth()
{
	Scene* scene;
	cudaMallocManaged(&scene, sizeof(Scene));

	scene->objects = nullptr;
	scene->objectsCount = 0;

	return scene;
}

__host__ Scene* transfertSceneToDevice(Scene* scene)
{
	assert(scene != nullptr);

	Scene* gpuScene  = createSceneOnDevice();
	Object* sObjects = scene->objects;
	Light* sLights   = scene->lights;


	if (scene->objectsCount > 0)
	{
		Object* gpuSceneObjects;
		cudaMalloc(&gpuSceneObjects, sizeof(Object) * scene->objectsCount);
		cudaMemcpy((void*)gpuSceneObjects, (void*)scene->objects, sizeof(Object) * scene->objectsCount, cudaMemcpyHostToDevice);
		
		scene->objects = gpuSceneObjects;
	}

	if (scene->lightsCount > 0)
	{
		Light* gpuSceneLights;
		cudaMalloc(&gpuSceneLights, sizeof(Light) * scene->lightsCount);
		cudaMemcpy((void*)gpuSceneLights, (void*)scene->lights, sizeof(Light) * scene->lightsCount, cudaMemcpyHostToDevice);

		scene->lights = gpuSceneLights;
	}

	cudaMemcpy((void*)gpuScene, (void*)scene, sizeof(Scene), cudaMemcpyHostToDevice);
	
	scene->objects = sObjects;
	scene->lights  = sLights;

	return gpuScene;
}

__host__ void freeSceneOnHost(Scene** scene)
{
	assert(scene != nullptr && *scene != nullptr);

	//Freeing scene objects first
	if ((*scene)->objects != nullptr)
	{
		free((*scene)->objects);
		(*scene)->objects = nullptr;
	}


	if ((*scene)->lights != nullptr)
	{
		free((*scene)->lights);
		(*scene)->lights = nullptr;
	}


	//Freeing scene afterwards
	free(*scene);
	*scene = nullptr;
}

__host__ void freeSceneOnDevice(Scene* scene)
{
	cudaFree(scene);
}
