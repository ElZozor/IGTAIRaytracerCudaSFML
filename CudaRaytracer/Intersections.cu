#include "Intersections.cuh"


static bool gpuInitialized = false;
static sf::Uint8* p;

#define acne_eps 1e-4f;


struct CamParameters
{
    float delta_y;
    glm::vec3 dy;
    glm::vec3 ray_delta_y;

    float delta_x;
    glm::vec3 dx;
    glm::vec3 ray_delta_x;
};




#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
 inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__device__ bool intersectPlane(Ray* ray, Object* obj, Intersection* intersection)
{
    const Plane& plane = obj->Geometry.plane;
    const glm::vec3& d = ray->dir;
    const glm::vec3& O = ray->orig;
    const glm::vec3& n = plane.normal;
    const float& D = plane.dist;

    if (glm::dot(d, n) == 0)
    {
        return false;
    }

    const float t = -((glm::dot(O, n) + D) / glm::dot(d, n));

    if (t < ray->tmin || t > ray->tmax)
    {
        return false;
    }

    intersection->material = &obj->mat;
    intersection->normal   = plane.normal;
    intersection->position = (ray->dir * t) + ray->orig;

    ray->tmax = t;

    return true;
}





__device__ bool intersectSphere(Ray* ray, Object* obj, Intersection* intersection)
{
    const Sphere& sphere = obj->Geometry.sphere;
    const float& R = sphere.radius;
    const glm::vec3& d = ray->dir;
    const glm::vec3 OC = ray->orig - sphere.center;

    const float b = 2.f * glm::dot(d, OC);
    const float c = (glm::dot(OC, OC) - R * R);

    const float delta = (b * b) - 4.f * c;
    if (delta < 0)
    {
        return false;
    }

    float t;
    if (delta == 0)
    {
        t = (-b) / 2.f;
    }
    else
    {
        float t1 = ((-b) - sqrtf(delta)) / 2.f;
        float t2 = ((-b) + sqrtf(delta)) / 2.f;

        if (t1 > t2)
        {
            const float t3 = t1;
            t1 = t2;
            t2 = t3;
        }

        if (t1 > ray->tmin&& t1 < ray->tmax)
        {
            t = t1;
        }
        else
        {
            t = t2;
        }
    }

    if (t < ray->tmin || t > ray->tmax)
    {
        return false;
    }

    intersection->material = &obj->mat;
    intersection->position = (ray->dir * t) + ray->orig;
    intersection->normal = glm::normalize(intersection->position - sphere.center);

    ray->tmax = t;

    return true;
}


__device__ bool intersectsObject(Ray* ray, Object* object, Intersection* intersection)
{
    switch (object->type)
    {
    case SPHERE :
        return intersectSphere(ray, object, intersection);
        
    case PLANE :
        return intersectPlane(ray, object, intersection);
    }

    return false;
}


__device__ bool intersectsScene(Scene* scene, Ray* ray, Intersection* intersection)
{
    bool intersectsScene = false;
    for (size_t i = 0; i < scene->objectsCount; ++i)
    {
        bool intersected = intersectsObject(ray, scene->objects + i, intersection);
        intersectsScene |= intersected;
    }

    return intersectsScene;
}



__device__ bool intersectsAnObject(Scene* scene, Ray* ray, Intersection* intersection)
{
    for (size_t i = 0; i < scene->objectsCount; ++i)
    {
        if (intersectsObject(ray, &scene->objects[i], intersection))
        {
            return true;
        }
    }

    return false;
}





__device__ inline float RDM_chiplus(float c) {
    return (c > 0.f) ? 1.f : 0.f; 
}

/** Normal Distribution Function : Beckmann
 * NdotH : Norm . Half
 */

__device__ float RDM_Beckmann(float NdotH, float alpha)
{
    const float cosSquared = NdotH * NdotH;
    const float tanOHSquared = (1.f - cosSquared) / cosSquared;
    const float alphaSquared = alpha * alpha;
    const float numerateur = expf((-tanOHSquared) / (alphaSquared));
    const float denominateur = M_PI * alphaSquared * (cosSquared * cosSquared);


    return RDM_chiplus(NdotH) * (numerateur / denominateur);
}

// Fresnel term computation. Implantation of the exact computation. we can use
// the Schlick approximation
// LdotH : Light . Half
__device__ float RDM_Fresnel(float cosOi, float n1, float n2)
{

    const float n1dn2 = (n1 / n2);
    float sin2Ot = (n1dn2 * n1dn2) * (1 - (cosOi * cosOi));
    if (sin2Ot > 1.f)
    {
        return 1.f;
    }

    const float cosOt = sqrt(1.f - sin2Ot);

    const float rs = (powf(n1 * cosOi - n2 * cosOt, 2.f)) / (powf(n1 * cosOi + n2 * cosOt, 2.f));
    const float rp = (powf(n1 * cosOt - n2 * cosOi, 2.f)) / (powf(n1 * cosOt + n2 * cosOi, 2.f));

    return 0.5f * (rs + rp);
}

// DdotH : Dir . Half
// HdotN : Half . Norm
__device__ float RDM_G1(float DdotH, float DdotN, float alpha)
{
    const float tanOx = (sqrtf(1.f - (DdotN * DdotN)) / DdotN);
    const float b = (1.f / (alpha * tanOx));
    const float k = (DdotH / DdotN);

    if (b < 1.6f)
    {
        return RDM_chiplus(k) * ((3.535f * b + 2.181f * (b * b)) / (1.f + 2.276f * b + 2.577f * (b * b)));
    }

    return RDM_chiplus(k);
}

// LdotH : Light . Half | v
// LdotN : Light . Norm | l
// VdotH : View . Half    | h
// VdotN : View . Norm    | n
__device__ float RDM_Smith(float LdotH, float LdotN, float VdotH, float VdotN,
    float alpha)
{

    const float G1A = RDM_G1(LdotH, LdotN, alpha);
    const float G1B = RDM_G1(VdotH, VdotN, alpha);

    return G1A * G1B;
}

// Specular term of the Cook-torrance bsdf
// LdotH : Light . Half
// NdotH : Norm . Half
// VdotH : View . Half
// LdotN : Light . Norm
// VdotN : View . Norm
__device__ color3 RDM_bsdf_s(float LdotH, float NdotH, float VdotH, float LdotN,
    float VdotN, Material* m)
{
    const color3& ks = m->specularColor;
    const float D = RDM_Beckmann(NdotH, m->roughness);
    const float F = RDM_Fresnel(LdotH, 1.f, m->IOR);
    const float G = RDM_Smith(LdotH, LdotN, VdotH, VdotN, m->roughness);


    return ks * ((D * F * G) / (4.f * LdotN * VdotN));
}

 // diffuse term of the cook torrance bsdf
__device__ color3 RDM_bsdf_d(Material *m)
 {
     return m->diffuseColor / M_PIf32;
 }


// diffuse term of the cook torrance bsdf
__device__ color3 RDM_bsdf_d(const color3& c)
{
    return c / M_PIf32;
}

__device__ color3 RDM_bsdf(float LdotH, float NdotH, float VdotH, float LdotN,
    float VdotN, Material* m, const color3& color)
{
    const color3 rightTerm = RDM_bsdf_s(LdotH, NdotH, VdotH, LdotN, VdotN, m);
    const color3 leftTerm = RDM_bsdf_d(color);

    return (leftTerm + rightTerm);
}

__device__ color3 shade(glm::vec3 n, glm::vec3 v, glm::vec3 l, color3 lc, Material* mat, const color3& color)
{
    const glm::vec3 h = normalize(v + l);
    const float LdotH = dot(l, h), NdotH = dot(n, h),
        VdotH = dot(v, h), LdotN = dot(l, n), VdotN = dot(v, n);

    color3 ret = lc * RDM_bsdf(LdotH, NdotH, VdotH, LdotN, VdotN, mat, color) * LdotN;

    
    return glm::clamp(ret, 0.f, 1.f);
}






__device__ color3 computeShadows(Scene* scene, Ray* ray, Intersection* intersection)
{
    color3 color(0);
    for (size_t i = 0; i < scene->lightsCount; ++i)
    {
        const Light& light = scene->lights[i];
        const glm::vec3& L = light.position;
        const glm::vec3& P = intersection->position;
        const glm::vec3 l = glm::normalize(L - P);

        Ray shadowRay;
        shadowRay.orig = intersection->position;
        shadowRay.dir = l;
        shadowRay.tmin = acne_eps;
        shadowRay.tmax = glm::distance(L, intersection->position);

        Intersection shadowIntersection;
        if (!intersectsAnObject(scene, &shadowRay, &shadowIntersection))
        {
            color += shade(intersection->normal, -ray->dir, l, light.color, intersection->material, intersection->material->diffuseColor);
        }
    }

    return color;
}



__device__ inline Ray computeReflectionRay(Ray* ray, Intersection* intersection)
{
    const glm::vec3 reflectionDir = glm::reflect(ray->dir, intersection->normal);
    const int add = glm::max(1, 10 - int(intersection->material->IOR * 10));

    Ray reflectionRay;
    reflectionRay.orig = intersection->position + (reflectionDir * 0.001f);
    reflectionRay.dir = reflectionDir;
    reflectionRay.tmin = acne_eps;
    reflectionRay.tmax = 100000.f;
    reflectionRay.depth = ray->depth + add;

    return reflectionRay;
}




__device__ color3 traceRay(int depth, Scene* scene, Ray* ray)
{
    if (depth > 3)
    {
        return BLACK;
    }

    Intersection intersection;
    if (!intersectsScene(scene, ray, &intersection))
    {
        return scene->skyColor;
    }

    color3 color = computeShadows(scene, ray, &intersection);
    Ray reflectionRay = computeReflectionRay(ray, &intersection);

    //const color3 cr = BLACK; 
    const float F = glm::min(1.f, RDM_Fresnel(glm::dot(reflectionRay.dir, intersection.normal), 1.f, intersection.material->IOR));
    const color3 cr = traceRay(depth + 1, scene, &reflectionRay);

    return color + F * cr * intersection.material->specularColor;
}




__device__ color3 traceRay(Scene* scene, Ray* ray)
{
    Ray r = *ray;
    color3 pixel(0);
    float frac = 1.f;
    for (int i = 0; i < 3; ++i)
    {
        Intersection intersection;
        if (!intersectsScene(scene, ray, &intersection))
        {
            return pixel;
        }

        color3 color = computeShadows(scene, ray, &intersection);
        r = computeReflectionRay(&r, &intersection);

        const float F = glm::min(1.f, RDM_Fresnel(glm::dot(r.dir, intersection.normal), 1.f, intersection.material->IOR));
        pixel += (color + (1.f - intersection.material->IOR) * F * intersection.material->specularColor) * frac;

        frac *= intersection.material->IOR;
    }

    return pixel;
}





/**
 * @brief Transform a pixel into it's greyscale value
 *
 */
#define GREY_SCALE(color) (                         \
      color[0] * 0.07f                              \
    + color[1] * 0.72f                              \
    + color[2] * 0.21f                              \
)


__device__ void renderPixel(Scene* scene, sf::Uint8* pixels, sf::Int8* greyScaleImage, CamParameters* cam, float aa)
{
    if (aa < 1) return;

    const int height      = int(scene->camera.height);
    const int width       = int(scene->camera.width);
    const float y         = float((blockIdx.x * blockDim.x + threadIdx.x) / scene->camera.width);
    const float x         = float((blockIdx.x * blockDim.x + threadIdx.x) % scene->camera.width);
    const float aspect    = 1.f / scene->camera.aspect;
    const unsigned int pg = unsigned int((height - y) * width + x);
    const unsigned int p  = pg * 4;

    if (p > height* width * 4)
    {
        return;
    }

    float add = 1.f / aa;
    color3 c(0);
    for (float xx = x; xx < x + 1.f; xx += add)
    {
        for (float yy = y; yy < y + 1.f; yy += add)
        {
            const glm::vec3 raydir = scene->camera.center + (cam->ray_delta_x)+(cam->ray_delta_y)+xx * (cam->dx)+yy * (cam->dy);
            Ray ray;
            ray.orig = scene->camera.position;
            ray.dir = glm::normalize(raydir);
            ray.tmin = 0;
            ray.tmax = 100000.f;
            ray.depth = 0;

            c += traceRay(0, scene, &ray);
        }
    }

    c *= add * add;
    
    
    pixels[p+0] = int(glm::min(c.x, 1.f) * 255.f);
    pixels[p+1] = int(glm::min(c.y, 1.f) * 255.f);
    pixels[p+2] = int(glm::min(c.z, 1.f) * 255.f);
    pixels[p+3] = 255;

    if (greyScaleImage != nullptr)
    {
        greyScaleImage[pg] = GREY_SCALE((pixels + p));
    }
}



__global__ void renderPixelFromHost(Scene* scene, sf::Uint8* pixels, sf::Int8* greyScaleImage, CamParameters* cam)
{
    renderPixel(scene, pixels, greyScaleImage, cam, 1.f);
}












/*
   _____       _          _
  / ____|     | |        | |
 | (___   ___ | |__   ___| |
  \___ \ / _ \| '_ \ / _ \ |
  ____) | (_) | |_) |  __/ |
 |_____/ \___/|_.__/ \___|_|
*/

/**
 * @brief Detect the horizontal edges for  \
 * the sobel operator
 *
 */
#define SOBEL_X_VALUE(greyscale, y, x, width, height) (     \
      greyscale[(y - 1) * width + (x - 1)]                  \
    + greyscale[(y + 0) * width + (x - 1)] * (+2)           \
    + greyscale[(y + 1) * width + (x - 1)]                  \
    + greyscale[(y - 1) * width + (x + 1)] * (-1)           \
    + greyscale[(y + 0) * width + (x + 1)] * (-2)           \
    + greyscale[(y + 1) * width + (x + 1)] * (-1)           \
)

 /**
  * @brief Detect the vertical edges for  \
  * the sobel operator
  *
  */
#define SOBEL_Y_VALUE(greyscale, y, x, width, height) (     \
      greyscale[(y - 1) * width + (x - 1)]                  \
    + greyscale[(y - 1) * width + (x + 0)] * (+2)           \
    + greyscale[(y - 1) * width + (x + 1)]                  \
    + greyscale[(y + 1) * width + (x - 1)] * (-1)           \
    + greyscale[(y + 1) * width + (x + 0)] * (-2)           \
    + greyscale[(y + 1) * width + (x + 1)] * (-1)           \
)


  /**
   * @brief Mix both horizontal and vertical values
   *
   */
#define SOBEL_VALUE(greyscale, y, x, width, height) (           \
      glm::abs(SOBEL_X_VALUE(greyscale, y, x, width, height))   \
    + glm::abs(SOBEL_Y_VALUE(greyscale, y, x, width, height))   \
)




/**
    * @brief Improve the render time by detecting edges for superslamping
    *
    * @param img           The resulting image
    * @param scene         The scene which contains the objects
    * @param tree          The KDTree used to compute colisions
    * @param ray_delta_x   The dx of the ray
    * @param ray_delta_y   The dy of the ray
    * @param dx            The dx
    * @param dy            The dy
    */
__global__ void sobelImprovedSuperslamping(sf::Int8* greyscaleImage, sf::Uint8* pixels, Scene* scene, CamParameters* cam)
{
    const int height = int(scene->camera.height);
    const int width  = int(scene->camera.width);
    const int y      = int((blockIdx.x * blockDim.x + threadIdx.x) / width);
    const int x      = int((blockIdx.x * blockDim.x + threadIdx.x) % width);
    const int p      = (y * width + x) * 4;
    const int pg     = (y * width + x) * 4;

    if (p > (height * width * 4))
    {
        return;
    }

    if (x == 0 || x == (width - 1))
    {
        return;
    }

    if (y == 0 || y == (height - 1))
    {
        return;
    }


    if (SOBEL_VALUE(greyscaleImage, (height - y), x, width, height) > 0)
    {
        renderPixel(scene, pixels, nullptr, cam, 0);
    }
}











__host__ CamParameters* allocParameters()
{
    CamParameters* params;
    cudaMallocManaged(&params, sizeof(CamParameters));

    return params;
}


__host__ void getParameters(Scene* scene, CamParameters* params)
{
    const float height  = scene->camera.height;
    const float width   = scene->camera.width;
    const float aspect  = 1.f / (scene->camera.aspect);

    params->delta_y     = 1.f / (height * 0.5f);
    params->dy          = params->delta_y * aspect * scene->camera.ydir;
    params->ray_delta_y = (0.5f - float(height) * 0.5f) / (float(height) * 0.5f) * aspect * scene->camera.ydir;

    params->delta_x     = 1.f / (width * 0.5f);
    params->dx          = params->delta_x * scene->camera.xdir;
    params->ray_delta_x = (0.5f - float(width) * 0.5f) / (float(width) * 0.5f) * scene->camera.xdir;
}






__host__ sf::Uint8* renderScene(Scene* scene)
{   
    static sf::Uint8* pixels;               // Final image
    static sf::Uint8* sobelImage;           // SuperslampedImage via sobel operator
    static sf::Int8*  greyscaleImage;       // Greyscale image for sobel operator
    static CamParameters* camParameters;    // Avoid computation of scene parameters for each threads

    static sf::Clock clock;                 // FPS counter
    

    if (!gpuInitialized)
    {
        cudaMalloc(&pixels, sizeof(sf::Uint8) * s_width * s_height * 4);
        cudaMalloc(&greyscaleImage, sizeof(sf::Int8) * s_width * s_height);
        camParameters = allocParameters();
        p = new sf::Uint8[s_width * s_height * 4];

        gpuInitialized = true;
    }
    
    getParameters(scene, camParameters);

    //printf("Rendering scene... (blocks: %d, threads: %d)\n", s_height, s_width);
    size_t thread_number = 512;
    size_t block_number = s_height * s_width / thread_number + 1;
    renderPixelFromHost<<<block_number, thread_number>>>(scene, pixels, greyscaleImage, camParameters);
    
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    //sobelImprovedSuperslamping<<<block_number, thread_number>>>(greyscaleImage, pixels, scene, camParameters);
    //gpuErrchk(cudaPeekAtLastError());
    //gpuErrchk(cudaDeviceSynchronize());


    //printf("Scene rendered ! \n");
    cudaMemcpy((void*)p, (void*)pixels, sizeof(sf::Uint8) * s_width * s_height * 4, cudaMemcpyDeviceToHost);

    printf("FPS: %f\r", 1.f / clock.restart().asSeconds());

    
    return p;
}