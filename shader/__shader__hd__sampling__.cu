#include "OptiXGlobalHelper.h"
#include "../demo/demo2/SamplingMethods/rtHDISTSamplingParam.h"

#include <curand.h>
#include <curand_kernel.h>

extern "C" __constant__ OptiXHDParamSamplingBased optixLaunchParams;

enum
{
    SURFACE_RAY_TYPE = 0,
    RAY_TYPE_COUNT
};

struct Payload_t
{
    float minDist;
    int hitCount;
};

float3 getRayDir()
{
    int samples = optixLaunchParams.Target.vSize * optixLaunchParams.samplingRate;
    switch (optixLaunchParams.samplingMethod)
    {
    case VERTEX:
        break;
    case HEMISPHERE:
        break;
    case AABB:
        break;

    default:
        break;
    }

    return make_float3(1,0,0);
}

extern "C" __global__ void __raygen__program__()
{
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int idx = iy * optixGetLaunchDimensions().x + ix;

    if (idx >= optixLaunchParams.querySize)
        return;

    float3 rayOrigin = optixLaunchParams.queryPoints[idx];
    
    curandState localState;
    curand_init(optixLaunchParams.randomseed, idx, 0, &localState);

    //float3 rayDir = getRayDir();
    float3 rayDir = {1.0f, 0.0f, 0.0f};

    float tmin = 1e-8f;
    float tmax = 1e-7f;
    uint hitCount = 0;
    float3 target;

    Payload_t payload;

    uint32_t u0, u1;
    packPointer(&payload, u0, u1);

    optixTrace(optixLaunchParams.traversable,
               rayOrigin,
               rayDir,
               tmin, // tmin
               tmax, // tmax
               0.0f, // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE, // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,    // SBT offset
               RAY_TYPE_COUNT,      // SBT stride
               SURFACE_RAY_TYPE,    // missSBTIndex
               u0, u1);

    if(tmax >= 1e7f || hitCount == 0){
        tmax = -1;
    }

    optixLaunchParams.Result.distance[idx] = tmax;
    optixLaunchParams.Result.pos[idx] = target;
}

extern "C" __global__ void __miss__radiance()
{
    Payload_t &prd = *(Payload_t *)getPRD<Payload_t>();

    prd.minDist = 1e16f;
}

extern "C" __global__ void __anyhit__radiance()
{
}

extern "C" __global__ void __intersection__radiance()
{
}

extern "C" __global__ void __closesthit__radiance()
{
    Payload_t &prd = *(Payload_t *)getPRD<Payload_t>();

    prd.minDist = optixGetRayTmax();
}