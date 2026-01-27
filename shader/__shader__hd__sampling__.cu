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

__device__
    float3
    getRayDir()
{
    switch (optixLaunchParams.samplingMethod)
    {
    case VERTEX:
        break;
    case SPHERE:
        break;
    case AABB:
        break;

    default:
        break;
    }

    return make_float3(1, 0, 0);
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

    int samples = optixLaunchParams.Target.vSize * optixLaunchParams.samplingRate;
    int interval = optixLaunchParams.Target.vSize / samples;

    float3 targetBoxMin = make_float3(optixLaunchParams.Target.aabb.minX,
                                      optixLaunchParams.Target.aabb.minY,
                                      optixLaunchParams.Target.aabb.minZ);
    float3 targetBoxMax = make_float3(optixLaunchParams.Target.aabb.maxX,
                                      optixLaunchParams.Target.aabb.maxY,
                                      optixLaunchParams.Target.aabb.maxZ);
    float3 targetBoxSize = targetBoxMax - targetBoxMin;

    float tMin = 1e-8f;
    float tMax = 1e7f;
    uint hitCount = 0;
    float3 target;
    float3 rayDir;
    Payload_t payload;
    payload.minDist = tMax;
    payload.hitCount = 0;

    uint32_t u0, u1;
    packPointer(&payload, u0, u1);

    for (int i = 0; i < samples; i++)
    {
        if (optixLaunchParams.samplingMethod == VERTEX)
        {
            int offset = (optixLaunchParams.offset == -1) ? curand(&localState) % interval : optixLaunchParams.offset;
            int targetIdx = i * interval + offset;

            if (targetIdx >= optixLaunchParams.Target.vSize)
                break;

            if (length(optixLaunchParams.queryPoints[idx] - optixLaunchParams.Target.vertices[targetIdx]) >= tMax)
                continue;

            rayDir = optixLaunchParams.Target.vertices[targetIdx] - optixLaunchParams.queryPoints[idx];
            rayDir = normalize(rayDir);
        }
        else if(optixLaunchParams.samplingMethod == SPHERE)
        {
            float u = curand_uniform(&localState) * 2 - 1;
            float v = curand_uniform(&localState) * 2 - 1;
            float r = curand_uniform(&localState) * 2 - 1;

            float3 sample = make_float3(u, v, r);
            rayDir = normalize(sample);
        }
        else if(optixLaunchParams.samplingMethod == AABB)
        {
            float u = curand_uniform(&localState);
            float v = curand_uniform(&localState);
            float r = curand_uniform(&localState);

            float3 targetPos = targetBoxSize * make_float3(u, v, r) + targetBoxMin;

            rayDir = targetPos - rayOrigin;
            rayDir = normalize(rayDir);
        }

        optixTrace(optixLaunchParams.traversable,
                   rayOrigin,
                   rayDir,
                   tMin, // tmin
                   tMax, // tmax
                   0.0f, // rayTime
                   OptixVisibilityMask(255),
                   OPTIX_RAY_FLAG_NONE, // OPTIX_RAY_FLAG_NONE,
                   SURFACE_RAY_TYPE,    // SBT offset
                   RAY_TYPE_COUNT,      // SBT stride
                   SURFACE_RAY_TYPE,    // missSBTIndex
                   u0, u1);

        if (payload.minDist > 0 && payload.minDist < tMax)
        {
            tMax = payload.minDist;
            target = rayOrigin + rayDir * tMax;
            hitCount++;
        }
    }

    if (tMax >= 1e7f || hitCount == 0)
    {
        tMax = -1;
    }

    optixLaunchParams.Result.distance[idx] = tMax;
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