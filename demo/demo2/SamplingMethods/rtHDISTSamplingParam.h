#pragma once
#include "3rdParty/optix7support.h"

enum OptiXHDistSamplingMethod{
    SPHERE,
    AABB,
    VERTEX
};

struct OptiXHDParamSamplingBased{
    float3* queryPoints;
    float3* queryNormals; //if needed
    size_t querySize;

    struct{
        float3* vertices;
        size_t vSize;
        uint3* indices;
        size_t tSize;
        OptixAabb aabb;
    }Target;

    struct{ //number of distance is same with querySize
        float* distance;
        float3* pos;
    }Result;

    float samplingRate = 1.0f;
    int samplingCount = 0;
    int offset = 0; // if -1 is rand
    unsigned int randomseed;

    OptixTraversableHandle traversable;

    OptiXHDistSamplingMethod samplingMethod = VERTEX;
};