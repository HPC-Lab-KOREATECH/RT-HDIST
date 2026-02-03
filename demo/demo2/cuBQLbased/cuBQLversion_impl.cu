#include "cuBQLbased.h"

#include "3rdParty/CUDABuffer.h"
#include "3rdParty/TimeChecker.h"

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/queries/triangleData/closestPointOnAnyTriangle.h"
#include "cuBQL/queries/triangleData/lineOfSight.h"

#include "ReduceUtils.h"

using cuBQL::divRoundUp;

// cubql HD
__global__ void computeTrianglesAndBoxes(const float3 *vertices, const uint3 *indices, cuBQL::Triangle *triangles, cuBQL::box3f *boxes, size_t numTriangles)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numTriangles)
        return;

    uint3 triIdx = indices[idx];
    float3 v0 = vertices[triIdx.x];
    float3 v1 = vertices[triIdx.y];
    float3 v2 = vertices[triIdx.z];

    triangles[idx] = cuBQL::Triangle{cuBQL::vec3f{v0.x, v0.y, v0.z},
                                     cuBQL::vec3f{v1.x, v1.y, v1.z},
                                     cuBQL::vec3f{v2.x, v2.y, v2.z}};

    boxes[idx] = triangles[idx].bounds();
}

__global__ void runQueries(cuBQL::bvh3f trianglesBVH, const cuBQL::Triangle *triangles, const float3 *queryPoints, float *outDistance, float3 *outPos, size_t numQueriues)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numQueriues)
        return;

    cuBQL::vec3f queryPoint = {queryPoints[idx].x, queryPoints[idx].y, queryPoints[idx].z};

    cuBQL::triangles::CPAT cpat;
    cpat.runQuery(triangles, trianglesBVH, queryPoint);

    outDistance[idx] = sqrtf(cpat.sqrDist);
    outPos[idx] = make_float3(cpat.P.x, cpat.P.y, cpat.P.z);
}

float cubqlHD(HDGPUParam<HDMODE::TRIANGLE> &dA, HDGPUParam<HDMODE::TRIANGLE> &dB, float3 &cand1, float3 &cand2,
              std::map<std::string, float> &timeParam)
{
    size_t numTrianglesB = dB.tSize;
    float HD = 0.0f;

    CUDABuffer dBoxesB;
    dBoxesB.alloc(sizeof(cuBQL::box3f) * numTrianglesB);
    CUDABuffer dTrianglesB;
    dTrianglesB.alloc(sizeof(cuBQL::Triangle) * numTrianglesB);

    size_t blockSize = 256;
    size_t numBlocks = divRoundUp(numTrianglesB, blockSize);
    cuBQL::bvh3f bvhB;

    auto GASBuildTime = SPIN::TimeCheck([&]
                                        {
        computeTrianglesAndBoxes<<<numBlocks, blockSize>>>(dB.vert, dB.tri,
             (cuBQL::Triangle*)dTrianglesB.d_pointer(),
             (cuBQL::box3f*)dBoxesB.d_pointer(), numTrianglesB);
        cudaDeviceSynchronize();
        cuBQL::gpuBuilder(bvhB, (cuBQL::box3f*)dBoxesB.d_pointer(), numTrianglesB, cuBQL::BuildConfig()); });
    timeParam["01_GAS_build"] = GASBuildTime;
    std::cout << "Build Time : " << GASBuildTime << " ms" << std::endl;

    size_t numQueries = dA.vSize;
    CUDABuffer dTargets;
    dTargets.alloc(sizeof(float3) * numQueries);
    CUDABuffer dDistances;
    dDistances.alloc(sizeof(float) * numQueries);
    numBlocks = divRoundUp(numQueries, blockSize);

    auto ComputeTime = SPIN::TimeCheck([&]
                                       {
                                           runQueries<<<numBlocks, blockSize>>>(bvhB,
                                                                                (cuBQL::Triangle *)dTrianglesB.d_pointer(),
                                                                                dA.vert,
                                                                                (float *)dDistances.d_pointer(),
                                                                                (float3 *)dTargets.d_pointer(),
                                                                                numQueries);
                                           cudaDeviceSynchronize();
                                           size_t maxIDX;
                                           float tmp = getMaximumF((float *)dDistances.d_pointer(), numQueries, maxIDX);
                                           HD = tmp;

                                           cudaMemcpy(&cand1, dA.vert + maxIDX, sizeof(float3) * 1, cudaMemcpyDeviceToHost);
                                           cudaMemcpy(&cand2, (float3 *)dTargets.d_pointer() + maxIDX, sizeof(float3) * 1, cudaMemcpyDeviceToHost);
                                       });
    timeParam["02_HD_Compute"] = ComputeTime;
    std::cout << "Compute Time : " << ComputeTime << " ms" << std::endl;

    cuBQL::cuda::free(bvhB);
    dBoxesB.free();
    dTrianglesB.free();
    dDistances.free();
    dTargets.free();

    return HD;
}

// cubql clustered HD (Same process)
__global__ void computeRadiusBoxes(const float3 *vertices, cuBQL::box3f *boxes, size_t numPoints, float radius){
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints)
        return;

    float3 v = vertices[idx];
    cuBQL::vec3f lower = cuBQL::vec3f{v.x - radius, v.y - radius, v.z - radius};
    cuBQL::vec3f upper = cuBQL::vec3f{v.x + radius, v.y + radius, v.z + radius};
    boxes[idx] = cuBQL::box3f{lower, upper};
}

__global__ void runRadiusQueries(cuBQL::bvh3f boxBVH, const cuBQL::box3f *leafs, const float3 *queryPoints, float *outDistance, float3 *outPos, size_t numQueriues)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numQueriues)
        return;

    cuBQL::vec3f queryPoint = {queryPoints[idx].x, queryPoints[idx].y, queryPoints[idx].z};
    cuBQL::vec3f rayDir = cuBQL::vec3f{1.0f, 0.0f, 0.0f};

    cuBQL::ray3f ray(queryPoint, rayDir, 1e-8f, 1e-7f); // very short ray
    
}