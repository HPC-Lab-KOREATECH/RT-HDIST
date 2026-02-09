#include "cuBQLbased.h"

#include "3rdParty/CUDABuffer.h"
#include "3rdParty/TimeChecker.h"

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/queries/triangleData/closestPointOnAnyTriangle.h"
#include "cuBQL/queries/triangleData/lineOfSight.h"

#include "ReduceUtils.h"
#include "MortonUtils.h"
#include "AABBSupport.h"

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
    //std::cout << "Build Time : " << GASBuildTime << " ms" << std::endl;

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
                                           cudaMemcpy(&cand2, (float3 *)dTargets.d_pointer() + maxIDX, sizeof(float3) * 1, cudaMemcpyDeviceToHost); });
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
__global__ void computeRadiusBoxes(const float3 *vertices, cuBQL::box3f *boxes, size_t numPoints, float radius)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPoints)
        return;

    float3 v = vertices[idx];
    cuBQL::vec3f lower = cuBQL::vec3f{v.x - radius, v.y - radius, v.z - radius};
    cuBQL::vec3f upper = cuBQL::vec3f{v.x + radius, v.y + radius, v.z + radius};
    boxes[idx] = cuBQL::box3f{lower, upper};
}

enum HDistOptimizeState
{
    FILTERING,
    COMPUTING
};

struct Payload_t
{
    float3 originVtx;
    float minDist;
    float rawMinDist;
    // float3 scaledQuery;
    uint targetIdx;

    bool terminated;
};

__global__ void runRadiusQueries(cuBQL::bvh3f boxBVH, const cuBQL::box3f *leafs, const float3 *queryPoints,
                                 const float3 *target, const size_t *clusterInfo, OptixAabb targetBound,
                                 float *outDistance, uint *targetIDX, size_t numQueriues,
                                 float3 targetVoxelSize, float targetEPS, HDistOptimizeState state)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numQueriues)
        return;

    cuBQL::vec3f queryPoint = {queryPoints[idx].x, queryPoints[idx].y, queryPoints[idx].z};
    cuBQL::vec3f rayDir = cuBQL::vec3f{1.0f, 0.0f, 0.0f};

    cuBQL::ray3f ray(queryPoint, rayDir, 1e-8f, 1e-7f); // very short ray

    Payload_t prd;
    prd.minDist = 1e8f;
    prd.rawMinDist = 1e8f;
    prd.targetIdx = -1;
    prd.terminated = false;

    if (state == COMPUTING)
    {
        float3 boundMin = {targetBound.minX, targetBound.minY, targetBound.minZ};
        float3 originVtx = queryPoints[idx] * targetVoxelSize + boundMin;
        prd.originVtx = originVtx;
    }

    auto perBox = [leafs, ray, state, targetEPS, &prd, target, clusterInfo](uint32_t leafID)
    {
        cuBQL::box3f box = leafs[leafID];
        if (cuBQL::rayIntersectsBox(ray, box))
        {
            float hitDist = length(box.center() - ray.origin);

            if (hitDist < targetEPS)
            {
                if (state == COMPUTING)
                {
                    size_t last = clusterInfo[leafID + 1];

                    for (int i = clusterInfo[leafID]; i < last; i++)
                    {
                        float rawdist = length(prd.originVtx - target[i]);

                        if (prd.rawMinDist >= rawdist)
                        {
                            prd.rawMinDist = rawdist;
                            prd.targetIdx = i;
                        }
                    }
                    return CUBQL_CONTINUE_TRAVERSAL;
                }
                else
                {
                    prd.rawMinDist = 1.0;
                    prd.terminated = true;
                    return CUBQL_TERMINATE_TRAVERSAL;
                }
            }
            return CUBQL_CONTINUE_TRAVERSAL;
        }
    };

    cuBQL::fixedRayQuery::forEachPrim(perBox, boxBVH, ray);

    if (prd.rawMinDist < 1e8f)
    {
        targetIDX[idx] = prd.targetIdx;
        outDistance[idx] = prd.rawMinDist;
    }
    else
    {
        targetIDX[idx] = -1;
        outDistance[idx] = -1;
    }
}

float cubqlClusterHD(HDGPUParam<HDMODE::TRIANGLE> &dA, HDGPUParam<HDMODE::TRIANGLE> &dB, float3 &cand1, float3 &cand2,
                     float _eps, BYTE bitCount,
                     std::map<std::string, float> &timeParam)
{
    AABBCluster target;

    OptixAabb targetAABB = computeAABB_device(dB.vert, dB.vSize);
    OptixAabb sourceAABB = computeAABB_device(dA.vert, dA.vSize);

    OptixAabb totalAABB = merge(sourceAABB, targetAABB);

    const float3 voxelS = (aabb2max(targetAABB) - aabb2min(targetAABB)) / (1 << bitCount);
    float delimiter = fmaxf(fmaxf(voxelS.x, voxelS.y), voxelS.z);
    const float3 voxelSize = make_float3(delimiter, delimiter, delimiter);

    target.number_of_points = dB.vSize;
    cudaMalloc(&target.points, sizeof(float3) * target.number_of_points);
    cudaMemcpy(target.points, dB.vert, sizeof(float3) * dB.vSize, cudaMemcpyDeviceToDevice);

    cudaMalloc(&target.pointAabb, sizeof(OptixAabb) * target.number_of_points);

    float3 *gpuRepresentative;
    auto TimeGenerateIndexSpace = SPIN::TimeCheck([&]()
                                                  { genUniformClusterAabb(target.points, target.number_of_points, target.cluster, gpuRepresentative, target.number_of_cluster, target.clusterInfo, bitCount); });
    timeParam["IndexSpaceBuildTime"] += TimeGenerateIndexSpace;
    std::cout << "Number of cluster : " << target.number_of_cluster << std::endl;

    target.representative.resize(target.number_of_cluster);
    cudaMemcpy(target.representative.data(), gpuRepresentative, sizeof(float3) * target.number_of_cluster, cudaMemcpyDeviceToHost);

    float HD = 0.0f;

    float eps = _eps;

    size_t remains = dA.vSize;
    size_t previous_remains = remains;
    size_t pprevious_remains = remains;

    float3 *queries;
    cudaMalloc(&queries, sizeof(float3) * remains);
    cudaMemcpy(queries, dA.vert, sizeof(float3) * remains, cudaMemcpyDeviceToDevice);

    float3 *previous;
    cudaMalloc(&previous, sizeof(float3) * remains);

    float3 *pprevious;
    cudaMalloc(&pprevious, sizeof(float3) * remains);

    size_t *gpuClusterInfo;
    cudaMalloc(&gpuClusterInfo, sizeof(size_t) * target.clusterInfo.size());
    cudaMemcpy(gpuClusterInfo, target.clusterInfo.data(), sizeof(size_t) * target.clusterInfo.size(), cudaMemcpyHostToDevice);

    OptixAabb *aabbBuffer;
    cudaMalloc(&aabbBuffer, sizeof(OptixAabb) * target.number_of_cluster);

    ToIndexSpace(dA.vert, dA.vSize, queries, targetAABB, voxelSize);
    cudaMemcpy(previous, queries, sizeof(float3) * remains, cudaMemcpyDeviceToDevice);
    cudaMemcpy(pprevious, queries, sizeof(float3) * remains, cudaMemcpyDeviceToDevice);

    float ceneterLength = length(aabb2center(targetAABB) - aabb2center(sourceAABB));
    float aabblength = length(distance(sourceAABB, targetAABB));

    eps = (int)(ceneterLength / delimiter * 0.5f) * sqrtf(3);
    uint3 number_of_voxels_maximum_inspace = make_uint3(aabb2size(totalAABB) / voxelSize);

    CUDABuffer dBoxesB;
    dBoxesB.alloc(sizeof(cuBQL::box3f) * target.number_of_cluster);

    cuBQL::bvh3f bvhB;
    while (remains > 0)
    {
        CUDABuffer dDistances;
        CUDABuffer dIdx;
        float3 *reduced;
        size_t reduced_size;

        size_t blockSize = 256;
        size_t numBlocks = divRoundUp(target.number_of_cluster, blockSize);
        auto TimeFilteringStep = SPIN::TimeCheck([&]()
                                                 {
                                                     auto GASBuildTime = SPIN::TimeCheck([&]
                                                                                         {
                                                        computeRadiusBoxes<<<numBlocks, blockSize>>>(gpuRepresentative,
                                                         (cuBQL::box3f*)dBoxesB.d_pointer(),
                                                         target.number_of_cluster,
                                                         eps);
                                                         cudaDeviceSynchronize();
                                                    cuBQL::gpuBuilder(bvhB, (cuBQL::box3f*)dBoxesB.d_pointer(), target.number_of_cluster, cuBQL::BuildConfig()); });
                                                     timeParam["01_GAS_build"] += GASBuildTime;
                                                     //std::cout << "Build Time : " << GASBuildTime << " ms" << std::endl;

                                                     dDistances.alloc(sizeof(float) * remains);
                                                     dIdx.alloc(sizeof(uint) * remains);

                                                     size_t numQueries = remains;
                                                     numBlocks = divRoundUp(numQueries, blockSize);

                                                     auto ComputeTime = SPIN::TimeCheck([&]
                                                                                        {
                                           runRadiusQueries<<<numBlocks, blockSize>>>(bvhB,
                                                                                (cuBQL::box3f*)dBoxesB.d_pointer(),
                                                                                queries,
                                                                                dB.vert,
                                                                                gpuClusterInfo,
                                                                                targetAABB,                                                                                
                                                                                (float *)dDistances.d_pointer(),
                                                                                (uint *)dIdx.d_pointer(),
                                                                                remains,
                                                                                voxelSize,
                                                                                eps,
                                                                                FILTERING
                                                                        );
                                           cudaDeviceSynchronize(); });
                                                     timeParam["02_HD_Compute"] += ComputeTime;
                                                     //std::cout << "Compute Time : " << ComputeTime << " ms" << std::endl;

                                                     eps += sqrt(3);
                                                     reduceWithDistance(queries, remains, (float *)dDistances.d_pointer(), reduced, reduced_size, false);
                                                     cudaMemcpy(pprevious, previous, sizeof(float3) * previous_remains, cudaMemcpyDeviceToDevice);
                                                     cudaMemcpy(previous, queries, sizeof(float3) * remains, cudaMemcpyDeviceToDevice);

                                                     dDistances.free();
                                                     dIdx.free();
                                                     cuBQL::cuda::free(bvhB); // for rebuild
                                                 });
        timeParam["FilteringTime"] += TimeFilteringStep;
        if (reduced_size == 0)
        {
            auto TimeComputing = SPIN::TimeCheck([&]()
                                                 {
                    size_t maxIDX;
                    float tmp;

                    float* distanceBuffer;
                    uint* idxBuffer;
                    cudaMalloc(&distanceBuffer, sizeof(float) * pprevious_remains);
                    cudaMalloc(&idxBuffer, sizeof(uint) * pprevious_remains);

                    auto GASBuildTime = SPIN::TimeCheck([&]
                                                                                         {
                                                        computeRadiusBoxes<<<numBlocks, blockSize>>>(gpuRepresentative,
                                                         (cuBQL::box3f*)dBoxesB.d_pointer(),
                                                         target.number_of_cluster,
                                                         eps);
                                                         cudaDeviceSynchronize();
                                                    cuBQL::gpuBuilder(bvhB, (cuBQL::box3f*)dBoxesB.d_pointer(), target.number_of_cluster, cuBQL::BuildConfig()); });
                                                     timeParam["01_GAS_build"] += GASBuildTime;
                                                    //std::cout << "Build Time : " << GASBuildTime << " ms" << std::endl;

                          size_t numQueries = pprevious_remains;
                                                     numBlocks = divRoundUp(numQueries, blockSize);

                                                     auto ComputeTime = SPIN::TimeCheck([&]
                                                                                        {
                                           runRadiusQueries<<<numBlocks, blockSize>>>(bvhB,
                                                                                (cuBQL::box3f*)dBoxesB.d_pointer(),
                                                                                pprevious,
                                                                                dB.vert,
                                                                                gpuClusterInfo,
                                                                                targetAABB,                                                                                
                                                                                distanceBuffer,
                                                                                idxBuffer,
                                                                                pprevious_remains,
                                                                                voxelSize,
                                                                                eps,
                                                                                COMPUTING
                                                                        );
                                           CUDA_SYNC_CHECK(); });

                    std::cout << "Remains : " << remains << ", 2step-previous :" << pprevious_remains << std::endl;
                    tmp = getMaximumF(distanceBuffer, pprevious_remains, maxIDX);
                    CUDA_SYNC_CHECK();
                    std::cout << eps << " " << tmp << std::endl;
                    uint targetIDX;
                    cudaMemcpy(&targetIDX, idxBuffer + maxIDX, sizeof(uint), cudaMemcpyDeviceToHost);
                    std::cout << eps << " " << tmp << std::endl;

                    HD = fmaxf(tmp, HD);
                    cudaMemcpy(&cand1, pprevious + maxIDX, sizeof(float3) * 1, cudaMemcpyDeviceToHost);
                    cand1 = cand1 * voxelSize + aabb2min(targetAABB);
                    cudaMemcpy(&cand2, target.points + targetIDX, sizeof(float3) * 1, cudaMemcpyDeviceToHost);
                    std::cout << length(cand1 - cand2) << std::endl;

                    cudaFree(distanceBuffer);
                    cudaFree(idxBuffer);
                    cuBQL::cuda::free(bvhB); 
                });
            timeParam["ComputingTime"] += TimeComputing;
        }
        else
        {
            cudaMemcpy(queries, reduced, sizeof(float3) * reduced_size, cudaMemcpyDeviceToDevice);
        }
        pprevious_remains = previous_remains;
        previous_remains = remains;
        remains = reduced_size;

        cudaFree(reduced);
    }

    cudaFree(aabbBuffer);
    cudaFree(queries);
    cudaFree(gpuRepresentative);
    cudaFree(gpuClusterInfo);
    cudaFree(previous);
    cudaFree(pprevious);

    return HD;
}
