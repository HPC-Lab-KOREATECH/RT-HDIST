#include "rtMeshGAS.h"

__global__ void _kernel_accumulate_vertex_normals_atomic_(
    const float3* __restrict__ vertex,
    const uint3* __restrict__ indice,
    size_t tSize,
    float3* __restrict__ vnormal
){
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= tSize) return;

    uint3 tri = indice[tID];
    float3 n = cross(vertex[tri.y] - vertex[tri.x], vertex[tri.z] - vertex[tri.x]);

    float len = length(n);
    if(len <= 1e-20f) return;

    atomicAdd(&vnormal[tri.x].x, n.x); atomicAdd(&vnormal[tri.x].y, n.y); atomicAdd(&vnormal[tri.x].z, n.z);
    atomicAdd(&vnormal[tri.y].x, n.x); atomicAdd(&vnormal[tri.y].y, n.y); atomicAdd(&vnormal[tri.y].z, n.z);
    atomicAdd(&vnormal[tri.z].x, n.x); atomicAdd(&vnormal[tri.z].y, n.y); atomicAdd(&vnormal[tri.z].z, n.z);
}

__global__ void _kernel_vector_normalize_(float3* __restrict__ vnormal, size_t vSize){
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= vSize) return;

    float3 n = vnormal[tID];
    float len = length(n);

    if(len>1e-20f){
        float inv = 1.0f/len;
        vnormal[tID] = n*inv;
    }else{
        vnormal[tID] = {0.0, 0.0, 0.0};
    }

}

void computeVertexNormals(const HDGPUParam<HDMODE::TRIANGLE> &deviceMesh, float3* &vertexNormalOutput){
    cudaMalloc(&vertexNormalOutput, sizeof(float3)*deviceMesh.tSize);

    int block = 256;
    int gridT = ceil(deviceMesh.tSize/256.)+1;
    int gridV = ceil(deviceMesh.vSize/256.)+1;

    _kernel_accumulate_vertex_normals_atomic_<<<gridT, block>>>(deviceMesh.vert, deviceMesh.tri, deviceMesh.tSize, vertexNormalOutput);
    _kernel_vector_normalize_<<<gridV, block>>>(vertexNormalOutput, deviceMesh.vSize);
    cudaDeviceSynchronize(); //if needed
}