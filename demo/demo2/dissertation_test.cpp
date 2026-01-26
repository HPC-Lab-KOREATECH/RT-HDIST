#include <iostream>
#include <filesystem>

#include "3rdParty/helper_math.h"
#include "OptiX_Base.h"
#include "rtHausdorffQCluster.h"

#include "3rdParty/TimeChecker.h"
#include "3rdParty/Logger.h"
#include "3rdParty/IO.h"

#include "GlobalParam.h"

#include "PointCloud.h"
#include "Object_t.h"

#include "MortonUtils.h"
#include "ReduceUtils.h"
#include "AABBSupport.h"

#include "rtHDISTSamplingParam.h"
#include "rtMeshGAS.h"

uint random_seed;

void buildQClusterShader();
void buildSamplingShdaer();

void alloc_and_upload(const TriangleMesh &model, HDGPUParam<HDMODE::TRIANGLE> &dModel)
{
    dModel.vSize = model.vertex.size();
    dModel.tSize = model.index.size();
    cudaMalloc(&dModel.vert, sizeof(float3) * dModel.vSize);
    cudaMalloc(&dModel.tri, sizeof(uint3) * dModel.tSize);

    cudaMemcpy(dModel.vert, model.vertex.data(), sizeof(float3) * dModel.vSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dModel.tri, model.index.data(), sizeof(uint3) * dModel.tSize, cudaMemcpyHostToDevice);
}

// Sampling-based method, dA = query, dB = target
float directHD(OptiXHDProgram &program, HDGPUParam<HDMODE::TRIANGLE> dA, HDGPUParam<HDMODE::TRIANGLE> dB,
               float3 &cand1, float3 &cand2, OptiXHDistSamplingMethod method, float samplingRate /*it is count for HEMISPHERE*/,
               std::map<std::string, float> &timeParam);

int main(int argc, char *argv[])
{
    std::cout << "Device name : " << optixGlobalParams.deviceProps.name << std::endl;

    buildQClusterShader();
    buildSamplingShdaer();

    verifyArguments(argc, argv);

    std::random_device rd;
    random_seed = (globalParams["seed"] >= 0) ? globalParams["seed"] : rd();
    std::cout << "Random seed : " << random_seed << std::endl;
    random_machine = std::mt19937(random_seed);

    SPIN::Logger log;

    Object_t hA = IO::read<Object_t, SPIN::OBJ>(inputFilePaths[0]);

    HDGPUParam<HDMODE::TRIANGLE> dA;
    HDGPUParam<HDMODE::TRIANGLE> dB;
    alloc_and_upload(*hA.model->meshes[0], dA);

    float3 boxTranslate = {0, 0, 0};
    OptixAabb aabb = computeAABB_device(dA.vert, dA.vSize);
    float3 aabbsize = aabb2size(aabb);

    if (globalParams["obj_count"] == 1)
    {
        if (globalParams["translate_ratio"])
        {
            globalTransform = globalTransformRatio * make_float3(aabbsize.x, 0, 0);
            boxTranslate = globalTransform;
        }
        Object_t hB = IO::read<Object_t, SPIN::OBJ>(inputFilePaths[0]);
        for (auto &v : hB.model->meshes[0]->vertex)
        {
            v += boxTranslate;
        }
        alloc_and_upload(*hB.model->meshes[0], dB);
    }
    else
    {
        Object_t hB = IO::read<Object_t, SPIN::OBJ>(inputFilePaths[1]);
        alloc_and_upload(*hB.model->meshes[0], dB);
    }

    std::cout << "For Test" << std::endl;
    OptiXHDistSamplingMethod testTarget[]{VERTEX, HEMISPHERE, AABB};

    for (const auto &mtd : testTarget)
    {
        std::map<std::string, float> timeParam;
        float3 cand1, cand2;
        float HD = directHD(static_cast<OptiXHDProgram &>(*optixGlobalParams.programList["SamplingBased"]),
                            dA, dB, cand1, cand2, mtd, 0.001f, timeParam);

        std::cout << "HDIST : " << HD << std::endl;
    }

    std::cout << "Hello world!" << std::endl;

    return -1;
}

float directHD(OptiXHDProgram &program, HDGPUParam<HDMODE::TRIANGLE> dA, HDGPUParam<HDMODE::TRIANGLE> dB,
               float3 &cand1, float3 &cand2, OptiXHDistSamplingMethod method, float samplingRate /*it is count for HEMISPHERE*/,
               std::map<std::string, float> &timeParam)
{
    float HD;

    MeshGAS targetGAS;
    auto GASBuildTime = SPIN::TimeCheck([&]
                                        {
        targetGAS.gpuMesh = &dB;
        targetGAS.build(); });
    timeParam["01_GAS_build"] = GASBuildTime;

    std::cout << "Build Time : " << GASBuildTime << " ms" << std::endl;

    OptixAabb targetAABB = computeAABB_device(dB.vert, dB.vSize);

    CUDABuffer launchParamBuffer;
    launchParamBuffer.alloc(sizeof(OptiXHDParamSamplingBased));

    CUDABuffer distanceBuffer;
    CUDABuffer posBuffer;
    distanceBuffer.alloc(sizeof(float) * dA.vSize);
    posBuffer.alloc(sizeof(float3) * dA.vSize);

    OptiXHDParamSamplingBased hdparam;
    { // param build
        hdparam.samplingMethod = method;
        hdparam.queryPoints = dA.vert;
        hdparam.querySize = dA.vSize;

        if (method == HEMISPHERE)
        {
            // build vertex normal
            computeVertexNormals(dA, hdparam.queryNormals);
        }

        hdparam.randomseed = random_seed;

        hdparam.samplingRate = samplingRate;

        hdparam.offset = 17;

        hdparam.Target.vertices = dB.vert;
        hdparam.Target.vSize = dB.vSize;
        hdparam.Target.indices = dB.tri;
        hdparam.Target.tSize = dB.tSize;
        hdparam.Target.aabb = targetAABB;

        hdparam.traversable = targetGAS.gas;

        hdparam.Result.distance = (float *)distanceBuffer.d_pointer();
        hdparam.Result.pos = (float3 *)posBuffer.d_pointer();
    }

    launchParamBuffer.upload(&hdparam, 1);

    // Compute HD
    auto ComputeTime = SPIN::TimeCheck([&]
                                       {
        program.Launches(launchParamBuffer, make_uint3(dA.vSize, 1, 1));
        size_t maxIDX;
        float tmp = getMaximumF(hdparam.Result.distance, dA.vSize, maxIDX);

        HD = tmp;
        cudaMemcpy(&cand1, dA.vert + maxIDX, sizeof(float3)*1, cudaMemcpyDeviceToHost);
        cudaMemcpy(&cand2, hdparam.Result.pos + maxIDX, sizeof(float3)*1, cudaMemcpyDeviceToHost); });
    timeParam["02_HD_Compute"] = ComputeTime;
    std::cout << "Compute Time : " << ComputeTime << " ms" << std::endl;

    if (method == HEMISPHERE)
    {
        cudaFree(hdparam.queryNormals);
    }

    return HD;
}

void buildQClusterShader()
{
    OptiXProgramCompileOption hdShaderOption;
    hdShaderOption.fileName = "__shader__hd__qcluster__";
    hdShaderOption.filePath = "";
    hdShaderOption.rayCount = 1;
    hdShaderOption.launchParamName = "optixLaunchParams";
    hdShaderOption.rayGenName = "__raygen__program__";
    hdShaderOption.missProgramNames = {"__miss__radiance"};
    hdShaderOption.hitProgramCount = 1;
    hdShaderOption.hitProgramNames = {{"__intersection__radiance", "__anyhit__radiance", "__closesthit__radiance"}};

    OptiXHDProgram *HDProgram = new OptiXHDProgram(hdShaderOption);

    optixGlobalParams.programList["QCluster"] = HDProgram;
}

void buildSamplingShdaer()
{
    OptiXProgramCompileOption hdShaderOption;
    hdShaderOption.fileName = "__shader__hd__sampling__";
    hdShaderOption.filePath = "";
    hdShaderOption.rayCount = 1;
    hdShaderOption.launchParamName = "optixLaunchParams";
    hdShaderOption.rayGenName = "__raygen__program__";
    hdShaderOption.missProgramNames = {"__miss__radiance"};
    hdShaderOption.hitProgramCount = 1;
    hdShaderOption.hitProgramNames = {{"__intersection__radiance", "__anyhit__radiance", "__closesthit__radiance"}};

    OptiXHDProgram *HDProgram = new OptiXHDProgram(hdShaderOption);

    optixGlobalParams.programList["SamplingBased"] = HDProgram;
}
