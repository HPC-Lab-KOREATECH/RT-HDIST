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
#include "cuBQLbased.h"

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

void runSamplingBasedHDISTTest(const HDGPUParam<HDMODE::TRIANGLE> &dA, const HDGPUParam<HDMODE::TRIANGLE> &dB);

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

    // For fast debug
    // {
    //     inputFilePaths[0] = "../" + inputFilePaths[0];
    //     if (inputFilePaths.size() > 1)
    //         inputFilePaths[1] = "../" + inputFilePaths[1];
    // }

    std::random_device rd;
    random_seed = (globalParams["seed"] >= 0) ? globalParams["seed"] : rd();
    std::cout << "Random seed : " << random_seed << std::endl;
    random_machine = std::mt19937(random_seed);

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

    { // Boot up
        std::map<std::string, float> timeParam;
        float3 cand1, cand2;
        float HD = directHD(static_cast<OptiXHDProgram &>(*optixGlobalParams.programList["SamplingBased"]),
                            dA, dB, cand1, cand2, VERTEX, 0.001f, timeParam);
    }

    std::map<std::string, float> timeParam;
    float3 cand1, cand2;
    float HD = 0.0f;
    auto cubqlHDTime = SPIN::TimeCheck([&]()
                                       {
                                                       float3 cand1_t, cand2_t;
                                                       float HD1 = cubqlHD(dA, dB, cand1, cand2, timeParam);
                                                       float HD2 = cubqlHD(dB, dA, cand1_t, cand2_t, timeParam);
                                                       //float HD1 = cubqlClusterHD(dA, dB, cand1, cand2, 0,globalParams["grid_1"], timeParam);
                                                       //float HD2 = cubqlClusterHD(dB, dA, cand1_t, cand2_t,0,globalParams["grid_1"], timeParam);

                                                        if(HD2>HD1){
                                                            cand1 = cand1_t;
                                                            cand2 = cand2_t;
                                                        }
                                                        HD = fmaxf(HD1,HD2); });
    std::cout << "HDIST : " << HD << std::endl;
    std::cout << "Total Time : " << cubqlHDTime << " ms" << std::endl;
    // runSamplingBasedHDISTTest(dA, dB);

    SPIN::Logger log;

    log.data["00_TYPE"].push_back("cuBQL_NN_HD");
    log.data["01_DISTACNE"].push_back(HD);
    log.data["02_PERFORMANCE"].push_back(cubqlHDTime);
    log.data["03_Sampling_Rate"].push_back("-");
    log.data["03_Detail_01_Build_Time"].push_back(timeParam["01_GAS_build"]);
    log.data["03_Detail_02_Intersection_Time"].push_back(timeParam["02_HD_Compute"]);
    log.data["03_Detail_03_Filter_Time"].push_back(timeParam["FilteringTime"]);
    log.data["03_Detail_04_Computing_Time"].push_back(timeParam["ComputingTime"]);

    auto float3ToString = [](float3 data)
    {
        std::string res = std::to_string(data.x) + std::string(", ") + std::to_string(data.y) + ", " + std::to_string(data.z);
        return res;
    };
    log.data["04_cand_1"].push_back(float3ToString(cand1));
    log.data["04_cand_2"].push_back(float3ToString(cand2));

    bool fileExists = std::filesystem::exists(loggerPath);

    std::ofstream logOut(loggerPath, std::ios::app);
    if (!fileExists)
    {
        logOut << log;
    }
    else
    {
        int t_size = log.data.begin()->second.size();
        for (int i = 0; i < t_size; i++)
        {
            for (auto &v : log.data)
            {
                std::visit([&logOut](auto &&arg)
                           { logOut << arg << ";"; }, v.second[i]);
            }
            logOut << std::endl;
        }
    }
    logOut.close();

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

        hdparam.randomseed = random_seed;

        hdparam.samplingRate = samplingRate;

        hdparam.offset = -1;

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

    return HD;
}

void runSamplingBasedHDISTTest(const HDGPUParam<HDMODE::TRIANGLE> &dA, const HDGPUParam<HDMODE::TRIANGLE> &dB)
{
    SPIN::Logger log;
    std::cout << "For Test" << std::endl;
    OptiXHDistSamplingMethod testTarget[]{VERTEX, SPHERE, AABB};

    // Boot up
    {
        std::map<std::string, float> timeParam;
        float3 cand1, cand2;
        float HD = directHD(static_cast<OptiXHDProgram &>(*optixGlobalParams.programList["SamplingBased"]),
                            dA, dB, cand1, cand2, VERTEX, 0.001f, timeParam);
    }

    std::vector<std::string> mtdString = {"Sphere", "AABB", "Vertex"};

    float samplingRate = 0.016f;
    for (int i = 0; i < 5; i++)
    {
        for (const auto &mtd : testTarget)
        {
            std::map<std::string, float> timeParam;
            float3 cand1, cand2;
            float HD = 0.0f;

            auto RTSamplingTimes = SPIN::TimeCheck([&]()
                                                   {
                                                       float3 cand1_t, cand2_t;
                                                       float HD1 = directHD(static_cast<OptiXHDProgram &>(*optixGlobalParams.programList["SamplingBased"]),
                                                                     dA, dB, cand1, cand2, mtd, samplingRate, timeParam);
                                                       float HD2 = directHD(static_cast<OptiXHDProgram &>(*optixGlobalParams.programList["SamplingBased"]),
                                                                            dB, dA, cand1_t, cand2_t, mtd, samplingRate, timeParam);

                                                        if(HD2>HD1){
                                                            cand1 = cand1_t;
                                                            cand2 = cand2_t;
                                                        }
                                                        HD = fmaxf(HD1,HD2); });
            std::cout << "HDIST : " << HD << std::endl;
            std::cout << "Total Time : " << RTSamplingTimes << " ms" << std::endl;

            log.data["00_TYPE"].push_back(mtdString[mtd]);
            log.data["01_DISTACNE"].push_back(HD);
            log.data["02_PERFORMANCE"].push_back(RTSamplingTimes);
            log.data["03_Sampling_Rate"].push_back(samplingRate * 100);
            log.data["03_Detail_01_Build_Time"].push_back(timeParam["01_GAS_build"]);
            log.data["03_Detail_02_Compute_Time"].push_back(timeParam["02_HD_Compute"]);

            auto float3ToString = [](float3 data)
            {
                std::string res = std::to_string(data.x) + std::string(", ") + std::to_string(data.y) + ", " + std::to_string(data.z);
                return res;
            };
            log.data["04_cand_1"].push_back(float3ToString(cand1));
            log.data["04_cand_2"].push_back(float3ToString(cand2));
        }
        samplingRate += 0.001f;
    }

    bool fileExists = std::filesystem::exists(loggerPath);

    std::ofstream logOut(loggerPath, std::ios::app);

    if (!fileExists)
    {
        logOut << log;
    }
    else
    {
        int t_size = log.data.begin()->second.size();
        for (int i = 0; i < t_size; i++)
        {
            for (auto &v : log.data)
            {
                std::visit([&logOut](auto &&arg)
                           { logOut << arg << ";"; }, v.second[i]);
            }
            logOut << std::endl;
        }
    }
    logOut.close();
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
