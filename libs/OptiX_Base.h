#pragma once
#include "3rdParty/optix7support.h"
#include "3rdParty/CUDABuffer.h"
#include <vector>
#include <string>
#include <map>
#include <fstream>

class OptiXProgramCompileOption {
public:
    std::string filePath;
    std::string fileName;

    int rayCount = 0;
    std::string launchParamName;
    std::string rayGenName;
    std::vector<std::string> missProgramNames;

    int hitProgramCount = 0;
    std::vector<std::vector<std::string>> hitProgramNames;
};

class OptiXPrograms {
public:
    std::string shaderName = "";

    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions = {};

    OptixModule                 module;
    OptixModuleCompileOptions   moduleCompileOptions = {};

    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    OptiXPrograms() {}
    OptiXPrograms(OptiXProgramCompileOption programOption);

    void Launches(CUDABuffer paramBuffer, uint3 layout, CUstream stream = 0);
};

class OptiXGlobalParam {
public:
	CUcontext cudaContext;
	cudaDeviceProp deviceProps;
	OptixDeviceContext optixContext;

    std::map<std::string, OptiXPrograms*> programList;

	OptiXGlobalParam();
};

extern OptiXGlobalParam optixGlobalParams;

static bool readSourceFile(std::string& str, const std::string& filename)
{
    // Try to open file
    std::ifstream file(filename.c_str(), std::ios::binary);
    if (file.good())
    {
        // Found usable source file
        std::vector<unsigned char> buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});
        str.assign(buffer.begin(), buffer.end());
        return true;
    }
    return false;
}