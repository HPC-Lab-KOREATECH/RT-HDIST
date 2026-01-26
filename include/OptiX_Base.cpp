#include "OptiX_Base.h"
#include <iostream>
#include <optix_function_table_definition.h>

OptiXGlobalParam optixGlobalParams;

static void context_log_cb(unsigned int level,
    const char* tag,
    const char* message,
    void*)
{
    fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

OptiXGlobalParam::OptiXGlobalParam() {
	cudaFree(0);

    int numDevices;
    cudaGetDeviceCount(&numDevices);

    if (numDevices == 0)
        throw std::runtime_error("no cuda devices");

    OPTIX_CHECK(optixInit());
    std::cout << "OptiX Initialized" << std::endl;

    const int deviceID = 0;
    CUDA_CHECK(SetDevice(deviceID));

    cudaGetDeviceProperties(&deviceProps, deviceID);
    CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS)
        fprintf(stderr, "Error querying current context: error code %d\n", cuRes);

    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, context_log_cb, nullptr, 4));
    std::cout << "OptiX context created" << std::endl;
}

OptiXPrograms::OptiXPrograms(OptiXProgramCompileOption programOption){
    shaderName = programOption.fileName;

    moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = programOption.launchParamName.c_str();

    pipelineLinkOptions.maxTraceDepth = 31;

    std::string* ptx = new std::string();

    if (readSourceFile(*ptx, programOption.filePath+programOption.fileName+".ptx")) {
    }

    const std::string ptxCode = ptx->c_str();

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(optixGlobalParams.optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        ptxCode.c_str(),
        ptxCode.size(),
        log, &sizeof_log,
        &module
    ));

    //RayGen Program
    {
        raygenPGs.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDesc.raygen.module = module;
        pgDesc.raygen.entryFunctionName = programOption.rayGenName.c_str();

        // OptixProgramGroup raypg;
        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixGlobalParams.optixContext,
            &pgDesc,
            1,
            &pgOptions,
            log, &sizeof_log,
            &raygenPGs[0]
        ));
        //if (sizeof_log > 1) PRINT(log);
    }

    //miss Program
    {
        missPGs.resize(programOption.rayCount);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.miss.module = module;
        char log[2048];
        size_t sizeof_log = sizeof(log);

        for (int i = 0; i < programOption.rayCount; i++) {
            pgDesc.miss.entryFunctionName = programOption.missProgramNames[i].c_str();

            OPTIX_CHECK(optixProgramGroupCreate(optixGlobalParams.optixContext,
                &pgDesc,
                1,
                &pgOptions,
                log, &sizeof_log,
                &missPGs[i]
            ));
        }
    }

    //hitgroup Program
    {
        hitgroupPGs.resize(programOption.rayCount * programOption.hitProgramCount);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        char log[2048];
        size_t sizeof_log = sizeof(log);

        for (int i = 0; i < programOption.hitProgramCount; i++) {
            for (int j = 0; j < programOption.rayCount; j++) {
                std::vector<std::string> programNames = programOption.hitProgramNames[i];
                int gIdx = j * 3;

                pgDesc.hitgroup.moduleIS = (programNames[gIdx+0].empty())?nullptr:module;
                pgDesc.hitgroup.entryFunctionNameIS = (programNames[gIdx + 0].empty()) ? nullptr : programNames[gIdx + 0].c_str();
                pgDesc.hitgroup.moduleAH = (programNames[gIdx + 1].empty()) ? nullptr : module;
                pgDesc.hitgroup.entryFunctionNameAH = (programNames[gIdx + 1].empty()) ? nullptr : programNames[gIdx + 1].c_str();
                pgDesc.hitgroup.moduleCH = (programNames[gIdx + 2].empty()) ? nullptr : module;
                pgDesc.hitgroup.entryFunctionNameCH = (programNames[gIdx + 2].empty()) ? nullptr : programNames[gIdx + 2].c_str();

                OPTIX_CHECK(optixProgramGroupCreate(optixGlobalParams.optixContext,
                    &pgDesc,
                    1,
                    &pgOptions,
                    log, &sizeof_log,
                    &hitgroupPGs[i*programOption.rayCount + j]
                ));
            }
        }
    }

    //make pipeline
    {
        std::vector<OptixProgramGroup> programGroups;
        for (auto pg : raygenPGs)
            programGroups.push_back(pg);
        for (auto pg : missPGs)
            programGroups.push_back(pg);
        for (auto pg : hitgroupPGs)
            programGroups.push_back(pg);

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(optixGlobalParams.optixContext,
            &pipelineCompileOptions,
            &pipelineLinkOptions,
            programGroups.data(),
            (int)programGroups.size(),
            log, &sizeof_log,
            &pipeline
        ));
        //if (sizeof_log > 1) PRINT(log);

        OPTIX_CHECK(optixPipelineSetStackSize
        (/* [in] The pipeline to configure the stack size for */
            pipeline,
            /* [in] The direct stack size requirement for direct
               callables invoked from IS or AH. */
            2 * 1024,
            /* [in] The direct stack size requirement for direct
               callables invoked from RG, MS, or CH.  */
            2 * 1024,
            /* [in] The continuation stack requirement. */
            2 * 1024,
            /* [in] The maximum depth of a traversable graph
               passed to trace. */
            1));
        //if (sizeof_log > 1) PRINT(log);
    }
}

void OptiXPrograms::Launches(CUDABuffer paramBuffer, uint3 layout, CUstream stream){
    OPTIX_CHECK(optixLaunch(pipeline, stream,
        paramBuffer.d_pointer(),
        paramBuffer.sizeInBytes,
        &sbt,
        layout.x,
        layout.y,
        layout.z
    ));
    cudaStreamSynchronize(stream);
}