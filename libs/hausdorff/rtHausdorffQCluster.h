#pragma once

#include "3rdParty/helper_math.h"
#include "OptiX_Base.h"

class GlobalHausdorffTestParm {
public:
	BYTE bitCount = 4;


	size_t number_of_source_point;
	size_t number_of_target_point;
};

class OptiXHDProgram : public OptiXPrograms {
public:
	OptiXHDProgram(OptiXProgramCompileOption programOption);
};

enum class HDMODE{
	POINT,
	TRIANGLE
};

template <HDMODE>
class HDGPUParam {};

template<>
class HDGPUParam<HDMODE::POINT> {
public:
	float3* vert;
	size_t vSize = 0;

	void samplingVTX(std::vector<float3>& vtx, std::vector<uint3>& tris);
};

template<>
class HDGPUParam<HDMODE::TRIANGLE> {
public:
	float3* vert;
	uint3* tri;
	size_t vSize = 0;
	size_t tSize = 0;
};

float qclusterHD(OptiXHDProgram& program, HDGPUParam<HDMODE::POINT> A, HDGPUParam<HDMODE::POINT> B, float3& cand1, float3& cand2, float _eps, BYTE bitCount, std::map<std::string, float>& timeParam);

void upload(std::vector<float3> &HostData, HDGPUParam<HDMODE::POINT>& DeviceData);
void download(HDGPUParam<HDMODE::POINT>& DeviceData, std::vector<float3> & HostData);

class AABB_GAS {
public:
	CUDABuffer asBuffer;
	CUDABuffer tempBuffer;
	CUDABuffer outputBuffer;
	size_t temp_buffer_size;
	size_t output_buffer_size;

	OptixTraversableHandle handle;

	void build(OptixAabb* aabbs, size_t aabbSize) {
		OptixTraversableHandle asHandle{ 0 };

		OptixBuildInput aabb_input = {};
		uint32_t aabbInputFlags = {};

		CUdeviceptr aabbPtr = (CUdeviceptr)aabbs;

		aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
		aabb_input.customPrimitiveArray.aabbBuffers = &aabbPtr;
		aabb_input.customPrimitiveArray.flags = &aabbInputFlags;

		aabb_input.customPrimitiveArray.numSbtRecords = 1;
		aabb_input.customPrimitiveArray.numPrimitives = aabbSize;

		aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
		aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
		aabb_input.customPrimitiveArray.primitiveIndexOffset = 0;

		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
		accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

		OptixAccelBufferSizes blasBufferSize;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(
			optixGlobalParams.optixContext,
			&accelOptions,
			&aabb_input,
			1,
			&blasBufferSize));

		CUDABuffer compactedSizeBuffer;
		compactedSizeBuffer.alloc(sizeof(uint64_t));

		OptixAccelEmitDesc emitDesc;
		emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitDesc.result = compactedSizeBuffer.d_pointer();

		temp_buffer_size = blasBufferSize.tempSizeInBytes;
		output_buffer_size = blasBufferSize.outputSizeInBytes;
		tempBuffer.alloc(blasBufferSize.tempSizeInBytes);
		outputBuffer.alloc(blasBufferSize.outputSizeInBytes);

		cudaEvent_t start, end;
		cudaEventCreate(&start); cudaEventCreate(&end);

		cudaEventRecord(start, 0);
		OPTIX_CHECK(optixAccelBuild(optixGlobalParams.optixContext,
			/* stream */0,
			&accelOptions,
			&aabb_input,
			1,
			tempBuffer.d_pointer(),
			tempBuffer.sizeInBytes,

			outputBuffer.d_pointer(),
			outputBuffer.sizeInBytes,

			&asHandle,

			&emitDesc, 1
		));
		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);
		//CUDA_SYNC_CHECK();

		float times;
		cudaEventElapsedTime(&times, start, end);

		//std::cout << "Build accel time : " << times << "ms" << std::endl;

		cudaEventDestroy(start); cudaEventDestroy(end);

		// ==================================================================
		// perform compaction
		// ==================================================================
		uint64_t compactedSize;
		compactedSizeBuffer.download(&compactedSize, 1);

		asBuffer.alloc(compactedSize);
		OPTIX_CHECK(optixAccelCompact(optixGlobalParams.optixContext,
			/*stream:*/0,
			asHandle,
			asBuffer.d_pointer(),
			asBuffer.sizeInBytes,
			&asHandle));
		//CUDA_SYNC_CHECK();

		// ==================================================================
		// aaaaaand .... clean up
		// ==================================================================
		compactedSizeBuffer.free();

		handle = asHandle;
		//std::cout << traversable << std::endl;
	}

	void refit(OptixAabb* aabbs, size_t aabbsize) {
		OptixBuildInput aabb_input = {};
		uint32_t aabbInputFlags = {};

		CUdeviceptr aabbPtr = (CUdeviceptr)aabbs;

		aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
		aabb_input.customPrimitiveArray.aabbBuffers = &aabbPtr;
		aabb_input.customPrimitiveArray.flags = &aabbInputFlags;

		aabb_input.customPrimitiveArray.numSbtRecords = 1;
		aabb_input.customPrimitiveArray.numPrimitives = aabbsize;

		aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
		aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
		aabb_input.customPrimitiveArray.primitiveIndexOffset = 0;

		OptixAccelBuildOptions accelOptions = {};
		accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
		accelOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;

		OPTIX_CHECK(optixAccelBuild(optixGlobalParams.optixContext,
			0,
			&accelOptions,
			&aabb_input,
			1,
			tempBuffer.d_pointer(),
			tempBuffer.sizeInBytes,

			outputBuffer.d_pointer(),
			outputBuffer.sizeInBytes,

			&handle,

			nullptr, 0
		));
		//CUDA_SYNC_CHECK();
	}

	~AABB_GAS() {
		asBuffer.free();
		outputBuffer.free(); // << the UNcompacted, temporary output buffer
		tempBuffer.free();
	}
};

class AABBCluster {
public:
	float3* points;
	OptixAabb* cluster;
	OptixAabb* pointAabb;

	size_t number_of_cluster;
	size_t number_of_points;

	std::vector<size_t> clusterInfo;
	std::vector<float3> representative;

	~AABBCluster() {
		cudaFree(points);
		cudaFree(cluster);
		cudaFree(pointAabb);
	}
};