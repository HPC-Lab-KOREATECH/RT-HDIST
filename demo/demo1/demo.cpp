#include <iostream>
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


void buildQClusterShader();

float3 randomDir() {
	float3 dir;

	std::normal_distribution<float> nd(0.0f, 1.0f);
	dir.x = nd(random_machine);
	dir.y = nd(random_machine);
	dir.z = nd(random_machine);

	dir /= length(dir);

	return dir;
}

int main(int argc, char* argv[])
{
	std::cout << "Device name : " << optixGlobalParams.deviceProps.name << std::endl;
    buildQClusterShader();

	verifyArguments(argc, argv);

    std::random_device rd;
	unsigned int seed = (globalParams["seed"] >= 0) ? globalParams["seed"]:rd();
	std::cout << "Random seed : " << seed << std::endl;
	random_machine = std::mt19937(seed);


    SPIN::Logger log;

    Object_t hA = IO::read<Object_t, SPIN::OBJ>(inputFilePaths[0]);


	HDGPUParam<HDMODE::POINT> dA;
	HDGPUParam<HDMODE::POINT> dB;
	cudaMalloc(&dA.vert, sizeof(float3) * hA.model->meshes[0]->vertex.size());
	upload(hA.model->meshes[0]->vertex, dA);

	float3 boxTranslate = { 0,0,0 };
	OptixAabb aabb = computeAABB_device(dA.vert, dA.vSize);
	float3 aabbSize = aabb2size(aabb);

	if (globalParams["obj_count"] == 1){
		if (globalParams["translate_ratio"]) {
			globalTransform = globalTransformRatio * make_float3(aabbSize.x, 0, 0);
			boxTranslate = globalTransform;
	    }
			Object_t hB = IO::read<Object_t, SPIN::OBJ>(inputFilePaths[0]);
		    for(auto &v : hB.model->meshes[0]->vertex){
				v += boxTranslate;
			}
			cudaMalloc(&dB.vert, sizeof(float3) * hB.model->meshes[0]->vertex.size());
			upload(hB.model->meshes[0]->vertex, dB);
	    }
	else{
		Object_t hB = IO::read<Object_t, SPIN::OBJ>(inputFilePaths[1]);
		cudaMalloc(&dB.vert, sizeof(float3) * hB.model->meshes[0]->vertex.size());
		upload(hB.model->meshes[0]->vertex, dB);
	}

	log.data["00_source_count"].push_back((int)dA.vSize);
	log.data["00_target_count"].push_back((int)dB.vSize);

	float3 Amin = aabb2min(aabb);
	float3 Amax = aabb2min(aabb);

	log.data["01_source_aabb_min_x"].push_back((float)Amin.x);
	log.data["01_source_aabb_min_y"].push_back((float)Amin.y);
	log.data["01_source_aabb_min_z"].push_back((float)Amin.z);

	log.data["01_source_aabb_max_x"].push_back((float)Amax.x);
	log.data["01_source_aabb_max_y"].push_back((float)Amax.y);
	log.data["01_source_aabb_max_z"].push_back((float)Amax.z);

	aabb = computeAABB_device(dB.vert, dB.vSize);
	float3 Bmin = aabb2min(aabb);
	float3 Bmax = aabb2min(aabb);

	log.data["01_target_aabb_min_x"].push_back((float)Bmin.x);
	log.data["01_target_aabb_min_y"].push_back((float)Bmin.y);
	log.data["01_target_aabb_min_z"].push_back((float)Bmin.z);

	log.data["01_target_aabb_max_x"].push_back((float)Bmax.x);
	log.data["01_target_aabb_max_y"].push_back((float)Bmax.y);
	log.data["01_target_aabb_max_z"].push_back((float)Bmax.z);

	log.data["02_Bit_count_AtoB"].push_back(globalParams["grid_1"]);
	log.data["02_Bit_count_BtoA"].push_back(globalParams["grid_2"]);

    float HD;
	float3 cand1, cand2;
	std::map<std::string, float> timeParam;

	timeParam["IndexSpaceBuildTime"] = 0;
	timeParam["FilteringTime"] = 0;
	timeParam["ComputingTime"] = 0;

	auto RTQclusterTimes = SPIN::TimeCheck([&]() {
		float3 tempCand1, tempCand2;
		float HD1 = qclusterHD(static_cast<OptiXHDProgram&>(*optixGlobalParams.programList["QCluster"]), dA, dB, cand1, cand2, sqrt(3), globalParams["grid_1"], timeParam);
		float HD2 = qclusterHD(static_cast<OptiXHDProgram&>(*optixGlobalParams.programList["QCluster"]), dB, dA, tempCand1, tempCand2, sqrt(3), globalParams["grid_2"], timeParam);
		std::cout << HD1 << ", " << HD2 << std::endl;

		if (HD2 > HD1) {
			cand1 = tempCand1;
			cand2 = tempCand2;
		}

		HD = fmaxf(HD1, HD2);
    });
    
	log.data["04_HD"].push_back(HD);
	log.data["05_Performance(ms)"].push_back(RTQclusterTimes);

	log.data["05__01_index_time"].push_back(timeParam["IndexSpaceBuildTime"]);
	log.data["05__02_filtering_time"].push_back(timeParam["FilteringTime"]);
	log.data["05__03_computing_time"].push_back(timeParam["ComputingTime"]);

	log.data["08_cand1_x"].push_back(cand1.x);
	log.data["08_cand1_y"].push_back(cand1.y);
	log.data["08_cand1_z"].push_back(cand1.z);

	log.data["08_cand2_x"].push_back(cand2.x);
	log.data["08_cand2_y"].push_back(cand2.y);
	log.data["08_cand2_z"].push_back(cand2.z);

	std::cout << cand1.x << " " << cand1.y << " " << cand1.z << std::endl;
	std::cout << cand2.x << " " << cand2.y << " " << cand2.z << std::endl;

	//Logger write---------------------------------------------------------------------------------------------------------------------------------------//
	bool fileExists = std::filesystem::exists(loggerPath);

	std::ofstream logOut(loggerPath, std::ios::app);

	if (!fileExists) {
		logOut << log;
	}
	else {
		int t_size = log.data.begin()->second.size();
		for (int i = 0; i < t_size; i++) {
			for (auto& v : log.data) {
				std::visit([&logOut](auto&& arg) {logOut << arg << ";"; }, v.second[i]);
			}
			logOut << std::endl;
		}
	}
	logOut.close();

	return 0;
}

void buildQClusterShader(){
	OptiXProgramCompileOption hdShaderOption;
	hdShaderOption.fileName = "__shader__hd__qcluster__";
	hdShaderOption.filePath = "";
	hdShaderOption.rayCount = 1;
	hdShaderOption.launchParamName = "optixLaunchParams";
	hdShaderOption.rayGenName = "__raygen__program__";
	hdShaderOption.missProgramNames = { "__miss__radiance" };
	hdShaderOption.hitProgramCount = 1;
	hdShaderOption.hitProgramNames = { {"__intersection__radiance", "__anyhit__radiance", "__closesthit__radiance"} };

	OptiXHDProgram* HDProgram = new OptiXHDProgram(hdShaderOption);

	optixGlobalParams.programList["QCluster"] = HDProgram;
}
