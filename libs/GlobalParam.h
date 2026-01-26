#pragma once
#include <algorithm>
#include <random>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include "3rdParty/helper_math.h"

enum LaunchMode {
	RTCLUSTER, // 0
};

extern std::mt19937 random_machine;
extern std::map<std::string, int> globalParams;

extern std::string loggerPath;
extern float3 globalTransform;
extern float globalTransformRatio;
extern std::vector<std::string> inputFilePaths;

void verifyArguments(int argc, char* argv[]);

// #ifdef GLOBALPARAM_IMPLEMENTATION
// void verifyArguments(int argc, char* argv[]) {
// 	std::vector<std::string> args;
// 	args.reserve(argc);

// 	for (int i = 0; i < argc; i++) {
// 		args.push_back(std::string(argv[i]));
// 	}

// 	auto it = std::find(args.begin(), args.end(), "-seed");
// 	if (it != args.end()) {
// 		globalParams["seed"] = std::stoi(*(it + 1));
// 	}
// 	else {
// 		globalParams["seed"] = -1;
// 	}

// 	it = std::find(args.begin(), args.end(), "-random");
// 	if (it != args.end()) {
// 		globalParams["random"] = true;
// 	}
// 	else {
// 		globalParams["random"] = false;
// 	}
	
// 	it = std::find(args.begin(), args.end(), "-t");
// 	if (it != args.end()) {
// 		globalParams["translate"] = true;
// 		globalTransform.x = std::stof(*(it + 1));
// 		globalTransform.y = std::stof(*(it + 2));
// 		globalTransform.z = std::stof(*(it + 3));
// 	}
// 	else
// 	{
// 		globalParams["translate"] = false;
// 	}

// 	if (globalParams["random"]) {
// 		it = std::find(args.begin(), args.end(), "-count");
// 		if (it != args.end()) {
// 			globalParams["count"] = std::stoi(*(it + 1));
// 		}
// 		else
// 		{
// 			globalParams["count"] = 1000;
// 		}
// 	}
// 	else {
// 		it = std::find(args.begin(), args.end(), "-path");
// 		if (it != args.end()) {
// 			globalParams["obj_count"] = std::stoi(*(it + 1));
// 			for (int i = 0; i < globalParams["obj_count"]; i++) {
// 				inputFilePaths.push_back(*(it + 2 + i));
// 			}
// 		}
// 		else {
// 			globalParams["obj_count"] = 2;
// 			inputFilePaths.push_back("../../Dataset/Dragon_decimates.obj");
// 			inputFilePaths.push_back("../../Dataset/Dragon_decimates_trans.obj");
			
// 			globalParams["translate_ratio"] = true;
// 			globalTransformRatio = 0.5f;
// 		}
// 	}

// 	it = std::find(args.begin(), args.end(), "-grid");
// 	if (it != args.end()) {
// 		globalParams["grid_1"] = std::stoi(*(it + 1));
// 		if (globalParams["obj_count"] == 2)
// 			globalParams["grid_2"] = std::stoi(*(it + 2));
// 		else
// 			globalParams["grid_2"] = globalParams["grid_1"];
// 	}
// 	else {
// 		globalParams["grid_1"] = 4;
// 		globalParams["grid_2"] = 4;
// 	}

// 	it = std::find(args.begin(), args.end(), "-tr");
// 	if (it != args.end()) {
// 		globalParams["translate_ratio"] = true;
// 		globalTransformRatio = std::stof(*(it + 1));
// 	}

// 	it = std::find(args.begin(), args.end(), "-log");
// 	if (it != args.end()) {
// 		globalParams["logger"] = true;
// 		loggerPath = *(it + 1);
// 	}
// 	else
// 	{
// 		globalParams["logger"] = false;
// 		loggerPath = "testLog.csv";
// 	}

// 	it = std::find(args.begin(), args.end(), "-mode");
// 	if (it != args.end()) {
// 		globalParams["mode"] = std::stoi(*(it + 1));
// 	}
// 	else {
// 		globalParams["mode"] = 0;
// 	}
// }
// #endif

extern std::vector<float3> basicCUBE;
extern std::vector<uint3> CUBEIDX;
