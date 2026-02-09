#pragma once
#include "3rdParty/helper_math.h"
#include "rtHausdorffQCluster.h"

float cubqlHD(HDGPUParam<HDMODE::TRIANGLE> &dA, HDGPUParam<HDMODE::TRIANGLE> &dB, float3 &cand1, float3 &cand2,
              std::map<std::string, float> &timeParam);
float cubqlClusterHD(HDGPUParam<HDMODE::TRIANGLE> &dA, HDGPUParam<HDMODE::TRIANGLE> &dB, float3 &cand1, float3 &cand2,
                     float _eps, BYTE bitCount,
                     std::map<std::string, float> &timeParam);