// host.cpp 与 device.cu 共享的 launch params 结构。
// 字段布局必须两边完全一致（device 侧 __constant__ VisParams params，
// pipelineLaunchParamsVariableName = "params"）。
#pragma once

#include <vector_types.h>

struct VisParams {
    unsigned long long handle;        // OptixTraversableHandle（单 GAS traversable）
    const float3* origins;            // 每射线起点（已偏移 eps）
    const float3* dirs;               // 每射线方向（单位向量）
    const unsigned* rayFace;          // 每射线所属目标面 id
    const unsigned* primFace;         // 每三角形所属面 id
    unsigned char* outVisible;        // 输出：1=可见(miss) 0=被遮挡(hit)
    unsigned numRays;
    float grazingT;                   // 同面掠射忽略阈值（= 1e-3 × 包围盒对角线）
    unsigned sameFaceAll;             // 1=view 模式：同面命中一律忽略（防平行视向贴面假遮挡）
};
