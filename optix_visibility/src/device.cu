// OptiX device 程序：目标面可见性遮挡检测。
// 由 host 程序在运行时用 NVRTC 编译为 PTX（见 host.cpp 的 compileDeviceSrc），
// 编译选项须与 OptiX 编程指南的 NVRTC 节一致（-default-device -rdc=true -D__x86_64 等）。
//
// 语义：
//   raygen  —— 每线程 1 条采样射线，payload_0 初始 1（假定可见）；
//   anyhit  —— 命中三角形：若属于射线自身面且距离 < grazingT（邻面掠射/采样点
//              所在三角形自交），忽略继续；否则 payload_0=0 并终止；
//   miss    —— 不做任何事（保持 1，可见）。
// 射线标志 TERMINATE_ON_FIRST_HIT | DISABLE_CLOSESTHIT：遮挡检测只需首个命中布尔。

#include <optix_device.h>
#include "params.h"

extern "C" {
__constant__ VisParams params;
}

static __inline__ __device__ void traceVisibility(float3 origin, float3 dir,
                                                  unsigned rayFace, unsigned& visible) {
    unsigned p0 = visible;  // 1 = 假定可见（miss 保持）
    unsigned p1 = rayFace;
    optixTrace(params.handle,
               origin, dir,
               0.0f, 1e16f, 0.0f,
               OptixVisibilityMask(255u),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               0u, 1u, 0u,  // sbtOffset, sbtStride, missSbtIndex
               p0, p1);
    visible = p0;
}

extern "C" __global__ void __raygen__rg() {
    const unsigned idx = optixGetLaunchIndex().x;
    if (idx >= params.numRays) return;
    const float3 o = params.origins[idx];
    const float3 d = params.dirs[idx];
    unsigned vis = 1u;
    traceVisibility(o, d, params.rayFace[idx], vis);
    params.outVisible[idx] = (unsigned char)vis;
}

extern "C" __global__ void __anyhit__ah() {
    const unsigned rayFace = optixGetPayload_1();
    const unsigned hitFace = params.primFace[optixGetPrimitiveIndex()];
    // 同面命中：近距掠射（采样点所在三角形/邻三角形）总是忽略；
    // view 模式下同面命中一律忽略（射线与面平行时贴面滑行会产生大 t 假遮挡）
    if (hitFace == rayFace &&
        (params.sameFaceAll != 0u || optixGetRayTmax() < params.grazingT)) {
        optixIgnoreIntersection();
        return;
    }
    optixSetPayload_0(0u);  // 被遮挡
    optixTerminateRay();
}

extern "C" __global__ void __miss__ms() {
    // 保持 payload_0 = 1（可见）
}
