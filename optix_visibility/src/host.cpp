// OptiX 目标面可见性遮挡检测 —— host 程序。
//
// 用法:
//   optix_visibility.exe <mesh.bin> <result.bin>   跑遮挡检测并写结果
//   optix_visibility.exe --info                    打印 GPU / OptiX 版本信息
//
// 流程: 读 PVMESH → CUDA/OptiX 初始化 → NVRTC 运行时编译 device.cu →
//       module/PG/pipeline/SBT → 上传缓冲 → 构建单 GAS(BVH) → optixLaunch →
//       读回逐射线结果 → 按面归约 → 写 result.bin
//
// mesh.bin/result.bin 格式见 src/pvmesh.h（与 api/visibility_bridge.py 对应）。

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda_runtime.h>
#include <nvrtc.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "params.h"
#include "pvmesh.h"

namespace fs = std::filesystem;

// ---------------- 错误检查宏 ----------------

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        cudaError_t err_ = (call);                                                           \
        if (err_ != cudaSuccess) {                                                           \
            std::fprintf(stderr, "[cuda] %s failed: %s\n", #call, cudaGetErrorString(err_)); \
            return 1;                                                                        \
        }                                                                                    \
    } while (0)

#define NVRTC_CHECK(call)                                                            \
    do {                                                                             \
        nvrtcResult err_ = (call);                                                   \
        if (err_ != NVRTC_SUCCESS) {                                                 \
            std::fprintf(stderr, "[nvrtc] %s failed: %s\n", #call, nvrtcGetErrorString(err_)); \
            return 1;                                                                \
        }                                                                            \
    } while (0)

#define OPTIX_CHECK(call)                                              \
    do {                                                               \
        OptixResult r_ = (call);                                       \
        if (r_ != OPTIX_SUCCESS) {                                     \
            std::fprintf(stderr, "[optix] %s failed: %s\n", #call,     \
                         optixGetErrorString(r_));                     \
            return 1;                                                  \
        }                                                              \
    } while (0)

static char* g_log = nullptr;
static size_t g_log_size = 0;
#define OPTIX_CHECK_LOG(call)                                                          \
    do {                                                                               \
        char log_[2048];                                                               \
        size_t log_size_ = sizeof(log_);                                               \
        OptixResult r_ = (call);                                                       \
        if (r_ != OPTIX_SUCCESS) {                                                     \
            std::fprintf(stderr, "[optix] %s failed: %s\nlog:\n%s\n", #call,           \
                         optixGetErrorString(r_),                                      \
                         (log_size_ > 0) ? log_ : "(no log)");                         \
            return 1;                                                                  \
        }                                                                              \
        g_log = log_;                                                                  \
        g_log_size = log_size_;                                                        \
    } while (0)

static void context_log_cb(unsigned level, const char* tag, const char* message, void*) {
    if (level <= 3)  // 4=trace 噪声大，只透出 warning 及以上
        std::fprintf(stderr, "[optix-ctx][%u][%s] %s\n", level, tag, message);
}

// ---------------- NVRTC 编译 device.cu ----------------

// 运行时把 device.cu 编译为 PTX（绕开 nvcc 与 MSVC 版本耦合）。
// CUDA include 目录由 CUDA_PATH 环境变量推断，OptiX include 由 OPTIX_PATH 或默认安装路径推断。
// srcDir 会作为附加 include 路径（NVRTC 不按源文件目录解析 "xxx.h" 引号 include）。
static int compileDeviceSrc(const std::string& src, const std::string& srcName,
                            const std::string& srcDir, std::string& ptx, std::string& err) {
    const char* cudaPath = std::getenv("CUDA_PATH");
    std::string cudaInclude = cudaPath ? (std::string(cudaPath) + "\\include")
                                       : std::string("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.4\\include");
    const char* optixPath = std::getenv("OPTIX_PATH");
    std::string optixInclude = optixPath ? (std::string(optixPath) + "\\include")
                                         : std::string("C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 9.1.0\\include");
    if (!fs::exists(optixInclude)) {
        err = "OptiX include dir not found: " + optixInclude + " (set OPTIX_PATH)";
        return 1;
    }

    nvrtcProgram prog = nullptr;
    NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), srcName.c_str(), 0, nullptr, nullptr));

    std::string arch = "--gpu-architecture=compute_86";
    std::string incCuda = "-I" + cudaInclude;
    std::string incOptix = "-I" + optixInclude;
    std::string incSrc = "-I" + srcDir;
    const char* opts[] = {
        arch.c_str(),
        "--std=c++17",
        "--use_fast_math",
        "-default-device",
        "-rdc=true",
        "-D__x86_64",
        "-DNOMINMAX",
        incCuda.c_str(),
        incOptix.c_str(),
        incSrc.c_str(),
    };
    nvrtcResult rc = nvrtcCompileProgram(prog, (int)(sizeof(opts) / sizeof(opts[0])), opts);
    size_t logSize = 0;
    nvrtcGetProgramLogSize(prog, &logSize);
    if (logSize > 1) {
        std::string log(logSize, '\0');
        nvrtcGetProgramLog(prog, log.data());
        std::fprintf(stderr, "[nvrtc] log:\n%s\n", log.c_str());
    }
    if (rc != NVRTC_SUCCESS) {
        err = "NVRTC compile failed: " + std::string(nvrtcGetErrorString(rc));
        nvrtcDestroyProgram(&prog);
        return 1;
    }

    size_t ptxSize = 0;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
    ptx.resize(ptxSize);
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
    nvrtcDestroyProgram(&prog);
    return 0;
}

// ---------------- main ----------------

int main(int argc, char* argv[]) {
    std::vector<std::string> args(argv + 1, argv + argc);
    const bool infoMode = !args.empty() && args[0] == "--info";

    // ---- CUDA / OptiX 初始化 ----
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());

    cudaDeviceProp prop{};
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int driverVer = 0;
    cudaDriverGetVersion(&driverVer);

    if (infoMode) {
        std::printf("{\"success\":true,\"gpu\":\"%s\",\"sm\":\"%d.%d\",\"sms\":%d,"
                    "\"cuda_driver\":\"%d.%d\",\"optix_version\":\"%d.%d.%d\"}\n",
                    prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                    driverVer / 1000, (driverVer % 1000) / 10, OPTIX_VERSION / 10000,
                    (OPTIX_VERSION % 10000) / 100, OPTIX_VERSION % 100);
        return 0;
    }

    if (args.size() < 2) {
        std::fprintf(stderr, "usage: optix_visibility.exe <mesh.bin> <result.bin> [--same-face-all] | --info\n");
        return 1;
    }
    const std::string meshPath = args[0];
    const std::string resultPath = args[1];
    bool sameFaceAll = false;
    for (size_t i = 2; i < args.size(); ++i)
        if (args[i] == "--same-face-all") sameFaceAll = true;
    const auto t0 = std::chrono::steady_clock::now();

    // ---- 读 mesh ----
    PvMesh mesh;
    std::string err;
    if (!pvReadMesh(meshPath.c_str(), mesh, err)) {
        std::fprintf(stderr, "[mesh] %s\n", err.c_str());
        return 1;
    }
    const unsigned numRays = (unsigned)mesh.numSamples;
    std::fprintf(stderr, "[mesh] verts=%llu tris=%llu faces=%llu rays=%u\n",
                 (unsigned long long)mesh.numVerts, (unsigned long long)mesh.numTris,
                 (unsigned long long)mesh.numFaces, numRays);
    if (mesh.numTris == 0 || numRays == 0) {
        std::fprintf(stderr, "[mesh] empty mesh or no samples\n");
        return 1;
    }

    // 包围盒对角线（grazingT 依据）
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (const auto& v : mesh.verts) {
        for (int k = 0; k < 3; ++k) {
            const float c = (&v.x)[k];
            lo[k] = std::min(lo[k], c);
            hi[k] = std::max(hi[k], c);
        }
    }
    const float diag = std::sqrt((hi[0] - lo[0]) * (hi[0] - lo[0]) +
                                 (hi[1] - lo[1]) * (hi[1] - lo[1]) +
                                 (hi[2] - lo[2]) * (hi[2] - lo[2]));

    // ---- OptiX context ----
    OptixDeviceContext context = nullptr;
    {
        OptixDeviceContextOptions opts = {};
        opts.logCallbackFunction = &context_log_cb;
        opts.logCallbackLevel = 3;
        OPTIX_CHECK(optixDeviceContextCreate(/*cuCtx=*/0, &opts, &context));
    }

    // ---- 设备缓冲 ----
    const auto tUp0 = std::chrono::steady_clock::now();
    CUdeviceptr dVerts = 0, dTris = 0, dTriFace = 0, dOrigins = 0, dDirs = 0, dRayFace = 0, dOut = 0;
    CUDA_CHECK(cudaMalloc((void**)&dVerts, mesh.verts.size() * sizeof(PvVec3)));
    CUDA_CHECK(cudaMalloc((void**)&dTris, mesh.tris.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&dTriFace, mesh.triFace.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&dOrigins, mesh.samples.size() * sizeof(float) * 3));
    CUDA_CHECK(cudaMalloc((void**)&dDirs, mesh.samples.size() * sizeof(float) * 3));
    CUDA_CHECK(cudaMalloc((void**)&dRayFace, mesh.samples.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&dOut, mesh.samples.size()));
    CUDA_CHECK(cudaMemcpy((void*)dVerts, mesh.verts.data(), mesh.verts.size() * sizeof(PvVec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)dTris, mesh.tris.data(), mesh.tris.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)dTriFace, mesh.triFace.data(), mesh.triFace.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    // samples 是 {origin,dir,faceId} 交织布局，逐字段拷贝
    {
        std::vector<float> origins(mesh.samples.size() * 3), dirs(mesh.samples.size() * 3);
        std::vector<uint32_t> rayFace(mesh.samples.size());
        for (size_t i = 0; i < mesh.samples.size(); ++i) {
            const PvSample& s = mesh.samples[i];
            origins[i * 3 + 0] = s.origin.x; origins[i * 3 + 1] = s.origin.y; origins[i * 3 + 2] = s.origin.z;
            dirs[i * 3 + 0] = s.dir.x; dirs[i * 3 + 1] = s.dir.y; dirs[i * 3 + 2] = s.dir.z;
            rayFace[i] = s.faceId;
        }
        CUDA_CHECK(cudaMemcpy((void*)dOrigins, origins.data(), origins.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)dDirs, dirs.data(), dirs.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)dRayFace, rayFace.data(), rayFace.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }
    const long long upMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now() - tUp0).count();

    // ---- 构建 GAS（OptiX 内部 BVH） ----
    const auto tGas0 = std::chrono::steady_clock::now();
    OptixTraversableHandle gasHandle = 0;
    {
        OptixAccelBuildOptions accelOpts = {};
        accelOpts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accelOpts.operation = OPTIX_BUILD_OPERATION_BUILD;

        uint32_t triFlag = OPTIX_GEOMETRY_FLAG_NONE;
        OptixBuildInput bi = {};
        bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        bi.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        bi.triangleArray.vertexStrideInBytes = 0;
        bi.triangleArray.numVertices = (unsigned)mesh.numVerts;
        bi.triangleArray.vertexBuffers = &dVerts;
        bi.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        bi.triangleArray.indexStrideInBytes = 0;
        bi.triangleArray.numIndexTriplets = (unsigned)mesh.numTris;
        bi.triangleArray.indexBuffer = dTris;
        bi.triangleArray.flags = &triFlag;
        bi.triangleArray.numSbtRecords = 1;

        OptixAccelBufferSizes sizes = {};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOpts, &bi, 1, &sizes));
        CUdeviceptr dTemp = 0, dOut_ = 0;
        CUDA_CHECK(cudaMalloc((void**)&dTemp, sizes.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc((void**)&dOut_, sizes.outputSizeInBytes));
        OPTIX_CHECK(optixAccelBuild(context, /*stream=*/0, &accelOpts, &bi, 1,
                                    dTemp, sizes.tempSizeInBytes,
                                    dOut_, sizes.outputSizeInBytes,
                                    &gasHandle, nullptr, 0));
        CUDA_CHECK(cudaFree((void*)dTemp));
        // dOut_ 保留至 launch 结束（traversable 引用它）
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    const long long gasMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - tGas0).count();
    std::fprintf(stderr, "[gas] built in %lld ms\n", gasMs);

    // ---- NVRTC 编译 device.cu → module ----
    const auto tPtx0 = std::chrono::steady_clock::now();
    std::string devSrc;
    std::string devSrcPath;
    {
        // 优先 exe 同目录 device.cu，其次 ../src/device.cu
        char exeBuf[MAX_PATH];
        GetModuleFileNameA(nullptr, exeBuf, MAX_PATH);
        fs::path exeDir = fs::path(exeBuf).parent_path();
        std::vector<fs::path> candidates = {exeDir / "device.cu", exeDir / ".." / "src" / "device.cu"};
        bool found = false;
        for (const auto& c : candidates) {
            std::error_code ec;
            if (fs::exists(c, ec)) {
                FILE* f = nullptr;
                if (fopen_s(&f, c.string().c_str(), "rb") == 0 && f) {
                    std::string data;
                    char buf[65536];
                    size_t n;
                    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) data.append(buf, n);
                    fclose(f);
                    devSrc = std::move(data);
                    devSrcPath = fs::absolute(c).string();
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            std::fprintf(stderr, "[ptx] device.cu not found next to exe or in ../src\n");
            return 1;
        }
    }
    std::string ptx;
    // 程序名给完整路径便于日志定位；include 靠 -I<源目录> 解析 "params.h"
    if (compileDeviceSrc(devSrc, devSrcPath,
                         fs::path(devSrcPath).parent_path().string(), ptx, err) != 0) {
        std::fprintf(stderr, "[ptx] %s\n", err.c_str());
        return 1;
    }
    const long long ptxMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - tPtx0).count();
    std::fprintf(stderr, "[ptx] NVRTC compiled in %lld ms (%zu bytes)\n", ptxMs, ptx.size());

    OptixModule module = nullptr;
    OptixPipelineCompileOptions pipeCompileOpts = {};
    {
        OptixModuleCompileOptions modCompileOpts = {};  // 默认优化等级
        pipeCompileOpts.usesMotionBlur = false;
        pipeCompileOpts.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeCompileOpts.numPayloadValues = 2;
        pipeCompileOpts.numAttributeValues = 2;
        pipeCompileOpts.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeCompileOpts.pipelineLaunchParamsVariableName = "params";
        pipeCompileOpts.pipelineLaunchParamsSizeInBytes = sizeof(VisParams);
        OPTIX_CHECK_LOG(optixModuleCreate(context, &modCompileOpts, &pipeCompileOpts,
                                          ptx.data(), ptx.size(), g_log, &g_log_size, &module));
    }

    // ---- 程序组 + pipeline ----
    OptixProgramGroup pgRaygen = nullptr, pgMiss = nullptr, pgHit = nullptr;
    {
        OptixProgramGroupOptions pgOpts = {};
        OptixProgramGroupDesc desc[3] = {};
        desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc[0].raygen.module = module;
        desc[0].raygen.entryFunctionName = "__raygen__rg";
        desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc[1].miss.module = module;
        desc[1].miss.entryFunctionName = "__miss__ms";
        desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc[2].hitgroup.moduleAH = module;
        desc[2].hitgroup.entryFunctionNameAH = "__anyhit__ah";
        OptixProgramGroup pgs[3] = {};
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context, desc, 3, &pgOpts,
                                                g_log, &g_log_size, pgs));
        pgRaygen = pgs[0];
        pgMiss = pgs[1];
        pgHit = pgs[2];
    }

    OptixPipeline pipeline = nullptr;
    {
        OptixPipelineLinkOptions linkOpts = {};
        linkOpts.maxTraceDepth = 1;
        OptixProgramGroup pgs[3] = {pgRaygen, pgMiss, pgHit};
        OPTIX_CHECK_LOG(optixPipelineCreate(context, &pipeCompileOpts, &linkOpts, pgs, 3,
                                            g_log, &g_log_size, &pipeline));
    }

    // ---- SBT ----
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) SbtRec {
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    OptixShaderBindingTable sbt = {};
    {
        SbtRec rgRec, msRec, hgRec;
        OPTIX_CHECK(optixSbtRecordPackHeader(pgRaygen, &rgRec));
        OPTIX_CHECK(optixSbtRecordPackHeader(pgMiss, &msRec));
        OPTIX_CHECK(optixSbtRecordPackHeader(pgHit, &hgRec));
        CUdeviceptr dRg = 0, dMs = 0, dHg = 0;
        CUDA_CHECK(cudaMalloc((void**)&dRg, sizeof(SbtRec)));
        CUDA_CHECK(cudaMalloc((void**)&dMs, sizeof(SbtRec)));
        CUDA_CHECK(cudaMalloc((void**)&dHg, sizeof(SbtRec)));
        CUDA_CHECK(cudaMemcpy((void*)dRg, &rgRec, sizeof(SbtRec), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)dMs, &msRec, sizeof(SbtRec), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)dHg, &hgRec, sizeof(SbtRec), cudaMemcpyHostToDevice));
        sbt.raygenRecord = dRg;
        sbt.missRecordBase = dMs;
        sbt.missRecordStrideInBytes = sizeof(SbtRec);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = dHg;
        sbt.hitgroupRecordStrideInBytes = sizeof(SbtRec);
        sbt.hitgroupRecordCount = 1;
    }

    // ---- launch ----
    VisParams params = {};
    params.handle = (unsigned long long)gasHandle;
    params.origins = (const float3*)dOrigins;
    params.dirs = (const float3*)dDirs;
    params.rayFace = (const unsigned*)dRayFace;
    params.primFace = (const unsigned*)dTriFace;
    params.outVisible = (unsigned char*)dOut;
    params.numRays = numRays;
    params.grazingT = 1e-3f * diag;
    params.sameFaceAll = sameFaceAll ? 1u : 0u;

    CUdeviceptr dParams = 0;
    CUDA_CHECK(cudaMalloc((void**)&dParams, sizeof(VisParams)));
    CUDA_CHECK(cudaMemcpy((void*)dParams, &params, sizeof(VisParams), cudaMemcpyHostToDevice));

    // 诊断：先填 0xEE，launch 后若仍有 0xEE 说明 raygen 未覆盖该射线
    CUDA_CHECK(cudaMemset((void*)dOut, 0xEE, mesh.samples.size()));
    const auto tTrace0 = std::chrono::steady_clock::now();
    // 注意：OptiX launch 维度 = raygen 调用总数（非 CUDA block 数）
    OPTIX_CHECK(optixLaunch(pipeline, /*stream=*/0, dParams, sizeof(VisParams), &sbt,
                            numRays, 1, 1));
    CUDA_CHECK(cudaDeviceSynchronize());
    const long long traceMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - tTrace0).count();

    // ---- 读回 + 按面归约 ----
    std::vector<unsigned char> outVisible(mesh.samples.size());
    CUDA_CHECK(cudaMemcpy(outVisible.data(), (const void*)dOut, mesh.samples.size(), cudaMemcpyDeviceToHost));
    {
        size_t unwritten = 0;
        for (unsigned char v : outVisible) unwritten += (v == 0xEE) ? 1 : 0;
        std::fprintf(stderr, "[diag] unwritten rays: %zu / %zu\n", unwritten, outVisible.size());
    }

    std::vector<uint32_t> visCount(size_t(mesh.numFaces), 0), totalCount(size_t(mesh.numFaces), 0);
    for (size_t i = 0; i < mesh.samples.size(); ++i) {
        const uint32_t f = mesh.samples[i].faceId;
        if (f < totalCount.size()) {
            totalCount[f]++;
            visCount[f] += outVisible[i] ? 1u : 0u;
        }
    }
    if (!pvWriteResult(resultPath.c_str(), visCount, totalCount, err)) {
        std::fprintf(stderr, "[result] %s\n", err.c_str());
        return 1;
    }

    uint32_t blockedRays = 0;
    for (unsigned char v : outVisible) blockedRays += v ? 0u : 1u;

    const long long totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::steady_clock::now() - t0).count();

    std::printf("{\"success\":true,\"gpu\":\"%s\",\"tris\":%llu,\"faces\":%llu,\"traced\":%u,"
                "\"blocked_rays\":%u,\"upload_ms\":%lld,\"gas_ms\":%lld,\"ptx_ms\":%lld,"
                "\"trace_ms\":%lld,\"total_ms\":%lld}\n",
                prop.name, (unsigned long long)mesh.numTris, (unsigned long long)mesh.numFaces,
                numRays, blockedRays, upMs, gasMs, ptxMs, traceMs, totalMs);

    // ---- 清理 ----
    cudaFree((void*)dParams);
    cudaFree((void*)dVerts);
    cudaFree((void*)dTris);
    cudaFree((void*)dTriFace);
    cudaFree((void*)dOrigins);
    cudaFree((void*)dDirs);
    cudaFree((void*)dRayFace);
    cudaFree((void*)dOut);
    return 0;
}
