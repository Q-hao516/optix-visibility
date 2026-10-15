// PVMESH v1 二进制格式（与 api/visibility_bridge.py 的 write_pvmesh 严格对应）。
//
//   magic u32 'PVM1'(0x31564D50) | version u32 = 1
//   u64 numVerts | u64 numTris | u64 numFaces | u64 numSamples
//   verts   f32x3 × numVerts        （所有面合并顶点）
//   tris    u32x3 × numTris         （全局顶点索引）
//   faceId  u32   × numTris         （每三角形所属面索引）
//   faces   {u32 triOffset, u32 triCount} × numFaces
//   samples {f32x3 origin, f32x3 dir, u32 faceId} × numSamples
//     （origin 已由 Python 侧沿法线/视向偏移 eps，避免自交）
//
// 结果文件 result.bin:
//   numFaces × {u32 visibleCount, u32 totalCount}
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct PvVec3 {
    float x, y, z;
};

struct PvSample {
    PvVec3 origin;
    PvVec3 dir;
    uint32_t faceId;
};

struct PvMesh {
    uint64_t numVerts = 0;
    uint64_t numTris = 0;
    uint64_t numFaces = 0;
    uint64_t numSamples = 0;
    std::vector<PvVec3> verts;
    std::vector<uint32_t> tris;      // 3 × numTris（展平）
    std::vector<uint32_t> triFace;   // numTris
    std::vector<uint32_t> faceTriOffset;  // numFaces
    std::vector<uint32_t> faceTriCount;   // numFaces
    std::vector<PvSample> samples;
};

static const uint32_t PV_MAGIC = 0x31564D50u;  // 'PVM1' little-endian
static const uint32_t PV_VERSION = 1u;

// 读 mesh.bin；成功返回 true，失败时 err 给出原因
inline bool pvReadMesh(const char* path, PvMesh& m, std::string& err) {
    FILE* f = nullptr;
    if (fopen_s(&f, path, "rb") != 0 || !f) {
        err = "cannot open mesh file: " + std::string(path);
        return false;
    }
    auto fail = [&](const std::string& msg) {
        if (f) fclose(f);
        err = msg;
        return false;
    };
    auto rd = [&](void* dst, size_t n) { return fread(dst, 1, n, f) == n; };

    uint32_t magic = 0, version = 0;
    if (!rd(&magic, 4) || magic != PV_MAGIC) return fail("bad magic (not a PVMESH file)");
    if (!rd(&version, 4) || version != PV_VERSION) return fail("unsupported PVMESH version");

    uint64_t counts[4];
    if (!rd(counts, sizeof(counts))) return fail("failed to read counts");
    m.numVerts = counts[0];
    m.numTris = counts[1];
    m.numFaces = counts[2];
    m.numSamples = counts[3];

    m.verts.resize(size_t(m.numVerts));
    m.tris.resize(size_t(m.numTris) * 3);
    m.triFace.resize(size_t(m.numTris));
    m.faceTriOffset.resize(size_t(m.numFaces));
    m.faceTriCount.resize(size_t(m.numFaces));
    m.samples.resize(size_t(m.numSamples));

    if (!rd(m.verts.data(), m.numVerts * sizeof(PvVec3))) return fail("failed to read verts");
    if (!rd(m.tris.data(), m.numTris * 3 * sizeof(uint32_t))) return fail("failed to read tris");
    if (!rd(m.triFace.data(), m.numTris * sizeof(uint32_t))) return fail("failed to read triFace");
    if (!rd(m.faceTriOffset.data(), m.numFaces * sizeof(uint32_t))) return fail("failed to read faceTriOffset");
    if (!rd(m.faceTriCount.data(), m.numFaces * sizeof(uint32_t))) return fail("failed to read faceTriCount");
    if (!rd(m.samples.data(), m.numSamples * sizeof(PvSample))) return fail("failed to read samples");

    fclose(f);
    return true;
}

// 写 result.bin（交错布局：numFaces × {u32 visible, u32 total}，与 Python 侧对齐）
inline bool pvWriteResult(const char* path, const std::vector<uint32_t>& visible,
                          const std::vector<uint32_t>& total, std::string& err) {
    if (visible.size() != total.size()) {
        err = "visible/total size mismatch";
        return false;
    }
    FILE* f = nullptr;
    if (fopen_s(&f, path, "wb") != 0 || !f) {
        err = "cannot open result file: " + std::string(path);
        return false;
    }
    for (size_t i = 0; i < visible.size(); ++i) {
        if (fwrite(&visible[i], sizeof(uint32_t), 1, f) != 1 ||
            fwrite(&total[i], sizeof(uint32_t), 1, f) != 1) {
            fclose(f);
            err = "failed to write result.bin";
            return false;
        }
    }
    fclose(f);
    return true;
}
