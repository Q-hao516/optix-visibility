# OptiX Visibility — 零件面可见性/遮挡检测

基于 **NVIDIA OptiX 9** 的 CAD 零件面级可见性分析工具：读取 STEP 模型，对每个面做 GPU 光追遮挡检测，输出**无遮挡 / 部分遮挡 / 完全遮挡**三态分类，并配套 React 三色可视化页面。

## 处理流水线

```
OCCT (pythonocc) 读取 STEP
      ↓
提取目标面并逐面三角化（BRepMesh，含法线朝向修正）
      ↓
面积加权撒点 + 生成采样射线（numpy 矢量化）
      ↓
写入 PVMESH 二进制 → 交给 OptiX 光追程序
      ↓
OptiX 构建单层 GAS (BVH) + optixLaunch 并行遮挡检测
      ↓
按面统计可见射线比例 → 三态分类
      ↓
visibility.json（数据） + visibility.glb（每面一个 node，三色着色）→ React 页面可视化
```

### 两种检测模式

| 模式 | 射线方向 | 用途 |
|---|---|---|
| `normal` 法线半球 | 每采样点沿外法线余弦加权半球采样 N 条 | 自遮挡检测（凹面自阴影计入，如内孔壁） |
| `view` 指定视向 | 每采样点沿给定视向（默认 +Z）发射 1 条 | 装夹/观察方向可见性（+Z 视向下顶面应全绿、底面全红，可作为正确性 sanity） |

### 分类阈值

`visible_ratio = 可见射线数 / 总射线数`：`≥ 0.95` 无遮挡（clear），`≤ 0.05` 完全遮挡（blocked），其余为部分遮挡（partial）。

## 目录结构

```
optix-visibility/
├── optix_visibility/          # C++/CUDA/OptiX 光追程序（Windows）
│   ├── build.ps1              # 一键构建脚本（MSVC host + NVRTC 运行时 JIT）
│   └── src/
│       ├── host.cpp           # OptiX host：GAS 构建 / pipeline / launch / 按面归约
│       ├── device.cu          # OptiX device：raygen / anyhit / miss（NVRTC 编译）
│       ├── params.h           # host/device 共享 launch params
│       └── pvmesh.h           # PVMESH 二进制格式读写
├── bridge/
│   └── visibility_bridge.py   # Python 桥接：OCCT 网格化 / 采样 / spawn exe / GLB 导出
├── backend/
│   └── visibility.ts          # Express 路由（/api/visibility/files|run|result|glb）
├── frontend/
│   └── Visibility.tsx         # React 页面：三色 3D 视图 + 面拾取聚焦 + 分组面列表
└── docs/
```

## 构建光追程序

### 前置要求（Windows）

- NVIDIA GPU（本机实测 RTX 3090）+ 较新驱动（OptiX 9.x 要求 R580+）
- [OptiX 9.1.0 SDK](https://developer.nvidia.com/optix/downloads)（默认安装路径自动识别，或设 `OPTIX_PATH`）
- CUDA Toolkit 12.x（本机 12.4；设 `CUDA_PATH`）
- Visual Studio（含 MSVC v14x 与 Windows SDK；脚本用 vswhere 自动定位）

> 注意：**无需安装 nvcc 可用的 MSVC 版本约束** —— device 代码由 host 程序在运行时用 **NVRTC** JIT 编译为 PTX（`--gpu-architecture=compute_86 -rdc=true` 等），host 代码为纯 C++17，由本机 MSVC 直接编译，彻底绕开 nvcc 与新版 MSVC 的版本检查耦合。若 GPU 非 Ampere（CC 8.6），改 `host.cpp` 中 `--gpu-architecture=compute_86` 一处即可。

### 构建

```powershell
cd optix_visibility
powershell -ExecutionPolicy Bypass -File build.ps1
# 产物: build\optix_visibility.exe（附带 device.cu、params.h、NVRTC DLL）
# 自检: build\optix_visibility.exe --info  →  打印 GPU / CUDA 驱动 / OptiX 版本
```

## 使用

### 1. CLI 直接调用

```bash
# Python 侧：STEP → mesh.bin（采样射线）
python bridge/visibility_bridge.py <step路径> <输出目录> --mode view --dir 0,0,1 --density 8

# 光追：
optix_visibility\build\optix_visibility.exe <输出目录>\mesh.bin <输出目录>\result.bin [--same-face-all]
```

view 模式由桥接脚本自动附加 `--same-face-all`（同面命中一律忽略，防平行视向贴面假遮挡）。

### 2. HTTP 接口（Express）

把 `backend/visibility.ts` 注册进 Express 应用（如 `app.use('/api/visibility', visibilityRoutes)`），提供：

| 接口 | 说明 |
|---|---|
| `GET /api/visibility/files` | 列出源 STEP 目录下的零件 |
| `POST /api/visibility/run` | body `{filename, mode, dir, density, raysPerSample, linDefl}`，spawn 桥接脚本并返回完整结果；GPU 进程全局互斥，并发请求返回 409 |
| `GET /api/visibility/result?filename=` | 读取缓存结果 `data/visibility/<name>/visibility.json` |
| `GET /api/visibility/glb?filename=&v=` | 三色着色 GLB（每面一个 node，名为 `face_<i>`） |

结果缓存于 `data/visibility/<文件名>/`：`mesh.bin` / `result.bin` / `visibility.json` / `visibility.glb`。

### 3. 前端页面

把 `frontend/Visibility.tsx` 放入 `src/pages/`，在路由表加 `/visibility`。页面提供零件下拉、模式切换、参数调节、运行按钮、三色 3D 视图（点击面拾取 + 相机聚焦）、按状态分组的面列表与统计卡片。

## PVMESH v1 二进制格式

固定偏移、小端，Python `ndarray.tofile` 直写、C++ `fread` 直读：

```
magic u32 'PVM1'(0x31564D50) | version u32 = 1
u64 numVerts | u64 numTris | u64 numFaces | u64 numSamples
verts   f32x3 × numVerts            （所有面合并顶点）
tris    u32x3 × numTris             （全局顶点索引）
faceId  u32   × numTris             （每三角形所属面索引）
faces   {u32 triOffset, u32 triCount} × numFaces
samples {f32x3 origin, f32x3 dir, u32 faceId} × numSamples
```

结果文件 `result.bin`：`numFaces × {u32 visible, u32 total}`（交错布局）。

## 集成注意事项

- 所有本机相关配置均通过环境变量注入，代码中不含任何硬编码路径：
  - `FEATUREFOX_HOME` — `featurefox` 包所在目录（`bridge/visibility_bridge.py` 依赖它提供 STEP → solid 与面元数据、逐面三角化；未设置时脚本直接退出并提示）
  - `VISIBILITY_PYTHON` — 桥接脚本使用的 Python 解释器（`backend/visibility.ts`，默认 `python`）
  - `VISIBILITY_STEP_DIR` — STEP 源文件目录（`backend/visibility.ts`，默认 `<项目根>/data/step`）
- Python 侧需 `pythonocc-core`、`numpy`、`trimesh`（GLB 导出）。
- 采样射线在 Python 侧生成（每面点数 `clamp(area × density, 8, 4096)`，面积加权选三角形 + 重心坐标撒点；法线半球为余弦加权）。全局射线上限 400 万条，超限按比例抽稀。

## 关键实现细节（踩坑记录）

1. **`optixLaunch` 的 gridDims 是 raygen 调用总数**，不是 CUDA 的 block 数（每格一个 launch index，硬件内部自动按 warp 调度）。传 CUDA 风格的 block 数会**只执行一条射线且无任何报错**——输出缓冲是未初始化显存，结果全错。建议 launch 前把输出 `memset` 成哨兵值用于自检。
2. **NVRTC 不按源文件目录解析引号 include**，需显式加 `-I<源码目录>`；且 `-I` 与路径必须连写。
3. OptiX 9 的 `OptixPipelineCompileOptions.pipelineLaunchParamsSizeInBytes` 为必填字段。
4. **view 模式的自交处理**：起点沿外法线额外偏移 `2ε`（ε = 1e-4 × 包围盒对角线）使射线从材料外发出，配合同面命中一律忽略，可消除「与视向平行的立面在顶面边界处被浮点命中」的假遮挡；normal 模式则只忽略近距掠射（t < 1e-3 × 对角线），保留凹面真自遮挡语义。
5. **three.js `traverse` 回调里不能重新挂载节点**（会让正在被遍历的 `children` 数组缩短，产生 undefined 崩溃），只收集引用、遍历结束后整体挂载。

## 性能实测（RTX 3090）

| 零件 | 面数 | 射线数 | 总耗时 | 说明 |
|---|---|---|---|---|
| 小零件（34KB STEP） | 11 | 5,592 | 0.4s | 含 NVRTC 编译 ~80ms |
| 大装配（4.9MB STEP） | 561 | 192,096 | 23.3s | 其中 OCCT 三角化 20.1s，GPU 光追仅 1–2s |

瓶颈在 CPU 侧的 OCCT 三角化与采样，GPU 光追本身接近瞬时；算法复杂度 O(采样射线数)，与三角形数解耦（OptiX BVH 加速）。

## License

MIT
