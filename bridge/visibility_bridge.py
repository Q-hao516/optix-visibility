"""可见性检测桥接脚本。

流水线: OCCT 读 STEP → 逐面三角化（featurefox._tessellate_solid）→ 面积/法线统计
        → 生成采样射线（normal=法线半球余弦加权 / view=指定视向）→ 写 mesh.bin
        → spawn optix_visibility.exe（OptiX GPU 光追遮挡检测）→ 读 result.bin
        → 按面分类 无遮挡/部分遮挡/完全遮挡 → visibility.json + visibility.glb

用法:
  python visibility_bridge.py <stepPath> <outDir>
       [--mode normal|view] [--dir 0,0,1] [--density 8] [--raysPerSample 24]
       [--linDefl 0.08] [--exe <optix_visibility.exe>]

  --mode normal   每采样点沿外法线半球余弦加权采样 raysPerSample 条射线（自遮挡检测）
  --mode view     每采样点沿 -视向 发射 1 条射线（如 +Z 视向：顶面应全绿、底面全红）
  --density       面采样密度（点/cm²，默认 8）；单面点数 clamp 在 [8, 4096]
  --linDefl       OCCT 三角化线性偏差（mm，默认 0.08）

输出 outDir: mesh.bin / result.bin / visibility.json / visibility.glb
stdout 末行: {"success":true,"num_faces":N,"counts":{...},"elapsed_s":..,"glb_url":"..","v":..}
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import struct
import subprocess
import sys
import time

import numpy as np

# featurefox 包所在目录（提供 STEP → solid 与逐面三角化），通过环境变量 FEATUREFOX_HOME 配置
_ff_home = os.environ.get("FEATUREFOX_HOME", "")
if not _ff_home:
    sys.stderr.write(
        "[visibility] 未设置环境变量 FEATUREFOX_HOME，请将其指向 featurefox 包所在目录\n"
    )
    sys.exit(2)
HD = pathlib.Path(_ff_home)
if str(HD) not in sys.path:
    sys.path.insert(0, str(HD))

from featurefox.aag import build_aag_from_step, _SURF_TYPE_TO_STR, SURF_TYPE_ID  # noqa: E402
from featurefox.export import _tessellate_solid  # noqa: E402

PV_MAGIC = 0x31564D50  # 'PVM1' little-endian
PV_VERSION = 1

# 分类阈值（visible_ratio = visible/total）
TH_CLEAR = 0.95   # >= → 无遮挡
TH_BLOCK = 0.05   # <= → 完全遮挡，否则部分遮挡

# 三态着色（与前端图例一致）
STATUS_COLOR = {
    "clear":   (0.18, 0.75, 0.44),   # 绿
    "partial": (0.95, 0.77, 0.06),   # 黄
    "blocked": (0.91, 0.29, 0.24),   # 红
    "no_mesh": (0.55, 0.55, 0.55),   # 灰（未网格化/退化面）
}

_ID_TO_STR = {SURF_TYPE_ID[k]: v for k, v in _SURF_TYPE_TO_STR.items()}
_SURFACE_CN = {"plane": "平面", "cylinder": "圆柱", "cone": "圆锥", "torus": "圆环"}
RAY_CAP = 4_000_000  # 全局射线数上限（超限按比例缩密度）


def _log(msg: str) -> None:
    print(f"[visibility] {msg}", file=sys.stderr, flush=True)


# ---------------- 采样 ----------------

def _sample_points_on_face(verts: np.ndarray, tris: np.ndarray, density_mm2: float,
                           rng: np.random.Generator):
    """按面积加权在三角形上均匀撒点，返回 (pts(N,3), nrms(N,3), area)。"""
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(cross, axis=1)
    tri_area = 0.5 * norm
    area = float(tri_area.sum())
    if area <= 1e-9 or len(tris) == 0:
        return None
    safe = np.maximum(norm, 1e-12)
    tri_nrm = cross / safe[:, None]
    n_pts = int(np.clip(round(area * density_mm2), 8, 4096))
    prob = tri_area / tri_area.sum()
    sel = rng.choice(len(tris), size=n_pts, p=prob)
    r1 = rng.random(n_pts)
    r2 = rng.random(n_pts)
    sa = np.sqrt(r1)
    u = (1.0 - sa)[:, None]
    w = (sa * (1.0 - r2))[:, None]
    q = (sa * r2)[:, None]
    pts = v0[sel] * u + v1[sel] * w + v2[sel] * q
    nrms = tri_nrm[sel]
    return pts, nrms, area


def _orthonormal_basis(nrms: np.ndarray):
    """每条法线的切向/副切向，用于半球采样定向。"""
    n = nrms / np.maximum(np.linalg.norm(nrms, axis=1, keepdims=True), 1e-12)
    helper = np.zeros_like(n)
    mask = np.abs(n[:, 2]) < 0.9
    helper[mask, 2] = 1.0
    helper[~mask, 0] = 1.0
    t = np.cross(n, helper)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    b = np.cross(n, t)
    return n, t, b


def _cosine_hemisphere_dirs(nrms: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """每点 k 条余弦加权半球方向，返回 (N*k, 3)。"""
    n, t, b = _orthonormal_basis(nrms)
    rep = np.repeat(np.arange(len(n)), k)
    total = len(n) * k
    cos_t = rng.random(total)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t * cos_t))
    phi = rng.random(total) * (2.0 * np.pi)
    d_local = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)
    dirs = t[rep] * d_local[:, 0:1] + b[rep] * d_local[:, 1:2] + n[rep] * d_local[:, 2:3]
    return dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)


# ---------------- PVMESH ----------------

def write_pvmesh(path: pathlib.Path, verts: np.ndarray, tris: np.ndarray,
                 tri_face: np.ndarray, face_tri_offset: np.ndarray,
                 face_tri_count: np.ndarray, origins: np.ndarray, dirs: np.ndarray,
                 ray_face: np.ndarray) -> None:
    num_verts, num_tris, num_faces, num_samples = len(verts), len(tris), len(face_tri_offset), len(ray_face)
    sample_dtype = np.dtype([("o", "<f4", 3), ("d", "<f4", 3), ("fi", "<u4")])
    samples = np.empty(num_samples, dtype=sample_dtype)
    samples["o"] = origins
    samples["d"] = dirs
    samples["fi"] = ray_face
    with open(path, "wb") as f:
        f.write(struct.pack("<II", PV_MAGIC, PV_VERSION))
        f.write(struct.pack("<QQQQ", num_verts, num_tris, num_faces, num_samples))
        np.ascontiguousarray(verts, dtype="<f4").tofile(f)
        np.ascontiguousarray(tris, dtype="<u4").tofile(f)
        np.ascontiguousarray(tri_face, dtype="<u4").tofile(f)
        np.ascontiguousarray(face_tri_offset, dtype="<u4").tofile(f)
        np.ascontiguousarray(face_tri_count, dtype="<u4").tofile(f)
        samples.tofile(f)


def read_result(path: pathlib.Path, num_faces: int):
    arr = np.fromfile(path, dtype=np.dtype([("visible", "<u4"), ("total", "<u4")]))
    if len(arr) != num_faces:
        raise RuntimeError(f"result.bin 面数不符: got {len(arr)} expect {num_faces}")
    return arr["visible"], arr["total"]


# ---------------- GLB ----------------

def build_glb(out_path: pathlib.Path, verts: np.ndarray, tris: np.ndarray,
              face_tri_offset: np.ndarray, face_tri_count: np.ndarray,
              faces_meta: list[dict]) -> None:
    """每面一个 node（face_<i>），按状态着色（顶点色）。"""
    import trimesh

    scene = trimesh.Scene()
    for meta in faces_meta:
        fi = meta["fi"]
        if not meta["mesh_ok"]:
            continue
        off, cnt = int(face_tri_offset[fi]), int(face_tri_count[fi])
        if cnt == 0:
            continue
        ftris = tris[off:off + cnt]
        used = ftris.reshape(-1)
        uniq, inv = np.unique(used, return_inverse=True)
        lv = verts[uniq]
        lt = inv.reshape(-1, 3)
        m = trimesh.Trimesh(vertices=lv, faces=lt, process=False)
        color = STATUS_COLOR[meta["status"]]
        vcol = np.ones((len(lv), 4), dtype=np.uint8) * 255
        vcol[:, 0] = int(color[0] * 255)
        vcol[:, 1] = int(color[1] * 255)
        vcol[:, 2] = int(color[2] * 255)
        m.visual.vertex_colors = vcol
        scene.add_geometry(m, node_name=f"face_{fi}", geom_name=f"face_{fi}")

    data = scene.export(file_type="glb")
    if isinstance(data, bytes):
        out_path.write_bytes(data)
    else:
        with open(out_path, "wb") as f:
            f.write(data)


# ---------------- main ----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step_path")
    ap.add_argument("out_dir")
    ap.add_argument("--mode", choices=["normal", "view"], default="normal")
    ap.add_argument("--dir", default="0,0,1", help="view 模式视向，如 0,0,1")
    ap.add_argument("--density", type=float, default=8.0, help="点/cm²")
    ap.add_argument("--raysPerSample", type=int, default=24)
    ap.add_argument("--linDefl", type=float, default=0.08)
    ap.add_argument("--exe", default=None)
    args = ap.parse_args()

    t0 = time.time()
    step_path = pathlib.Path(args.step_path)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "view":
        view = np.array([float(x) for x in args.dir.split(",")], dtype=np.float64)
        if len(view) != 3 or np.linalg.norm(view) < 1e-9:
            print(json.dumps({"success": False, "error": f"非法视向: {args.dir}"}))
            sys.exit(1)
        view = view / np.linalg.norm(view)
        k_per_pt = 1
    else:
        view = None
        k_per_pt = max(1, int(args.raysPerSample))

    # ---- 1. OCCT 读 STEP + 逐面三角化 ----
    _log(f"reading STEP: {step_path.name}")
    aag = build_aag_from_step(str(step_path))
    solid = getattr(aag, "_solid")
    face_meshes = _tessellate_solid(solid, lin_defl=args.linDefl)

    # ---- 2. 合并网格（全局顶点/三角形 + 每三角形面 id） ----
    verts_list: list[np.ndarray] = []
    tris_list: list[np.ndarray] = []
    tri_face_list: list[np.ndarray] = []
    face_tri_offset = np.zeros(len(aag.faces), dtype=np.uint32)
    face_tri_count = np.zeros(len(aag.faces), dtype=np.uint32)

    v_cursor = 0
    t_cursor = 0
    faces_meta: list[dict] = []
    for fi, f in enumerate(aag.faces):
        stype = _ID_TO_STR.get(f.surface_type, "other")
        meta = {
            "fi": fi,
            "tsid": f.tshape_addr,
            "surface_type": stype,
            "surface_cn": _SURFACE_CN.get(stype, "其他"),
            "area": round(float(f.area), 3),
            "mesh_ok": False,
            "ratio": None,
            "status": "no_mesh",
        }
        mesh = face_meshes.get(f.tshape_addr)
        if mesh is not None and mesh[1].size > 0:
            fv, ft = mesh
            verts_list.append(np.asarray(fv, dtype=np.float32))
            tris_list.append(np.asarray(ft, dtype=np.uint32) + v_cursor)
            tri_face_list.append(np.full(len(ft), fi, dtype=np.uint32))
            face_tri_offset[fi] = t_cursor
            face_tri_count[fi] = len(ft)
            v_cursor += len(fv)
            t_cursor += len(ft)
            meta["mesh_ok"] = True
        faces_meta.append(meta)

    if v_cursor == 0:
        print(json.dumps({"success": False, "error": "STEP 无可三角化面"}))
        sys.exit(1)

    verts = np.concatenate(verts_list)
    tris = np.concatenate(tris_list)
    tri_face = np.concatenate(tri_face_list)
    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    eps = 1e-4 * diag
    _log(f"mesh: verts={len(verts)} tris={len(tris)} diag={diag:.2f}mm eps={eps:.5f}")

    # ---- 3. 采样射线 ----
    density_mm2 = args.density / 100.0  # 点/cm² → 点/mm²
    rng = np.random.default_rng(0)
    origins_all, dirs_all, face_all = [], [], []
    for meta in faces_meta:
        if not meta["mesh_ok"]:
            continue
        fi = meta["fi"]
        off, cnt = int(face_tri_offset[fi]), int(face_tri_count[fi])
        res = _sample_points_on_face(verts, tris[off:off + cnt], density_mm2, rng)
        if res is None:
            continue
        pts, nrms, _ = res
        if args.mode == "normal":
            dirs = _cosine_hemisphere_dirs(nrms, k_per_pt, rng)
            origins = np.repeat(pts, k_per_pt, axis=0) + np.repeat(nrms, k_per_pt, axis=0) * eps
            face_ids = np.full(len(dirs), fi, dtype=np.uint32)
        else:
            # 射线从面指向观察者（+view）：命中其它几何 → 被遮挡；miss → 可见。
            # 起点沿外法线额外偏 2ε（出材料），防平行视向射线在顶面边界 fp 命中
            origins = pts + (view + 2.0 * nrms) * eps
            dirs = np.tile(view, (len(pts), 1)).astype(np.float32)
            face_ids = np.full(len(pts), fi, dtype=np.uint32)
        origins_all.append(origins)
        dirs_all.append(dirs)
        face_all.append(face_ids)

    origins_arr = np.concatenate(origins_all).astype(np.float32)
    dirs_arr = np.concatenate(dirs_all).astype(np.float32)
    face_arr = np.concatenate(face_all)

    # 全局射线数上限保护
    if len(face_arr) > RAY_CAP:
        keep = np.linspace(0, len(face_arr) - 1, RAY_CAP).astype(np.int64)
        origins_arr, dirs_arr, face_arr = origins_arr[keep], dirs_arr[keep], face_arr[keep]
        _log(f"rays capped to {RAY_CAP}")

    _log(f"samples: {len(face_arr)} rays ({args.mode} mode)")

    # ---- 4. 写 mesh.bin ----
    mesh_bin = out_dir / "mesh.bin"
    write_pvmesh(mesh_bin, verts, tris, tri_face, face_tri_offset, face_tri_count,
                 origins_arr, dirs_arr, face_arr)
    _log(f"mesh.bin: {mesh_bin.stat().st_size / 1e6:.1f} MB")

    # ---- 5. spawn optix_visibility.exe ----
    exe = pathlib.Path(args.exe) if args.exe else \
        pathlib.Path(__file__).resolve().parent.parent / "optix_visibility" / "build" / "optix_visibility.exe"
    if not exe.exists():
        print(json.dumps({"success": False, "error": f"optix_visibility.exe 不存在: {exe}"}))
        sys.exit(1)
    result_bin = out_dir / "result.bin"
    exe_args = [str(exe), str(mesh_bin), str(result_bin)]
    if args.mode == "view":
        exe_args.append("--same-face-all")  # 同面命中一律忽略（平行视向贴面假遮挡）
    _log(f"running OptiX tracer: {exe.name}")
    proc = subprocess.run(exe_args, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        print(json.dumps({"success": False, "error": f"OptiX 光追失败: {tail}"}))
        sys.exit(1)
    for line in (proc.stderr or "").splitlines():
        if line.strip():
            _log(line.strip())

    # ---- 6. 分类 ----
    vis, tot = read_result(result_bin, len(aag.faces))
    for meta in faces_meta:
        fi = meta["fi"]
        if tot[fi] > 0:
            meta["ratio"] = round(float(vis[fi]) / float(tot[fi]), 4)
            meta["status"] = ("clear" if meta["ratio"] >= TH_CLEAR
                              else "blocked" if meta["ratio"] <= TH_BLOCK else "partial")

    counts = {"clear": 0, "partial": 0, "blocked": 0, "no_mesh": 0}
    for meta in faces_meta:
        counts[meta["status"]] += 1

    # ---- 7. 输出 JSON + GLB ----
    elapsed = round(time.time() - t0, 2)
    summary = {
        "success": True,
        "filename": step_path.name,
        "num_faces": len(aag.faces),
        "num_mesh_faces": sum(1 for m in faces_meta if m["mesh_ok"]),
        "counts": counts,
        "params": {
            "mode": args.mode,
            "dir": args.dir if args.mode == "view" else None,
            "density": args.density,
            "raysPerSample": k_per_pt,
            "linDefl": args.linDefl,
            "rays": int(len(face_arr)),
        },
        "elapsed_s": elapsed,
        "faces": faces_meta,
    }
    (out_dir / "visibility.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8")

    glb_path = out_dir / "visibility.glb"
    build_glb(glb_path, verts, tris, face_tri_offset, face_tri_count, faces_meta)

    v = int(glb_path.stat().st_mtime)
    print(json.dumps({
        "success": True,
        "num_faces": len(aag.faces),
        "counts": counts,
        "elapsed_s": elapsed,
        "rays": int(len(face_arr)),
        "glb": str(glb_path),
        "glb_url": f"/api/visibility/glb?filename={step_path.name}&v={v}",
        "v": v,
    }))


if __name__ == "__main__":
    main()
