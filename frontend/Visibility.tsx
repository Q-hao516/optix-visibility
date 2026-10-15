import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import {
  ArrowLeft, Loader2, Eye, Play, AlertCircle, ChevronDown, ChevronRight,
} from 'lucide-react'
import {
  fetchVisibilityFiles, runVisibility, visibilityGlbUrl,
  type VisibilitySummary, type VisibilityFace, type VisibilityStatus,
} from '../api'

const STATUS_META: Record<VisibilityStatus, { cn: string; dot: string; text: string; border: string }> = {
  clear:   { cn: '无遮挡',   dot: 'bg-emerald-500', text: 'text-emerald-400', border: 'border-emerald-500/40' },
  partial: { cn: '部分遮挡', dot: 'bg-amber-500',   text: 'text-amber-400',   border: 'border-amber-500/40' },
  blocked: { cn: '完全遮挡', dot: 'bg-rose-500',    text: 'text-rose-400',    border: 'border-rose-500/40' },
  no_mesh: { cn: '未网格化', dot: 'bg-zinc-500',    text: 'text-zinc-400',    border: 'border-zinc-500/40' },
}
const STATUS_ORDER: VisibilityStatus[] = ['clear', 'partial', 'blocked', 'no_mesh']

export default function Visibility() {
  const [files, setFiles] = useState<string[]>([])
  const [selFile, setSelFile] = useState('')
  const [mode, setMode] = useState<'normal' | 'view'>('normal')
  const [dir, setDir] = useState('0,0,1')
  const [density, setDensity] = useState(8)
  const [raysPerSample, setRaysPerSample] = useState(24)
  const [linDefl, setLinDefl] = useState(0.08)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const [summary, setSummary] = useState<VisibilitySummary | null>(null)
  const [selFace, setSelFace] = useState<number | null>(null)
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})

  const viewRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<{
    scene: THREE.Scene
    camera: THREE.PerspectiveCamera
    renderer: THREE.WebGLRenderer
    controls: OrbitControls
    meshGroup: THREE.Group
  } | null>(null)
  const faceMeshesRef = useRef<Map<number, { mesh: THREE.Mesh; origMat: THREE.Material }>>(new Map())
  const selFaceRef = useRef<number | null>(null)

  // ---------- 文件列表 ----------
  useEffect(() => {
    fetchVisibilityFiles().then((r) => {
      if (r.success && r.files) {
        setFiles(r.files)
        if (r.files.length && !selFile) setSelFile(r.files[0])
      }
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ---------- three.js 场景初始化（一次） ----------
  useEffect(() => {
    const view = viewRef.current
    if (!view) return
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x1a1d24)
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1e7)
    camera.up.set(0, 0, 1) // STEP 通常 Z 朝上
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    view.appendChild(renderer.domElement)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.15
    scene.add(new THREE.AmbientLight(0xffffff, 0.75))
    const l1 = new THREE.DirectionalLight(0xffffff, 0.7)
    l1.position.set(1, 2, 1)
    scene.add(l1)
    const l2 = new THREE.DirectionalLight(0xffffff, 0.4)
    l2.position.set(-1, -0.5, -1)
    scene.add(l2)
    const meshGroup = new THREE.Group()
    scene.add(meshGroup)
    sceneRef.current = { scene, camera, renderer, controls, meshGroup }

    const onResize = () => {
      const cw = view.clientWidth, ch = view.clientHeight
      if (!cw || !ch) return
      camera.aspect = cw / ch
      camera.updateProjectionMatrix()
      renderer.setSize(cw, ch)
    }
    onResize()
    window.addEventListener('resize', onResize)
    let animId = 0
    const animate = () => {
      animId = requestAnimationFrame(animate)
      controls.update()
      renderer.render(scene, camera)
    }
    animate()

    // 点击拾取 face_<i>（区分点击与拖拽）
    const raycaster = new THREE.Raycaster()
    const ndc = new THREE.Vector2()
    let downX = 0, downY = 0, dragging = false
    const onDown = (e: PointerEvent) => { downX = e.clientX; downY = e.clientY; dragging = false }
    const onMove = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - downX, e.clientY - downY) > 5) dragging = true
    }
    const onUp = (e: PointerEvent) => {
      if (dragging) return
      const rect = renderer.domElement.getBoundingClientRect()
      ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
      ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(ndc, camera)
      const pickMeshes = [...faceMeshesRef.current.values()].map((r) => r.mesh)
      const hits = raycaster.intersectObjects(pickMeshes, false)
      if (!hits.length) { setSelFace(null); return }
      const m = hits[0].object.name.match(/^face_(\d+)$/)
      if (m) setSelFace(parseInt(m[1], 10))
    }
    renderer.domElement.addEventListener('pointerdown', onDown)
    renderer.domElement.addEventListener('pointermove', onMove)
    renderer.domElement.addEventListener('pointerup', onUp)

    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener('resize', onResize)
      renderer.domElement.removeEventListener('pointerdown', onDown)
      renderer.domElement.removeEventListener('pointermove', onMove)
      renderer.domElement.removeEventListener('pointerup', onUp)
      controls.dispose()
      renderer.dispose()
      view.removeChild(renderer.domElement)
      sceneRef.current = null
    }
  }, [])

  // ---------- 高亮选中面 ----------
  useEffect(() => {
    selFaceRef.current = selFace
    for (const [fi, rec] of faceMeshesRef.current) {
      const mat = rec.mesh.material as THREE.MeshBasicMaterial
      if (fi === selFace) {
        mat.color.set(0xffffff)
      } else {
        mat.color.set(0xdddddd)
      }
    }
  }, [selFace])

  // ---------- 加载 GLB（summary 变化时） ----------
  useEffect(() => {
    const sr = sceneRef.current
    if (!sr || !summary?.glb_url) return
    const { scene, camera, controls, meshGroup } = sr
    const loader = new GLTFLoader()
    let disposed = false
    loader.load(
      summary.glb_url,
      (gltf) => {
        if (disposed) return
        // 清空旧模型
        while (meshGroup.children.length) meshGroup.remove(meshGroup.children[0])
        faceMeshesRef.current.clear()
        // 注意：traverse 期间不能重新挂载节点（会让被遍历的 children 数组缩短产生 undefined），
        // 这里只收集引用/加边线，最后整体挂载 gltf.scene
        gltf.scene.traverse((obj) => {
          const mesh = obj as THREE.Mesh
          if (!mesh.isMesh || mesh.type === 'LineSegments') return
          const m = mesh.name.match(/^face_(\d+)$/)
          if (!m) return
          const fi = parseInt(m[1], 10)
          const origMat = mesh.material as THREE.Material
          // 用 Basic 材质展示顶点色（GLB 已按状态着色）
          mesh.material = new THREE.MeshBasicMaterial({ vertexColors: true })
          const edges = new THREE.EdgesGeometry(mesh.geometry as THREE.BufferGeometry, 25)
          const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x333a44 }))
          mesh.add(line)
          faceMeshesRef.current.set(fi, { mesh, origMat })
        })
        meshGroup.add(gltf.scene)
        // 相机适配
        const box = new THREE.Box3().setFromObject(gltf.scene)
        const center = box.getCenter(new THREE.Vector3())
        const size = box.getSize(new THREE.Vector3())
        const maxDim = Math.max(size.x, size.y, size.z)
        if (maxDim > 0 && isFinite(maxDim)) {
          gltf.scene.position.set(-center.x, -center.y, -center.z)
          const dist = maxDim * 1.5
          camera.position.set(dist * 0.55, -dist * 0.55, dist * 0.65)
          controls.target.set(0, 0, 0)
          controls.update()
        }
        setSelFace(null)
      },
      undefined,
      (err) => {
        console.error('[visibility] GLB load error:', err)
        setError(`3D 模型加载失败: ${err instanceof Error ? err.message : String(err)}`)
      },
    )
    return () => { disposed = true }
  }, [summary])

  // ---------- 运行检测 ----------
  const handleRun = useCallback(async () => {
    if (!selFile || running) return
    setRunning(true)
    setError('')
    try {
      const r = await runVisibility({
        filename: selFile,
        mode,
        dir: mode === 'view' ? dir : undefined,
        density,
        raysPerSample: mode === 'normal' ? raysPerSample : 1,
        linDefl,
      })
      if (r.success) {
        setSummary(r)
      } else {
        setError(r.error || '检测失败')
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRunning(false)
    }
  }, [selFile, running, mode, dir, density, raysPerSample, linDefl])

  const faceOf = (fi: number | null): VisibilityFace | null =>
    fi == null ? null : summary?.faces.find((f) => f.fi === fi) ?? null
  const selected = faceOf(selFace)

  // 点击列表行 → 选中并聚焦
  const focusFace = (fi: number) => {
    setSelFace(fi)
    const rec = faceMeshesRef.current.get(fi)
    const sr = sceneRef.current
    if (!rec || !sr) return
    const box = new THREE.Box3().setFromObject(rec.mesh)
    const center = box.getCenter(new THREE.Vector3())
    const sphere = box.getBoundingSphere(new THREE.Sphere())
    const dist = Math.max(sphere.radius * 4, 1e-3)
    // 保持当前视线方向，仅重新聚焦
    const dirV = new THREE.Vector3().subVectors(sr.camera.position, sr.controls.target).normalize()
    sr.controls.target.copy(center)
    sr.camera.position.copy(center).addScaledVector(dirV, dist)
    sr.controls.update()
  }

  const grouped = (st: VisibilityStatus): VisibilityFace[] =>
    (summary?.faces || []).filter((f) => f.status === st)

  return (
    <div className="mx-auto min-h-screen max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
      <header className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link
            to="/"
            className="flex h-9 w-9 items-center justify-center rounded-xl border border-[var(--border)] text-[var(--text-secondary)] transition-colors hover:text-[var(--text)]"
          >
            <ArrowLeft className="h-4 w-4" />
          </Link>
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-[var(--accent-from)] to-[var(--accent-to)] shadow-lg shadow-[var(--accent-shadow)]">
            <Eye className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">可见性检测</h1>
            <p className="text-xs text-[var(--text-secondary)]">
              OptiX GPU 光追遮挡检测 · 无遮挡 / 部分遮挡 / 完全遮挡
            </p>
          </div>
        </div>
      </header>

      {/* 控制栏 */}
      <div className="mb-4 rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-4">
        <div className="flex flex-wrap items-end gap-4">
          <label className="flex min-w-56 flex-1 flex-col gap-1">
            <span className="text-xs text-[var(--text-secondary)]">零件 STEP</span>
            <select
              value={selFile}
              onChange={(e) => setSelFile(e.target.value)}
              className="rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] px-3 py-2 text-sm text-[var(--text)] outline-none"
            >
              {files.map((f) => <option key={f} value={f}>{f}</option>)}
            </select>
          </label>

          <div className="flex flex-col gap-1">
            <span className="text-xs text-[var(--text-secondary)]">模式</span>
            <div className="flex gap-1 rounded-lg border border-[var(--border)] p-0.5">
              {(['normal', 'view'] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`rounded-md px-3 py-1.5 text-xs transition-colors ${
                    mode === m ? 'bg-[var(--bg-surface-hover)] text-[var(--accent)]' : 'text-[var(--text-muted)] hover:text-[var(--text)]'
                  }`}
                >
                  {m === 'normal' ? '法线半球' : '指定视向'}
                </button>
              ))}
            </div>
          </div>

          {mode === 'view' && (
            <label className="flex w-32 flex-col gap-1">
              <span className="text-xs text-[var(--text-secondary)]">视向 (x,y,z)</span>
              <input
                value={dir}
                onChange={(e) => setDir(e.target.value)}
                className="rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] px-3 py-2 text-sm text-[var(--text)] outline-none"
              />
            </label>
          )}

          {mode === 'normal' && (
            <label className="flex w-28 flex-col gap-1">
              <span className="text-xs text-[var(--text-secondary)]">射线/点</span>
              <input
                type="number"
                min={1}
                max={256}
                value={raysPerSample}
                onChange={(e) => setRaysPerSample(Math.max(1, Math.min(256, Number(e.target.value) || 1)))}
                className="rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] px-3 py-2 text-sm text-[var(--text)] outline-none"
              />
            </label>
          )}

          <label className="flex w-28 flex-col gap-1">
            <span className="text-xs text-[var(--text-secondary)]">密度 (点/cm²)</span>
            <input
              type="number"
              min={1}
              max={64}
              value={density}
              onChange={(e) => setDensity(Math.max(1, Math.min(64, Number(e.target.value) || 1)))}
              className="rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] px-3 py-2 text-sm text-[var(--text)] outline-none"
            />
          </label>

          <label className="flex w-28 flex-col gap-1">
            <span className="text-xs text-[var(--text-secondary)]">三角化偏差</span>
            <input
              type="number"
              min={0.01}
              max={0.5}
              step={0.01}
              value={linDefl}
              onChange={(e) => setLinDefl(Math.max(0.01, Math.min(0.5, Number(e.target.value) || 0.08)))}
              className="rounded-lg border border-[var(--border)] bg-[var(--bg-surface)] px-3 py-2 text-sm text-[var(--text)] outline-none"
            />
          </label>

          <button
            onClick={handleRun}
            disabled={running || !selFile}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-br from-[var(--accent-from)] to-[var(--accent-to)] px-5 py-2.5 text-sm font-medium text-white shadow transition-opacity disabled:opacity-50"
          >
            {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
            {running ? '检测中…' : '运行检测'}
          </button>
        </div>
        {error && (
          <div className="mt-3 flex items-center gap-2 rounded-lg bg-rose-500/10 px-3 py-2 text-xs text-rose-400">
            <AlertCircle className="h-3.5 w-3.5 shrink-0" />
            {error}
          </div>
        )}
      </div>

      {/* 统计卡片 */}
      {summary && (
        <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
          {STATUS_ORDER.map((st) => (
            <button
              key={st}
              onClick={() => setCollapsed((c) => ({ ...c, [st]: false }))}
              className={`rounded-xl border bg-[var(--bg-elevated)] p-3 text-left transition-colors ${STATUS_META[st].border}`}
            >
              <div className="flex items-center gap-2">
                <span className={`h-2.5 w-2.5 rounded-full ${STATUS_META[st].dot}`} />
                <span className="text-xs text-[var(--text-secondary)]">{STATUS_META[st].cn}</span>
              </div>
              <div className={`mt-1 text-2xl font-bold ${STATUS_META[st].text}`}>{summary.counts?.[st] ?? 0}</div>
            </button>
          ))}
          <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-3">
            <div className="text-xs text-[var(--text-secondary)]">总面数</div>
            <div className="mt-1 text-2xl font-bold text-[var(--text)]">{summary.num_faces}</div>
          </div>
          <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-3">
            <div className="text-xs text-[var(--text-secondary)]">耗时 / 射线</div>
            <div className="mt-1 text-lg font-bold text-[var(--text)]">
              {(summary.server_elapsed_s ?? summary.elapsed_s ?? 0).toFixed(2)}s
              <span className="ml-1 text-xs font-normal text-[var(--text-muted)]">
                {summary.params?.rays ?? summary.rays ?? '—'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* 主体：3D 视口 + 面列表 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_320px]">
        <div
          ref={viewRef}
          className="relative h-[560px] overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)]"
        >
          {running && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/30">
              <Loader2 className="h-8 w-8 animate-spin text-[var(--accent)]" />
            </div>
          )}
          {!summary && !running && (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-[var(--text-muted)]">
              选择零件并运行检测，结果将按三色渲染
            </div>
          )}
          {selected && (
            <div className="absolute left-3 top-3 z-10 rounded-lg border border-[var(--border)] bg-black/60 px-3 py-2 text-xs text-white backdrop-blur">
              <span className="font-semibold">face_{selected.fi}</span>
              <span className="mx-2 text-[var(--text-muted)]">|</span>
              {selected.surface_cn}
              <span className="mx-2 text-[var(--text-muted)]">|</span>
              {selected.area} mm²
              <span className="mx-2 text-[var(--text-muted)]">|</span>
              可见率 {selected.ratio != null ? `${(selected.ratio * 100).toFixed(1)}%` : '—'}
              <span className="mx-2 text-[var(--text-muted)]">|</span>
              <span className={STATUS_META[selected.status].text}>{STATUS_META[selected.status].cn}</span>
            </div>
          )}
        </div>

        <div className="max-h-[560px] space-y-2 overflow-y-auto rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-3">
          {!summary && <div className="p-4 text-center text-xs text-[var(--text-muted)]">暂无结果</div>}
          {STATUS_ORDER.map((st) => {
            const list = grouped(st)
            if (!list.length) return null
            const isCollapsed = collapsed[st]
            return (
              <div key={st}>
                <button
                  onClick={() => setCollapsed((c) => ({ ...c, [st]: !c[st] }))}
                  className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-xs text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-surface-hover)]"
                >
                  {isCollapsed ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                  <span className={`h-2 w-2 rounded-full ${STATUS_META[st].dot}`} />
                  <span className="font-medium">{STATUS_META[st].cn}</span>
                  <span className="ml-auto text-[var(--text-muted)]">{list.length}</span>
                </button>
                {!isCollapsed && (
                  <div className="mt-1 space-y-0.5">
                    {list.map((f) => (
                      <button
                        key={f.fi}
                        onClick={() => focusFace(f.fi)}
                        className={`flex w-full items-center gap-2 rounded-lg px-3 py-1.5 text-left text-xs transition-colors ${
                          selFace === f.fi
                            ? 'bg-[var(--accent-bg-15)] text-[var(--accent)]'
                            : 'text-[var(--text-secondary)] hover:bg-[var(--bg-surface-hover)]'
                        }`}
                      >
                        <span className="w-12 shrink-0 font-mono">#{f.fi}</span>
                        <span className="w-14 shrink-0">{f.surface_cn}</span>
                        <span className="w-20 shrink-0 text-right tabular-nums">{f.area} mm²</span>
                        <span className="ml-auto tabular-nums">
                          {f.ratio != null ? `${(f.ratio * 100).toFixed(0)}%` : '—'}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
