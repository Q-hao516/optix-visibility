// 可见性检测 API：OptiX GPU 光追遮挡检测。
// 依赖 api/visibility_bridge.py + optix_visibility/build/optix_visibility.exe。
// 结果缓存到 data/visibility/<safeKey>/：mesh.bin / result.bin / visibility.json / visibility.glb
import { Router, type Request, type Response } from 'express'
import { spawn } from 'child_process'
import path from 'path'
import fs from 'fs'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const router = Router()

// 桥接脚本解释器（需含 pythonocc-core / numpy / trimesh），可用环境变量 VISIBILITY_PYTHON 覆盖
const PYTHON = process.env.VISIBILITY_PYTHON || 'python'
const ROOT = path.join(__dirname, '..', '..')
// STEP 源文件目录，可用环境变量 VISIBILITY_STEP_DIR 覆盖
const machineSourceDir = process.env.VISIBILITY_STEP_DIR || path.join(ROOT, 'data', 'step')
const bridgeScript = path.join(__dirname, '..', 'visibility_bridge.py')
const visRoot = path.join(ROOT, 'data', 'visibility')

// GPU 光追进程全局互斥：同一时刻只跑一个检测，第二个请求直接 409
let running: Promise<unknown> | null = null
// 进程内结果缓存（避免重复读盘）
const lastResult = new Map<string, Record<string, unknown>>()
const lastError = new Map<string, { code: number | null; error: string; elapsed_s: number }>()

function safeKey(filename: string): string {
  return encodeURIComponent(path.basename(filename))
}
function visDir(safe: string): string {
  return path.join(visRoot, safe)
}

function glbMtimeV(safe: string): number {
  try {
    return fs.statSync(path.join(visDir(safe), 'visibility.glb')).mtimeMs
  } catch {
    return 0
  }
}

// GET /api/visibility/files — 列出可检测的源 STEP 文件
router.get('/files', (_req: Request, res: Response) => {
  try {
    if (!fs.existsSync(machineSourceDir)) {
      res.json({ success: true, files: [] })
      return
    }
    const files: string[] = []
    for (const e of fs.readdirSync(machineSourceDir, { withFileTypes: true })) {
      if (!e.isFile()) continue
      const ext = path.extname(e.name).toLowerCase()
      if (ext === '.step' || ext === '.stp') files.push(e.name)
    }
    files.sort((a, b) => a.localeCompare(b))
    res.setHeader('Cache-Control', 'no-store')
    res.json({ success: true, files })
  } catch {
    res.status(500).json({ success: false, error: '列出源文件失败' })
  }
})

// POST /api/visibility/run — body: { filename, mode?, dir?, density?, raysPerSample?, linDefl? }
router.post('/run', (req: Request, res: Response) => {
  try {
    const filename = String(req.body?.filename || '')
    const safe = path.basename(filename === path.basename(filename) ? filename : '')
    if (!safe) {
      res.status(400).json({ success: false, error: '缺少 filename' })
      return
    }
    const stepPath = path.join(machineSourceDir, safe)
    if (!fs.existsSync(stepPath)) {
      res.status(404).json({ success: false, error: '零件的 STEP 文件不存在' })
      return
    }
    const mode = req.body?.mode === 'view' ? 'view' : 'normal'
    const dir = String(req.body?.dir || '0,0,1')
    const density = Number(req.body?.density ?? 8)
    const raysPerSample = Number(req.body?.raysPerSample ?? 24)
    const linDefl = Number(req.body?.linDefl ?? 0.08)

    if (running) {
      res.status(409).json({ success: false, error: '已有可见性检测在运行，请稍后再试' })
      return
    }

    const outDir = visDir(safe)
    fs.mkdirSync(outDir, { recursive: true })

    const args = [
      bridgeScript, stepPath, outDir,
      '--mode', mode,
      '--dir', dir,
      '--density', String(density),
      '--raysPerSample', String(raysPerSample),
      '--linDefl', String(linDefl),
    ]
    const startedAt = Date.now()
    running = new Promise<void>((resolve) => {
      const proc = spawn(PYTHON, args, {
        cwd: path.join(__dirname, '..'),
        stdio: ['ignore', 'pipe', 'pipe'],
      })
      let out = ''
      let err = ''
      proc.stdout?.on('data', (d) => (out += d.toString()))
      proc.stderr?.on('data', (d) => (err += d.toString()))
      // 前端等待上限 10 分钟
      const timer = setTimeout(() => {
        try { proc.kill() } catch { /* 忽略 */ }
      }, 10 * 60 * 1000)
      proc.on('error', () => { resolve() })
      proc.on('close', (code) => {
        clearTimeout(timer)
        const elapsedS = (Date.now() - startedAt) / 1000
        if (code === 0) {
          try {
            // bridge 的 stdout 末行只是摘要；完整 per-face 数据在 visibility.json
            const jp = path.join(outDir, 'visibility.json')
            const full = JSON.parse(fs.readFileSync(jp, 'utf-8')) as Record<string, unknown>
            full.glb_url = `/api/visibility/glb?filename=${encodeURIComponent(safe)}&v=${glbMtimeV(safe)}`
            full.server_elapsed_s = Number(elapsedS.toFixed(2))
            lastResult.set(safe, full)
            lastError.delete(safe)
          } catch (e) {
            console.error(`[visibility] ${safe} 读取 visibility.json 失败:`, e instanceof Error ? e.message : e)
            lastError.set(safe, { code, error: '检测结果文件读取失败', elapsed_s: Number(elapsedS.toFixed(2)) })
          }
        } else {
          const trimmed = (err || out).trim().slice(-800)
          console.error(`[visibility] ${safe} failed code=${code}: ${trimmed}`)
          lastError.set(safe, { code, error: trimmed, elapsed_s: Number(elapsedS.toFixed(2)) })
        }
        resolve()
      })
    })
    running.finally(() => { running = null })

    // 等检测结束后统一返回（前端 loading 由请求耗时体现）
    void running.then(() => {
      const cached = lastResult.get(safe)
      if (cached) {
        res.setHeader('Cache-Control', 'no-store')
        res.json(cached)
        return
      }
      const e = lastError.get(safe)
      res.status(500).json({
        success: false,
        error: e?.error || '可见性检测失败',
        elapsed_s: e?.elapsed_s,
      })
    })
  } catch (e) {
    res.status(500).json({ success: false, error: e instanceof Error ? e.message : String(e) })
  }
})

// GET /api/visibility/result?filename=.. — 读取缓存的检测结果
router.get('/result', (req: Request, res: Response) => {
  try {
    const filename = String(req.query.filename || '')
    const safe = path.basename(filename === path.basename(filename) ? filename : '')
    if (!safe) {
      res.status(400).json({ success: false, error: '缺少 filename 参数' })
      return
    }
    const jp = path.join(visDir(safe), 'visibility.json')
    if (!fs.existsSync(jp)) {
      res.status(404).json({ success: false, error: '尚未生成检测结果' })
      return
    }
    const summary = JSON.parse(fs.readFileSync(jp, 'utf-8'))
    summary.glb_url = `/api/visibility/glb?filename=${encodeURIComponent(safe)}&v=${glbMtimeV(safe)}`
    res.setHeader('Cache-Control', 'no-store')
    res.json(summary)
  } catch (e) {
    res.status(500).json({ success: false, error: e instanceof Error ? e.message : String(e) })
  }
})

// GET /api/visibility/glb?filename=..&v=.. — 三色着色 GLB（每面一个 node face_<i>）
router.get('/glb', (req: Request, res: Response) => {
  try {
    const filename = String(req.query.filename || '')
    const safe = path.basename(filename === path.basename(filename) ? filename : '')
    const fp = path.join(visDir(safe), 'visibility.glb')
    if (!safe || !fs.existsSync(fp)) {
      res.status(404).json({ success: false, error: '检测模型未生成' })
      return
    }
    res.setHeader('Content-Type', 'model/gltf-binary')
    res.setHeader('Cache-Control', 'public, max-age=86400')
    res.sendFile(fp)
  } catch {
    res.status(500).json({ success: false, error: '获取检测模型失败' })
  }
})

export default router
