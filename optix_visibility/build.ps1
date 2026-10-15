# optix_visibility.exe 构建脚本（MSVC host + NVRTC 运行时 JIT device.cu）。
# 用法: powershell -ExecutionPolicy Bypass -File build.ps1
# 产物: optix_visibility\build\optix_visibility.exe（附 device.cu 与 NVRTC DLL）

$ErrorActionPreference = 'Stop'

# ---- 定位 Visual Studio (MSVC) ----
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vsRoot  = & $vswhere -latest -property installationPath
if (-not $vsRoot) { throw '未找到 Visual Studio（vswhere）' }
$msvcVer = Get-ChildItem (Join-Path $vsRoot 'VC\Tools\MSVC') -Directory |
    Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty Name
$vcTools = Join-Path $vsRoot "VC\Tools\MSVC\$msvcVer"
$clExe   = Join-Path $vcTools 'bin\Hostx64\x64\cl.exe'
Write-Host "MSVC: $clExe"

# ---- 定位 Windows SDK ----
$sdkKey = Get-ItemProperty 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Microsoft SDKs\Windows\v10.0' -ErrorAction Stop
$sdkRoot = $sdkKey.InstallationFolder        # 来自注册表，安装位置因机器而异
# 用 Include 下的实际目录名（注册表版本号可能少一级，如 10.0.26100 vs 10.0.26100.0）
$sdkVer = Get-ChildItem (Join-Path $sdkRoot 'Include') -Directory |
    Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty Name
Write-Host "WinSDK: $sdkRoot ($sdkVer)"

# ---- 定位 CUDA / OptiX ----
$cudaRoot = $env:CUDA_PATH
if (-not $cudaRoot) { $cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4' }
$optixRoot = $env:OPTIX_PATH
if (-not $optixRoot) { $optixRoot = 'C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0' }
if (-not (Test-Path (Join-Path $optixRoot 'include\optix.h'))) { throw "未找到 OptiX SDK: $optixRoot" }
Write-Host "CUDA: $cudaRoot"
Write-Host "OptiX: $optixRoot"

# ---- 环境变量 ----
$env:INCLUDE = @(
    (Join-Path $vcTools 'include'),
    (Join-Path $sdkRoot "Include\$sdkVer\ucrt"),
    (Join-Path $sdkRoot "Include\$sdkVer\um"),
    (Join-Path $sdkRoot "Include\$sdkVer\shared"),
    (Join-Path $sdkRoot "Include\$sdkVer\winrt"),
    (Join-Path $sdkRoot "Include\$sdkVer\cppwinrt")
) -join ';'
$env:LIB = @(
    (Join-Path $vcTools 'lib\x64'),
    (Join-Path $sdkRoot "Lib\$sdkVer\um\x64"),
    (Join-Path $sdkRoot "Lib\$sdkVer\ucrt\x64"),
    (Join-Path $cudaRoot 'lib\x64')
) -join ';'

# ---- 编译 ----
$here = $PSScriptRoot
$buildDir = Join-Path $here 'build'
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null

& $clExe /nologo /O2 /std:c++17 /EHsc /utf-8 /W3 `
    "/I$(Join-Path $here 'src')" `
    "/I$optixRoot\include" `
    "/I$cudaRoot\include" `
    /Fe:"$buildDir\optix_visibility.exe" /Fo:"$buildDir\" `
    (Join-Path $here 'src\host.cpp') `
    /link cudart_static.lib nvrtc.lib
if ($LASTEXITCODE -ne 0) { throw "cl.exe 编译失败 ($LASTEXITCODE)" }

# ---- 拷贝运行时依赖 ----
Copy-Item (Join-Path $here 'src\device.cu') (Join-Path $buildDir 'device.cu') -Force
Copy-Item (Join-Path $here 'src\params.h') (Join-Path $buildDir 'params.h') -Force
foreach ($dll in @('nvrtc64_120_0.dll', 'nvJitLink_120_0.dll', 'nvrtc-builtins64_124.dll')) {
    $src = Join-Path $cudaRoot "bin\$dll"
    if (Test-Path $src) { Copy-Item $src (Join-Path $buildDir $dll) -Force }
}

Write-Host "`n构建完成: $buildDir\optix_visibility.exe"
& (Join-Path $buildDir 'optix_visibility.exe') --info
