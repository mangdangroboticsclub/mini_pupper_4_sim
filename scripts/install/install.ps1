<#
.SYNOPSIS
    Create the mp4-sim Python virtual environment (Windows).

.DESCRIPTION
      gpu mode  - mjlab / MuJoCo Warp parallel training, ONNX export, and the
                  CPU MuJoCo + Gymnasium path. Needs an NVIDIA GPU; torch is
                  taken from the cu128 wheel index.
      cpu mode  - MuJoCo + Gymnasium + Stable-Baselines3 training and ONNX
                  export/runtime. No CUDA, no mjlab.

    Pins live in requirements\constraints.txt. Once mp4-sim has a
    pyproject.toml it is installed editable as well.

.PARAMETER Mode
    auto (default) | gpu | cpu.  auto picks gpu when nvidia-smi is on PATH.

.PARAMETER VenvDir
    Venv location, relative to the repo root. Default: .venv

.PARAMETER Python
    Interpreter spec passed to `uv venv --python`. Default: 3.12

.PARAMETER DryRun
    Resolve only; pass --dry-run to uv and skip the import check.

.PARAMETER SkipVerify
    Do not run the post-install import / CUDA check.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1
    powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1 -Mode cpu
    powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1 -Mode gpu -DryRun
#>
[CmdletBinding()]
param(
    [ValidateSet("auto", "gpu", "cpu")]
    [string]$Mode = "auto",
    [string]$VenvDir = ".venv",
    [string]$Python = "3.12",
    [switch]$DryRun,
    [switch]$SkipVerify
)

$ErrorActionPreference = "Stop"

$Root     = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ReqDir   = Join-Path $Root "requirements"
$VenvPath = Join-Path $Root $VenvDir
$Py       = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found. Install: powershell -c `"irm https://astral.sh/uv/install.ps1 | iex`""
}

if ($Mode -eq "auto") {
    $Mode = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { "gpu" } else { "cpu" }
    Write-Host "[mp4-sim] auto-detected mode: $Mode"
}
$TorchBackend = if ($Mode -eq "gpu") { "cu128" } else { "cpu" }

$env:PYTHONUTF8 = "1"
$env:WANDB_MODE = "disabled"   # mjlab pulls wandb in; keep it offline

Write-Host "[mp4-sim] root : $Root"
Write-Host "[mp4-sim] venv : $VenvPath"
Write-Host "[mp4-sim] mode : $Mode  (torch backend: $TorchBackend)"

& uv venv --python $Python --allow-existing $VenvPath
if ($LASTEXITCODE -ne 0) { throw "uv venv failed" }

$UvArgs = @("--python", $Py, "-c", (Join-Path $ReqDir "constraints.txt"),
            "--torch-backend=$TorchBackend", "-r", (Join-Path $ReqDir "$Mode.txt"))
if (Test-Path (Join-Path $Root "pyproject.toml")) { $UvArgs += @("-e", $Root) }
if ($DryRun) { $UvArgs += "--dry-run" }

Write-Host "[mp4-sim] installing $Mode requirements ..."
& uv pip install @UvArgs
if ($LASTEXITCODE -ne 0) { throw "uv pip install failed" }

if ($DryRun) {
    Write-Host "[mp4-sim] dry run complete - nothing installed."
    exit 0
}

if (-not $SkipVerify) {
    Write-Host "[mp4-sim] verifying imports ..."
    & $Py (Join-Path $PSScriptRoot "verify.py") $Mode
    if ($LASTEXITCODE -ne 0) { throw "[mp4-sim] verification failed (see above)" }
}

Write-Host ""
Write-Host "[mp4-sim] done. Activate with:"
Write-Host "    $VenvPath\Scripts\Activate.ps1"
