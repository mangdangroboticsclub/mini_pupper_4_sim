# Isaac Lab 3.0 with the standalone URDF importer, on Windows.
# See docs/IMPLEMENTATION.md.
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts\install.ps1 [-Venv PATH]
#   Venv defaults to <repo>\.venv.
#   A path argument overrides that. MP4_ISAAC_VENV does the same when no
#   argument is given.
#
# Isaac Lab 3 lists Windows 11. The virtualenv uses Scripts\python.exe.

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Venv
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Show-Usage {
    Write-Host "Usage: powershell -ExecutionPolicy Bypass -File scripts\install.ps1 [-Venv PATH]"
    Write-Host "Create a Python 3.12 virtualenv and install Isaac Lab plus this package."
    Write-Host "Venv defaults to $Root\.venv."
    Write-Host "Set MP4_ISAAC_VENV to choose a directory when no argument is passed."
}

if ($Venv -in @("-h", "--help", "-?", "/?")) {
    Show-Usage
    exit 0
}

if (-not $Venv) {
    if ($env:MP4_ISAAC_VENV) {
        $Venv = $env:MP4_ISAAC_VENV
    } else {
        $Venv = Join-Path $Root ".venv"
    }
}
if (-not [System.IO.Path]::IsPathRooted($Venv)) {
    $Venv = Join-Path (Get-Location) $Venv
}
$Venv = [System.IO.Path]::GetFullPath($Venv)

function Invoke-Uv {
    & uv @args
    if ($LASTEXITCODE -ne 0) {
        throw "uv failed with exit code $LASTEXITCODE"
    }
}

function Add-UvToPath {
    $dir = Join-Path $env:USERPROFILE ".local\bin"
    $exe = Join-Path $dir "uv.exe"
    if (Test-Path $exe) {
        $env:Path = "$dir;$env:Path"
    }
}

$caption = (Get-CimInstance Win32_OperatingSystem).Caption
if ($caption -notmatch "Windows 11") {
    Write-Warning "Isaac Lab 3 lists Windows 11 as its Windows platform. This host is $caption."
}

$longPaths = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -ErrorAction SilentlyContinue
if ($longPaths.LongPathsEnabled -ne 1) {
    Write-Warning "Windows long paths are off. Isaac Lab's wheels need them. From an elevated PowerShell: New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Add-UvToPath
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "installing uv for the current user"
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    Add-UvToPath
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not on PATH after install. Open a new PowerShell window and run this script again."
}

$Python = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Invoke-Uv venv --python 3.12 $Venv
}

if (-not $env:UV_HTTP_TIMEOUT) {
    $env:UV_HTTP_TIMEOUT = "600"
}
if (-not $env:UV_CONCURRENT_DOWNLOADS) {
    $env:UV_CONCURRENT_DOWNLOADS = "2"
}

$Overrides = "https://raw.githubusercontent.com/isaac-sim/IsaacLab/v3.0.0/tools/wheel_builder/uv-overrides.txt"

Invoke-Uv pip install -U pip --python $Python
Invoke-Uv pip install "isaaclab[importers]==3.0.0" `
    --python $Python `
    --overrides $Overrides `
    --index https://pypi.nvidia.com `
    --torch-backend cu130 `
    --index-strategy unsafe-best-match
# Isaac Lab's published wheel pulls torch 2.11.0+cu130. A CUDA 12.9 driver
# cannot open that wheel. Same torch, CUDA 12.8 build. A newer driver can
# load this build as well.
Invoke-Uv pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 `
    --python $Python `
    --index-url https://download.pytorch.org/whl/cu128
Invoke-Uv pip install -e $Root --python $Python

Write-Host "ready: $Python"
