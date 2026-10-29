#!/usr/bin/env bash
# Isaac Lab 3.0 with the standalone URDF importer, on the Linux filesystem.
# See docs/IMPLEMENTATION.md.
set -euo pipefail

VENV="${MP4_ISAAC_VENV:-$HOME/.venvs/mp4-isaac}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OVERRIDES="https://raw.githubusercontent.com/isaac-sim/IsaacLab/v3.0.0/tools/wheel_builder/uv-overrides.txt"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.12 "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# The CUDA wheels are large. A short timeout drops them on a slow link.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-2}"

uv pip install -U pip
uv pip install "isaaclab[importers]==3.0.0" \
  --overrides "$OVERRIDES" \
  --index https://pypi.nvidia.com \
  --torch-backend cu130 \
  --index-strategy unsafe-best-match
# Isaac Lab's published wheel pulls torch 2.11.0+cu130. This WSL driver is CUDA
# 12.9 (576.83), so that wheel cannot open the GPU. Same torch, CUDA 12.8 build.
uv pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install -e "$ROOT"

echo "ready: $VENV/bin/python"
