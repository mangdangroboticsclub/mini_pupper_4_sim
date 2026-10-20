#!/usr/bin/env bash
# Create the mp4-sim Python virtual environment (Linux / macOS).
#
#   gpu mode - mjlab / MuJoCo Warp parallel training, ONNX export, and the CPU
#              MuJoCo + Gymnasium path. Needs an NVIDIA GPU; torch comes from
#              the cu130 wheel index.
#   cpu mode - the CPU simulator, servo models and ONNX export/runtime.
#              No CUDA, no mjlab. Only mode on macOS.
#
# Usage:
#   bash scripts/install/install.sh [--mode auto|gpu|cpu] [--venv .venv]
#                                   [--python 3.12] [--dry-run] [--skip-verify]
#
# Headless Linux rendering: export MUJOCO_GL=egl.
set -euo pipefail

MODE="auto"
VENV_DIR=".venv"
PYTHON="3.12"
DRY_RUN=0
SKIP_VERIFY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)        MODE="$2"; shift 2 ;;
    --venv)        VENV_DIR="$2"; shift 2 ;;
    --python)      PYTHON="$2"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    --skip-verify) SKIP_VERIFY=1; shift ;;
    -h|--help)     sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REQ_DIR="$ROOT/requirements"
VENV_PATH="$ROOT/$VENV_DIR"
PY="$VENV_PATH/bin/python"

command -v uv >/dev/null 2>&1 || {
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }

OS="$(uname -s)"
if [[ "$MODE" == "auto" ]]; then
  if [[ "$OS" != "Darwin" ]] && command -v nvidia-smi >/dev/null 2>&1; then MODE=gpu; else MODE=cpu; fi
  echo "[mp4-sim] auto-detected mode: $MODE"
fi
if [[ "$OS" == "Darwin" && "$MODE" == "gpu" ]]; then
  echo "[mp4-sim] gpu mode is not supported on macOS (no CUDA)" >&2; exit 1
fi

# macOS torch wheels on PyPI are already CPU/MPS builds.
TORCH_FLAGS=()
if [[ "$OS" != "Darwin" ]]; then
  if [[ "$MODE" == "gpu" ]]; then TORCH_FLAGS=(--torch-backend=cu130); else TORCH_FLAGS=(--torch-backend=cpu); fi
fi

export PYTHONUTF8=1
export WANDB_MODE=disabled

echo "[mp4-sim] root : $ROOT"
echo "[mp4-sim] venv : $VENV_PATH"
echo "[mp4-sim] mode : $MODE  (${TORCH_FLAGS[*]:-pypi torch})"

uv venv --python "$PYTHON" --allow-existing "$VENV_PATH"

ARGS=(--python "$PY" -c "$REQ_DIR/constraints.txt" "${TORCH_FLAGS[@]}" -r "$REQ_DIR/$MODE.txt")
[[ -f "$ROOT/pyproject.toml" ]] && ARGS+=(-e "$ROOT")
[[ $DRY_RUN -eq 1 ]] && ARGS+=(--dry-run)

echo "[mp4-sim] installing $MODE requirements ..."
uv pip install "${ARGS[@]}"

if [[ $DRY_RUN -eq 1 ]]; then
  echo "[mp4-sim] dry run complete - nothing installed."; exit 0
fi

if [[ "$MODE" == "gpu" ]]; then
  # mujoco_warp prints a warning per heightfield overflow from its GPU kernel; remove the warning (the physics is unchanged)
  echo "[mp4-sim] patching mujoco_warp ..."
  "$PY" -m mp4_sim.gpu.patch_warp
fi

if [[ $SKIP_VERIFY -eq 0 ]]; then
  echo "[mp4-sim] verifying imports ..."
  "$PY" "$SCRIPT_DIR/verify.py" "$MODE"
fi

echo
echo "[mp4-sim] done. Activate with:  source $VENV_PATH/bin/activate"
