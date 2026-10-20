#!/bin/sh
# The rough-ground and stair training chain: step 1 trains the rough-ground policy, steps 2-6 the stair policy. Each
# step fine-tunes the previous step's checkpoint with the options below. GPU training is not deterministic: a rerun
# gives a comparable policy, not the same weights.
#
#   sh scripts/recipes/stairs.sh [LAST_STEP]     # default 6; LAST_STEP=1 stops after the rough-ground policy
#
# Run from the repository root. Environment: PYTHON (default .venv/bin/python), OUT (default runs/recipes), ENVS
# (default 1024), INIT (the warm start, default policies/flat_omni). Steps 1-6 run 1200 + 5 x 1000 iterations: about
# 2.4 s per iteration at 1024 environments on an RTX 5080, about 4 h in all. A step whose last checkpoint exists is
# skipped when the script is run again; an interrupted step's run directory must be deleted first.
# Export afterwards, e.g.:
#   python -m mp4_sim.export.onnx --checkpoint runs/recipes/<step 1 run>/model_700.pt  --out policies/my_rough
#   python -m mp4_sim.export.onnx --checkpoint runs/recipes/<step 6 run>/model_4500.pt --out policies/my_stairs
set -eu
LAST=${1:-6}
PY="env -u PYTHONPATH WANDB_MODE=${WANDB_MODE:-disabled} ${PYTHON:-.venv/bin/python} -B"
OUT=${OUT:-runs/recipes}
COMMON="--variant warm_start --heading loop --foot-softness 0.005 0.02 --friction 0.4 1.3 \
 --vx-max 0.12 --clearance apex --clearance-weight -2 --stumble -0.5 --progress 2 --envs ${ENVS:-1024} --out $OUT"
NARROW="--terrain stairs_narrow --stair-exit --tilt-limit 1.0 --heading-law pi --heading-wz 0.4"

run_of() {  # the newest run directory with this seed and tag
  ls -d "$OUT"/*_s"$1"_"$2" 2>/dev/null | tail -n 1
}
step() {  # step number, seed, tag, its last checkpoint, resume checkpoint ("" for the warm start), options
  n=$1 seed=$2 tag=$3 last=$4 resume=$5; shift 5
  [ "$n" -le "$LAST" ] || exit 0
  run=$(run_of "$seed" "$tag")
  if [ -n "$run" ]; then
    [ -f "$run/$last" ] || { echo "step $n: $run has no $last (interrupted?): delete it and run again" >&2; exit 1; }
    echo "step $n: $run done, skipped"; return
  fi
  echo "step $n: $tag"
  if [ -n "$resume" ]; then
    [ -f "$resume" ] || { echo "missing $resume" >&2; exit 1; }
    set -- "$@" --resume "$resume"
  fi
  $PY -m mp4_sim.gpu.train $COMMON "$@" --seed "$seed" --tag "$tag"
}

step 1 4 rough model_1199.pt "" --init "${INIT:-policies/flat_omni}" --terrain rough --history 10 --iters 1200
step 2 5 stairs model_1699.pt "$(run_of 4 rough)/model_700.pt" --terrain stairs --iters 1000
step 3 5 stairs_exit model_2499.pt "$(run_of 5 stairs)/model_1500.pt" --terrain stairs --stair-exit --iters 1000
step 4 5 stairs_narrow model_2899.pt "$(run_of 5 stairs_exit)/model_1900.pt" $NARROW --iters 1000
step 5 6 stairs_narrow_r40 model_3898.pt "$(run_of 5 stairs_narrow)/model_2899.pt" $NARROW --max-init-level 9 \
  --narrow-risers 0.020 0.040 --iters 1000
step 6 6 stairs_narrow_r45 model_4897.pt "$(run_of 6 stairs_narrow_r40)/model_3898.pt" $NARROW --max-init-level 9 \
  --narrow-risers 0.025 0.045 --iters 1000
