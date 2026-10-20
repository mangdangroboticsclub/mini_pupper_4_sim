# Training the rough-ground and stair policies

`scripts/recipes/stairs.sh` trains the two Mini Pupper v4 terrain policies
shipped in `policies/`: `rough_terrain` (step 1) and `stairs` (step 6). It needs the
`gpu` environment ([INSTALL.md](INSTALL.md)), including the MuJoCo Warp
patch.

```bash
sh scripts/recipes/stairs.sh          # all six steps
sh scripts/recipes/stairs.sh 1        # stop after step 1 (the rough-ground policy)
```

Run it from the repository root. Environment variables:

| Variable | Default | |
|---|---|---|
| `PYTHON` | `.venv/bin/python` | Interpreter |
| `OUT` | `runs/recipes` | Where run directories go |
| `ENVS` | `1024` | Parallel environments |
| `INIT` | `policies/flat_omni` | The policy step 1 starts from |
| `WANDB_MODE` | `disabled` | W&B logging needs `--logger wandb` on a manual `mp4_sim.gpu.train` run |

At 1024 environments one iteration takes about 2.4 s on an RTX 5080; the six
steps (6200 iterations) take about 4 h. GPU training is not deterministic: a
rerun gives a comparable policy, not the same weights.

Running the script again skips every step whose last checkpoint exists. A
step that was interrupted leaves a run directory without it: delete that
directory and run the script again.

## The steps

Each step after the first fine-tunes a checkpoint of the step before it. All
steps share these options: forward speed commands up to 0.12 m/s, a heading
loop on most environments, randomised foot contact softness (0.005-0.02 s)
and friction (0.4-1.3), a foot-clearance penalty at the swing apex, a
penalty on feet pushed sideways, and a reward for forward progress. Every
terrain has 10 difficulty levels; an environment moves up a level after an
episode that walked 0.8 m forward without falling, and down after a fall or
an episode that walked less than half its commanded distance.

| Step | Terrain | Starts from | Iterations | Last checkpoint |
|---|---|---|---|---|
| 1 | `rough` | `policies/flat_omni` | 1200 | `model_1199.pt` |
| 2 | `stairs` | step 1, `model_700.pt` | 1000 | `model_1699.pt` |
| 3 | `stairs`, ending at the top of the stairs | step 2, `model_1500.pt` | 1000 | `model_2499.pt` |
| 4 | `stairs_narrow` | step 3, `model_1900.pt` | 1000 | `model_2899.pt` |
| 5 | `stairs_narrow`, risers 20-40 mm | step 4, `model_2899.pt` | 1000 | `model_3898.pt` |
| 6 | `stairs_narrow`, risers 25-45 mm | step 5, `model_3898.pt` | 1000 | `model_4897.pt` |

A resumed run continues the iteration count, so checkpoint numbers keep
rising across steps. `policies/rough_terrain` is step 1's `model_700.pt`;
`policies/stairs` is step 6's `model_4500.pt`.

1. **Rough ground.** Starts from the flat-ground omnidirectional policy (see
   [Warm start](#warm-start)) with a 10-frame observation history. The
   `rough` terrain mixes flat ground, low-poly relief (some of it on a
   plinth with vertical edges up to 25 mm), slopes up to 20°, scattered boxes
   and small pyramid stairs (risers 5-30 mm).
2. **Stairs.** Half the tiles become square stairs around the spawn point,
   climbing or descending, every riser and tread drawn at random (nominal
   riser 5-60 mm rising with the level, ±30 % per step, treads 50-200 mm);
   the rest keep the rough-ground mix.
3. **Stair exit.** The same terrain, but an episode on stairs ends as a
   success once the robot is off them. Levels go up on that, and down on a
   fall or on never leaving the start platform.
4. **Narrow flights.** 20 % of the tiles become narrow straight flights of 4
   or 5 steps (nominal riser 10-35 mm, ±15 % per step, treads 45-80 mm,
   0.17-0.30 m wide, a 0.20-0.40 m landing) with drops at the sides and past
   the landing. On these the robot always walks forward at 0.06-0.12 m/s and
   holds the heading it started with. From here the heading loop is the
   robot's PI law, limited to 0.4 rad/s, and an episode ends as a fall only
   past 1.0 rad of tilt (a steep flight tilts the body by itself).
5. **Higher risers.** Narrow-flight risers of 20-40 mm; environments start
   anywhere in the 10 levels (levels are not saved in a checkpoint).
6. **Higher risers again.** 25-45 mm.

Step 1 is seed 4, steps 2-4 seed 5, steps 5-6 seed 6.

### Warm start

Step 1 uses `--variant warm_start`, which starts from an exported policy
(`--init`, default `policies/flat_omni`) instead of random weights:

- **Action scale.** `flat_omni` acts with `(0.32, 0.45, 0.42)` rad per unit
  action, the terrain task with `(0.25, 0.50, 0.50)`. With `k = old / new`
  per joint, the output layer is multiplied by `k` and the first layer's
  previous-action inputs divided by `k`, so the network commands the same
  joint targets in radians from the same robot state.
- **History.** The first layer is widened to the 450-value input: the
  original weights on each term's newest frame, zeros on the older frames.
  The network acts exactly as the original until training uses the history.
- **Anchor.** For the first 50 iterations only the critic trains. PPO then
  adds a loss term pulling the actor's mean action toward the warm-start
  policy's, computed on the noise-free 45-value observation: weight 5 for 50
  iterations, then falling linearly to 0 over 300 (`--beta0`,
  `--beta-hold`, `--beta-decay`). From iteration 400 on it is plain PPO.
- Exploration noise starts at 0.1 and the learning rate at 1e-4.

Steps 2-6 resume with `--resume`: actor, critic, optimizer state, iteration
count and the observation-noise schedule come from the checkpoint, the
history from its run's `config.json`. They train with plain PPO (no anchor)
from a learning rate of 1e-4.

## Running one step

The script calls `mp4_sim.gpu.train`. Step 1 by hand:

```bash
python -m mp4_sim.gpu.train --variant warm_start --init policies/flat_omni --heading loop \
  --foot-softness 0.005 0.02 --friction 0.4 1.3 --vx-max 0.12 --clearance apex \
  --clearance-weight -2 --stumble -0.5 --progress 2 --envs 1024 --out runs/recipes \
  --terrain rough --history 10 --iters 1200 --seed 4 --tag rough
```

A later step adds its terrain options and `--resume <run>/model_<k>.pt`
(see the `step` lines at the end of the script). Each policy's
`config.json` in `policies/` records the command that trained it.

Each run writes `<OUT>/<date>_<time>_warm_start_bam_t5ub06_s<seed>_<tag>/`
(`bam_t5ub06` is the fitted servo model): `config.json` (every option, the
commit, file hashes), a checkpoint `model_<k>.pt` every 50 iterations and at
the end, TensorBoard event files, and `git/` (the repository's commit and
uncommitted diff, written by the training runner).

```bash
tensorboard --logdir runs/recipes
```

## Main training options

The options the recipe uses (`python -m mp4_sim.gpu.train --help` lists
all of them):

| Option | Recipe value | Meaning |
|---|---|---|
| `--variant` | `warm_start` | Preset: warm-starts from `--init`, with the fitted servo model (`bam_t5ub06`) |
| `--init` | `policies/flat_omni` | Exported policy (45 inputs) to warm-start from and anchor to |
| `--resume` | steps 2-6 | Checkpoint to fine-tune; overrides `--init` and `--history` |
| `--history` | `10` | Observation frames in the actor's input |
| `--terrain` | `rough`, `stairs`, `stairs_narrow` | Terrain preset |
| `--iters` | `1200`, `1000` | Iterations in this run |
| `--envs` | `1024` | Parallel environments |
| `--seed` | `4`, `5`, `6` | Training seed |
| `--tag`, `--out` | | Run directory name suffix and parent |
| `--heading` | `loop` | Most environments get their yaw-rate command from a heading loop, as on the robot |
| `--heading-law` | `pi` (steps 4-6) | `p`: `clip(2 err, ±0.6)`; `pi`: the robot's law (`heading.pi_step`) |
| `--heading-wz` | `0.4` | `pi`: the loop's output limit (rad/s) |
| `--vx-max` | `0.12` | Top of the forward speed command (m/s) |
| `--foot-softness` | `0.005 0.02` | Range of the feet's contact time constant, drawn per environment (s) |
| `--friction` | `0.4 1.3` | Range of the feet's friction |
| `--clearance` | `apex` | Penalise a swing apex below 25 mm over the terrain under and 3 cm ahead of the foot, at touchdown |
| `--clearance-weight` | `-2` | Per metre of apex shortfall per touchdown |
| `--stumble` | `-0.5` | Per second per foot pushed mostly sideways |
| `--progress` | `2` | Per second of forward progress (1 = walking at the command) |
| `--stair-exit` | steps 3-6 | Stair episodes end as a success once off the stairs |
| `--tilt-limit` | `1.0` (steps 4-6) | Tilt from vertical that ends an episode as a fall (rad; default 0.7) |
| `--max-init-level` | `9` (steps 5-6) | Environments start on levels 0 to this (default 2) |
| `--narrow-risers` | steps 5-6 | `stairs_narrow`: nominal riser range from easiest to hardest level (m; default 0.010 0.035) |

## Exporting a checkpoint

```bash
python -m mp4_sim.export.onnx --checkpoint runs/recipes/<step 1 run>/model_700.pt --out policies/my_rough
python -m mp4_sim.export.verify policies/my_rough
```

The exporter reads the action scale, history and command source from the
run's `config.json` next to the checkpoint; a run trained with `--heading
loop` gets a `heading_loop` command with the robot's PI gains. It writes
`policy.onnx` and `fixture.npz`; `verify` checks them against
[POLICY_CONTRACT.md](POLICY_CONTRACT.md).

## Evaluating

`mp4_sim.gpu.evaluate` scores an exported policy in simulation, per terrain
and level, on the same seeds for every policy: success, distance, falls,
tilt and servo current. Environments start facing +x, commanded `--vx`
(default 0.10 m/s) for `--seconds` (default 12), under observation noise on
randomised robots. `--heading-loop` supplies the yaw rate from the robot's
PI law.

Rough ground (success: no fall and at least `--goal`, default 0.6 m, of
forward travel):

```bash
python -m mp4_sim.gpu.evaluate --policy policies/rough_terrain --terrain rough --heading-loop \
  --json runs/eval/rough_terrain.json
```

A straight flight built from given dimensions (success: all four feet on the
landing without a fall):

```bash
python -m mp4_sim.gpu.evaluate --policy policies/stairs --terrain stairs_built --risers 0.025 \
  --tread 0.05 --steps 4 --width 0.20 --landing 0.25 --heading-loop --tilt-limit 1.0
```

`--risers` takes several heights, one flight each. `--json` writes the
options, per-cell results and per-trial outcomes; `--video out.mp4` renders
environment 0 (on a headless machine set `MUJOCO_GL=egl`). Servo currents
are model estimates: compare them between policies.
