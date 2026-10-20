"""PPO on the SKU2 terrain task (``mp4_sim.gpu.terrain_task``), from scratch or from a walking policy with an
imitation term that fades out.

    python -m mp4_sim.gpu.train --variant scratch --seed 1
    ... --variant warm_start --seed 1   # warm start and anchor: policies/flat_omni, rescaled; beta 5 -> 0

Variants (presets; every option can be set explicitly and goes into ``<run>/config.json``):
- scratch: the ``bam_t5ub06`` actuator model, from scratch, the 45-float observation.
- scratch_history: scratch with a 5-step observation history in the actor.
- warm_start: scratch, warm-started from an exported omni policy (``--init``, an ONNX policy directory; default
  ``policies/flat_omni``), its output layer rescaled per joint to this task's action scale, and anchored to it with
  beta ``--beta0`` held ``--beta-hold`` iterations then fading to 0 over ``--beta-decay``. The rough-ground and stair
  recipe (scripts/recipes/stairs.sh) starts here.
- stock_servo: scratch with the stock servo model.

``--resume <run>/model_k.pt`` fine-tunes a terrain checkpoint instead: actor, critic, std, optimizer, the iteration
and the noise ramp's step counter come from it, the history from its run's config; plain PPO (no anchor), lr0 1e-4.

Network 128-128-64 ELU for actor and critic; PPO lr 1e-3 adaptive (KL 0.01), gamma 0.99, lambda 0.95, clip 0.2,
entropy 0.005, 5 epochs x 4 minibatches, 24 steps per env per iteration. From scratch the exploration std starts at
``--init-std`` 0.5 (learned); from a warm start at 0.1. Runs are written under ``runs/terrain`` (``--out``).

Training refuses to start unless mujoco_warp is patched (``python -m mp4_sim.gpu.patch_warp``; ``--allow-unpatched``
overrides).
"""
import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import (
    MjlabOnPolicyRunner,
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)

from .. import paths
from ..contract import OBS_TERMS
from ..export.onnx import load_policy
from ..robot.sku2 import masses_path
from .env_cfg import describe, noise_profile
from .terrain_task import TERRAIN_ACTION_SCALE, TERRAINS, TerrainOptions, make_terrain_env_cfg

REPO = paths.REPO
sha256 = paths.sha256
PREV_ACTION = slice(33, 45)        # contract.OBS_TERMS: the previous action in the 45-float observation
VARIANTS = {
    "scratch": dict(actuator="bam_t5ub06", history=0, init=None),
    "scratch_history": dict(actuator="bam_t5ub06", history=5, init=None),
    "warm_start": dict(actuator="bam_t5ub06", history=0, init="policies/flat_omni"),
    "stock_servo": dict(actuator="stock", history=0, init=None),
}


def policy_to_checkpoint(policy_dir: Path, new_scale, out: Path, history: int = 0) -> Path:
    """An exported policy (``policy.onnx``, a 45-float observation) as an rsl_rl actor checkpoint in a new action
    scale: output rows times k = old/new per joint,
    and the first layer's previous-action columns divided by k, so the same robot state gives the same target in
    radians. With ``history`` > 0 the first layer is widened to the stacked observation (``contract.HistoryStacker``:
    each term's frames oldest to newest): the original weights on each term's newest frame, zeros on the older
    frames, so the warm start acts exactly as the original until training uses the history."""
    pol = load_policy(policy_dir)
    if pol["history"] != 1:
        raise ValueError(f"{policy_dir}: a warm start needs a policy without history (it has {pol['history']})")
    n = len(pol["weights"])
    k = np.asarray(pol["scale"], dtype=np.float32) / np.asarray(new_scale, dtype=np.float32)
    sd = {}
    for i in range(n):
        w, b = pol["weights"][i].astype(np.float32), pol["biases"][i].astype(np.float32)
        if i == 0:   # the observed previous action is target / scale: in the new units it is k times larger
            w = w.copy()
            w[:, PREV_ACTION] = w[:, PREV_ACTION] / k[None, :]
            if history:
                wide, col = np.zeros((w.shape[0], w.shape[1] * history), dtype=np.float32), 0
                for _, sl in OBS_TERMS:
                    size = sl.stop - sl.start
                    wide[:, col + (history - 1) * size: col + history * size] = w[:, sl]
                    col += history * size
                w = wide
        if i == n - 1:
            w, b = w * k[:, None], b * k
        sd[f"mlp.{2 * i}.weight"], sd[f"mlp.{2 * i}.bias"] = torch.tensor(w), torch.tensor(b)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state_dict": sd, "source": Path(policy_dir).name, "rescale": k.tolist()}, out)
    return out


def prev_action_columns(env, group: str) -> np.ndarray:
    """The columns of ``group``'s flattened observation holding the previous action, (frames, 12): each term's history
    is stored frame by frame (oldest first), so frame f's joint j is at the term's offset + 12 f + j."""
    om = env.observation_manager
    off = 0
    for name, dims in zip(om.active_terms[group], om.group_obs_term_dim[group]):
        size = int(np.prod(dims))
        if name == "prev_action":
            return off + np.arange(size).reshape(-1, 12)
        off += size
    raise KeyError(f"no prev_action in the {group!r} observation")


def rescale_checkpoint(src: Path, env, old_scale, new_scale, out: Path) -> Path:
    """A terrain checkpoint in a new action scale, acting exactly as before in radians: with k = old / new per joint,
    the actor's output rows and its std times k, the previous-action input columns (every history frame, in the actor
    and the critic) divided by k; Adam's moments transformed with their parameters (m / s, v / s^2 for p -> s p), so the
    optimizer state stays consistent. The changed columns are saved in the checkpoint under ``rescale``."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    k = torch.tensor(np.asarray(old_scale, dtype=np.float32) / np.asarray(new_scale, dtype=np.float32))
    scales = {"actor_state_dict": {}, "critic_state_dict": {}}
    cols = {g: prev_action_columns(env, g) for g in ("actor", "critic")}
    a, c = ck["actor_state_dict"], ck["critic_state_dict"]
    last = max((n for n in a if n.startswith("mlp.") and n.endswith(".weight")), key=lambda n: int(n.split(".")[1]))
    sa = {n: torch.ones_like(v) for n, v in a.items()}
    sa[last] = k[:, None].expand_as(a[last]).clone()
    sa[last.replace("weight", "bias")] = k.clone()
    sa["distribution.std_param"] = k.clone()
    for g, sd, sc in (("actor", a, sa), ("critic", c, None)):
        if sc is None:
            sc = {n: torch.ones_like(v) for n, v in sd.items()}
        for f in range(cols[g].shape[0]):
            sc["mlp.0.weight"][:, cols[g][f]] = 1.0 / k[None, :]
        scales["actor_state_dict" if g == "actor" else "critic_state_dict"] = sc
    for key, sc in scales.items():
        for n in ck[key]:
            ck[key][n] = ck[key][n] * sc[n]
    params = [scales["actor_state_dict"][n] for n in a] + [scales["critic_state_dict"][n] for n in c]
    st = ck["optimizer_state_dict"]["state"]
    for i, s in enumerate(params):
        if i in st:
            st[i]["exp_avg"] = st[i]["exp_avg"] / s.to(st[i]["exp_avg"].device)
            st[i]["exp_avg_sq"] = st[i]["exp_avg_sq"] / (s ** 2).to(st[i]["exp_avg_sq"].device)
    ck["rescale"] = {"source": str(src), "k": k.tolist(), "old_scale": list(old_scale), "new_scale": list(new_scale),
                     "prev_action_cols": {g: cols[g].tolist() for g in cols}}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    return out


def widen_critic(src: Path, env, out: Path, extra: int) -> Path:
    """A terrain checkpoint whose critic now also observes ``extra`` trailing inputs (the cross-track drift,
    ``track_errors``, appended last to the critic group): zero columns, so the value is unchanged until training uses
    them; the optimizer state is dropped (its first-layer moments no longer fit)."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    w = ck["critic_state_dict"]["mlp.0.weight"]
    want = int(np.prod(env.observation_manager.group_obs_dim["critic"]))
    assert w.shape[1] + extra == want, (w.shape, extra, want)
    ck["critic_state_dict"]["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], extra)], dim=1)
    ck.pop("optimizer_state_dict", None)
    ck["critic_widened"] = {"source": str(src), "extra_inputs": extra}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    return out


def runner_cfg(seed, iters, init_std, anchored: bool, logger="tensorboard", wandb_project="mp4-sim"):
    return RslRlOnPolicyRunnerCfg(
        seed=seed, num_steps_per_env=24, max_iterations=iters, save_interval=50,
        experiment_name="mp4_sim_terrain", logger=logger, wandb_project=wandb_project, clip_actions=1.0,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
        actor=RslRlModelCfg(hidden_dims=(128, 128, 64), activation="elu", obs_normalization=False,
                            distribution_cfg={"class_name": "GaussianDistribution", "init_std": init_std,
                                              "std_type": "scalar"}),
        critic=RslRlModelCfg(hidden_dims=(128, 128, 64), activation="elu", obs_normalization=False),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
            num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive", gamma=0.99,
            lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
            **({"class_name": "mp4_sim.gpu.algorithms:AnchorPPO"} if anchored else {})),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True,
                    help="scratch: from scratch; scratch_history: from scratch with a 5-step history; warm_start: from "
                         "the omni policy (the rough-ground and stair recipe); stock_servo: from scratch with the "
                         "stock servo model")
    ap.add_argument("--actuator",
                    help="actuator model: `bam_t5ub06` (BAM models fitted to the MD01 servos) or `stock`")
    ap.add_argument("--history", type=int)
    ap.add_argument("--init", help="an exported policy directory (policy.onnx, 45 inputs) to warm-start from and "
                                   "anchor to; relative to the repository root unless absolute")
    ap.add_argument("--resume", help="a terrain run's model_<k>.pt to fine-tune (overrides --init and --history)")
    ap.add_argument("--action-scale", type=float, nargs=3, metavar=("ABD", "HIP", "KNEE"),
                    help="rad per unit action (default 0.25 0.5 0.5); with --resume from another scale the checkpoint "
                         "is rescaled to act as before (rescale_checkpoint)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--init-std", type=float)
    ap.add_argument("--beta0", type=float, default=5.0)
    ap.add_argument("--beta-hold", type=int, default=50)
    ap.add_argument("--beta-decay", type=int, default=300)
    ap.add_argument("--critic-warmup", type=int,
                    help="value-only updates before the actor moves (default 50 with --init, else 0); the anchor's hold "
                         "starts after them")
    ap.add_argument("--lr0", type=float,
                    help="the adaptive learning rate's start (default 1e-4 with --init or --resume, else 1e-3)")
    ap.add_argument("--noise", default="n2", choices=("n1", "n2"))
    ap.add_argument("--masses", help="mass file in robot/assets/sku_2 (default: the terrain task's, 478 g)")
    ap.add_argument("--terrain", default="rough_basic", choices=TERRAINS)
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--terrain-seed", type=int, default=0)
    ap.add_argument("--heading", default="integral", choices=("integral", "loop"),
                    help="integral: the HeadingTrack penalty; loop: wz from a simulated heading loop, as on the robot")
    ap.add_argument("--alive", type=float, default=0.0, help="reward weight per second not terminated")
    ap.add_argument("--terminated", type=float, default=0.0,
                    help="reward weight on a fall (not a time-out); rewards are scaled by dt, so -250 is -5 per fall")
    ap.add_argument("--progress", type=float, default=0.0, help="weight of the linear progress term (per second)")
    ap.add_argument("--stumble", type=float, default=0.0, help="weight per second per foot pushed mostly sideways")
    ap.add_argument("--clearance", default="speed", choices=("speed", "apex"),
                    help="speed: |h - 30 mm| x foot speed; apex: the swing apex's deficit below 25 mm at touchdown, "
                         "against the terrain under and 3 cm ahead of the foot")
    ap.add_argument("--clearance-weight", type=float, default=-4.0, help="apex: per metre of deficit per touchdown")
    ap.add_argument("--clearance-target", type=float, default=0.025, help="apex: the swing apex wanted (m)")
    ap.add_argument("--tilt-limit", type=float, default=0.7,
                    help="rad: the tilt from vertical that ends an episode as a fall (a 27-degree flight pitches the "
                         "body by itself)")
    ap.add_argument("--stair-exit", action="store_true",
                    help="jittered-stair tiles end the episode once off the stairs (a success): promotion on that, "
                         "demotion on a fall or never leaving the platform (terrain_task.stair_exit)")
    ap.add_argument("--heading-law", choices=("p", "pi"), default="p",
                    help="the simulated heading loop: p = mjlab's clip(2 err, +-0.6); pi = the robot's (heading.pi_step: "
                         "PI with anti-windup, clipped at --heading-wz)")
    ap.add_argument("--heading-wz", type=float, default=0.4, help="pi: the loop's output limit (rad/s)")
    ap.add_argument("--cross-track", type=float, default=0.0,
                    help="weight per second of the clipped cross-track drift (|drift| / --cross-track-clip, at most 1; "
                         "the critic also observes the drift; a resumed critic is widened)")
    ap.add_argument("--cross-track-clip", type=float, default=0.10, help="m: the drift at which that term saturates")
    ap.add_argument("--yaw-catch", type=float, nargs=2, metavar=("LO", "HI"),
                    help="N m: yaw torque pulses on the base, |torque| U(LO, HI), random sign, 0.2-0.4 s every 3-6 s "
                         "(terrain_task.YawCatch: a foot catching a step edge)")
    ap.add_argument("--heading-reward", type=float, default=0.0,
                    help="weight per second of heading_align, exp(-(err / 0.25)^2), on the heading loop's envs")
    ap.add_argument("--yaw-catch-after", type=int, default=0,
                    help="iterations of this run before the yaw pulses start")
    ap.add_argument("--foot-softness", type=float, nargs=2, help="the feet's solref timeconst range drawn per env (s)")
    ap.add_argument("--friction", type=float, nargs=2, help="the feet's friction range (default 0.5 1.3)")
    ap.add_argument("--vx-max", type=float, help="the forward command's top (default 0.15 m/s)")
    ap.add_argument("--abduction", type=float, default=0.0,
                    help="weight per second of the abductions' squared distance from the stand (rad^2)")
    ap.add_argument("--leg-contact", type=float, default=-0.5,
                    help="weight per second per thigh or shin capsule touching the terrain")
    ap.add_argument("--narrow-risers", type=float, nargs=2,
                    help="--terrain stairs_narrow: the flights' nominal riser range, difficulty 0 to 1 (m; default 0.010 0.035)")
    ap.add_argument("--max-init-level", type=int, default=2,
                    help="envs start on terrain levels drawn from 0..this (levels are not saved in a checkpoint: a "
                         "resumed policy re-climbs from there)")
    ap.add_argument("--out", default=str(paths.RUNS / "terrain"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard",
                    help="wandb also writes the TensorBoard events (needs `wandb login` and WANDB_MODE not disabled)")
    ap.add_argument("--wandb-project", default="mp4-sim")
    ap.add_argument("--wandb-entity", help="the W&B user or team (default: the account's default entity)")
    ap.add_argument("--allow-unpatched", action="store_true",
                    help="train even if mujoco_warp is not patched (python -m mp4_sim.gpu.patch_warp)")
    a = ap.parse_args()
    if not a.allow_unpatched:
        from .patch_warp import is_patched
        if not is_patched():
            raise SystemExit("mujoco_warp is not patched: run python -m mp4_sim.gpu.patch_warp")
    if a.logger == "wandb" and os.environ.get("WANDB_MODE") in ("disabled", "offline"):
        raise SystemExit("--logger wandb with WANDB_MODE=%s: unset it to log online" % os.environ["WANDB_MODE"])
    if a.wandb_entity:
        os.environ["WANDB_USERNAME"] = a.wandb_entity           # rsl_rl reads the entity from it
    v = dict(VARIANTS[a.variant])
    for k in ("actuator", "history", "init"):
        if getattr(a, k) is not None:
            v[k] = getattr(a, k)
    if a.resume:
        src = json.loads((Path(a.resume).parent / "config.json").read_text())
        v.update(actuator=src["task"]["arm"], history=src["variant_settings"]["history"], init=None)
    init_std = a.init_std if a.init_std is not None else (0.1 if v["init"] else 0.5)
    warmup = a.critic_warmup if a.critic_warmup is not None else (50 if v["init"] else 0)
    lr0 = a.lr0 if a.lr0 is not None else (1.0e-4 if v["init"] or a.resume else 1.0e-3)

    opt = TerrainOptions(arm=v["actuator"], num_envs=a.envs, noisy=True, noise_ramp=True, randomise=True,
                         noise_end=noise_profile(a.noise), history=v["history"],
                         **({"masses": a.masses} if a.masses else {}),
                         terrain=a.terrain, rows=a.rows, terrain_seed=a.terrain_seed,
                         action_scale=tuple(a.action_scale) * 4 if a.action_scale else TERRAIN_ACTION_SCALE,
                         heading=a.heading, alive=a.alive,
                         terminated=a.terminated, progress=a.progress, stumble=a.stumble, clearance=a.clearance,
                         clearance_weight=a.clearance_weight, clearance_target=a.clearance_target,
                         stair_exit=a.stair_exit, heading_law=a.heading_law,
                         heading_wz=a.heading_wz, cross_track=a.cross_track, cross_track_clip=a.cross_track_clip,
                         yaw_catch=tuple(a.yaw_catch) if a.yaw_catch else None, tilt_limit=a.tilt_limit,
                         abduction=a.abduction, leg_contact=a.leg_contact, heading_reward=a.heading_reward,
                         max_init_level=a.max_init_level,
                         narrow_risers=tuple(a.narrow_risers) if a.narrow_risers else None,
                         foot_softness=tuple(a.foot_softness) if a.foot_softness else None,
                         **({"friction_range": tuple(a.friction)} if a.friction else {}),
                         **({"vx_max": a.vx_max} if a.vx_max is not None else {}))
    if a.yaw_catch:     # the env step counter continues from a resumed checkpoint (mjlab's runner restores it)
        ck = torch.load(a.resume, map_location="cpu", weights_only=False) if a.resume else {}
        s0 = ((ck.get("infos") or {}).get("env_state") or {}).get("common_step_counter", 24 * int(ck.get("iter", 0)))
        opt.yaw_catch_start = int(s0) + 24 * a.yaw_catch_after    # 24 env steps per iteration (runner_cfg)
    env_cfg = make_terrain_env_cfg(opt)
    env_cfg.seed = a.seed
    name = f"{datetime.now():%Y%m%d_%H%M%S}_{a.variant}_{v['actuator']}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    log_dir = Path(a.out) / name
    log_dir.mkdir(parents=True, exist_ok=True)

    anchored = v["init"] is not None
    cfg = asdict(runner_cfg(a.seed, a.iters, init_std, anchored, logger=a.logger, wandb_project=a.wandb_project))
    cfg["algorithm"]["learning_rate"] = lr0
    ckpt = None
    if anchored:
        init_dir = REPO / v["init"]
        ckpt = policy_to_checkpoint(init_dir, opt.action_scale, log_dir / "init_rescaled.pt", history=v["history"])
        # the anchor reads the plain 45-float observation group: its network keeps the original input layer
        anchor = ckpt if not v["history"] else policy_to_checkpoint(init_dir, opt.action_scale,
                                                                      log_dir / "anchor_rescaled.pt")
        cfg["algorithm"].update(anchor_path=str(anchor), anchor_sha256=sha256(anchor), anchor_beta0=a.beta0,
                                anchor_hold_iters=warmup + a.beta_hold, anchor_decay_iters=a.beta_decay,
                                anchor_beta_floor=0.0, anchor_obs_group="anchor", actor_warmup_iters=warmup)

    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    venv = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = MjlabOnPolicyRunner(venv, cfg, str(log_dir), "cuda:0")
    if anchored:
        sd = torch.load(ckpt, map_location="cuda:0", weights_only=False)["actor_state_dict"]
        missing, unexpected = runner.alg.actor.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        assert all("normalizer" in k or "std" in k for k in missing), missing
        with torch.no_grad():
            runner.alg.actor.distribution.std_param.fill_(init_std)
    resume = a.resume
    if resume:
        old = tuple(src["task"]["options"]["action_scale"])
        if not np.allclose(old, opt.action_scale):
            resume = str(rescale_checkpoint(Path(resume), env, old, opt.action_scale, log_dir / "resume_rescaled.pt"))
        widened = False
        if opt.cross_track and not src["task"]["options"].get("cross_track", 0.0):
            resume = str(widen_critic(Path(resume), env, log_dir / "resume_critic_widened.pt", extra=3))
            widened = True
        runner.load(resume, map_location="cuda:0",           # strict: the same network, history and observations
                    load_cfg={"actor": True, "critic": True, "optimizer": not widened, "iteration": True})

    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                                text=True).stdout.strip())
    (log_dir / "config.json").write_text(json.dumps({
        "task": describe(opt), "terrain_task": True, "variant": a.variant, "variant_settings": v, "runner": cfg,
        "algorithm_class": f"{type(runner.alg).__module__}.{type(runner.alg).__name__}",
        "masses_sha256": sha256(masses_path(opt.masses)),
        "init": v["init"], "init_rescaled_sha256": sha256(ckpt) if ckpt else None, "init_std": init_std,
        "resume": a.resume, "resume_sha256": sha256(Path(a.resume)) if a.resume else None,
        "resume_rescaled": resume if resume != a.resume else None,
        "critic_warmup": warmup, "lr0": lr0,
        "noise": a.noise, "repo_commit": commit, "repo_dirty": dirty, "argv": sys.argv}, indent=1, default=str))
    runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=True)
    print("RUN_DIR", log_dir)


if __name__ == "__main__":
    main()
