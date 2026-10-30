"""PPO (rsl_rl) on the SKU2 velocity task in Isaac Lab, Newton MuJoCo Warp physics.

    python -m mp4_isaac.train --envs 1024 --iters 1500 --seed 1
    python -m mp4_isaac.train --init flat_omni --iters 600 --tag warm     # start from a shipped policy
    python -m mp4_isaac.train --resume runs/<run>/model_1499.pt --iters 500

Each run writes ``runs/<date>_<time>_<terrain>_s<seed>[_tag]/``: ``config.json`` (every option, the source hashes,
the command line), a checkpoint ``model_<k>.pt`` every 50 iterations and at the end, and TensorBoard events. Export
a checkpoint with ``python -m mp4_isaac.export``; ``python -m mp4_isaac.verify`` checks the ONNX file.

Network 128-128-64 ELU for actor and critic; PPO lr 1e-3 adaptive (KL 0.01), gamma 0.99, lambda 0.95,
clip 0.2, entropy 0.005, 5 epochs x 4 minibatches, 24 steps per env per iteration. ``--init`` loads a shipped
45-input policy into the actor, rescaled per joint to this task's action scale (and widened to the history), starts
the exploration std at 0.1 and the learning rate at 1e-4, and trains only the critic for ``--critic-warmup``
iterations.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--envs", type=int, default=1024)
parser.add_argument("--iters", type=int, default=1500)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--terrain", default="flat", choices=("flat", "rough"))
parser.add_argument("--history", type=int, default=0, help="actor observation frames (0: the 45-value contract)")
parser.add_argument("--action-scale", type=float, nargs=3, default=(0.25, 0.50, 0.50), metavar=("ABD", "HIP", "KNEE"))
parser.add_argument("--vx-max", type=float, default=0.15)
parser.add_argument("--vy-max", type=float, default=0.0)
parser.add_argument("--wz-max", type=float, default=0.6)
parser.add_argument("--heading", default="none", choices=("none", "loop"),
                    help="loop: most envs get their yaw rate from a heading P loop (exported as the robot's heading loop)")
parser.add_argument("--progress", type=float, default=0.0, help="weight per second of forward progress")
parser.add_argument("--abduction", type=float, default=0.0, help="weight of the abductions' distance from the stand")
parser.add_argument("--tilt-limit", type=float, default=0.7)
parser.add_argument("--spawn-z", type=float, default=0.10)
parser.add_argument("--no-heatsink", action="store_true", help="450 g instead of the 478 g heatsink budget")
parser.add_argument("--noise", default="n2", choices=("n1", "n2"))
parser.add_argument("--init", help="a shipped policy name or directory (policy.onnx, 45 inputs) to start the actor from")
parser.add_argument("--resume", help="a run's model_<k>.pt to continue (network, optimizer, iteration)")
parser.add_argument("--init-std", type=float)
parser.add_argument("--lr0", type=float)
parser.add_argument("--critic-warmup", type=int)
parser.add_argument("--out", default=None, help="parent of the run directory (default: runs/ in this repo)")
parser.add_argument("--tag", default="")
parser.add_argument("--logger", default="tensorboard", choices=("tensorboard", "wandb"))
add_launcher_args(parser)
parser.set_defaults(visualizer=["none"])
args_cli = parser.parse_args()

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
    check_rsl_rl_version,
    create_rsl_rl_runner,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab.utils.dict import class_to_dict

from .contract import OBS_TERMS
from .onnx_policy import load_policy
from .paths import ASSETS, REPO, contract_json, fit_json, interface_usd, servo_map_json, sha256
from .task import TaskOptions, make_env_cfg

PREV_ACTION = slice(33, 45)


def policy_dir(name: str) -> Path:
    path = Path(name)
    if path.is_dir() or path.suffix == ".onnx":
        return path
    return ASSETS / "policies" / name


def policy_to_actor(src: Path, new_scale, history: int) -> dict:
    """A shipped 45-input policy as an rsl_rl actor state dict in this task's action scale.

    With ``k = old / new`` per joint the output rows are multiplied by ``k`` and the previous-action inputs divided
    by ``k``, so the same robot state gives the same target in radians. With a history the first layer is widened
    to the term-major stack: the original weights on each term's newest frame, zeros on the older frames.
    """
    pol = load_policy(src)
    if pol["history"] != 1:
        raise ValueError(f"{src}: a warm start needs a policy without history (it has {pol['history']})")
    k = np.asarray(pol["scale"], dtype=np.float32) / np.asarray(new_scale, dtype=np.float32)
    n = len(pol["weights"])
    sd = {}
    for i in range(n):
        w, b = pol["weights"][i].astype(np.float32).copy(), pol["biases"][i].astype(np.float32).copy()
        if i == 0:
            w[:, PREV_ACTION] /= k[None, :]
            if history > 1:
                wide, col = np.zeros((w.shape[0], w.shape[1] * history), dtype=np.float32), 0
                for _, sl in OBS_TERMS:
                    size = sl.stop - sl.start
                    wide[:, col + (history - 1) * size: col + history * size] = w[:, sl]
                    col += history * size
                w = wide
        if i == n - 1:
            w, b = w * k[:, None], b * k
        sd[f"mlp.{2 * i}.weight"], sd[f"mlp.{2 * i}.bias"] = torch.tensor(w), torch.tensor(b)
    return sd


def runner_cfg(seed: int, iters: int, init_std: float, lr0: float, device: str, logger: str) -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        seed=seed,
        device=device,
        num_steps_per_env=24,
        max_iterations=iters,
        save_interval=50,
        experiment_name="mp4_isaac",
        empirical_normalization=False,
        obs_groups={"actor": ["actor"], "critic": ["critic"]},
        clip_actions=1.0,
        logger=logger,
        wandb_project="mp4-isaac",
        actor=RslRlMLPModelCfg(
            hidden_dims=[128, 128, 64], activation="elu", obs_normalization=False,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=init_std, std_type="scalar"),
        ),
        critic=RslRlMLPModelCfg(hidden_dims=[128, 128, 64], activation="elu", obs_normalization=False),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
            num_learning_epochs=5, num_mini_batches=4, learning_rate=lr0, schedule="adaptive", gamma=0.99,
            lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
        ),
    )


def critic_warmup(alg, iters: int) -> None:
    """The first ``iters`` updates train only the critic: the actor's parameters take no gradient, and the adaptive
    learning rate is held (an unmoved actor's KL is ~0, which would raise it)."""
    if iters <= 0:
        return
    actor = getattr(alg, "_raw_actor", alg.actor)
    update = alg.update
    count = {"n": 0}

    def wrapped(*a, **kw):
        frozen = count["n"] < iters
        kl = alg.desired_kl
        for p in actor.parameters():
            p.requires_grad_(not frozen)
        if frozen:
            alg.desired_kl = None
        count["n"] += 1
        try:
            return update(*a, **kw)
        finally:
            for p in actor.parameters():
                p.requires_grad_(True)
            alg.desired_kl = kl

    alg.update = wrapped


def git_state() -> tuple[str, bool]:
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True,
                                text=True).stdout.strip())
    return commit, dirty


def sources(opt: TaskOptions) -> dict:
    usd = interface_usd("sku_2")
    masses = ASSETS / "sku_2" / ("sku2_masses_478g.json" if opt.heatsink else "sku2_masses.json")
    files = {"contract": contract_json(), "servo_map": servo_map_json(), "masses": masses,
             "fit_abd_knee": fit_json("abd"), "fit_hip": fit_json("hip")}
    out = {k: {"file": str(Path(p).relative_to(REPO)).replace("\\", "/"), "sha256": sha256(p)}
           for k, p in files.items()}
    out["usd"] = str(usd.relative_to(REPO)).replace("\\", "/") if usd else None
    return out


def main() -> None:
    a = args_cli
    history = a.history
    if a.resume:
        prior = json.loads((Path(a.resume).parent / "config.json").read_text())
        history = int(prior["variant_settings"]["history"])
    init = policy_dir(a.init) if a.init and not a.resume else None
    init_std = a.init_std if a.init_std is not None else (0.1 if init else 0.5)
    lr0 = a.lr0 if a.lr0 is not None else (1.0e-4 if init else 1.0e-3)
    warmup = a.critic_warmup if a.critic_warmup is not None else (50 if init else 0)

    opt = TaskOptions(
        num_envs=a.envs, terrain=a.terrain, history=history, action_scale=tuple(a.action_scale) * 4,
        vx_max=a.vx_max, vy_max=a.vy_max, wz_max=a.wz_max, heading=a.heading, progress=a.progress,
        abduction=a.abduction, tilt_limit=a.tilt_limit, spawn_z=a.spawn_z, heatsink=not a.no_heatsink,
        noise=a.noise,
    )
    env_cfg = make_env_cfg(opt)
    env_cfg.seed = a.seed
    device = getattr(a, "device", None) or "cuda:0"
    env_cfg.sim.device = device

    out = Path(a.out) if a.out else REPO / "runs"
    name = f"{datetime.now():%Y%m%d_%H%M%S}_{a.terrain}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
    log_dir = out / name
    log_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(log_dir)

    with launch_simulation(env_cfg, a):
        env = ManagerBasedRLEnv(cfg=env_cfg)
        venv = RslRlVecEnvWrapper(env, clip_actions=1.0)
        agent = runner_cfg(a.seed, a.iters, init_std, lr0, device, a.logger)
        agent = handle_deprecated_rsl_rl_cfg(agent, check_rsl_rl_version())
        runner = create_rsl_rl_runner(venv, agent, log_dir=str(log_dir))
        actor = getattr(runner.alg, "_raw_actor", runner.alg.actor)
        if init is not None:
            sd = policy_to_actor(init, opt.action_scale, history)
            missing, unexpected = actor.load_state_dict(sd, strict=False)
            if unexpected or any("mlp" in k for k in missing):
                raise RuntimeError(f"warm start does not fit the actor: missing {missing}, unexpected {unexpected}")
            with torch.no_grad():
                actor.distribution.std_param.fill_(init_std)
            print(f"[mp4_isaac] actor from {init} (rescaled to {list(a.action_scale)})")
        if a.resume:
            runner.load(a.resume, map_location=device)
        critic_warmup(runner.alg, warmup)

        commit, dirty = git_state()
        (log_dir / "config.json").write_text(json.dumps({
            "task": {"name": f"isaac_{a.terrain}", "simulator": "isaaclab_newton_mjwarp", "arm": "bam_t5ub06",
                     "options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(opt).items()},
                     "decimation": env_cfg.decimation, "physics_dt": env_cfg.sim.dt},
            "variant_settings": {"history": history},
            "runner": class_to_dict(agent),
            "init": str(init) if init else None,
            "init_sha256": sha256(init / "policy.onnx" if init and init.is_dir() else init) if init else None,
            "init_std": init_std, "lr0": lr0, "critic_warmup": warmup,
            "resume": a.resume, "resume_sha256": sha256(a.resume) if a.resume else None,
            "sources": sources(opt), "repo_commit": commit, "repo_dirty": dirty, "argv": sys.argv,
        }, indent=1, default=str))
        print(f"RUN_DIR {log_dir}")
        with contextlib.suppress(KeyboardInterrupt):
            runner.learn(num_learning_iterations=a.iters, init_at_random_ep_len=True)
        env.close()


if __name__ == "__main__":
    main()
