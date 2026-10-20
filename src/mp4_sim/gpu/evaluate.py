"""Score a policy on terrain in simulation: success, distance, falls, tilt and servo current per (terrain, level) cell,
on the same seeds for every policy, so that two policies compare trial by trial.

    python -m mp4_sim.gpu.evaluate --policy policies/<name> --terrain rough --heading-loop --json runs/eval/<name>.json
    python -m mp4_sim.gpu.evaluate --policy policies/<name> --terrain stairs_built --risers 0.025 --tread 0.05 \
        --steps 4 --width 0.20 --landing 0.25 --heading-loop --tilt-limit 1.0
    python -m mp4_sim.gpu.evaluate --policy policies/<name> --terrain stairs_built --envs 16 --video runs/video/x.mp4
    python -m mp4_sim.gpu.evaluate --throughput 2048 --actuator bam_t5ub06          # env-steps/s, zero actions

``--policy`` is an exported policy directory (``policy.onnx``; its manifest gives the action scale, the history and
whether it observes the whole command); it runs as a torch forward pass of the ONNX weights.

Each env starts at its cell's spawn point facing +x, commanded ``--vx`` forward (wz = 0, or the heading loop's output
for a policy that observes a yaw-rate command), for ``--seconds``, under the measured observation noise on randomised
robots (friction, COM, mass, policy-to-servo delay, pushes off). Walking out from a pyramid's centre, "slope_up" and "stairs_up"
(platform on top) are descents; "slope_down" and "stairs_down" are climbs.

Per cell: success = no fall and at least ``--goal`` m of forward displacement; distance (median); falls; the p99 of the
tilt low-passed at 0.1 s (the robot cuts servo power above 35° of tilt); for BAM actuator models the motor current (torque / kt) as the
share of samples above 500 mA and the episodes that stay above ``--guard-ma`` (1000) for 0.5 s (a servo current
guard of that form). Servo currents are model estimates: compare them between policies.

The printed line per cell: success; "on the strip", success without drifting more than ``--side`` sideways; "off the
side", the trials that did drift that far; the median end lateral offset; fall, stall (neither a fall nor a success)
and the median distance; the tilt p99 and the share of trials past the 35° cut; the p99 foot penetration; for BAM
models the share of samples over 500 mA and "trip<guard-ma>", the trials that would trip the current guard; on built
stairs "WIDTH", success with every stance foot centre within half the width (``--width-rule``), then the risers
climbed, "legs", the trials with a thigh or shin on the terrain, and "sat h/k", the share of steps the hip and knee
actions are clipped; "psi max", the median and p90 of each trial's peak heading error; "sat", the mean share of steps
with the heading loop's output at its limit.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from .. import paths
from ..export.onnx import load_policy
from ..heading import ANTIWINDUP, WZ_MAX, lateral_step, pi_step
from . import terrain as terrain_mod
from .env_cfg import noise_profile
from .terrain_task import (
    FOOT_R,
    STAIR_BUILT,
    TerrainOptions,
    _foot_site_ids,
    foot_heights_centre,
    make_terrain_env_cfg,
    sub_terrains,
)

DT = 0.02
JOINT_ORDER = json.loads(paths.CONTRACT_JSON.read_text())["joint_order"]
TILT_CUT_DEG = 35.0
STAIR_FOOT_PAST = 0.01          # m: a foot centre this far past a riser counts as over it


class TorchPolicy:
    """An exported ``policy.onnx`` (ELU MLP) on the GPU, batched: the ONNX weights in torch."""

    def __init__(self, path: Path, device: str):
        p = load_policy(path)
        self.layers = [(torch.tensor(w, device=device), torch.tensor(b, device=device))
                       for w, b in zip(p["weights"], p["biases"])]
        self.manifest = p["manifest"]
        self.history = p["history"]
        self.scale = tuple(float(x) for x in p["scale"])

    @torch.no_grad()
    def __call__(self, x):
        for i, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if i < len(self.layers) - 1:
                x = torch.nn.functional.elu(x)
        return x


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


WZ_TRAIN = 0.6                  # rad/s: the yaw-rate range of training's P-only heading law (--heading-law p)


def parse_kick(text: str) -> tuple[float, float, float]:
    t0, dur, rate = (float(v) for v in text.split(":"))
    return t0, dur, rate


def parse_yaw_kick(text: str) -> tuple[float, float, float, float]:
    t0, t1, torque, dur = (float(v) for v in text.split(":"))
    return t0, t1, torque, dur


def yaw_quat(a):
    """(w, x, y, z) of a rotation by ``a`` (rad) about the world z axis."""
    return torch.stack([torch.cos(a / 2), torch.zeros_like(a), torch.zeros_like(a), torch.sin(a / 2)], -1)


def quat_mul(p, q):
    pw, px, py, pz = p.unbind(-1)
    qw, qx, qy, qz = q.unbind(-1)
    return torch.stack([pw * qw - px * qx - py * qy - pz * qz, pw * qx + px * qw + py * qz - pz * qy,
                        pw * qy - px * qz + py * qw + pz * qx, pw * qz + px * qy - py * qx + pz * qw], -1)


def alive_now(fell, done, reached):
    return ~(fell | done | reached)


def yaw_of(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def assign_cells(env, cells):
    """Put env i on cells[i % len(cells)] = (row, col): its terrain level, type and spawn origin."""
    t = env.scene.terrain
    n = env.num_envs
    rows = torch.tensor([cells[i % len(cells)][0] for i in range(n)], device=env.device)
    cols = torch.tensor([cells[i % len(cells)][1] for i in range(n)], device=env.device)
    t.terrain_levels[:] = rows
    t.terrain_types[:] = cols
    t.env_origins[:] = t.terrain_origins[rows, cols]
    return rows.cpu().numpy(), cols.cpu().numpy()


def bam_currents(env):
    """Motor current per joint (A, |torque| / kt) from each BAM actuator group, in the contract's joint order (a knee's
    tendon goes to its leg's calf column), or None for other actuator models."""
    robot = env.scene["robot"]
    out = torch.zeros(env.num_envs, 12, device=env.device)
    found = False
    for act in getattr(robot, "_actuators", []):
        last = getattr(act, "_last", None)
        bam = getattr(act, "_bam_model", None)
        if last is None or bam is None:
            continue
        found = True
        kt = float(bam.kt.value)
        cols = [JOINT_ORDER.index(n.replace("_knee_drive", "_calf")) for n in act.target_names]
        out[:, cols] = last["torque"].abs() / kt
    return out if found else None


def build(opt: TerrainOptions, render: bool = False, seed: int | None = None, camera: tuple | None = None):
    """The env, seeded before it is built: its startup events (friction, COM, mass, actuator draws) run inside the
    constructor, so a seed set afterwards changes only the resets and the noise, not the randomised robots."""
    cfg = make_terrain_env_cfg(opt)
    cfg.seed = seed
    if render:
        cfg.viewer.width, cfg.viewer.height = 960, 540
        cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = camera or (0.55, -18.0, 135.0)
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if render else None)


@torch.no_grad()
def evaluate(args) -> dict:
    pol = TorchPolicy(Path(args.policy), "cuda:0") if args.policy else None
    omni = bool(pol and pol.manifest.get("omni"))
    arm = args.arm
    if arm is None and pol is not None:
        arm = (pol.manifest.get("task") or {}).get("actuator")
        if arm is None:
            from ..actuators.arms import trained_arm
            arm = trained_arm(args.policy)
        if arm is None:
            print(f"{args.policy}: no actuator in its manifest or config.json; using stock (set --actuator)")
    scale = pol.scale if pol else TerrainOptions().action_scale
    stair_kw = {}
    if args.terrain == "stairs_built":
        stair_kw = {"stair_risers": tuple(args.risers), "stair_tread": args.tread, "stair_steps": args.steps,
                    "stair_width": args.width, "stair_landing": args.landing}
    opt = TerrainOptions(arm=arm or "stock", num_envs=args.envs, terrain=args.terrain, rows=args.rows,
                         terrain_seed=args.terrain_seed, noisy=not args.clean, noise_ramp=False,
                         noise_end=noise_profile(args.noise), randomise=not args.clean, push_mps=0.0,
                         fixed_command=(args.vx, 0.0, 0.0), level=0, action_scale=scale,
                         episode_s=args.seconds + 5.0, visuals=bool(args.video),
                         **({"masses": args.masses} if args.masses else {}), self_collision=not args.no_self_collision,
                         history=pol.history if pol and pol.history > 1 else 0,
                         **({"foot_solref": tuple(args.foot_solref)} if args.foot_solref else {}),
                         **stair_kw, tilt_limit=args.tilt_limit)
    names = list(sub_terrains(opt))
    stairs = args.terrain == "stairs_built"
    rows = [0] if stairs else (args.levels or list(range(args.rows)))
    cols = [names.index(c) for c in args.only] if args.only else list(range(len(names)))
    cells = [(r, c) for c in cols for r in rows]
    env = build(opt, render=bool(args.video), seed=args.seed, camera=tuple(args.camera) if args.camera else None)
    torch.manual_seed(args.seed)
    obs, _ = env.reset()
    rr, cc = assign_cells(env, cells)
    obs, _ = env.reset()
    n, steps = env.num_envs, int(round(args.seconds / DT))
    robot = env.scene["robot"]
    origin = env.scene.env_origins.clone()
    yaw0 = yaw_of(robot.data.root_link_quat_w)
    gen = torch.Generator(device=env.device).manual_seed(int(args.seed) + 7919)   # start noise and disturbances
    if args.start_noise:   # the start jig's tolerance: a lateral offset and a yaw error per trial, uniform
        dy = (2 * torch.rand(n, device=env.device, generator=gen) - 1) * args.start_noise[0]
        dpsi = (2 * torch.rand(n, device=env.device, generator=gen) - 1) * math.radians(args.start_noise[1])
        pose = torch.cat([robot.data.root_link_pos_w, robot.data.root_link_quat_w], -1).clone()
        pose[:, 1] += dy
        pose[:, 3:7] = quat_mul(yaw_quat(dpsi), pose[:, 3:7])
        robot.write_root_link_pose_to_sim(pose)
        yaw0 = wrap(yaw0 + dpsi)   # the heading loop holds the start heading, aim error included (as on the robot)
    integ = torch.zeros(n, device=env.device)
    y_hat = torch.zeros(n, device=env.device)                     # the lateral loop's dead-reckoned offset (m)
    train_law = args.heading_law == "train"
    wz_lim = WZ_TRAIN if train_law else args.heading_wz_max
    psi_max = torch.zeros(n, device=env.device)                   # the largest |heading - start heading| (rad)
    psi_hist = torch.zeros(25, n, device=env.device)              # the last 0.5 s of heading, for its fastest change
    psi_d05 = torch.zeros(n, device=env.device)                   # the largest heading change over 0.5 s (rad)
    wz_sat = torch.zeros(n, device=env.device)                    # live steps with the loop's output at its limit
    live_steps = torch.zeros(n, device=env.device)
    ykick = parse_yaw_kick(args.yaw_kick) if args.yaw_kick else None
    if ykick is not None:      # a catch: a yaw torque on the body for a while, at a random time per trial, random sign
        kick_k = torch.round((ykick[0] + (ykick[1] - ykick[0]) * torch.rand(n, device=env.device, generator=gen)) / DT).long()
        kick_s = torch.where(torch.rand(n, device=env.device, generator=gen) < 0.5, -1.0, 1.0)
        kick_n = max(1, int(round(ykick[3] / DT)))
        base_id = [robot.body_names.index("base")]
        zero3 = torch.zeros(n, 1, 3, device=env.device)
    if stairs:  # each env's risers (x from its spawn), from the generated layouts; padded with +inf
        lay = [terrain_mod.STAIR_LAYOUTS[(names[int(c)], int(r))] for r, c in zip(rr, cc)]
        n_ris = max(len(L["risers_x"]) for L in lay)
        risers_x = torch.full((n, n_ris), float("inf"), device=env.device)
        for i, L in enumerate(lay):
            risers_x[i, :len(L["risers_x"])] = torch.tensor(L["risers_x"], device=env.device)
        n_steps = torch.tensor([len(L["risers_x"]) for L in lay], device=env.device)
        landing_x = torch.tensor([L["risers_x"][-1] for L in lay], device=env.device)
        feet = _foot_site_ids(env)
        x_first = risers_x[:, 0]
        foot_env = torch.zeros(n, device=env.device)        # the largest |y| of any foot centre over the stairs
        foot_env_st = torch.zeros(n, device=env.device)     # ... of a stance foot on a tread (centre within its radius + 2 mm of it)
        climbed = torch.zeros(n, dtype=torch.long, device=env.device)   # risers all four feet have crossed (max)
        leg_steps = torch.zeros(n, device=env.device)                   # policy steps with a thigh/shin on the terrain
        alive_steps = torch.zeros(n, device=env.device)
        sat = torch.zeros(n, 3, device=env.device)                      # steps with |a| >= 0.98: abduction, hip, knee
        cur_max = []
    kick = parse_kick(args.heading_kick) if args.heading_kick else None
    kick_peak = torch.zeros(n, device=env.device)                    # the largest error the kick caused (rad)
    overshoot = torch.zeros(n, device=env.device)                    # the largest error past the line after it (rad)
    cmd = env.command_manager.get_term("twist")
    fell = torch.zeros(n, dtype=torch.bool, device=env.device)
    cause = torch.zeros(n, dtype=torch.long, device=env.device)   # a fall's first termination: 1 tilt, 2 body contact, 3 both
    done = torch.zeros(n, dtype=torch.bool, device=env.device)   # ended without falling (left the grid)
    reached = torch.zeros(n, dtype=torch.bool, device=env.device)  # --stop-at-goal: past the goal, the trial is over
    dist = torch.zeros(n, device=env.device)
    lat = torch.zeros(n, device=env.device)
    lat_max = torch.zeros(n, device=env.device)
    t_goal = torch.full((n,), float("nan"), device=env.device)    # time the base first passes the goal (s)
    tilt_lp = torch.zeros(n, device=env.device)
    tilts, cur_over5, pens = [], [], []
    over_guard_s = torch.zeros(n, 12, device=env.device)
    guard_trip = torch.zeros(n, dtype=torch.bool, device=env.device)
    frames = []
    alpha = DT / 0.1
    t0 = time.time()
    for k in range(steps):
        live0 = alive_now(fell, done, reached)
        psi = wrap(yaw_of(robot.data.root_link_quat_w) - yaw0)
        psi_max = torch.where(live0, torch.maximum(psi_max, psi.abs()), psi_max)
        if k >= 25:
            psi_d05 = torch.where(live0, torch.maximum(psi_d05, wrap(psi - psi_hist[k % 25]).abs()), psi_d05)
        psi_hist[k % 25] = psi
        if ykick is not None:      # the torque is held from kick_k for kick_n steps (written every step: resets clear it)
            on = live0 & (k >= kick_k) & (k < kick_k + kick_n)
            tq = zero3.clone()
            tq[:, 0, 2] = torch.where(on, kick_s * ykick[2], torch.zeros_like(kick_s))
            robot.write_external_wrench_to_sim(zero3, tq, body_ids=base_id)
        if omni and args.heading_loop:  # the robot's outer loop (heading.pi_step), on the true yaw, into the wz command
            target = torch.zeros(n, device=env.device)
            if args.lateral_loop:
                target, y_hat = lateral_step(y_hat, psi, args.vx, DT, args.lateral_loop)
            err = wrap(target - psi)
            if train_law:            # training's law (mjlab's heading command): P only, clipped at training's range
                wz = (2.0 * err).clamp(-WZ_TRAIN, WZ_TRAIN)
            else:
                wz, integ = pi_step(err, integ, DT, wz_max=args.heading_wz_max, antiwindup=args.heading_aw)
            wz_sat += (live0 & (wz.abs() >= wz_lim - 1e-4)).float()
            live_steps += live0.float()
            if kick is not None:     # an edge strike: a yaw rate of the robot's own, added within training's range
                t = k * DT
                if kick[0] <= t < kick[0] + kick[1]:
                    wz = (wz + kick[2]).clamp(-WZ_TRAIN, WZ_TRAIN)
                live = alive_now(fell, done, reached)
                s = -float(np.sign(kick[2]))                                 # the correction's direction
                if t >= kick[0]:
                    kick_peak = torch.where(live, torch.maximum(kick_peak, s * err), kick_peak)
                if t >= kick[0] + kick[1]:
                    overshoot = torch.where(live, torch.maximum(overshoot, -s * err), overshoot)
            cmd.vel_command_b[:, 2] = wz
        a = pol(obs["actor"]) if pol else torch.zeros(n, 12, device=env.device)
        obs, _, terminated, truncated, _ = env.step(a)
        alive = ~(fell | done | reached)
        ended = terminated | truncated
        g = robot.data.projected_gravity_b
        tilt = torch.rad2deg(torch.acos((-g[:, 2]).clamp(-1, 1)))
        tilt_lp = tilt_lp + alpha * (tilt - tilt_lp)
        tilts.append(torch.where(alive & ~ended, tilt_lp, torch.full_like(tilt_lp, float("nan"))))
        pos = robot.data.root_link_pos_w - origin
        # any episode end (a fall, or leaving the terrain grid: a truncation) auto-resets the env: freeze there
        dist = torch.where(alive & ~ended, pos[:, 0], dist)
        lat = torch.where(alive & ~ended, pos[:, 1], lat)
        lat_max = torch.where(alive & ~ended, torch.maximum(lat_max, pos[:, 1].abs()), lat_max)
        if not stairs:
            t_goal = torch.where(alive & ~ended & torch.isnan(t_goal) & (pos[:, 0] >= args.goal),
                                 torch.full_like(t_goal, (k + 1) * DT), t_goal)
        if stairs:
            live = alive & ~ended
            fx = (robot.data.site_pos_w[:, feet, 0] - origin[:, :1]).min(-1).values   # the rearmost foot
            now = (fx.unsqueeze(-1) > risers_x + STAIR_FOOT_PAST).sum(-1)
            climbed = torch.where(live, torch.maximum(climbed, now), climbed)
            fp = robot.data.site_pos_w[:, feet, :] - origin.unsqueeze(1)
            over = fp[:, :, 0] > x_first.unsqueeze(-1) - 0.01
            fy = torch.where(over, fp[:, :, 1].abs(), torch.zeros_like(fp[:, :, 1])).max(-1).values
            foot_env = torch.where(live, torch.maximum(foot_env, fy), foot_env)
            stance = (fp[:, :, 0] > x_first.unsqueeze(-1) + 0.01) & (foot_heights_centre(env) <= FOOT_R + 0.002)
            fs = torch.where(stance, fp[:, :, 1].abs(), torch.zeros_like(fp[:, :, 1])).max(-1).values
            foot_env_st = torch.where(live, torch.maximum(foot_env_st, fs), foot_env_st)
            lc = (env.scene["leg_contact"].data.found > 0).flatten(1).any(-1)
            leg_steps += (live & lc).float()
            alive_steps += live.float()
            sat += live.float().unsqueeze(-1) * (a.abs() >= 0.98).float().view(n, 4, 3).any(1).float()
            on_landing = live & (fx > landing_x + STAIR_FOOT_PAST)
            t_goal = torch.where(on_landing & torch.isnan(t_goal), torch.full_like(t_goal, (k + 1) * DT), t_goal)
        pen = (FOOT_R - foot_heights_centre(env)).clamp(min=0.0)         # how far each foot is pressed in
        pens.append(torch.where((alive & ~ended).unsqueeze(-1), pen, torch.full_like(pen, float("nan"))))
        cur = bam_currents(env)
        if cur is not None:
            cur_over5.append(torch.where((alive & ~ended).unsqueeze(-1), (cur > 0.5).float(), torch.full_like(cur, float("nan"))))
            over_guard_s = torch.where(cur > args.guard_ma / 1000.0, over_guard_s + DT, torch.zeros_like(over_guard_s))
            guard_trip |= alive & (over_guard_s > 0.5).any(-1)
            if stairs:
                cur_max.append(torch.where(alive & ~ended, cur.max(-1).values, torch.full_like(cur[:, 0], float("nan"))))
        new_fall = alive & terminated & ~truncated
        tm = env.termination_manager
        c_now = tm.get_term("tilt").long() + 2 * tm.get_term("body_contact").long()
        cause = torch.where(new_fall, c_now, cause)
        fell |= new_fall
        done |= alive & truncated
        if args.stop_at_goal:
            reached |= alive & ~ended & ((~torch.isnan(t_goal)) if stairs else (pos[:, 0] >= args.goal))
        if args.video:
            frames.append(env.render())
    wall = time.time() - t0
    tilt_all = torch.stack(tilts).cpu().numpy()
    fell, dist, guard_trip = fell.cpu().numpy(), dist.cpu().numpy(), guard_trip.cpu().numpy()
    lat, lat_max, t_goal = lat.cpu().numpy(), lat_max.cpu().numpy(), t_goal.cpu().numpy()
    kick_peak, overshoot = np.degrees(kick_peak.cpu().numpy()), np.degrees(overshoot.cpu().numpy())
    psi_max_deg, psi_d05_deg = np.degrees(psi_max.cpu().numpy()), np.degrees(psi_d05.cpu().numpy())
    wz_sat_share = (wz_sat / live_steps.clamp(min=1)).cpu().numpy()
    c5 = torch.stack(cur_over5).cpu().numpy() if cur_over5 else None
    pen_all = torch.stack(pens).cpu().numpy()                             # (t, env, 4)
    if stairs:  # success = all four feet on the landing (past the last riser) without a fall
        landed = ~np.isnan(t_goal)
        ok = ~fell & landed
        ok_strip = ok & (lat_max <= args.side)
        climbed_n, steps_n = climbed.cpu().numpy(), n_steps.cpu().numpy()
        leg_s = (leg_steps * DT).cpu().numpy()
        leg_any = (leg_steps > 0).cpu().numpy()
        sat_share = (sat / alive_steps.clamp(min=1).unsqueeze(-1)).cpu().numpy()
        cmax = torch.stack(cur_max).cpu().numpy() if cur_max else None
        fenv, fenv_st = foot_env.cpu().numpy(), foot_env_st.cpu().numpy()
        w_rule = args.width_rule if args.width_rule is not None else args.width
        ok_width = ok & (fenv_st <= w_rule / 2) if w_rule else None     # every stance foot centre on the treads
        ok_width_all = ok & (fenv <= w_rule / 2) if w_rule else None    # ... every foot, swing included
    else:
        ok = ~fell & (dist >= args.goal)
        ok_strip = ok & (lat_max <= args.side)
    table = []
    for (r, c) in cells:
        m = (rr == r) & (cc == c)
        if not m.any():                                                    # fewer environments than cells
            continue
        row = {"terrain": names[c], "level": int(r), "n": int(m.sum()),
               "success": float(np.mean(ok[m])),
               "fall": float(np.mean(fell[m])), "dist_med_m": float(np.median(dist[m])),
               "stall": float(np.mean(~fell[m] & ~ok[m])),                    # neither a fall nor a success
               "lat_end_med_mm": float(1000 * np.median(lat[m])), "off_side": float(np.mean(lat_max[m] > args.side)),
               "success_on_strip": float(np.mean(ok_strip[m])),
               "tilt_p99_deg": float(np.nanpercentile(tilt_all[:, m], 99)),
               "foot_pen_p99_mm": float(1000 * np.nanpercentile(pen_all[:, m], 99)),
               "tilt_over_cut": float(np.mean(np.nanmax(tilt_all[:, m], axis=0) > TILT_CUT_DEG)),
               "psi_max_med_deg": float(np.median(psi_max_deg[m])), "psi_max_p90_deg": float(np.percentile(psi_max_deg[m], 90)),
               "wz_sat_share_mean": float(np.mean(wz_sat_share[m]))}
        if c5 is not None:
            row["cur_over_500mA"] = float(np.nanmean(c5[:, m]))
            row["guard_trip"] = float(np.mean(guard_trip[m]))
        if stairs:
            L = terrain_mod.STAIR_LAYOUTS[(names[c], int(r))]
            row.update({"direction": L["direction"], "riser_mm": round(1000 * float(np.mean(L["rises"])), 2),
                        "tread_mm": round(1000 * float(np.mean(L["treads"])), 1) if L["treads"] else None,
                        "steps": len(L["rises"]), "rises_mm": [round(1000 * x, 2) for x in L["rises"]],
                        "risers_climbed_mean": float(np.mean(climbed_n[m])),
                        "t_landing_med_s": float(np.nanmedian(t_goal[m])) if np.any(m & landed) else None,
                        "leg_contact_trials": float(np.mean(leg_any[m])), "leg_contact_s_mean": float(np.mean(leg_s[m])),
                        "sat_abd": float(np.mean(sat_share[m, 0])), "sat_hip": float(np.mean(sat_share[m, 1])),
                        "sat_knee": float(np.mean(sat_share[m, 2])),
                        "stance_env_p50_mm": float(1000 * np.median(fenv_st[m]))})  # sat_*: share of steps an action is clipped; stance_env: largest |y| of a stance foot
            if ok_width is not None:
                row["success_width"] = float(np.mean(ok_width[m]))
                row["success_width_all_feet"] = float(np.mean(ok_width_all[m]))
            if cmax is not None:
                row["cur_max_p99_A"] = float(np.nanpercentile(cmax[:, m], 99))
        table.append(row)
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=paths.REPO).stdout.strip()
    res = {"options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in opt.__dict__.items()
                       if k not in ("noise_start", "noise_end")}, "repo_commit": commit, "argv": sys.argv,
           "policy": Path(args.policy).name if args.policy else None, "actuator": opt.arm, "foot_solref": opt.foot_solref,
           "masses": opt.masses, "action_scale": list(opt.action_scale), "omni": omni, "heading_loop": bool(omni and args.heading_loop),
           "heading": ({"law": "mjlab P-only clip(2 err, 0.6)" if train_law else "heading.pi_step on the true yaw",
                        "wz_max": wz_lim, "antiwindup": None if train_law else args.heading_aw,
                        "kick": args.heading_kick, "lateral_loop_k": args.lateral_loop}
                       if omni and args.heading_loop else None),
           "start_noise": args.start_noise, "yaw_kick": args.yaw_kick,
           "width": args.width if stairs else None, "landing": args.landing if stairs else None,
           "risers": args.risers if stairs else None, "tread": args.tread if stairs else None,
           "steps": args.steps if stairs else None,
           "width_rule": (args.width_rule if args.width_rule is not None else args.width) if stairs else None,
           "vx": args.vx, "seconds": args.seconds, "goal_m": args.goal, "stop_at_goal": bool(args.stop_at_goal), "noise": None if args.clean else args.noise,
           "seed": args.seed, "terrain_seed": args.terrain_seed, "envs": n, "guard_ma": args.guard_ma, "wall_s": wall, "cells": table,
           # per trial, in env order (the same seeds give the same randomised robots and tiles: pair policies trial by trial)
           "trials": {"level": rr.astype(int).tolist(), "terrain": [names[int(c)] for c in cc],
                      "success": ok.astype(int).tolist(),
                      "success_on_strip": ok_strip.astype(int).tolist(),
                      **({"risers_climbed": climbed_n.astype(int).tolist(), "steps": steps_n.astype(int).tolist(),
                          "leg_contact_s": np.round(leg_s, 2).tolist(),
                          "foot_env_m": np.round(fenv, 4).tolist(), "foot_env_stance_m": np.round(fenv_st, 4).tolist(),
                          **({"success_width": ok_width.astype(int).tolist(),
                              "success_width_all_feet": ok_width_all.astype(int).tolist()} if ok_width is not None else {})}
                         if stairs else {}),
                      "psi_max_deg": np.round(psi_max_deg, 2).tolist(), "psi_d05_deg": np.round(psi_d05_deg, 2).tolist(),
                      "wz_sat_share": np.round(wz_sat_share, 3).tolist(),
                      **({"yaw_kick_t_s": np.round(kick_k.cpu().numpy() * DT, 2).tolist(),
                          "yaw_kick_sign": kick_s.cpu().numpy().astype(int).tolist()} if ykick is not None else {}),
                      "fall": fell.astype(int).tolist(), "fall_cause": cause.cpu().numpy().astype(int).tolist(),
                      "dist_m": np.round(dist, 4).tolist(),
                      "lat_m": np.round(lat, 4).tolist(), "lat_max_m": np.round(lat_max, 4).tolist(),
                      "t_goal_s": [None if np.isnan(x) else round(float(x), 2) for x in t_goal],
                      **({"kick_peak_deg": np.round(kick_peak, 2).tolist(), "overshoot_deg": np.round(overshoot, 2).tolist()}
                         if kick is not None else {})}}
    if args.video:
        import imageio.v2 as imageio
        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.video, frames, fps=int(1 / DT))
        res["video"] = args.video
    env.close()
    return res


@torch.no_grad()
def throughput(args) -> None:
    opt = TerrainOptions(arm=args.arm or "stock", num_envs=args.throughput, terrain=args.terrain, rows=args.rows,
                         noisy=True, randomise=True, **({"masses": args.masses} if args.masses else {}))
    env = build(opt)
    env.reset()
    a = torch.zeros(env.num_envs, 12, device=env.device)
    for _ in range(20):
        env.step(a)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.throughput_steps):
        env.step(a)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"{opt.arm} on {args.terrain}: {args.throughput} envs, {args.throughput_steps} steps in {dt:.1f} s = "
          f"{args.throughput * args.throughput_steps / dt:,.0f} env-steps/s (policy steps, 20 physics steps each)")
    env.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", help="an exported policy directory (policy.onnx); none: zero actions")
    ap.add_argument("--actuator", dest="arm", metavar="ACTUATOR",
                    help="actuator model: `bam_t5ub06` (BAM models fitted to the MD01 servos) or `stock` (default: the "
                         "policy manifest's task.actuator, else stock)")
    ap.add_argument("--terrain", default="rough_basic",
                    choices=("rough_basic", "rough", "rough_steps", "stairs", "stairs_steep", "stairs_narrow", "flat",
                             "stairs_built"),
                    help="procedural families by level (rows), or stairs_built: straight flights from --risers, "
                         "--tread, --steps, --width, --landing, one column per riser; success there = all four feet "
                         "on the landing without a fall (--goal unused)")
    ap.add_argument("--rows", type=int, default=10)
    ap.add_argument("--tilt-limit", type=float, default=0.7,
                    help="rad: the tilt from vertical that ends a trial as a fall (training's 0.7; a 27-degree flight "
                         "pitches the body by itself)")
    ap.add_argument("--risers", type=float, nargs="+", default=list(STAIR_BUILT["risers"]),
                    help="stairs_built: the riser height(s) (m), one flight each (default 0.025)")
    ap.add_argument("--tread", type=float, default=STAIR_BUILT["tread"], help="stairs_built: the tread (m)")
    ap.add_argument("--steps", type=int, default=STAIR_BUILT["steps"], help="stairs_built: risers per flight")
    ap.add_argument("--levels", type=int, nargs="+")
    ap.add_argument("--only", nargs="+", help="terrain names")
    ap.add_argument("--envs", type=int, default=840, help="spread evenly over the cells")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--vx", type=float, default=0.10)
    ap.add_argument("--goal", type=float, default=0.6, help="m of forward displacement for success")
    ap.add_argument("--heading-loop", action="store_true")
    ap.add_argument("--heading-aw", choices=ANTIWINDUP, default="cond", help="the loop's anti-windup (heading.pi_step; none = no anti-windup)")
    ap.add_argument("--heading-wz-max", type=float, default=WZ_MAX, help="rad/s: the loop's output limit")
    ap.add_argument("--heading-kick", metavar="T0:DUR:RATE",
                    help="a heading disturbance (an edge strike): add RATE rad/s (- = right) to the wz command during "
                         "[T0, T0+DUR) s, the sum clipped to ±0.6 (the P law's range); records the disturbance's peak error and "
                         "the overshoot past the line")
    ap.add_argument("--heading-law", choices=("robot", "train"), default="robot",
                    help="robot: heading.pi_step (the robot's PI law, --heading-wz-max); train: mjlab's P-only "
                         "clip(2 err, ±0.6), the default training law (--heading-law p)")
    ap.add_argument("--lateral-loop", type=float, metavar="K",
                    help="rad/m: bias the heading target by -K x the dead-reckoned lateral offset (heading.lateral_step)")
    ap.add_argument("--yaw-kick", metavar="T0:T1:TORQUE:DUR",
                    help="a yaw torque pulse (a foot catching): +-TORQUE N m (random sign) on the body for DUR s, from "
                         "a random time in [T0, T1] s per trial (physical, unlike --heading-kick)")
    ap.add_argument("--start-noise", type=float, nargs=2, metavar=("DY", "DYAW"),
                    help="m and deg: a uniform lateral offset and yaw error at the start per trial (the placement "
                         "tolerance when the robot is set down, e.g. 0.01 2)")
    ap.add_argument("--width", type=float, default=STAIR_BUILT["width"],
                    help="stairs_built: the treads' width (m), drops at the sides (default 0.20)")
    ap.add_argument("--landing", type=float, default=STAIR_BUILT["landing"],
                    help="stairs_built: the top tread's length past the last riser (m; default 0.25)")
    ap.add_argument("--width-rule", type=float,
                    help="m: score success with every stance foot centre within half this width (default --width)")
    ap.add_argument("--side", type=float, default=0.15,
                    help="m of lateral offset counted as leaving the course's side")
    ap.add_argument("--noise", default="n2")
    ap.add_argument("--clean", action="store_true", help="no noise, nominal robot (no randomisation)")
    ap.add_argument("--masses", help="mass file in robot/assets/sku_2 (default: the terrain task's, 478 g)")
    ap.add_argument("--no-self-collision", action="store_true")
    ap.add_argument("--foot-solref", type=float, nargs=2, help="the feet's contact timeconst and dampratio")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--terrain-seed", type=int, default=0)
    ap.add_argument("--guard-ma", type=int, default=1000, help="the servo current guard scored (mA held 0.5 s)")
    ap.add_argument("--stop-at-goal", action="store_true",
                    help="a trial ends when the body passes the goal (as scored on the robot): later falls and "
                         "sideways drift no longer count")
    ap.add_argument("--camera", type=float, nargs=3, metavar=("DIST", "ELEV", "AZIM"),
                    help="video camera: distance (m), elevation and azimuth (deg); default 0.55 -18 135")
    ap.add_argument("--json", help="write the results (options, cells, per-trial outcomes) to this file")
    ap.add_argument("--video", help="render the run to this mp4 (env 0's camera; 960 x 540 at 50 fps)")
    ap.add_argument("--throughput", type=int, help="measure env-steps/s with this many envs instead")
    ap.add_argument("--throughput-steps", type=int, default=200, help="--throughput: the steps timed")
    args = ap.parse_args()
    if args.throughput:
        throughput(args)
        sys.stdout.flush()
        os._exit(0)
    res = evaluate(args)
    for row in res["cells"]:
        extra = (f" cur>500mA {100 * row['cur_over_500mA']:4.1f}% trip{args.guard_ma} {100 * row['guard_trip']:3.0f}%"
                 if "cur_over_500mA" in row else "")
        if "success_width" in row:
            extra += (f"  WIDTH {100 * row['success_width']:5.1f}% (all feet {100 * row['success_width_all_feet']:5.1f}%,"
                      f" stance env p50 {row['stance_env_p50_mm']:.0f} mm)")
        extra += f"  psi max p50/p90 {row['psi_max_med_deg']:.1f}/{row['psi_max_p90_deg']:.1f} deg sat {100 * row['wz_sat_share_mean']:.0f}%"
        if "riser_mm" in row:
            extra += (f"  riser {row['riser_mm']:4.1f} mm tread {row['tread_mm']} mm climbed {row['risers_climbed_mean']:.1f}"
                      f"/{row['steps']} legs {100 * row['leg_contact_trials']:3.0f}% sat h/k "
                      f"{100 * row['sat_hip']:.1f}/{100 * row['sat_knee']:.1f}%")
        print(f"{row['terrain']:12s} L{row['level']}: success {100 * row['success']:5.1f}% "
              f"(on the strip {100 * row['success_on_strip']:5.1f}%, off the side {100 * row['off_side']:4.1f}%, "
              f"lateral {row['lat_end_med_mm']:+5.0f} mm)  fall "
              f"{100 * row['fall']:5.1f}%  stall {100 * row['stall']:5.1f}%  dist {row['dist_med_m']:5.2f} m  "
              f"tilt p99 {row['tilt_p99_deg']:5.1f}°"
              f" (>{TILT_CUT_DEG:.0f}°: {100 * row['tilt_over_cut']:3.0f}%)  foot pressed in p99 "
              f"{row['foot_pen_p99_mm']:.1f} mm{extra}  n={row['n']}")
    print(f"wall {res['wall_s']:.0f} s")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
