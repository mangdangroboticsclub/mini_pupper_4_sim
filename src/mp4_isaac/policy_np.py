"""Numpy side of one policy step: the action limit, the velocity filter, and the observation history."""

from __future__ import annotations

import numpy as np

from .contract import ACTION_SIZE, BAND_DEG, F32, OBS_SIZE, OBS_TERMS, STEP_LIMIT_DEG, STATE_DT, Contract


class ActionDecoder:
    """The step limit acts against the previous applied target. ``reset`` starts at the stand."""

    def __init__(self, contract: Contract, scale, quantise: bool = True):
        self.c = contract
        self.scale = np.broadcast_to(np.asarray(scale, dtype=float), (ACTION_SIZE,)).copy()
        self.quantise = quantise
        self.band = np.radians(BAND_DEG)
        self.limit = np.radians(STEP_LIMIT_DEG)
        self.reset()

    def reset(self) -> None:
        self.prev_rel = np.zeros(ACTION_SIZE)
        self.prev_action = np.zeros(ACTION_SIZE)

    def __call__(self, raw_action) -> dict:
        a = np.clip(np.asarray(raw_action, dtype=float), -1.0, 1.0)
        rel = np.clip(self.scale * a, -self.band, self.band)
        rel = np.clip(rel, self.prev_rel - self.limit, self.prev_rel + self.limit)
        if self.quantise:
            _dd, rel = self.c.quantise_rel(rel)
        self.prev_rel = rel
        self.prev_action = rel / self.scale
        return {"rel": rel, "action": self.prev_action.copy()}


class VelocityFilter:
    """Velocity from the 200 Hz positions: a median of 3, then the slope over the last 4. Fed every 5 ms."""

    WEIGHTS = np.array([-1.5, -0.5, 0.5, 1.5]) / (5.0 * STATE_DT)

    def __init__(self, n: int = ACTION_SIZE):
        self.hist = None
        self.n = n

    def reset(self) -> None:
        self.hist = None

    def push(self, x) -> None:
        x = np.asarray(x, dtype=float)
        if self.hist is None:
            self.hist = np.repeat(x[None, :], 6, axis=0)
        else:
            self.hist = np.concatenate([self.hist[1:], x[None, :]], axis=0)

    def velocity(self) -> np.ndarray:
        if self.hist is None:
            return np.zeros(self.n)
        med = np.stack([np.median(self.hist[k:k + 3], axis=0) for k in range(4)])
        return self.WEIGHTS @ med


class HistoryStacker:
    """Each term's frames, oldest to newest. The first observation after a reset fills every slot."""

    def __init__(self, length: int):
        self.length = int(length)
        self.frames = None

    def reset(self) -> None:
        self.frames = None

    def __call__(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=F32)
        if self.frames is None:
            self.frames = [obs] * self.length
        else:
            self.frames = self.frames[1:] + [obs]
        h = np.stack(self.frames)
        return np.concatenate([h[:, sl].reshape(-1) for _, sl in OBS_TERMS]).astype(F32)


def assemble_observation(ang_vel, gravity, command, pos_rel, vel, prev_action) -> np.ndarray:
    obs = np.empty(OBS_SIZE, dtype=F32)
    obs[0:3] = ang_vel
    obs[3:6] = gravity
    obs[6:9] = (command, 0.0, 0.0) if np.ndim(command) == 0 else command
    obs[9:21] = pos_rel
    obs[21:33] = vel
    obs[33:45] = prev_action
    return obs
