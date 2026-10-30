"""The policy manifest embedded in an ONNX file under ``mp3_manifest_json``.

Required: ``robot``, ``observation_dim``, ``action_dim``, ``control_dt``, ``action_scale_rad``, ``contract_sha256``,
``omni``. A history is term-major (each term's frames, oldest to newest) and a reset repeats the first frame.
``command`` is ``{"source": "velocity"}`` or the heading PI loop. No value may be a path.
"""

from __future__ import annotations

import re

from . import contract, heading
from .paths import contract_json, sha256

MANIFEST_KEY = "mp3_manifest_json"
ROBOT = "MP3-SKU2"
REQUIRED = ("robot", "observation_dim", "action_dim", "control_dt", "action_scale_rad", "contract_sha256", "omni")
OPTIONAL = ("history", "frame_dim", "history_layout", "history_reset", "command", "task", "observation_layout",
            "servo_order", "action_clip", "band_deg", "step_limit_deg", "physics_dt", "normalization",
            "checkpoint_sha256")
OBSERVATION_LAYOUT = "ang_vel 3, gravity 3, command 3, pos 12, vel 12, prev_action 12"
COMMAND_SOURCES = ("velocity", "heading_loop")
_PATHLIKE = re.compile(r"^(/|~|[A-Za-z]:[\\/]|\\\\)|://|[\\/](home|Users|tmp|mnt)[\\/]")


def heading_loop_command(wz_max: float = heading.WZ_MAX) -> dict:
    """The heading PI loop (``heading.pi_step``) as the command source."""
    return {"source": "heading_loop", "law": "pi", "kp": heading.KP, "ki": heading.KI, "wz_max": float(wz_max),
            "i_max": heading.I_MAX, "antiwindup": "cond"}


def build(action_scale, *, omni: bool, history: int = 1, command: dict | None = None, task: dict | None = None,
          physics_dt: float = 0.001, **extra) -> dict:
    """A manifest for a 12-action policy. ``history`` of 0 or 1 is the 45-value frame."""
    history = max(int(history), 1)
    scale = [float(x) for x in action_scale]
    if len(scale) != contract.ACTION_SIZE:
        raise ValueError(f"action scale has {len(scale)} values, expected {contract.ACTION_SIZE}")
    m = {"robot": ROBOT, "observation_dim": contract.obs_size(history), "action_dim": contract.ACTION_SIZE,
         "control_dt": contract.CONTROL_DT, "action_scale_rad": scale,
         "contract_sha256": sha256(contract_json()), "omni": bool(omni),
         "history": history, "frame_dim": contract.OBS_SIZE, "history_layout": "term_major",
         "history_reset": "repeat_first", "command": command or {"source": "velocity"},
         "observation_layout": OBSERVATION_LAYOUT, "servo_order": contract.Contract.load().meta["servo_order"],
         "action_clip": [-1.0, 1.0], "band_deg": [float(x) for x in contract.BAND_DEG],
         "step_limit_deg": contract.STEP_LIMIT_DEG, "physics_dt": float(physics_dt), "normalization": "none"}
    if task is not None:
        m["task"] = task
    m.update(extra)
    validate(m)
    return m


def _strings(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)


def validate(m: dict, contract_file=None) -> dict:
    """Raise ``ValueError`` unless ``m`` is a valid manifest. ``contract_file`` also checks the contract hash."""
    missing = [k for k in REQUIRED if k not in m]
    unknown = sorted(set(m) - set(REQUIRED) - set(OPTIONAL))
    if missing or unknown:
        raise ValueError(f"manifest: missing {missing}, unknown {unknown}")
    errors = []
    if m["robot"] != ROBOT:
        errors.append(f"robot {m['robot']!r}")
    if m["action_dim"] != contract.ACTION_SIZE or len(m["action_scale_rad"]) != contract.ACTION_SIZE:
        errors.append("action_dim / action_scale_rad")
    if abs(float(m["control_dt"]) - contract.CONTROL_DT) > 1e-12:
        errors.append(f"control_dt {m['control_dt']}")
    history, frame = int(m.get("history", 1)), int(m.get("frame_dim", contract.OBS_SIZE))
    if history < 1 or m["observation_dim"] != frame * history:
        errors.append(f"observation_dim {m['observation_dim']} != frame_dim {frame} x history {history}")
    if m.get("history_layout", "term_major") != "term_major" or m.get("history_reset", "repeat_first") != "repeat_first":
        errors.append("history_layout / history_reset")
    cmd = m.get("command", {"source": "velocity"})
    if not isinstance(cmd, dict) or cmd.get("source") not in COMMAND_SOURCES:
        errors.append(f"command {cmd!r}")
    if not isinstance(m["omni"], bool):
        errors.append("omni")
    if m.get("normalization", "none") != "none":
        errors.append(f"normalization {m['normalization']!r}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(m["contract_sha256"])):
        errors.append("contract_sha256")
    elif contract_file is not None and m["contract_sha256"] != sha256(contract_file):
        errors.append("contract_sha256 differs from the contract file's")
    bad = [s for s in _strings(m) if _PATHLIKE.search(s)]
    if bad:
        errors.append(f"path-like values {bad}")
    if errors:
        raise ValueError("manifest: " + "; ".join(errors))
    return m
