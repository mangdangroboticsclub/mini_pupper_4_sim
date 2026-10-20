"""The policy manifest: what a loader needs to run an exported policy, embedded in the ONNX metadata under
``mp3_manifest_json`` (docs/POLICY_CONTRACT.md).

Required (a loader rejects the file unless they match): ``robot``, ``observation_dim``, ``action_dim``, ``control_dt``,
``action_scale_rad``, ``contract_sha256``, ``omni``. Optional, with defaults: ``history`` (1), ``frame_dim`` (45),
``history_layout`` ("term_major": each term's frames oldest to newest, the terms in order), ``history_reset``
("repeat_first": after a reset every slot holds the first frame), ``command`` (where the command comes from:
``{"source": "velocity"}``, or ``{"source": "heading_loop", "law": "pi", ...}`` with the loop's gains), and the
descriptive fields below. Only these keys are allowed, and no value may be a path or a host name.
"""
from __future__ import annotations

import re

from .. import contract, heading, paths

MANIFEST_KEY = "mp3_manifest_json"
ROBOT = "MP3-SKU2"
REQUIRED = ("robot", "observation_dim", "action_dim", "control_dt", "action_scale_rad", "contract_sha256", "omni")
OPTIONAL = ("history", "frame_dim", "history_layout", "history_reset", "command", "task", "observation_layout",
            "servo_order", "action_clip", "band_deg", "step_limit_deg", "physics_dt", "normalization",
            "checkpoint_sha256")
OBSERVATION_LAYOUT = "ang_vel 3, gravity 3, command 3, pos 12, vel 12, prev_action 12"
COMMAND_SOURCES = ("velocity", "heading_loop")
# a value that looks like an absolute path (POSIX, a Windows drive or share) or a URL
_PATHLIKE = re.compile(r"^(/|~|[A-Za-z]:[\\/]|\\\\)|://|[\\/](home|Users|tmp|mnt)[\\/]")


def heading_loop_command(wz_max: float = heading.WZ_MAX) -> dict:
    """The robot's heading loop (``heading.pi_step``) as the command source."""
    return {"source": "heading_loop", "law": "pi", "kp": heading.KP, "ki": heading.KI, "wz_max": float(wz_max),
            "i_max": heading.I_MAX, "antiwindup": "cond"}


def build(action_scale, *, omni: bool, history: int = 1, command: dict | None = None, task: dict | None = None,
          physics_dt: float = 0.001, **extra) -> dict:
    """A complete manifest for a 12-action policy on SKU2 with a history of ``history`` steps."""
    history = max(int(history), 1)
    scale = [float(x) for x in action_scale]
    assert len(scale) == contract.ACTION_SIZE, scale
    m = {"robot": ROBOT, "observation_dim": contract.obs_size(history), "action_dim": contract.ACTION_SIZE,
         "control_dt": contract.CONTROL_DT, "action_scale_rad": scale,
         "contract_sha256": paths.sha256(paths.CONTRACT_JSON), "omni": bool(omni),
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


def validate(m: dict, contract_json=None) -> dict:
    """Raise ``ValueError`` unless ``m`` is a valid manifest; ``contract_json``: also require its hash to match that
    contract file's."""
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
    elif contract_json is not None and m["contract_sha256"] != paths.sha256(contract_json):
        errors.append("contract_sha256 differs from the contract file's")
    bad = [s for s in _strings(m) if _PATHLIKE.search(s)]
    if bad:
        errors.append(f"path-like values {bad}")
    if errors:
        raise ValueError("manifest: " + "; ".join(errors))
    return m
