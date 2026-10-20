"""Plumbing for the viewer scene: the robot plus its blue ghost copy.

``assets/<sku>/scene.xml`` contains the robot *and* a second, massless copy of
it named ``g_*`` used as the "where the real servos actually are" ghost. The RL
model deliberately drops that copy; this small class addresses both, and
``tools/build_model.py`` uses it to pose the stand keyframe.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .kinematics import JOINT_ORDER
from .mapping import ServoMap

__all__ = ["GhostRig", "actuator_map"]


def actuator_map(model: mujoco.MjModel):
    """How each actuator drives the 12 joints.

    Returns ``(joint_of_actuator, tendons)`` where

    * ``joint_of_actuator`` maps an actuator index to an index into
      :data:`~mp4_sim.robot.kinematics.JOINT_ORDER` for the plain hinge actuators,
    * ``tendons`` is a list of ``(actuator index, [(joint index, coefficient), ...])``
      for the actuators that drive a *tendon* — that is the four knees, where
      the servo sets the shin's angle through the rod rather than the angle
      between thigh and calf.

    Reading this from the model (instead of hard-coding it) keeps it working if
    the CAD, the tendon coefficients or the actuator order ever change.
    """
    joint_of_actuator: dict[int, int] = {}
    tendons: list[tuple[int, list[tuple[int, float]]]] = []
    for i in range(model.nu):
        name = model.actuator(i).name
        if int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT):
            joint_of_actuator[i] = JOINT_ORDER.index(name)
            continue
        tendon = int(model.actuator_trnid[i, 0])
        terms = []
        for k in range(int(model.tendon_num[tendon])):
            adr = int(model.tendon_adr[tendon]) + k
            if int(model.wrap_type[adr]) != int(mujoco.mjtWrap.mjWRAP_JOINT):
                continue
            jname = model.joint(int(model.wrap_objid[adr])).name
            terms.append((JOINT_ORDER.index(jname), float(model.wrap_prm[adr])))
        tendons.append((i, terms))
    return joint_of_actuator, tendons


class GhostRig:
    """Joint/actuator addressing for the viewer's model (robot + ghost).

    ``model`` must have been built from the same SKU as the servo map
    (``servo_map``, else the map of ``sku``, else the global SKU's).
    """

    def __init__(self, model, servo_map=None, *, sku=None):
        self.model = model
        self.map = ServoMap(servo_map, sku=sku)
        self.qadr = np.array([model.joint(n).qposadr[0] for n in JOINT_ORDER])
        self.gqadr = np.array([model.joint("g_" + n).qposadr[0] for n in JOINT_ORDER])
        self.act_joint, self.act_tendon = actuator_map(model)
        self.root = int(model.joint("root").qposadr[0])
        self.groot = int(model.joint("g_root").qposadr[0])
        self.groot_dof = int(model.joint("g_root").dofadr[0])

    def servo_to_joint(self, servo_deg):
        """firmware servo angles (deg) -> model joint angles (rad)."""
        return self.map.servo_to_joint(servo_deg)

    def set_qpos(self, data, q, ghost: bool = False) -> None:
        """Place the joints (or the ghost copy's joints) at ``q``."""
        data.qpos[self.gqadr if ghost else self.qadr] = np.asarray(q, dtype=float).reshape(12)

    def set_ctrl(self, data, q) -> None:
        """Command the joints, routing the knees through their tendon."""
        q = np.asarray(q, dtype=float).reshape(12)
        for i, j in self.act_joint.items():
            data.ctrl[i] = q[j]
        for i, terms in self.act_tendon:
            data.ctrl[i] = sum(coef * q[j] for j, coef in terms)
