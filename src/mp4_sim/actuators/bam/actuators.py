# Copyright 2025 Marc Duclusaud & Grégoire Passault

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Trimmed to the MangDang MD01 / MD-300-150 family. Upstream also registers Dynamixel,
# Feetech, eRob, Waveshare and Unitree actuators here.

from .mangdang.actuator import (
    MD01PwmActuator,
    MD01PwmPinnedActuator,
    MD300Actuator,
)
from .testbench import Pendulum

actuators = {
    # MangDang MD01
    "md01v": lambda: MD01PwmActuator(Pendulum),
    "md01vp": lambda: MD01PwmPinnedActuator(Pendulum),
    # MangDang MD-300-150 (Rhoban/bam pull request #26): id_1 abduction, id_2 hip, id_3 knee
    "mangdang_md300_id_1": lambda: MD300Actuator(Pendulum),
    "mangdang_md300_id_2": lambda: MD300Actuator(Pendulum),
    "mangdang_md300_id_3": lambda: MD300Actuator(Pendulum),
}

__all__ = ["actuators"]
