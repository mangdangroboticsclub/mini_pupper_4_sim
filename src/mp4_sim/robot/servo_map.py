"""Mapping between servo angles and model joint angles.

Provides the per-SKU matrix ``A`` and stand pose used by the contract
(``targets = A @ (action * 40 degrees) + stand``), and the inverse used to
report servo positions and velocities relative to the stand pose.
"""
