# Measured on the Mini Pupper v4 (SKU2)

Data measured on a Mini Pupper v4 (SKU2) and used by the walking and get-up policies that run on the robot. These
files belong to the model and calibration of the
[`sku2-policies` branch of mini_pupper_4_recovery](https://github.com/mangdang-bam/mini_pupper_4_recovery/tree/sku2-policies):
file paths named inside them (for example `robot/assets/sku_2/pupper.xml`, `tools/make_contract.py`) are that
repository's. This repository's own SKU2 model (`src/mp4_sim/robot/assets/sku_2/`, its 450 g masses and contract) is
not the one they were measured against, and nothing here uses them.

| File | What it is |
|---|---|
| `sku2_masses.json` | The robot weighed (630.5 g) and its centre of mass from level and pitched weighings, with the base's centre of mass solved for the model |
| `sku2_joint_limits.json` | Joint limits traced by hand with the servos limp, per leg, in degrees from the stand and as model joint ranges |
| `sku2_joint_limits.sha256` | The limits file's SHA-256 |
| `sku2_hip_envelope.json` | The hip's limit as a function of the abduction, per leg, from the same traces |
| `sku2_contract.json` | The servo-to-policy contract the limits were measured with: signs, stand pose and calibration offsets. The limits file records the first 12 hex digits of its SHA-256 (`contract_sha256_prefix`) |
