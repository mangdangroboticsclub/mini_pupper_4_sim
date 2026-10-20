"""The policy contract shared by both backends and the exporter.

Single source for the values in docs/POLICY_CONTRACT.md: observation layout
and scales (45 values), servo order, action dimension and scale (12 values,
40 degrees), control and physics periods, and the manifest embedded in every
ONNX export under ``mp3_manifest_json``.

Nothing else in the package should hard-code these values.
"""
