# mp4-isaac

The Mini Pupper v4 (SKU2) as an Isaac Lab USD asset, imported from the CAD URDF.

The CAD URDFs, part meshes, screen, weighed masses, servo fits, and the
`flat_omni` policy live in `assets/`. This package stages a URDF the NVIDIA
importer can resolve (CAD links plus the screen) and converts it with Isaac Lab
3.0's standalone URDF importer. No Kit install.

The design and the steps are in [docs/DESIGN.md](docs/DESIGN.md) and
[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

```bash
bash scripts/install.sh
PYTHONPATH=src python3 -m mp4_isaac.stage          # mesh check, no Isaac required
. /home/afreez/.venvs/mp4-isaac/bin/activate
python -m mp4_isaac.convert
python -m mp4_isaac.inspect
```

The virtualenv is on the Linux disk because the Windows drive this repo sits on
does not have room for the NVIDIA wheels.
