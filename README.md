# mp4-isaac

The Mini Pupper v4 (SKU2) as an Isaac Lab USD asset, imported from the CAD URDF.

The CAD URDFs, part meshes, screen, weighed masses, servo fits, and the
policies (`flat_omni`, `rough_terrain`, `stairs`) live in `assets/`. This package stages a URDF the NVIDIA
importer can resolve (CAD links plus the screen) and converts it with Isaac Lab
3.0's standalone URDF importer. No Kit install.

The design and the steps are in [docs/DESIGN.md](docs/DESIGN.md) and
[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

```bash
bash scripts/install.sh                 # virtualenv at .venv in this directory
bash scripts/install.sh /path/to/venv   # or any other directory
PYTHONPATH=src python3 -m mp4_isaac.stage          # mesh check, no Isaac required
. .venv/bin/activate
python -m mp4_isaac.convert
python -m mp4_isaac.inspect
```

On Windows, the same install is `scripts/install.ps1`. It writes `Scripts\python.exe`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Venv D:\venvs\mp4-isaac
$env:PYTHONPATH = "src"; python -m mp4_isaac.stage
.\.venv\Scripts\Activate.ps1
python -m mp4_isaac.convert
python -m mp4_isaac.inspect
```

`MP4_ISAAC_VENV` selects the directory when either script is run without a path. Isaac Lab 3 lists Windows 11 for a native Windows install.
