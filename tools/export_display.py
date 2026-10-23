"""
export_display.py - pull the screen out of the SKU 1 SolidWorks assembly.

    python tools/export_display.py --assembly "path/to/最新装配体.SLDASM"

Windows only: it drives a local SolidWorks over COM (``pip install pywin32``),
reads the tessellation SolidWorks already shows for the two screen components
and writes them, in the assembly's own frame (metres), to

    cad/sku_1/display/display_bezel.stl    display_minipupper.SLDPRT, the frame
    cad/sku_1/display/display_panel.stl    屏幕.SLDPRT ("screen"), the glass
    cad/sku_1/display/source.json          what they were read from

Nothing in the assembly is changed, and an assembly that is already open is
read as it is. ``tools/build_display.py`` turns these into each SKU's overlay;
you only need to run this again if the screen changes in CAD.
"""
import argparse
import datetime
import json
import os
import struct

import numpy as np
import win32com.client

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, os.pardir, "cad", "sku_1", "display"))

#: output part -> the SolidWorks part file it is read from
PARTS = {"display_bezel": "display_minipupper.SLDPRT", "display_panel": "屏幕.SLDPRT"}

SW_DOC_ASSEMBLY = 2


def call(x):
    """pywin32's late binding exposes some zero-argument methods as properties."""
    if x is None or isinstance(x, win32com.client.CDispatch):
        return x
    return x() if callable(x) else x


def rgb(values):
    v = list(values or ())
    return [round(float(c), 4) for c in v[:3]] if len(v) >= 3 and min(v[:3]) >= 0 else None


def write_stl(path, triangles):
    n = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    np.divide(n, length, out=n, where=length > 0)
    records = np.zeros((len(triangles), 50), dtype=np.uint8)
    data = np.concatenate([n, triangles.reshape(-1, 9)], axis=1).astype("<f4")
    records[:, :48] = data.view(np.uint8).reshape(len(triangles), 48)
    with open(path, "wb") as fh:
        fh.write(b"export_display.py".ljust(80, b" ") + struct.pack("<I", len(triangles)) + records.tobytes())


def component_triangles(comp):
    """The component's faces as triangles in the assembly frame (m), and their colours."""
    t = np.array(comp.Transform2.ArrayData, dtype=float)
    R, p, s = t[0:9].reshape(3, 3), t[9:12], t[12]
    fallback = rgb(comp.MaterialPropertyValues) or rgb(call(comp.GetModelDoc2).MaterialPropertyValues)
    tris, colors = [], set()
    for body in comp.GetBodies2(0) or ():
        for face in body.GetFaces() or ():
            v = face.GetTessTriangles(True)
            if not v:
                continue
            # SolidWorks transforms row vectors: x_assembly = x_part @ R * s + p
            tris.append(((np.array(v, dtype=float).reshape(-1, 3) @ R) * s + p).reshape(-1, 3, 3))
            colors.add(tuple(rgb(face.MaterialPropertyValues) or fallback))
    return np.concatenate(tris), sorted(colors), t.tolist()


def main(argv=None):
    ap = argparse.ArgumentParser(description="SKU 1 SolidWorks assembly -> cad/sku_1/display/")
    ap.add_argument("--assembly", required=True, help="the SKU 1 .SLDASM")
    ap.add_argument("--out", default=OUT, help=f"default: {OUT}")
    a = ap.parse_args(argv)
    path = os.path.abspath(a.assembly)

    sw = win32com.client.Dispatch("SldWorks.Application")
    sw.Visible = True       # attaching to a running SolidWorks must not hide it
    doc = sw.GetOpenDocumentByName(path) or sw.OpenDoc(path, SW_DOC_ASSEMBLY)
    if doc is None:
        raise SystemExit(f"SolidWorks could not open {path}")
    if doc.GetUserPreferenceIntegerValue(31) != 0:   # swUnitsLinear: 0 = mm
        print("  note: the assembly is not in mm; the API reports metres regardless")

    by_file = {}
    for comp in doc.GetComponents(False):
        if not call(comp.IsSuppressed):
            by_file.setdefault(os.path.basename(call(comp.GetPathName)), []).append(comp)
    os.makedirs(a.out, exist_ok=True)
    source = {"about": "The screen, read from the SKU 1 SolidWorks assembly by tools/export_display.py. "
                       "STLs are in the assembly frame (y up, x toward the rear), metres.",
              "assembly": os.path.basename(path),
              "solidworks": call(sw.RevisionNumber),
              "exported": datetime.date.today().isoformat(),
              "parts": {}}
    for name, part in PARTS.items():
        comps = by_file.get(part, [])
        if len(comps) != 1:
            raise SystemExit(f"expected one unsuppressed {part} in the assembly, found {len(comps)}")
        tris, colors, transform = component_triangles(comps[0])
        write_stl(os.path.join(a.out, name + ".stl"), tris)
        source["parts"][name] = {"component": comps[0].Name2, "part": part, "triangles": len(tris),
                                 "cad_rgb": [list(c) for c in colors], "transform": transform}
        lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
        print(f"  {name:14s} <- {comps[0].Name2}: {len(tris)} triangles, "
              f"{np.round(lo * 1000, 1)} .. {np.round(hi * 1000, 1)} mm, colour {colors}")
    with open(os.path.join(a.out, "source.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(source, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print("wrote", a.out)
    print("now place it:  python tools/build_display.py")


if __name__ == "__main__":
    main()
