# Remote copy

The tree being edited lives on the other Windows machine. Pull it onto this
checkout with OpenSSH `scp` from PowerShell.

- Host: `afree@192.168.50.13`
- Remote root: `C:\Users\afree\Haris\mp4-isaac`
- Local root: `C:\Users\MangDang\Haris\mp4-isaac`

## Whole tree

Run this from `C:\Users\MangDang\Haris`, the parent of this folder. The
destination is the folder name, so `scp` writes into this project.

```powershell
scp -r afree@192.168.50.13:C:\Users\afree\Haris\mp4-isaac .\mp4-isaac
```

That copy includes `.git`, `.cache`, `__pycache__`, and
`src\mp4_isaac.egg-info`.

## One path

From this project directory, name the remote file and a local destination.
Backslashes match the remote Windows path, the same form as the whole-tree copy.

```powershell
scp afree@192.168.50.13:C:\Users\afree\Haris\mp4-isaac\docs\DESIGN.md .\docs\DESIGN.md
```

A directory needs `-r`:

```powershell
scp -r afree@192.168.50.13:C:\Users\afree\Haris\mp4-isaac\src .\src
```
