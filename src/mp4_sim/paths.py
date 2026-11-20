"""Where the package's files live: package-relative paths only, and the file hash every record uses."""
import hashlib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ASSETS = PACKAGE / "robot" / "assets"
SERVO_PROFILES = ASSETS / "servo_profiles.json"
ACTUATOR_PARAMS = PACKAGE / "actuators" / "params"
ENV_DATA = PACKAGE / "envs" / "data"
PREFIX = "mp4_sim:"           # how records name a file of this package

TEXT_SUFFIXES = {".xml", ".json", ".py", ".csv", ".md", ".txt"}


def sha256(path) -> str:
    """The file's digest; text sources are hashed with LF line endings, so a Windows checkout gives the same digest."""
    p = Path(path)
    b = p.read_bytes()
    if p.suffix.lower() in TEXT_SUFFIXES:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def name(path) -> str:
    """A file's name in records: ``mp4_sim:<path relative to the package>`` for a package file, else the path as
    given."""
    p = Path(path).resolve()
    if p.is_relative_to(PACKAGE):
        return PREFIX + p.relative_to(PACKAGE).as_posix()
    return str(path)


def resolve(name_or_path) -> Path:
    """The inverse of ``name``: an ``mp4_sim:`` name as a path in the package; anything else as a path."""
    s = str(name_or_path)
    if s.startswith(PREFIX):
        return PACKAGE / s[len(PREFIX):]
    return Path(s)
