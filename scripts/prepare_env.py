"""Create an overlay which inherits the ACTUAL launcher environment, including uv venvs."""
import hashlib
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import venv


def source_environment():
    paths = site.getsitepackages()
    if site.ENABLE_USER_SITE:
        paths.append(site.getusersitepackages())
    return {"executable": sys.executable, "prefix": sys.prefix,
            "version": list(sys.version_info[:3]),
            "site_packages": list(dict.fromkeys(str(Path(p).resolve()) for p in paths if Path(p).is_dir()))}


def interpreter(folder):
    return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def prepare_environment(root, source=None):
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError(f"Python 3.10-3.13 required; launcher is {sys.version.split()[0]}")
    root = Path(root).resolve()
    source = source if source is not None else source_environment()
    signature = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:12]
    folder = root / ".venv-molab"
    marker = folder / "fdit-environment.json"
    if folder.exists():
        # Keep existing environments intact, including unmarked overlays made by older launchers.
        same = marker.is_file() and json.loads(marker.read_text()) == source and interpreter(folder).is_file()
        if not same:
            folder = root / f".venv-molab-{signature}"
            marker = folder / "fdit-environment.json"
    if folder.exists() and not (marker.is_file() and json.loads(marker.read_text()) == source and interpreter(folder).is_file()):
        raise RuntimeError(f"Incomplete or incompatible overlay at {folder}; preserve it and use a fresh project directory")
    if not folder.exists():
        print(f"Creating project overlay using Python {sys.version.split()[0]}: {folder}", file=sys.stderr)
        venv.EnvBuilder(system_site_packages=True, with_pip=True).create(folder)
        child = interpreter(folder)
        purelib = subprocess.check_output([str(child), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip()
        # --system-site-packages alone inherits the base interpreter, NOT necessarily
        # packages installed in the invoking /tmp/uv-venv. Append those explicit paths
        # after the overlay's own packages so pinned dependencies can be overlaid.
        Path(purelib, "fdit_launcher.pth").write_text("\n".join(source["site_packages"]) + "\n", encoding="utf-8")
        marker.write_text(json.dumps(source, indent=2), encoding="utf-8")
    return interpreter(folder)


if __name__ == "__main__":
    print(prepare_environment(Path(__file__).resolve().parents[1]))
