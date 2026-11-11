import importlib.util
from pathlib import Path
import subprocess
import sys

from packaging.requirements import Requirement


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numpy_pin_matches_python_version():
    requirements = [Requirement(line) for line in Path("requirements.txt").read_text().splitlines()
                    if line and not line.startswith("#")]
    for version, expected in (("3.11", "==1.26.4"), ("3.12", "==1.26.4"), ("3.13", "==2.2.6")):
        selected = [r for r in requirements if r.name == "numpy" and r.marker.evaluate({"python_version": version})]
        assert len(selected) == 1 and str(selected[0].specifier) == expected


def test_overlay_inherits_parent_venv_packages_and_reuses_it(tmp_path):
    prepare = load_script("prepare_env")
    parent_packages = tmp_path / "parent-site-packages"
    parent_packages.mkdir()
    (parent_packages / "fdit_parent_probe.py").write_text("VALUE = 'inherited from launcher venv'\n")
    source = prepare.source_environment()
    source["site_packages"].append(str(parent_packages))
    child = prepare.prepare_environment(tmp_path, source)
    output = subprocess.check_output([str(child), "-c", "import fdit_parent_probe; print(fdit_parent_probe.VALUE)"], text=True)
    assert output.strip() == "inherited from launcher venv"
    assert prepare.prepare_environment(tmp_path, source) == child
    # An interpreter change selects another overlay without touching existing files.
    source = dict(source, prefix="another-launcher-environment")
    other = prepare.prepare_environment(tmp_path, source)
    assert other != child and child.is_file() and other.is_file()
