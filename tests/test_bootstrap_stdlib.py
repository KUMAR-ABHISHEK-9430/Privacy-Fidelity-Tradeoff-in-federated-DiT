"""Dependency-free bootstrap regressions; runnable without torch/pytest.

python -m unittest discover -s tests -p test_bootstrap_stdlib.py -v
"""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".cache").mkdir(exist_ok=True)

    def test_constraints_use_distribution_versions_and_fresh_process(self):
        check = script("check_install")
        commands = []
        fake = {"torch": SimpleNamespace(__version__="2.11.0+cu130", version=SimpleNamespace(cuda="13.0")),
                "torchvision": SimpleNamespace(__version__="0.26.0+cu130")}
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as temporary:
            previous = Path.cwd()
            try:
                os.chdir(temporary)
                with patch.dict(sys.modules, fake), patch.object(sys, "argv", ["check_install.py", "--install"]), \
                     patch.object(check.metadata, "version", side_effect=lambda n: {"torch": "2.11.0", "torchvision": "0.26.0"}[n]), \
                     patch.object(check.subprocess, "run", side_effect=lambda command, **kwargs: commands.append(command)):
                    check.main()
                constraint = Path(".cache/torch-constraints.txt").read_text()
                self.assertEqual(constraint, "torch==2.11.0\ntorchvision==0.26.0\n")
                self.assertEqual(len(commands), 3)
                self.assertIn("check_install.py", commands[-1][1])
                self.assertNotIn("--install", commands[-1])
                self.assertIn("--expected-torch", commands[-1])
            finally:
                os.chdir(previous)

    def test_outer_venv_site_packages_are_really_importable(self):
        prepare = script("prepare_env")
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as temporary:
            root = Path(temporary)
            parent = root / "outer-packages"
            parent.mkdir()
            (parent / "fdit_bootstrap_marker.py").write_text("VALUE = 130\n")
            source = prepare.source_environment()
            source["site_packages"].append(str(parent))
            child = prepare.prepare_environment(root, source)
            output = subprocess.check_output([str(child), "-c", "import fdit_bootstrap_marker; print(fdit_bootstrap_marker.VALUE)"], text=True)
            self.assertEqual(output.strip(), "130")
            self.assertEqual(prepare.prepare_environment(root, source), child)


if __name__ == "__main__":
    unittest.main()
