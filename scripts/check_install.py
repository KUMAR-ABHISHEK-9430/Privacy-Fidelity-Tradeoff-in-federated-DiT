"""Install experiment dependencies in an overlay, never mutate the CUDA torch stack."""
import argparse
import importlib.metadata as metadata
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise SystemExit("Pinned NumPy 1.26 stack requires Python 3.10-3.12; use a compatible CUDA Python environment (3.11 tested).")
    try:
        import torch
        import torchvision
        before = (torch.__version__, torchvision.__version__, torch.version.cuda)
        assert torch.version.cuda, "CPU-only torch installation"
    except Exception as exc:
        raise SystemExit(f"Working CUDA torch/torchvision must be preinstalled; will not replace them: {exc}")
    if args.install:
        # Transitive dependencies are resolved under exact torch/vision constraints.
        # A venv overlay in run_all.sh keeps base Molab packages untouched.
        constraint = Path(".cache/torch-constraints.txt")
        constraint.parent.mkdir(parents=True, exist_ok=True)
        constraint.write_text(f"torch=={before[0]}\ntorchvision=={before[1]}\n")
        subprocess.run([sys.executable, "-m", "pip", "install", "-c", str(constraint),
                        "-r", "requirements.txt"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-e", "."], check=True)
    for package in ("diffusers", "peft", "opacus", "torchmetrics", "torch_fidelity", "fdit"):
        __import__(package)
    assert metadata.version("torch") == before[0]
    assert metadata.version("torchvision") == before[1]
    print("Imports passed; CUDA PyTorch stack preserved:", before)


if __name__ == "__main__":
    main()
