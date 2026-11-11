"""Install experiment dependencies in an overlay, never mutate the CUDA torch stack."""
import argparse
import importlib.metadata as metadata
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--expected-torch")
    parser.add_argument("--expected-torchvision")
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        raise SystemExit("Python 3.10-3.13 required; Python 3.13 uses the pinned NumPy 2.2 environment.")
    try:
        import torch
        import torchvision
        # Molab can expose torch.__version__ == '2.11.0+cu130' while its
        # distribution metadata is '2.11.0'. pip resolves the latter.
        before = (metadata.version("torch"), metadata.version("torchvision"), torch.version.cuda)
        if args.expected_torch and before[0] != args.expected_torch:
            raise RuntimeError("CUDA torch version changed during dependency installation")
        if args.expected_torchvision and before[1] != args.expected_torchvision:
            raise RuntimeError("torchvision version changed during dependency installation")
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
        # The current process already imported torch/NumPy before pip changed
        # dependencies. Validate the final imports in a FRESH process, avoiding
        # mixing cached NumPy 2.4 modules with newly installed NumPy 2.2 files.
        if metadata.version("torch") != before[0] or metadata.version("torchvision") != before[1]:
            raise SystemExit("Dependency installation changed the protected CUDA torch stack")
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--expected-torch", before[0], "--expected-torchvision", before[1]], check=True)
        return
    for package in ("diffusers", "peft", "opacus", "torchmetrics", "torch_fidelity", "fdit"):
        __import__(package)
    assert metadata.version("torch") == before[0]
    assert metadata.version("torchvision") == before[1]
    print("Imports passed; CUDA PyTorch stack preserved:", before)


if __name__ == "__main__":
    main()
