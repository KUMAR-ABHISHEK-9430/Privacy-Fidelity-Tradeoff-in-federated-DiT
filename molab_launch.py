"""Run from marimo with subprocess.run([sys.executable, 'molab_launch.py'], check=True)."""
from pathlib import Path
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", help="Optional source repository to clone if --directory is absent")
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not args.directory.exists():
        if not args.repo_url:
            parser.error("--repo-url required for cloning")
        subprocess.run(["git", "clone", args.repo_url, str(args.directory)], check=True)
    subprocess.run(["bash", "run_all.sh"], cwd=args.directory, check=True)


if __name__ == "__main__":
    main()
