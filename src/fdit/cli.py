import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback

import torch
import yaml

from .checkpoints import atomic_json
from .config import load_config, fingerprint


class RuntimeGuard:
    def __init__(self):
        minutes = float(os.environ.get("MAX_RUNTIME_MINUTES", "570"))
        self.deadline = time.monotonic() + minutes * 60 if minutes > 0 else float("inf")
        self.stopped = False
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.stop)

    def stop(self, *_):
        self.stopped = True

    def __call__(self):
        if self.stopped or time.monotonic() >= self.deadline:
            raise InterruptedError("Runtime guard/signal: resume from latest durable logical step")


def run(args):
    from .preflight import cuda_preflight, smoke
    from .model import load_components
    from .lora import inject, adapter_state
    from .dataset import load_dataset, split_manifest
    from .partition import partition, statistics
    from .latent_cache import build_cache
    from .federated import train
    from .evaluate import evaluate
    from .plots import partition_plot
    from .report import aggregate
    from .backup import backup
    from safetensors.torch import save_file
    config = load_config(args.config, args.preset)
    root = Path(config["results_root"])
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"preset": args.preset, "config_hash": fingerprint(config),
                "status": "INCOMPLETE", "conditions": [dict(c, status="pending") for c in config["conditions"]]}
    # Preserve separate suite manifests so switching fast/full needs no YAML edits.
    mpath = root / "suites" / manifest["config_hash"] / "manifest.json"
    if mpath.exists():
        old = json.loads(mpath.read_text())
        if not args.resume:
            raise RuntimeError("Results exist. Use --resume")
        manifest = old
    manifest["status"] = "INCOMPLETE"
    if not args.smoke:
        atomic_json(root / "manifest.json", manifest)
    guard = RuntimeGuard()
    current = None
    try:
        print("FAST: low-budget pilot, not publishable definitive results" if args.preset == "fast" else "FULL: finite six-condition protocol")
        hardware = cuda_preflight()
        model, vae, scheduler, metadata = load_components(config)
        torch.manual_seed(config["seed"])
        model = inject(model, config["lora_rank"], config["lora_alpha"])
        initial = adapter_state(model)
        metadata.update(hardware=hardware, trainable_parameters=sum(v.numel() for v in initial.values()),
                        caveat="No guarantee every frozen component has never seen an overlapping image")
        smoke(model, vae, scheduler, config)
        if args.smoke:
            return
        guard()
        dataset = load_dataset(config["data_root"])
        split = split_manifest(dataset, config["seed"])
        # Record original IDs only in private/simulation cache, never in backup archive.
        atomic_json(Path(config["cache_root"]) / "split_manifest.json", split)
        parts = {kind: partition(split["indices"]["train"], dataset.targets, kind,
                                 config["partition_seed"], config["dirichlet_alpha"])
                 for kind in ("iid", "non_iid")}
        stats = {k: statistics(v, dataset.targets) for k, v in parts.items()}
        if stats["non_iid"]["mean_js"] < stats["iid"]["mean_js"] + .05:
            raise RuntimeError("Actual label skew is insufficient; refusing misleading comparison")
        atomic_json(root / "client_partition_stats.json", stats)
        partition_plot(stats, root / "client_partitions.png")
        cache, cache_hash = build_cache(dataset, split["indices"]["train"], vae, metadata,
                                        config["cache_root"], split["hash"], guard)
        for cell in manifest["conditions"]:
            current = cell
            identity = fingerprint({"config": config, "condition": cell["id"],
                                    "model_revision": metadata["model_revision"], "split": split["hash"],
                                    "cache": cache_hash, "partition": parts[cell["partition"]]})
            run_id = f"{cell['id']}_{identity[:12]}"
            if cell.get("run_id", run_id) != run_id:
                raise RuntimeError("Model/data fingerprint changed; refusing mixed results")
            cell["run_id"] = run_id
            if cell["status"] == "completed":
                required = ("fid_kid.json", "global_lora.safetensors", "privacy.json", "metadata.json", "checkpoint.pt")
                if all((root / run_id / name).exists() for name in required):
                    continue
                # A completed manifest cannot conceal missing or lost outputs.
                cell["status"] = "interrupted"
            folder = root / run_id
            folder.mkdir(parents=True, exist_ok=True)
            cell["status"] = "running"
            cell.pop("error", None)
            atomic_json(mpath, manifest)
            atomic_json(root / "manifest.json", manifest)
            (folder / "resolved_config.yaml").write_text(yaml.safe_dump(dict(config, condition={k: cell[k] for k in ("id", "partition", "dp", "epsilon") if k in cell})))
            atomic_json(folder / "metadata.json", dict(metadata, identity=identity, split_hash=split["hash"], cache_hash=cache_hash,
                                                       preprocess=split["preprocess"]))
            atomic_json(folder / "client_partition_stats.json", stats[cell["partition"]])
            perf = train(model, scheduler, cache, parts[cell["partition"]], config, cell, initial,
                         folder, identity, guard, args.resume)
            atomic_json(folder / "performance.json", {k: v for k, v in perf.items() if k != "privacy"})
            tmp = folder / "global_lora.tmp.safetensors"
            save_file(adapter_state(model), str(tmp))
            tmp.replace(folder / "global_lora.safetensors")
            evaluation_started = time.monotonic()
            metrics = evaluate(model, vae, scheduler, dataset, split["indices"], config, folder, guard)
            perf["evaluation_seconds"] = time.monotonic() - evaluation_started
            perf["wall_clock_seconds"] = perf["training_seconds"] + perf["evaluation_seconds"]
            if next(model.parameters()).device.type == "cuda":
                perf["maximum_gpu_memory_bytes"] = max(perf["maximum_gpu_memory_bytes"], torch.cuda.max_memory_allocated())
            atomic_json(folder / "performance.json", {k: v for k, v in perf.items() if k != "privacy"})
            atomic_json(folder / "fid_kid.json", metrics)
            cell["status"] = "completed"
            backup(aggregate(root, manifest))
        manifest["status"] = "COMPLETE"
        manifest.pop("error", None)
    except BaseException as exc:
        if current is not None and current["status"] != "completed":
            current["status"] = "interrupted" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed"
            current["error"] = f"{type(exc).__name__}: {exc}"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        (root / "diagnostics.txt").write_text(traceback.format_exc())
        raise
    finally:
        if not args.smoke:
            if manifest["status"] != "COMPLETE":
                (root / "INCOMPLETE").write_text("Resume with bash run_all.sh\n")
            else:
                (root / "INCOMPLETE").unlink(missing_ok=True)
            atomic_json(mpath, manifest)
            backup(aggregate(root, manifest))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "diagnose"])
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--preset", default="fast", choices=["fast", "full"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=1000, help="Centralized diagnostic stopping step; 500 or 1000 for initial tests")
    parser.add_argument("--federated-results", help="Read-only original results directory for diagnostic adapter comparisons")
    args = parser.parse_args()
    if args.command == "diagnose":
        if args.smoke:
            parser.error("Use run --smoke for isolated pretrained preflight")
        from .diagnostics import run as diagnose
        diagnose(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
