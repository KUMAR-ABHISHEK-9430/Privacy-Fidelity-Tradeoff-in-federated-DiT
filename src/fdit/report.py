import csv
import json
import tarfile
from pathlib import Path

from .checkpoints import atomic_json
from .plots import quality_plots

CAVEATS = """This is simulated record-level privacy on public EuroSAT, with add/remove adjacency and fixed disjoint clients. The server observes private client updates; no secure aggregation or client-level DP is claimed. All rounds compose within each client, and the disclosure bound is max client epsilon. Label-dependent partition construction/histograms are public simulation setup, outside the private mechanism. Frozen pretrained components have no guaranteed zero overlap. LoRA matrix averaging differs from dense-update averaging. Resizing 64x64 images to 256x256 adds no detail. Generated images are unconditional. Fast mode is a low-budget pilot. FID/KID with 96 or 512 samples on satellite images are noisy diagnostics. Timing and GPU memory are operational diagnostics outside the formal privacy claim. Non-cryptographic RNG is not production protection."""


def aggregate(root, manifest):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for cell in manifest["conditions"]:
        row = {"condition": cell["id"], "partition": cell["partition"], "status": cell["status"],
               "run_id": cell.get("run_id", ""), "target_epsilon": cell.get("epsilon", ""),
               "epsilon": None, "fid": None, "kid_mean": None, "generated_count": None,
               "training_seconds": None, "evaluation_seconds": None, "wall_clock_seconds": None,
               "maximum_gpu_memory_bytes": None,
               "warmup_adjusted_steps_per_second": None, "error": cell.get("error", "")}
        folder = root / cell.get("run_id", "__missing__")
        for name in ("fid_kid.json", "performance.json"):
            if (folder / name).exists():
                values = json.loads((folder / name).read_text())
                row.update({k: values[k] for k in row if k in values})
        if cell["dp"] and (folder / "privacy.json").exists():
            row["epsilon"] = json.loads((folder / "privacy.json").read_text())["epsilon"]
        rows.append(row)
    atomic_json(root / "manifest.json", manifest)
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [f"# Federated DiT {manifest['preset']} report", "", CAVEATS, "",
            f"Suite status: {manifest['status']}. {manifest.get('error', '')}", "",
            "Paired comparisons use identical initial adapters, splits, training/sampling seeds and budgets.", "",
            "| Condition | Status | Computed epsilon | FID | KID | Generated |",
            "|---|---|---|---|---|---|"]
    for r in rows:
        text.append(f"| {r['condition']} | {r['status']} | {r['epsilon']} | {r['fid']} | {r['kid_mean']} | {r['generated_count']} |")
        if r["error"]:
            text.append(f"\n{r['condition']}: {r['error']}\n")
    text.extend(["", "See summary.csv and each performance.json for measured hardware timings and memory.",
                 "Missing conditions do not support conclusions; no-DP runs have no reported epsilon."])
    (root / "summary.md").write_text("\n".join(text), encoding="utf-8")
    quality_plots(rows, root / "plots")
    archive = root.parent / "results_bundle.tar.gz"
    tmp = archive.with_suffix(".tmp")
    # Only allowlisted compact results; never data/cache or arbitrary files.
    allowed = {".json", ".yaml", ".csv", ".md", ".png", ".safetensors", ".pt", ".txt"}
    with tarfile.open(tmp, "w:gz") as tar:
        for path in sorted(root.rglob("*")):
            if path.is_file() and (path.suffix in allowed or path.name == "INCOMPLETE"):
                tar.add(path, arcname=str(Path("results") / path.relative_to(root)))
    tmp.replace(archive)
    return archive
