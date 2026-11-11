from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def partition_plot(stats, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, value) in zip(axes, stats.items()):
        data = np.array(value["frequencies"])
        bottom = np.zeros(3)
        for label in range(10):
            ax.bar(range(3), data[:, label], bottom=bottom, label=str(label))
            bottom += data[:, label]
        ax.set(title=f"{name}: mean JS={value['mean_js']:.3f} bits", xlabel="Client", ylabel="Class frequency", xticks=range(3))
    axes[-1].legend(ncol=5, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def quality_plots(rows, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for part in ("iid", "non_iid"):
        values = [r for r in rows if r["partition"] == part and r.get("epsilon") is not None and r.get("fid") is not None]
        ax.plot([r["epsilon"] for r in values], [r["fid"] for r in values], "o-", label=part)
    ax.set(xlabel="Computed record epsilon (DP cells only)", ylabel="Diagnostic FID")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "quality_vs_epsilon.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    values = [r for r in rows if r.get("fid") is not None]
    ax.bar([r["condition"] for r in values], [r["fid"] for r in values])
    ax.set(ylabel="Diagnostic FID", title="IID versus label skew; missing cells omitted")
    fig.tight_layout()
    fig.savefig(root / "iid_vs_non_iid.png")
    plt.close(fig)
