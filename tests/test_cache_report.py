import csv
import json
import tarfile
from types import SimpleNamespace

import numpy as np
import torch

from fdit.latent_cache import build_cache
from fdit.report import aggregate
from fdit.config import load_config


class ToyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=.25)
        self.encodes = 0

    def encode(self, images):
        assert not torch.is_grad_enabled()
        self.encodes += 1
        mean = images.mean(1, keepdim=True)[:, :, ::8, ::8].repeat(1, 4, 1, 1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=mean))


def test_mmap_cache_fingerprints_and_reuse(tmp_path):
    dataset = [(torch.ones(3, 256, 256) * k, 0) for k in range(5)]
    vae = ToyVAE()
    metadata = {"vae_id": "toy/vae", "model_revision": "revision1"}
    cache, key = build_cache(dataset, list(range(5)), vae, metadata, tmp_path, "split", shard_size=3)
    assert vae.encodes == 2
    torch.testing.assert_close(cache.batch([4, 1], device="cpu")[:, 0, 0, 0], torch.tensor([1., .25]))
    again, same = build_cache(dataset, list(range(5)), vae, metadata, tmp_path, "split", shard_size=3)
    assert same == key and vae.encodes == 2
    assert isinstance(np.load(cache.root / "000000.npy", mmap_mode="r"), np.memmap)
    metadata["model_revision"] = "revision2"
    _, changed = build_cache(dataset, list(range(5)), vae, metadata, tmp_path, "split", shard_size=3)
    assert changed != key


def test_partial_six_row_report_and_safe_archive(tmp_path):
    config = load_config("configs/experiment.yaml")
    manifest = {"preset": "fast", "status": "INCOMPLETE", "error": "test blocker",
                "conditions": [dict(c, status="pending") for c in config["conditions"]]}
    root = tmp_path / "results"
    root.mkdir()
    # Wrong file types are not included; real datasets live outside results.
    (root / "private.npy").write_bytes(b"excluded")
    archive = aggregate(root, manifest)
    rows = list(csv.DictReader((root / "summary.csv").open()))
    assert len(rows) == 6 and all(r["status"] == "pending" for r in rows)
    assert "test blocker" in (root / "summary.md").read_text()
    assert json.loads((root / "manifest.json").read_text())["status"] == "INCOMPLETE"
    with tarfile.open(archive) as tar:
        assert not any(n.endswith(".npy") for n in tar.getnames())
    assert (root / "plots/quality_vs_epsilon.png").exists()
