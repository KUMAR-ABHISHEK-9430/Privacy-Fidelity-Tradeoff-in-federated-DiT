import argparse
import json
from types import SimpleNamespace

import torch

from fdit import cli
from fdit.config import load_config
from test_lora_weights import TinyAttention
from test_dp_reference import loss


def test_six_cell_runner_and_completed_resume(tmp_path, monkeypatch):
    import fdit.preflight as preflight
    import fdit.model as model_api
    import fdit.dataset as datasets
    import fdit.latent_cache as caches
    import fdit.evaluate as evaluation
    import fdit.federated as fed
    import fdit.backup as backups
    config = load_config("configs/experiment.yaml")
    config.update(rounds=1, local_steps_per_round=2, expected_logical_batch_size=2,
                  lora_rank=1, lora_alpha=1,
                  eval_samples_per_condition=2, results_root=str(tmp_path / "results"),
                  cache_root=str(tmp_path / "cache"))
    monkeypatch.setattr(cli, "load_config", lambda *args: config)
    monkeypatch.setattr(preflight, "cuda_preflight", lambda: {"versions": {"torch": torch.__version__}})
    metadata = {"model_revision": "toy-test-only", "vae_id": "toy"}
    monkeypatch.setattr(model_api, "load_components", lambda c: (TinyAttention(), None, None, metadata.copy()))
    monkeypatch.setattr(preflight, "smoke", lambda *args: None)
    dataset = SimpleNamespace(targets=[k for k in range(10) for _ in range(30)],
                              samples=[(f"{k}/{i}.jpg", k) for k in range(10) for i in range(30)])
    monkeypatch.setattr(datasets, "load_dataset", lambda root: dataset)
    data = torch.randn(300, 2, 4, generator=torch.Generator().manual_seed(2))
    class Cache:
        def batch(self, ids, device="cpu"):
            return data[ids].to(device)
    monkeypatch.setattr(caches, "build_cache", lambda *args: (Cache(), "toy-cache"))
    monkeypatch.setattr(fed, "loss_vector", lambda m, s, x: loss(m, x))
    evaluated = []
    def evaluate(*args):
        evaluated.append(str(args[-2]))
        return {"fid": 42., "kid_mean": .01, "generated_count": 2}
    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    monkeypatch.setattr(backups, "backup", lambda archive: None)
    starts = []
    real_train = fed.train
    def train(*args, **kwargs):
        starts.append({k: v.clone() for k, v in args[6].items()})
        return real_train(*args, **kwargs)
    monkeypatch.setattr(fed, "train", train)
    args = argparse.Namespace(config="unused", preset="fast", resume=True, smoke=False)
    cli.run(args)
    manifest = json.loads((tmp_path / "results/manifest.json").read_text())
    assert manifest["status"] == "COMPLETE"
    assert len(manifest["conditions"]) == 6
    assert all(c["status"] == "completed" for c in manifest["conditions"])
    assert len(starts) == len(evaluated) == 6
    for state in starts[1:]:
        for name in state:
            torch.testing.assert_close(state[name], starts[0][name], rtol=0, atol=0)
    cli.run(args)
    assert len(starts) == len(evaluated) == 6  # Finished conditions really skip.


def test_preflight_blocker_writes_incomplete_six_row_manifest(tmp_path, monkeypatch):
    import pytest
    import fdit.preflight as preflight
    import fdit.backup as backups
    config = load_config("configs/experiment.yaml")
    config["results_root"] = str(tmp_path / "results")
    monkeypatch.setattr(cli, "load_config", lambda *args: config)
    def fail():
        raise RuntimeError("test CUDA compatibility blocker")
    monkeypatch.setattr(preflight, "cuda_preflight", fail)
    monkeypatch.setattr(backups, "backup", lambda archive: None)
    args = argparse.Namespace(config="unused", preset="fast", resume=True, smoke=False)
    with pytest.raises(RuntimeError, match="compatibility blocker"):
        cli.run(args)
    root = tmp_path / "results"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "INCOMPLETE"
    assert len(manifest["conditions"]) == 6
    assert all(c["status"] == "pending" for c in manifest["conditions"])
    assert (root / "INCOMPLETE").exists()
    assert "test CUDA compatibility blocker" in (root / "diagnostics.txt").read_text()
