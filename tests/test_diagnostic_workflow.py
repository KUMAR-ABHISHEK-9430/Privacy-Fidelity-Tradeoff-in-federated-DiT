import argparse
import json
from types import SimpleNamespace

import torch
import yaml
from safetensors.torch import save_file

from fdit import diagnostics as diag
from fdit.config import load_config
from fdit.dataset import split_manifest
from fdit.lora import adapter_state, inject
from fdit.model import verify_components
from test_diffusion_shapes import tiny_components


def test_diagnostic_workflow_is_paired_resumable_and_never_trains_federated(tmp_path, monkeypatch):
    import fdit.federated as federated
    import fdit.backup as backup
    config = load_config("configs/experiment.yaml")
    config.update(lora_rank=1, lora_alpha=1, cache_root=str(tmp_path / "cache"))
    settings = config["diagnostic"]
    settings.update(output_root=str(tmp_path / "diagnostics"), federated_conditions=["iid_nodp"], max_steps=2,
                    milestones=[1, 2], checkpoint_interval=1, grid_interval=1, log_interval=1,
                    batch_size=4, physical_batch_size=2, eval_samples=4, generation_batch_size=2,
                    validation_samples=4, vae_reconstruction_samples=4, preview_samples=2, aggregation_local_steps=1)
    dataset = SimpleNamespace(targets=[k for k in range(10) for _ in range(30)],
                              samples=[(f"{k}/{i}.jpg", k) for k in range(10) for i in range(30)])
    split = split_manifest(dataset, config["seed"])
    def components(c):
        torch.manual_seed(1)
        model, vae, scheduler = tiny_components()
        model.requires_grad_(False)
        vae.requires_grad_(False)
        meta = verify_components(model, vae, scheduler)
        meta.update(model_revision="test-revision", vae_id="toy/vae")
        return model, vae, scheduler, meta
    model, _, _, meta = components(config)
    model = inject(model, 1, 1)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.fill_(.01)
    source = tmp_path / "results"
    ref = source / "iid_nodp_test"
    ref.mkdir(parents=True)
    save_file(adapter_state(model), str(ref / "global_lora.safetensors"))
    (ref / "metadata.json").write_text(json.dumps(dict(meta, split_hash=split["hash"])))
    (ref / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (source / "manifest.json").write_text(json.dumps({"conditions": [{"id": "iid_nodp", "status": "completed", "run_id": ref.name}]}))
    source_bytes = (source / "manifest.json").read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("The diagnostic invoked federated training")
    monkeypatch.setattr(federated, "train", forbidden)
    monkeypatch.setattr(diag, "load_config", lambda *args: config)
    monkeypatch.setattr(diag, "load_components", components)
    monkeypatch.setattr(diag, "cuda_preflight", lambda: {"versions": {"torch": torch.__version__}})
    monkeypatch.setattr(diag, "load_dataset", lambda root: dataset)
    monkeypatch.setattr(backup, "backup", lambda archive: None)
    z = torch.randn(300, 4, 8, 8, generator=torch.Generator().manual_seed(5))
    class Cache:
        def batch(self, ids, device="cpu"):
            return z[ids].to(device)
    monkeypatch.setattr(diag, "build_cache", lambda *args: (Cache(), "toy-cache"))
    monkeypatch.setattr(diag, "vae_reconstruction", lambda *args: {"adequate_by_configured_psnr_heuristic": True})
    monkeypatch.setattr(diag, "validation_bank", lambda *args: {"latents": z[:4], "noise": torch.ones_like(z[:4]), "timesteps": torch.arange(4)})
    banks = []
    def evaluate(model, vae, scheduler, dataset, split, cfg, folder, guard, noise_bank, validation_bank, baseline_preview):
        banks.append(noise_bank.clone())
        with torch.no_grad():
            images = diag.probe_predict(model, noise_bank[:cfg["preview_samples"]])[:, :3].sigmoid()
        save_file({"images": images}, str(folder / "paired_samples.safetensors"))
        return {"fid": 335., "kid_mean": .12, "kid_std": .01, "generated_count": len(noise_bank),
                "public_validation_denoising_loss": .09, "paired_pixel_mae_vs_pretrained_preview": None}
    monkeypatch.setattr(diag, "evaluate", evaluate)
    args = argparse.Namespace(config="unused", steps=2, resume=True, federated_results=str(source))
    diag.run(args)
    output = next((tmp_path / "diagnostics").glob("dit_diagnostic_*"))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed_requested_steps" and manifest["central_steps"] == 2
    assert set(manifest["comparisons"]) == {"pretrained", "federated_iid_nodp"}
    assert set(manifest["central_checkpoints"]) == {"1", "2"}
    assert (output / "loss_and_fidelity_curves.png").exists()
    assert len(banks) == 4 and all(torch.equal(bank, banks[0]) for bank in banks)
    assert (source / "manifest.json").read_bytes() == source_bytes
    diag.run(args)
    assert len(banks) == 4  # Completed comparisons were skipped on resume.


def test_adaptation_assessment_does_not_claim_semantic_success():
    baseline = {"pretrained": {"metrics": {"fid": 300., "public_validation_denoising_loss": .1}}}
    central = {"500": {"metrics": {"fid": 200., "public_validation_denoising_loss": .08}}}
    result = diag.assessment(baseline, central, {"adequate_by_configured_psnr_heuristic": True},
                             {"min_relative_fid_improvement": .05, "min_relative_validation_improvement": .02})
    assert result["status"] == "numerical_improvement_candidate"
    assert "not established" in result["meaningful_domain_adaptation"]
