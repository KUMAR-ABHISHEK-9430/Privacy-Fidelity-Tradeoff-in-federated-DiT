"""One-command, non-private diagnostic workflow; never invokes the factorial suite."""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
import traceback

import torch
import yaml
from safetensors.torch import load_file, save_file

from .audit import (activation_audit, adapters_disabled, aggregation_round_audit,
                    frozen_fingerprint, gradient_update_audit, preserve_rng,
                    probe_predict, schedule_audit, state_distance)
from .centralized import train_centralized
from .checkpoints import atomic_json
from .config import fingerprint, load_config
from .dataset import load_dataset, split_manifest
from .diagnostic_data import (balanced_public_indices, initial_noise_bank, tensor_hash, validation_bank,
                              validation_loss, vae_reconstruction)
from .diffusion import generate
from .evaluate import evaluate
from .latent_cache import build_cache
from .lora import adapter_state, inject, load_adapter
from .model import load_components
from .partition import partition
from .preflight import cuda_preflight
from .plots import diagnostic_curves


def diagnostic_settings(config, requested_steps):
    settings = dict(config["diagnostic"])
    for key in ("max_steps", "checkpoint_interval", "grid_interval", "log_interval", "batch_size",
                "physical_batch_size", "generation_batch_size", "eval_samples", "validation_samples",
                "vae_reconstruction_samples", "probe_steps", "aggregation_local_steps", "preview_samples"):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"diagnostic.{key} must be a positive integer")
    if requested_steps < 1 or requested_steps > settings["max_steps"]:
        raise ValueError("Requested diagnostic steps must be within diagnostic.max_steps")
    if settings["probe_steps"] < 2:
        raise ValueError("Two probe steps are needed to test A gradients after zero-B initialization")
    if settings["precision"] not in ("fp32", "bf16"):
        raise ValueError("Unsupported diagnostic precision")
    if not 2 <= settings["eval_samples"] or settings["preview_samples"] > min(64, settings["eval_samples"]):
        raise ValueError("Invalid diagnostic evaluation/preview counts")
    if any(not isinstance(s, int) or not 1 <= s <= settings["max_steps"] for s in settings["milestones"]):
        raise ValueError("Invalid diagnostic evaluation milestones")
    settings.update({k: config[k] for k in ("seed", "sampler_seed", "noise_seed", "evaluation_seed")})
    return settings


def references(root, names):
    root = Path(root).resolve()
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Restore the original results directory first: {path}; diagnostics do not retrain federated models")
    manifest = json.loads(path.read_text())
    by_id = {c["id"]: c for c in manifest["conditions"]}
    result = []
    if not names:
        raise ValueError("Configure at least one existing federated condition")
    for name in names:
        cell = by_id.get(name)
        if cell is None or cell["status"] != "completed":
            raise ValueError(f"Federated reference {name} is missing/incomplete")
        if Path(cell["run_id"]).name != cell["run_id"]:
            raise ValueError("Unsafe reference run ID")
        folder = root / cell["run_id"]
        weights = folder / "global_lora.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(weights)
        meta = json.loads((folder / "metadata.json").read_text())
        resolved = yaml.safe_load((folder / "resolved_config.yaml").read_text())
        result.append({"condition": name, "folder": folder, "weights": weights, "metadata": meta,
                       "config": resolved, "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest()})
    if len({r["metadata"]["model_revision"] for r in result}) != 1:
        raise ValueError("Federated references use different pretrained revisions")
    return result


def atomic_adapter(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file(adapter_state(model), str(temporary))
    temporary.replace(path)


def inference_roundtrip(model, path, initial, probe):
    with torch.no_grad():
        expected = probe_predict(model, probe)
        trained = adapter_state(model)
        load_adapter(model, initial)
        load_adapter(model, load_file(str(path)))
        actual = probe_predict(model, probe)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    return {"safetensors_roundtrip_prediction_maxabs": float((actual - expected).abs().max()),
            "adapter_distance_from_initial": state_distance(initial, trained),
            **activation_audit(model, probe, require_effect=True)}


def assessment(entries, central, vae, settings):
    baseline = entries.get("pretrained", {}).get("metrics")
    complete = [(int(step), value) for step, value in central.items() if value.get("metrics")]
    if not baseline or not complete:
        return {"status": "not_evaluated", "meaningful_domain_adaptation": "not established"}
    step, latest = max(complete)
    quality = latest["metrics"]
    fid_gain = (baseline["fid"] - quality["fid"]) / max(abs(baseline["fid"]), 1e-12)
    val_gain = (baseline["public_validation_denoising_loss"] - quality["public_validation_denoising_loss"]) / max(abs(baseline["public_validation_denoising_loss"]), 1e-12)
    improved = (fid_gain >= settings["min_relative_fid_improvement"]
                and val_gain >= settings["min_relative_validation_improvement"])
    return {"status": "numerical_improvement_candidate" if improved else "no_clear_numerical_adaptation",
            "central_step": step, "relative_fid_improvement": fid_gain,
            "relative_public_validation_improvement": val_gain,
            "vae_adequate_by_psnr_heuristic": vae.get("adequate_by_configured_psnr_heuristic"),
            "thresholds_are_heuristics_not_significance_tests": True,
            "meaningful_domain_adaptation": "not established without visual target-domain review and repeated-seed uncertainty",
            "note": "Centralized and historical federated models have different training budgets; this is a learning feasibility diagnostic, not a controlled algorithm ranking"}


def write_report(folder, manifest, settings):
    rows = []
    for name, entry in manifest.get("comparisons", {}).items():
        if entry.get("metrics"):
            rows.append({"model": name, **entry["metrics"]})
    for step, entry in manifest.get("central_checkpoints", {}).items():
        if entry.get("metrics"):
            rows.append({"model": f"central_step_{step}", **entry["metrics"]})
    fields = ["model", "fid", "kid_mean", "kid_std", "public_validation_denoising_loss",
              "generated_count", "paired_pixel_mae_vs_pretrained_preview"]
    with (folder / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    result = assessment(manifest.get("comparisons", {}), manifest.get("central_checkpoints", {}),
                        manifest.get("vae_reconstruction", {}), settings)
    atomic_json(folder / "adaptation_assessment.json", result)
    lines = ["# DiT adaptation diagnostic", "", f"Status: **{manifest['status']}**. {manifest.get('error', '')}", "",
             "This workflow is NON-PRIVATE. It never trains the six-condition federated suite or changes privacy accounting.",
             "All reevaluated models use the same CPU-generated per-image noise bank, held-out real bank, scheduler and FP32 evaluation.", "",
             "| Model | FID | KID | Public validation MSE | Images | Paired pixel MAE vs pretrained preview |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['fid']:.5f} | {row['kid_mean']:.7f} | {row['public_validation_denoising_loss']:.8f} | {row['generated_count']} | {row.get('paired_pixel_mae_vs_pretrained_preview')} |")
    lines += ["", "## Interpretation", "", f"Numerical assessment: **{result['status']}**.",
              f"Meaningful domain adaptation: **{result['meaningful_domain_adaptation']}**.",
              "Inspect baseline, central 500/1000-step and federated grids for actual overhead land-cover appearance. Small-sample FID/KID and denoising MSE alone do not establish that semantic change.",
              "The stored first pilot remained ImageNet-like; the current comparison regenerates all available models under a common evaluation protocol rather than reusing its 96-image FID scores.",
              "", "## Audit artifacts", "",
              "- `component_audit.json`: null label, VAE scale/channels, matching betas, closed-form noising and epsilon/variance interpretation.",
              "- `gradient_audit.json`: temporary non-private TRAIN-record gradient/update check; zero initial A gradient is expected until B updates.",
              "- `aggregation_audit.json`: a fresh tiny non-private three-client replay comparing factor FedAvg and exact mean dense updates; not a retrospective reconstruction of discarded private client snapshots.",
              "- Each model folder: `activation_audit.json`, same-seed grid and fidelity metrics; centralized checkpoints also test saved-adapter reload predictions.",
              "- `vae_reconstruction.json` and `public_test_original_reconstruction.png`: public TEST-only reconstruction PSNR/SSIM and paired image grid.",
              "- `central/training_history.csv`: non-private mean training loss, gradient/update norms, timings; `validation_history.csv`: fixed public validation curve.",
              "- `frozen_audit.json`: hashes of all frozen DiT/VAE parameters and buffers before/after diagnostics.", "",
              "## Limitations", "",
              "A/B averaging is not dense-update averaging. A measured gap does not prove that it caused the historical image quality problem. Production aggregation remains unchanged.",
              "A zero-shot comparison isolates initialization; centralized versus historical federated results still differ in steps, exposure, optimizer resets and precision. Nine-cell IID/70-30/90-10 experiments remain outside this diagnostic.",
              "The original private-run epsilon records are not a guarantee for the combined diagnostic bundle, which includes non-private training and public reconstruction imagery.",
              "Checkpoint optimizer/RNG states are trusted continuation files; do not publish them as DP releases. Test originals are from public EuroSAT; no original training images are exported."]
    (folder / "diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    diagnostic_curves(folder, manifest)
    return result


def archive_diagnostic(folder):
    archive = folder.parent / f"{folder.name}.tar.gz"
    temporary = archive.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as tar:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and ".tmp" not in path.name:
                tar.add(path, arcname=str(Path(folder.name) / path.relative_to(folder)))
    temporary.replace(archive)
    return archive


def run(args):
    from .cli import RuntimeGuard
    from .backup import backup
    config = load_config(args.config, "fast")
    settings = diagnostic_settings(config, args.steps)
    refs = references(args.federated_results or settings["federated_results"], settings["federated_conditions"])
    # Pin the exact original backbone instead of silently evaluating a newer main revision.
    config["model_revision"] = refs[0]["metadata"]["model_revision"]
    for ref in refs:
        for key in ("model_id", "lora_rank", "lora_alpha"):
            if ref["config"][key] != config[key]:
                raise ValueError(f"Federated reference {key} differs from diagnostic setup")
    output_root = Path(settings["output_root"]).resolve()
    source_root = refs[0]["folder"].parent.resolve()
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("Diagnostics must write outside the historical results tree")
    hardware = cuda_preflight()
    source_hash = fingerprint({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(Path(__file__).parent.glob("*.py"))})
    identity = fingerprint({"config": config, "hardware_versions": hardware["versions"], "source": source_hash,
                            "references": {r["condition"]: r["weights_sha256"] for r in refs}})
    folder = output_root / f"dit_diagnostic_{identity[:12]}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "manifest.json"
    manifest = {"identity": identity, "status": "running", "requested_steps": args.steps,
                "planned_max_steps": settings["max_steps"], "comparisons": {}, "central_checkpoints": {}}
    if path.exists():
        if not args.resume:
            raise FileExistsError("Diagnostic exists; use --resume")
        manifest = json.loads(path.read_text())
        if manifest["identity"] != identity:
            raise ValueError("Diagnostic fingerprint mismatch")
    manifest.update(status="running", requested_steps=args.steps)
    manifest.pop("error", None)
    atomic_json(path, manifest)
    (folder / "resolved_config.yaml").write_text(yaml.safe_dump(dict(config, diagnostic_resolved=settings)), encoding="utf-8")
    guard = RuntimeGuard()
    try:
        print(f"[diagnostic] Output: {folder}; no federated training will run", flush=True)
        model, vae, scheduler, metadata = load_components(config)
        torch.manual_seed(config["seed"])
        model = inject(model, config["lora_rank"], config["lora_alpha"])
        initial = adapter_state(model)
        print("[audit] Hashing frozen backbone and VAE before training", flush=True)
        base_hash, vae_hash = frozen_fingerprint(model), frozen_fingerprint(vae)
        dataset = load_dataset(config["data_root"])
        split = split_manifest(dataset, config["seed"])
        if any(r["metadata"]["split_hash"] != split["hash"] for r in refs):
            raise ValueError("Original federated and diagnostic EuroSAT splits differ")
        for ref in refs:
            original = ref["metadata"]
            if original["null_class_id"] != 1000 or original["vae_scaling_factor"] != metadata["vae_scaling_factor"]:
                raise ValueError("Federated reference VAE/null-class mismatch")
            for key in ("num_train_timesteps", "beta_start", "beta_end", "beta_schedule", "prediction_type",
                        "trained_betas", "set_alpha_to_one", "steps_offset", "timestep_spacing", "clip_sample"):
                if original["scheduler"][key] != metadata["scheduler"][key]:
                    raise ValueError("Federated reference scheduler mismatch")
        metadata.update(hardware=hardware, source_fingerprint=source_hash, split_hash=split["hash"],
                        diagnostic_identity=identity, actual_eval_precision="fp32", actual_train_precision=settings["precision"])
        atomic_json(folder / "metadata.json", metadata)
        validation_ids = balanced_public_indices(split["indices"]["val"], dataset.targets,
                                                 settings["validation_samples"], config["evaluation_seed"])
        reconstruction_ids = balanced_public_indices(split["indices"]["test"], dataset.targets,
                                                     settings["vae_reconstruction_samples"], config["evaluation_seed"] + 1)
        print("[VAE] Auditing held-out public image reconstruction", flush=True)
        recon = vae_reconstruction(dataset, reconstruction_ids, vae, settings["vae_reconstruction_samples"],
                                   settings["generation_batch_size"], folder, settings["vae_min_psnr_db"], guard)
        manifest["vae_reconstruction"] = recon
        atomic_json(folder / "vae_reconstruction.json", recon)
        val = validation_bank(dataset, validation_ids, vae, scheduler, settings["validation_samples"],
                              config["evaluation_seed"], settings["generation_batch_size"], guard)
        bank = initial_noise_bank(settings["eval_samples"], model.config.in_channels, model.config.sample_size, config["evaluation_seed"])
        bank_info = {"initial_noise_sha256": tensor_hash(bank), "per_image_seeds": [config["evaluation_seed"] + i for i in range(len(bank))],
                     "validation_sha256": {k: tensor_hash(v) for k, v in val.items()},
                     "construction": "per-image CPU generators, shared across all models; same test reference permutation"}
        bank_info["public_validation_class_counts"] = {str(k): int(sum(dataset.targets[i] == k for i in validation_ids)) for k in sorted(set(dataset.targets))}
        bank_info["public_reconstruction_class_counts"] = {str(k): int(sum(dataset.targets[i] == k for i in reconstruction_ids)) for k in sorted(set(dataset.targets))}
        bank_info["public_bank_selection"] = "seeded macro-balanced held-out classes; avoids the old sorted first-96 class bias"
        atomic_json(folder / "paired_banks.json", bank_info)
        manifest["paired_banks"] = bank_info
        eval_config = dict(config, eval_samples_per_condition=settings["eval_samples"], sampling_steps=settings["sampling_steps"],
                           generation_batch_size=settings["generation_batch_size"], preview_samples=settings["preview_samples"])
        probe = bank[:5].to(next(model.parameters()).device)
        atomic_json(folder / "component_audit.json", schedule_audit(model, vae, scheduler, probe))
        zero = activation_audit(model, probe, require_effect=False)
        if zero["enabled_minus_disabled_maxabs"] != 0:
            raise RuntimeError("Initialized adapters alter the untouched pretrained model")
        cache, cache_hash = build_cache(dataset, split["indices"]["train"], vae, metadata, config["cache_root"], split["hash"], guard)
        manifest["cache_hash"] = cache_hash
        audit_ids = split["indices"]["train"][:settings["physical_batch_size"]]
        z = cache.batch(audit_ids, device=next(model.parameters()).device)
        atomic_json(folder / "gradient_audit.json", gradient_update_audit(model, scheduler, z, settings["learning_rate"], steps=settings["probe_steps"]))
        clients = partition(split["indices"]["train"], dataset.targets, "iid", config["partition_seed"])
        print("[audit] Tiny non-private three-client aggregation replay", flush=True)
        atomic_json(folder / "aggregation_audit.json", aggregation_round_audit(model, scheduler, cache, clients, settings, probe=probe))

        def evaluate_entry(name, entry, destination, disabled=False):
            destination.mkdir(parents=True, exist_ok=True)
            entry["folder"] = str(destination.relative_to(folder))
            if entry.get("status") == "completed":
                required = [destination / "fid_kid.json", destination / "paired_samples.safetensors"]
                if all(p.is_file() for p in required):
                    return entry["metrics"]
            guard()
            peer = None if name == "pretrained" else load_file(str(folder / "pretrained/paired_samples.safetensors"))["images"]
            print(f"[diagnostic] Paired evaluation: {name}", flush=True)
            with preserve_rng():
                if disabled:
                    with adapters_disabled(model):
                        quality = evaluate(model, vae, scheduler, dataset, split["indices"], eval_config, destination, guard,
                                           noise_bank=bank, validation_bank=val, baseline_preview=peer)
                else:
                    quality = evaluate(model, vae, scheduler, dataset, split["indices"], eval_config, destination, guard,
                                       noise_bank=bank, validation_bank=val, baseline_preview=peer)
            quality.update(paired_initial_noise_sha256=bank_info["initial_noise_sha256"], actual_eval_precision="fp32")
            atomic_json(destination / "fid_kid.json", quality)
            entry.update(status="completed", metrics=quality)
            atomic_json(path, manifest)
            return quality

        load_adapter(model, initial)
        baseline = manifest["comparisons"].setdefault("pretrained", {})
        baseline_metrics = evaluate_entry("pretrained", baseline, folder / "pretrained", disabled=True)
        atomic_json(folder / "pretrained/activation_audit.json", zero)
        for ref in refs:
            guard()
            name = "federated_" + ref["condition"]
            target = folder / name
            target.mkdir(exist_ok=True)
            load_adapter(model, load_file(str(ref["weights"])))
            effect = activation_audit(model, probe)
            atomic_json(target / "activation_audit.json", effect)
            if ref["condition"] == "iid_nodp":
                print("[audit] Factor-versus-dense replay starting from the existing IID non-private checkpoint", flush=True)
                replay = aggregation_round_audit(model, scheduler, cache, clients, settings, probe=probe)
                replay["starting_checkpoint_sha256"] = ref["weights_sha256"]
                atomic_json(folder / "aggregation_audit_from_federated.json", replay)
            entry = manifest["comparisons"].setdefault(name, {})
            entry.update(source_weights_sha256=ref["weights_sha256"], source_run_id=ref["folder"].name,
                         original_training_steps_per_client=ref["config"]["rounds"] * ref["config"]["local_steps_per_round"])
            evaluate_entry(name, entry, target)
        load_adapter(model, initial)

        def central_callback(step, trained):
            guard()
            destination = folder / "central" / f"step_{step:06d}"
            destination.mkdir(parents=True, exist_ok=True)
            entry = manifest["central_checkpoints"].setdefault(str(step), {})
            required = ("global_lora.safetensors", "preview_grid.png", "activation_audit.json")
            if not entry.get("preview_complete") or not all((destination / name).exists() for name in required):
                atomic_adapter(destination / "global_lora.safetensors", trained)
                value = validation_loss(trained, scheduler, val, settings["generation_batch_size"], guard)
                from torchvision.utils import save_image
                images = list(generate(trained, vae, scheduler, settings["preview_samples"], settings["sampling_steps"],
                                       config["evaluation_seed"], batch_size=settings["generation_batch_size"],
                                       initial_noise=bank[:settings["preview_samples"]], guard=guard))
                save_image(torch.cat(images), destination / "preview_grid.png", nrow=4)
                entry.update(preview_complete=True, public_validation_loss=value,
                             activation=inference_roundtrip(trained, destination / "global_lora.safetensors", initial, probe))
                atomic_json(destination / "activation_audit.json", entry["activation"])
                print(f"[central] Step {step}: fixed public validation MSE={value:.6f}", flush=True)
            if step in settings["milestones"] or step == args.steps:
                evaluate_entry(f"central_{step}", entry, destination)
            atomic_json(path, manifest)
            with (folder / "validation_history.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["step", "public_validation_mse"])
                writer.writerow([0, baseline_metrics["public_validation_denoising_loss"]])
                writer.writerows((s, e["public_validation_loss"]) for s, e in sorted(manifest["central_checkpoints"].items(), key=lambda pair: int(pair[0])) if "public_validation_loss" in e)
            write_report(folder, manifest, settings)

        train = train_centralized(model, scheduler, cache, split["indices"]["train"], settings, initial,
                                 folder / "central", identity, args.steps, guard, central_callback, args.resume)
        manifest["central_steps"] = train["steps"]
        print("[audit] Verifying all frozen backbone/VAE hashes after diagnostics", flush=True)
        final_base, final_vae = frozen_fingerprint(model), frozen_fingerprint(vae)
        atomic_json(folder / "frozen_audit.json", {"base_before": base_hash, "base_after": final_base,
                                                  "vae_before": vae_hash, "vae_after": final_vae})
        if final_base != base_hash or final_vae != vae_hash:
            raise RuntimeError("Frozen pretrained weights changed")
        manifest["status"] = "completed_requested_steps"
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        (folder / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        atomic_json(path, manifest)
        verdict = write_report(folder, manifest, settings)
        archive = archive_diagnostic(folder)
        print(f"[diagnostic] {manifest['status']}: {folder / 'diagnostic_report.md'}", flush=True)
        print(f"[diagnostic] Assessment: {verdict['status']}; download {archive}", flush=True)
        backup(archive)
