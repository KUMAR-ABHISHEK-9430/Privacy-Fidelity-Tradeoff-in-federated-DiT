import importlib.metadata

import numpy as np
import torch

from .diffusion import generate, loss_vector


@torch.no_grad()
def evaluate(model, vae, scheduler, dataset, split, config, folder, guard,
             noise_bank=None, validation_bank=None, baseline_preview=None):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchvision.utils import save_image
    from safetensors.torch import save_file
    from .diagnostic_data import balanced_public_indices, validation_loss
    n = config["eval_samples_per_condition"]
    if n > len(split["test"]) or n < 2:
        raise ValueError("Evaluation count exceeds held-out reference or is too small")
    print(f"[eval] {folder.name}: loading Inception features for {n} paired samples", flush=True)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to("cuda")
    kid = KernelInceptionDistance(subset_size=min(50, n), subsets=20, normalize=True).to("cuda")
    rng = np.random.default_rng(config["evaluation_seed"])
    ids = rng.permutation(split["test"])[:n]
    for start in range(0, n, 16):
        guard()
        images = (torch.stack([dataset[int(i)][0] for i in ids[start:start + 16]]).to("cuda") + 1) / 2
        fid.update(images, real=True)
        kid.update(images, real=True)
        print(f"[eval] Real reference features {min(start + 16, n)}/{n}", flush=True)
    grid = []
    count = 0
    for images in generate(model, vae, scheduler, n, config["sampling_steps"], config["evaluation_seed"],
                           batch_size=config.get("generation_batch_size", 4), initial_noise=noise_bank, guard=guard):
        guard()
        fid.update(images.to("cuda"), real=False)
        kid.update(images.to("cuda"), real=False)
        count += len(images)
        print(f"[eval] Generated and scored {count}/{n}", flush=True)
        if sum(len(g) for g in grid) < 64:
            grid.append(images)
    save_image(torch.cat(grid)[:64], folder / "sample_grid.png", nrow=8)
    preview = torch.cat(grid)[:config.get("preview_samples", 16)].contiguous()
    paired_mae = None
    if noise_bank is not None:
        save_file({"images": preview}, str(folder / "paired_samples.safetensors"))
        if baseline_preview is not None:
            if preview.shape != baseline_preview.shape:
                raise ValueError("Paired preview sizes differ")
            paired_mae = float((preview - baseline_preview).abs().mean())
    # Fixed seed gives paired public-validation noise/timesteps across conditions.
    gen = torch.Generator(device="cuda").manual_seed(config["evaluation_seed"])
    print("[eval] Computing fixed public-validation loss", flush=True)
    losses = []
    if validation_bank is not None:
        value = validation_loss(model, scheduler, validation_bank, config.get("generation_batch_size", 16), guard)
        losses = [value] * len(validation_bank["latents"])
    else:
        validation_ids = balanced_public_indices(split["val"], dataset.targets, min(96, len(split["val"])), config["evaluation_seed"])
        for idx in validation_ids:
            guard()
            image = dataset[idx][0][None].to("cuda")
            z = vae.encode(image).latent_dist.mean * vae.config.scaling_factor
            losses.append(loss_vector(model, scheduler, z, gen).item())
    torch.manual_seed(config["evaluation_seed"])
    torch.cuda.manual_seed_all(config["evaluation_seed"])
    print("[eval] Computing final KID/FID statistics", flush=True)
    km, ks = kid.compute()
    return {"fid": float(fid.compute()), "kid_mean": float(km), "kid_std": float(ks),
            "generated_count": count, "real_count": n, "sampling_steps": config["sampling_steps"],
            "generation_batch_size": config.get("generation_batch_size", 4),
            "paired_pixel_mae_vs_pretrained_preview": paired_mae,
            "public_validation_denoising_loss": float(np.mean(losses)),
            "validation_count": len(losses), "feature_space": "torch-fidelity Inception-v3 2048",
            "validation_selection": "fixed seeded macro-balanced held-out classes",
            "torchmetrics_version": importlib.metadata.version("torchmetrics"),
            "diagnostic_only": True, "kid_subset_size": min(50, n),
            "caveat": "Upscaled satellite images; small-sample FID/KID are noisy proxies, not stable conclusions."}
