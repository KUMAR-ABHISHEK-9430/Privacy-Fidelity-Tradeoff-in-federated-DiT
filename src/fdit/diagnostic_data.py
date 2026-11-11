"""Fixed PUBLIC validation, VAE reconstruction, and paired noise banks."""
import hashlib
import math

import numpy as np
import torch

from .diffusion import epsilon_channels


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def initial_noise_bank(n, channels, size, seed):
    # Per-image CPU generators make the bank independent of GPU physical grouping.
    return torch.stack([torch.randn(channels, size, size, generator=torch.Generator().manual_seed(seed + i))
                        for i in range(n)])


def balanced_public_indices(indices, labels, count, seed):
    """Public held-out macro-balanced bank; avoid sorted first-N single-class bias."""
    if count < 1 or count > len(indices):
        raise ValueError("Invalid held-out selection count")
    ids, labels = np.asarray(indices), np.asarray(labels)
    rng = np.random.default_rng(seed)
    classes = rng.permutation(np.unique(labels[ids]))
    pools = [rng.permutation(ids[labels[ids] == c]).tolist() for c in classes]
    selected = []
    while len(selected) < count:
        for pool in pools:
            if pool and len(selected) < count:
                selected.append(int(pool.pop()))
    return selected


@torch.no_grad()
def validation_bank(dataset, indices, vae, scheduler, count, seed, batch_size, guard):
    device = next(vae.parameters()).device
    ids = list(indices)[:count]
    if len(ids) != count:
        raise ValueError("Insufficient held-out validation records")
    latent = []
    for start in range(0, count, batch_size):
        guard()
        images = torch.stack([dataset[i][0] for i in ids[start:start + batch_size]]).to(device)
        z = vae.encode(images).latent_dist.mean * vae.config.scaling_factor
        latent.append(z.float().cpu())
    latent = torch.cat(latent)
    generator = torch.Generator().manual_seed(seed)
    return {"latents": latent, "noise": torch.randn(latent.shape, generator=generator),
            "timesteps": torch.randint(scheduler.config.num_train_timesteps, (count,), generator=generator)}


@torch.no_grad()
def validation_loss(model, scheduler, bank, batch_size=16, guard=lambda: None):
    device = next(model.parameters()).device
    mode = model.training
    model.eval()
    total = 0.
    try:
        for start in range(0, len(bank["latents"]), batch_size):
            guard()
            z, noise, t = [bank[k][start:start + batch_size].to(device) for k in ("latents", "noise", "timesteps")]
            prediction = model(scheduler.add_noise(z, noise, t), timestep=t, class_labels=torch.full_like(t, 1000)).sample
            total += float((epsilon_channels(prediction, z) - noise).float().square().flatten(1).mean(1).sum())
        return total / len(bank["latents"])
    finally:
        model.train(mode)


@torch.no_grad()
def vae_reconstruction(dataset, test_indices, vae, count, batch_size, folder, min_psnr, guard):
    from torchvision.utils import save_image
    from torchmetrics.functional.image import structural_similarity_index_measure
    if count > len(test_indices) or count < 1:
        raise ValueError("Invalid public VAE reconstruction sample count")
    device = next(vae.parameters()).device
    mse_total, mae_total, ssim_total, examples = 0., 0., 0., []
    pixels, records = 0, 0
    for start in range(0, count, batch_size):
        guard()
        images = torch.stack([dataset[i][0] for i in test_indices[start:start + min(batch_size, count - start)]]).to(device)
        latent = vae.encode(images).latent_dist.mean
        scaled = latent * vae.config.scaling_factor
        decoded = vae.decode(scaled / vae.config.scaling_factor).sample
        if decoded.shape != images.shape or scaled.shape[1] != vae.config.latent_channels:
            raise RuntimeError("VAE round-trip shape mismatch")
        target, reconstruction = (images + 1) / 2, (decoded.clamp(-1, 1) + 1) / 2
        diff = target - reconstruction
        mse_total += float(diff.square().sum())
        mae_total += float(diff.abs().sum())
        pixels += diff.numel()
        ssim_total += float(structural_similarity_index_measure(reconstruction, target, data_range=1.)) * len(images)
        records += len(images)
        if len(examples) < 32:
            for original, rebuilt in zip(target.cpu(), reconstruction.cpu()):
                if len(examples) < 32:
                    examples.extend([original, rebuilt])
        print(f"[VAE] Public held-out reconstructions {records}/{count}", flush=True)
    save_image(torch.stack(examples), folder / "public_test_original_reconstruction.png", nrow=8)
    mse = mse_total / pixels
    psnr = -10 * math.log10(mse) if mse > 0 else None
    return {"source": "public held-out EuroSAT TEST records; never training records", "count": count,
            "mse_0_1": mse, "mae_0_1": mae_total / pixels, "psnr_db": psnr,
            "ssim": ssim_total / records, "posterior": "mean", "scaling_factor": float(vae.config.scaling_factor),
            "threshold_psnr_db": min_psnr,
            "adequate_by_configured_psnr_heuristic": mse == 0 or psnr >= min_psnr,
            "caveat": "PSNR threshold is a diagnostic heuristic; inspect original/reconstruction pairs for semantic detail"}
