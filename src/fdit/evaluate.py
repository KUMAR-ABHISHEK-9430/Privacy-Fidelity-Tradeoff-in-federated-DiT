import importlib.metadata

import numpy as np
import torch

from .diffusion import generate, loss_vector


@torch.no_grad()
def evaluate(model, vae, scheduler, dataset, split, config, folder, guard):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchvision.utils import save_image
    n = config["eval_samples_per_condition"]
    if n > len(split["test"]) or n < 2:
        raise ValueError("Evaluation count exceeds held-out reference or is too small")
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to("cuda")
    kid = KernelInceptionDistance(subset_size=min(50, n), subsets=20, normalize=True).to("cuda")
    rng = np.random.default_rng(config["evaluation_seed"])
    ids = rng.permutation(split["test"])[:n]
    for start in range(0, n, 16):
        guard()
        images = (torch.stack([dataset[int(i)][0] for i in ids[start:start + 16]]).to("cuda") + 1) / 2
        fid.update(images, real=True)
        kid.update(images, real=True)
    grid = []
    count = 0
    for images in generate(model, vae, scheduler, n, config["sampling_steps"], config["evaluation_seed"]):
        guard()
        fid.update(images.to("cuda"), real=False)
        kid.update(images.to("cuda"), real=False)
        count += len(images)
        if sum(len(g) for g in grid) < 64:
            grid.append(images)
    save_image(torch.cat(grid)[:64], folder / "sample_grid.png", nrow=8)
    # Fixed seed gives paired public-validation noise/timesteps across conditions.
    gen = torch.Generator(device="cuda").manual_seed(config["evaluation_seed"])
    losses = []
    for idx in split["val"][:min(96, len(split["val"]))]:
        guard()
        image = dataset[idx][0][None].to("cuda")
        z = vae.encode(image).latent_dist.mean * vae.config.scaling_factor
        losses.append(loss_vector(model, scheduler, z, gen).item())
    torch.manual_seed(config["evaluation_seed"])
    torch.cuda.manual_seed_all(config["evaluation_seed"])
    km, ks = kid.compute()
    return {"fid": float(fid.compute()), "kid_mean": float(km), "kid_std": float(ks),
            "generated_count": count, "real_count": n, "sampling_steps": config["sampling_steps"],
            "public_validation_denoising_loss": float(np.mean(losses)),
            "validation_count": len(losses), "feature_space": "torch-fidelity Inception-v3 2048",
            "torchmetrics_version": importlib.metadata.version("torchmetrics"),
            "diagnostic_only": True, "kid_subset_size": min(50, n),
            "caveat": "Upscaled satellite images; small-sample FID/KID are noisy proxies, not stable conclusions."}
