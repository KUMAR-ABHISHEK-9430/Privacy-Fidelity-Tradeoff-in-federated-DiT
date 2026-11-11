import torch


def epsilon_channels(output, latent):
    if output.shape[0] != latent.shape[0] or output.shape[1] != 2 * latent.shape[1] or output.shape[2:] != latent.shape[2:]:
        raise ValueError("Expected concatenated epsilon/variance channels")
    return output[:, :latent.shape[1]]


def loss_vector(model, scheduler, latents, generator=None):
    noise = torch.randn(latents.shape, device=latents.device, generator=generator)
    timesteps = torch.randint(scheduler.config.num_train_timesteps, (len(latents),),
                              device=latents.device, generator=generator)
    noisy = scheduler.add_noise(latents, noise, timesteps)
    labels = torch.full_like(timesteps, 1000)
    output = model(noisy, timestep=timesteps, class_labels=labels).sample
    eps = epsilon_channels(output, latents)
    return (eps.float() - noise.float()).square().flatten(1).mean(1)


@torch.no_grad()
def generate(model, vae, scheduler, n, steps, seed, device=None, batch_size=4,
             initial_noise=None, guard=lambda: None):
    from diffusers import DDIMScheduler
    device = next(model.parameters()).device if device is None else torch.device(device)
    scheduler = DDIMScheduler.from_config(scheduler.config)
    scheduler.set_timesteps(steps, device=device)
    rng = torch.Generator(device=device).manual_seed(seed)
    shape = (n, model.config.in_channels, model.config.sample_size, model.config.sample_size)
    if initial_noise is not None and tuple(initial_noise.shape) != shape:
        raise ValueError("Paired initial noise bank has incorrect shape")
    model.eval()
    for start in range(0, n, batch_size):
        size = min(batch_size, n - start)
        guard()
        x = (initial_noise[start:start + size].to(device).clone() if initial_noise is not None else
             torch.randn((size, *shape[1:]), generator=rng, device=device))
        x *= scheduler.init_noise_sigma
        for t in scheduler.timesteps:
            guard()
            ts = t.expand(size)
            out = model(scheduler.scale_model_input(x, t), timestep=ts,
                        class_labels=torch.full_like(ts, 1000)).sample
            x = scheduler.step(epsilon_channels(out, x), t, x, eta=0.).prev_sample
        images = vae.decode(x / vae.config.scaling_factor).sample
        yield ((images.clamp(-1, 1) + 1) / 2).cpu()
