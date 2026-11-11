import importlib.metadata
import platform
import torch


def cuda_preflight():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required. Attach Molab GPU; keep the working CUDA torch installation.")
    try:
        x = torch.ones((32, 32), device="cuda")
        (x @ x).sum().item()
        torch.cuda.synchronize()
    except RuntimeError as exc:
        raise RuntimeError("CUDA kernels unavailable; install a Blackwell-compatible CUDA PyTorch stack") from exc
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured bf16 requires GPU support")
    versions = {}
    for name in ("torch", "torchvision", "diffusers", "peft", "opacus", "numpy", "torchmetrics", "transformers"):
        versions[name] = importlib.metadata.version(name)
    info = {"device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "bf16": True, "cuda": torch.version.cuda, "versions": versions,
            "python": platform.python_version()}
    print(info)
    return info


def smoke(model, vae, scheduler, config):
    """Real pretrained one-step DP integration; restores adapters and removes hooks."""
    from .lora import adapter_state, load_adapter
    from .diffusion import loss_vector, generate
    from .private import wrap, optimizer, private_step, RDPAccountant
    state = adapter_state(model)
    frozen = [(p, p.detach().flatten()[:64].clone()) for p in model.parameters() if not p.requires_grad]
    wrapped = wrap(model)
    rng = torch.Generator(device="cuda").manual_seed(config["noise_seed"])
    opt = optimizer(wrapped, config["learning_rate"], 1., config["max_grad_norm"], 2, rng)
    try:
        model.train()
        z = torch.randn((2, 4, 32, 32), device="cuda")
        # VAE encode/decode and channel/scaling checks happen on real components.
        with torch.no_grad():
            enc = vae.encode(torch.zeros((1, 3, 256, 256), device="cuda")).latent_dist.mean
            assert enc.shape == (1, 4, 32, 32)
        private_step(wrapped, opt, list(z.split(1)), lambda m, b: loss_vector(m, scheduler, b), RDPAccountant(), .1)
        assert any(not torch.equal(state[n], p) for n, p in adapter_state(model).items())
        assert all(torch.equal(p.detach().flatten()[:64], old) for p, old in frozen)
        private_step(wrapped, opt, [], lambda m, b: loss_vector(m, scheduler, b), RDPAccountant(), .1)
    finally:
        wrapped.remove_hooks()
        load_adapter(model, state)
    next(generate(model, vae, scheduler, 1, 1, config["evaluation_seed"]))
    print("Pretrained CUDA/PEFT/Opacus/VAE/one-step sampling smoke passed")
