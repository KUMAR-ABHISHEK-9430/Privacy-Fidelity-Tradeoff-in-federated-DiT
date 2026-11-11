"""Public/synthetic and NON-PRIVATE diagnostics. Never attach to a private trainer."""
from contextlib import contextmanager
import hashlib
import math

import torch
from torch.nn import functional as F
from peft.tuners.tuners_utils import BaseTunerLayer

from .checkpoints import rng_state, restore_rng
from .diffusion import epsilon_channels, loss_vector
from .federated import fedavg
from .lora import adapter_state, load_adapter


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def lora_layers(model):
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, BaseTunerLayer) and hasattr(m, "lora_A")]
    if not layers:
        raise RuntimeError("No LoRA layers found for audit")
    for name, layer in layers:
        if (list(layer.active_adapters) != ["default"] or layer.merged
                or layer.fan_in_fan_out or layer.lora_variant):
            raise RuntimeError(f"Audit supports unmerged standard default Linear LoRA only: {name}")
    return layers


@contextmanager
def adapters_disabled(model):
    layers = lora_layers(model)
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    enabled = [not m.disable_adapters for _, m in layers]
    try:
        for _, layer in layers:
            layer.enable_adapters(False)
        yield
    finally:
        for (_, layer), value in zip(layers, enabled):
            layer.enable_adapters(value)
        for name, param in model.named_parameters():
            param.requires_grad_(flags[name])


def frozen_fingerprint(model):
    """Stream all frozen parameters/buffers; never clone a full backbone."""
    digest = hashlib.sha256()
    tensors = [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
    tensors += [("buffer:" + n, b) for n, b in model.named_buffers()]
    for name, tensor in tensors:
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        flat = tensor.detach().reshape(-1)
        for chunk in flat.split(1_000_000):
            digest.update(chunk.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_versions(model):
    return {n: p._version for n, p in model.named_parameters() if not p.requires_grad}


def assert_frozen_versions(model, versions):
    if frozen_versions(model) != versions:
        raise RuntimeError("A frozen base parameter changed during adapter-only training")


def state_distance(left, right):
    if set(left) != set(right):
        raise ValueError("Adapter keys differ")
    return math.sqrt(sum(float((left[n].double() - right[n].double()).square().sum()) for n in left))


def gradient_norm(model, family=None):
    values = [p.grad.detach().float() for n, p in model.named_parameters()
              if p.requires_grad and p.grad is not None and (family is None or family in n)]
    if not values:
        return 0.
    flat = torch.cat([g.reshape(-1) for g in values])
    if not torch.isfinite(flat).all():
        raise RuntimeError("Non-finite non-private diagnostic gradient")
    return float(torch.linalg.vector_norm(flat))


def gradient_update_audit(model, scheduler, latents, lr, seed=2718, steps=2):
    """Temporary ordinary AdamW on public-simulation TRAIN records; restore everything."""
    initial = adapter_state(model)
    versions = frozen_versions(model)
    mode = model.training
    device = latents.device
    history = []
    try:
        with preserve_rng():
            model.train()
            rng = torch.Generator(device=device).manual_seed(seed)
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
            for step in range(steps):
                opt.zero_grad()
                objective = loss_vector(model, scheduler, latents, rng).mean()
                objective.backward()
                a, b = gradient_norm(model, "lora_A"), gradient_norm(model, "lora_B")
                if a == 0 and b == 0:
                    raise RuntimeError("All LoRA gradients are zero")
                before = adapter_state(model)
                opt.step()
                delta = state_distance(before, adapter_state(model))
                if delta == 0:
                    raise RuntimeError("Nonzero LoRA gradients produced no parameter update")
                assert_frozen_versions(model, versions)
                history.append({"step": step + 1, "nonprivate_mean_loss": float(objective.detach()),
                                "A_gradient_l2": a, "B_gradient_l2": b, "parameter_update_l2": delta})
            if not any(r["A_gradient_l2"] > 0 for r in history):
                raise RuntimeError("LoRA A has no gradient after multiple diagnostic steps")
    finally:
        model.zero_grad(set_to_none=True)
        load_adapter(model, initial)
        model.train(mode)
    return {"scope": "temporary non-private train-record probe; not a DP measurement", "steps": history,
            "base_unchanged_by_version_check": True,
            "note": "A gradients can be zero on the first step when B initializes to zero"}


def probe_predict(model, latents):
    n = len(latents)
    times = torch.tensor([0, 250, 500, 750, 999], device=latents.device).repeat((n + 4) // 5)[:n]
    out = model(latents, timestep=times, class_labels=torch.full_like(times, 1000)).sample
    return epsilon_channels(out, latents).float()


@torch.no_grad()
def activation_audit(model, latents, require_effect=True, predict=probe_predict):
    mode = model.training
    model.eval()
    calls = {name: 0 for name, _ in lora_layers(model)}
    handles = []
    for name, layer in lora_layers(model):
        def hook(module, inputs, output, name=name):
            calls[name] += 1
        handles.append(layer.lora_B["default"].register_forward_hook(hook))
    try:
        active = predict(model, latents)
        for handle in handles:
            handle.remove()
        handles.clear()
        with adapters_disabled(model):
            disabled = predict(model, latents)
        active_again = predict(model, latents)
        torch.testing.assert_close(active, active_again, rtol=1e-5, atol=1e-6)
        delta = active - disabled
        if not torch.isfinite(delta).all() or any(n == 0 for n in calls.values()):
            raise RuntimeError("An adapter did not execute, or inference contains non-finite values")
        maximum = float(delta.abs().max())
        if require_effect and maximum == 0:
            raise RuntimeError("Loaded adapter has no measured inference effect with adapters enabled")
        return {"enabled_minus_disabled_maxabs": maximum,
                "enabled_minus_disabled_rms": float(delta.square().mean().sqrt()),
                "relative_rms": float(delta.square().mean().sqrt() / disabled.square().mean().sqrt().clamp_min(1e-12)),
                "adapter_layers_executed": sum(v > 0 for v in calls.values()),
                "restored_enabled_predictions": True, "effect_detected": maximum > 0}
    finally:
        for handle in handles:
            handle.remove()
        model.train(mode)


@torch.no_grad()
def schedule_audit(model, vae, scheduler, probe):
    from diffusers import DDIMScheduler
    from .model import verify_components
    metadata = verify_components(model, vae, scheduler)
    sampling = DDIMScheduler.from_config(scheduler.config)
    torch.testing.assert_close(scheduler.betas, sampling.betas, rtol=0, atol=0)
    t = torch.full((len(probe),), 500, device=probe.device, dtype=torch.long)
    noise = torch.ones_like(probe) * .37
    noisy = scheduler.add_noise(probe, noise, t)
    alpha = scheduler.alphas_cumprod.to(probe.device)[t].reshape(-1, 1, 1, 1)
    torch.testing.assert_close(noisy, alpha.sqrt() * probe + (1 - alpha).sqrt() * noise)
    sampling.set_timesteps(2, device=probe.device)
    ts = sampling.timesteps[0].expand(len(probe))
    out = model(probe, timestep=ts, class_labels=torch.full_like(ts, 1000)).sample
    epsilon = epsilon_channels(out, probe)
    stepped = sampling.step(epsilon, sampling.timesteps[0], probe).prev_sample
    if not torch.isfinite(stepped).all():
        raise RuntimeError("Non-finite DDIM sampling step")
    return {"components": metadata, "training_and_sampling_betas_identical": True,
            "add_noise_matches_closed_form": True, "output_shape": list(out.shape),
            "epsilon_shape": list(epsilon.shape), "sampling_step_finite": True,
            "variance_handling": "last latent_channels discarded; epsilon-only DDIM, no variance training"}


def _low_rank_norm_squared(b, a):
    # ||BA||_F^2 = tr((B^T B)(AA^T)); no dense 1152x1152 update needed.
    return float(((b.T @ b) * (a @ a.T).T).sum().clamp_min(0))


def aggregation_artifacts(states, sizes, scaling=1.):
    average = fedavg(states, sizes)
    weights = [n / sum(sizes) for n in sizes]
    layers, dense_squared, error_squared = {}, 0., 0.
    for key in average:
        if ".lora_A." not in key:
            continue
        bkey = key.replace(".lora_A.", ".lora_B.")
        aa, bb = average[key].double(), average[bkey].double()
        cat_a = torch.cat([s[key].double() * math.sqrt(w) for s, w in zip(states, weights)], dim=0)
        cat_b = torch.cat([s[bkey].double() * math.sqrt(w) for s, w in zip(states, weights)], dim=1)
        exact = _low_rank_norm_squared(cat_b, cat_a) * scaling**2
        difference = _low_rank_norm_squared(torch.cat([bb, cat_b], 1), torch.cat([aa, -cat_a], 0)) * scaling**2
        dense_squared += exact
        error_squared += difference
        layers[key.split(".lora_A.")[0]] = {"weighted_dense_update_frobenius": math.sqrt(exact),
                                          "factor_average_error_frobenius": math.sqrt(difference)}
    if not layers:
        raise ValueError("No paired LoRA A/B tensors")
    return {"layers": layers, "weighted_dense_update_frobenius": math.sqrt(dense_squared),
            "factor_average_error_frobenius": math.sqrt(error_squared),
            "relative_frobenius_error": math.sqrt(error_squared / dense_squared) if dense_squared > 0 else None,
            "production_fedavg_modified": False,
            "formula": "sum_k w_k B_k A_k versus (sum_k w_k B_k)(sum_k w_k A_k), scaled by alpha/r"}


@contextmanager
def exact_dense_mean_inference(model, states, sizes):
    """Scoped inference-only hooks: exact weighted dense LoRA updates, no base mutation."""
    handles = []
    weights = [n / sum(sizes) for n in sizes]
    try:
        for name, layer in lora_layers(model):
            if layer.disable_adapters:
                raise RuntimeError("Dense-mean comparison requires enabled adapters")
            device = layer.lora_A["default"].weight.device
            pairs = [(s[f"{name}.lora_A.default.weight"].to(device),
                      s[f"{name}.lora_B.default.weight"].to(device)) for s in states]
            def hook(module, inputs, output, pairs=pairs):
                x = inputs[0].float()
                a, b = module.lora_A["default"].weight, module.lora_B["default"].weight
                current = F.linear(F.linear(x, a), b)
                exact = sum(w * F.linear(F.linear(x, aa), bb) for w, (aa, bb) in zip(weights, pairs))
                return output + ((exact - current) * module.scaling["default"]).to(output.dtype)
            handles.append(layer.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def aggregation_round_audit(model, scheduler, cache, clients, config, predict=probe_predict, probe=None):
    """New, tiny NON-PRIVATE three-client replay; not a reconstruction of old DP updates."""
    original = adapter_state(model)
    mode = model.training
    versions = frozen_versions(model)
    device = next(model.parameters()).device
    states = []
    try:
        with preserve_rng():
            for k, ids in enumerate(clients):
                load_adapter(model, original)
                if state_distance(original, adapter_state(model)) != 0:
                    raise RuntimeError("Aggregation clients did not start from identical global adapters")
                model.train()
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"])
                if opt.state:
                    raise RuntimeError("Unexpected optimizer state at client initialization")
                chooser = torch.Generator().manual_seed(config["sampler_seed"] + k)
                rng = torch.Generator(device=device).manual_seed(config["noise_seed"] + k)
                for _ in range(config["aggregation_local_steps"]):
                    select = torch.randperm(len(ids), generator=chooser)[:config["physical_batch_size"]].tolist()
                    z = cache.batch([ids[i] for i in select], device=device)
                    opt.zero_grad()
                    loss_vector(model, scheduler, z, rng).mean().backward()
                    opt.step()
                states.append(adapter_state(model))
                del opt
            sizes = list(map(len, clients))
            load_adapter(model, fedavg(states, sizes))
            model.eval()
            if probe is None:
                probe = cache.batch(clients[0][:2], device=device)
            scales = {float(layer.scaling["default"]) for _, layer in lora_layers(model)}
            if len(scales) != 1:
                raise RuntimeError("Mixed LoRA scales require a per-layer audit")
            report = aggregation_artifacts(states, sizes, scales.pop())
            with torch.no_grad():
                factor = predict(model, probe)
                with exact_dense_mean_inference(model, states, sizes):
                    dense = predict(model, probe)
            report.update(scope="fresh non-private miniature round; historical local DP snapshots unavailable",
                          local_steps=config["aggregation_local_steps"],
                          factor_vs_dense_prediction_rms=float((factor - dense).float().square().mean().sqrt()),
                          common_start_verified=True, optimizer_reset_per_client=True)
            assert_frozen_versions(model, versions)
            return report
    finally:
        model.zero_grad(set_to_none=True)
        load_adapter(model, original)
        model.train(mode)
