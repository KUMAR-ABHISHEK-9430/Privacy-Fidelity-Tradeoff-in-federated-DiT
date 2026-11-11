"""Opacus hooks + DPOptimizer. No batch-gradient noise approximation."""
import torch
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from opacus.utils.uniform_sampler import UniformWithReplacementSampler


def calibrate(epsilon, delta, q, steps):
    return get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
                                sample_rate=q, steps=steps, accountant="rdp",
                                epsilon_tolerance=.001)


def poisson(n, q, generator, steps):
    if not 0 < q <= 1:
        raise ValueError("Invalid Poisson inclusion probability")
    return UniformWithReplacementSampler(num_samples=n, sample_rate=q,
                                          generator=generator, steps=steps)


def wrap(model):
    # SUM loss ensures correct record gradients even for variable physical sizes.
    # LoRA Linear hooks sum over token/spatial axes, retaining the record axis.
    return GradSampleModule(model, batch_first=True, loss_reduction="sum", strict=True)


def optimizer(model, lr, sigma, clip, expected_batch, generator):
    params = [p for p in model.parameters() if p.requires_grad]
    if any(p.dtype != torch.float32 for p in params):
        raise RuntimeError("DP trainable parameters must be FP32")
    return DPOptimizer(torch.optim.AdamW(params, lr=lr), noise_multiplier=sigma,
                       max_grad_norm=clip, expected_batch_size=expected_batch,
                       loss_reduction="mean", generator=generator, secure_mode=False)


def private_step(wrapped, opt, chunks, loss_fn, accountant, q):
    """One independent Poisson draw, zero or more physical chunks, one noise draw."""
    opt.zero_grad()
    if not chunks:
        for p in opt.params:
            p.grad = torch.zeros_like(p)
            p.grad_sample = torch.empty((0, *p.shape), device=p.device, dtype=torch.float32)
        opt.step()
        opt.zero_grad()
    else:
        for i, chunk in enumerate(chunks):
            loss_fn(wrapped, chunk).sum().backward()
            if any(p.grad_sample is None or p.grad_sample.dtype != torch.float32 for p in opt.params):
                raise RuntimeError("Missing or non-FP32 per-record gradients")
            opt.signal_skip_step(i < len(chunks) - 1)
            opt.step()
            opt.zero_grad()
    accountant.step(noise_multiplier=opt.noise_multiplier, sample_rate=q)


def privacy_record(accountants, steps, sigma, q, config, target):
    eps = [a.get_epsilon(config["delta"]) if s else 0. for a, s in zip(accountants, steps)]
    if max(eps) > target + .02:
        raise RuntimeError(f"Achieved epsilon {max(eps)} exceeds target {target}")
    return {"privacy_unit": "one image-label record", "adjacency": "add/remove",
            "epsilon": max(eps), "target_epsilon": target, "delta": config["delta"],
            "clients": [{"epsilon": e, "steps": s, "q": rate, "noise_multiplier": sig,
                         "expected_logical_batch_size": config["expected_logical_batch_size"],
                         "max_grad_norm": config["max_grad_norm"]}
                        for e, s, rate, sig in zip(eps, steps, q, sigma)],
            "accountant": "Opacus RDP", "rng": "non-cryptographic research simulation",
            "composition": "max across fixed disjoint client record populations; all rounds composed",
            "secure_aggregation": False, "client_level_privacy": False,
            "server_observes_client_updates": True}


def reference_gradients(model, x, loss_fn):
    """Slow correctness oracle ONLY; never a production backend."""
    params = [p for p in model.parameters() if p.requires_grad]
    rows = [[] for _ in params]
    for sample in x:
        grads = torch.autograd.grad(loss_fn(model, sample[None]).sum(), params)
        for row, grad in zip(rows, grads):
            row.append(grad.detach())
    return [torch.stack(row) for row in rows]


def reference_clipped(grads, clip, expected_batch):
    norms = torch.stack([g.flatten(1).square().sum(1) for g in grads]).sum(0).sqrt()
    factors = (clip / (norms + 1e-6)).clamp(max=1)
    return [(g * factors.reshape(-1, *([1] * (g.ndim - 1)))).sum(0) / expected_batch for g in grads]
