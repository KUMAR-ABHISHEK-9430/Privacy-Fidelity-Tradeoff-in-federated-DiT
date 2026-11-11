import copy
import torch
import pytest
from opacus.accountants import RDPAccountant
from opacus.optimizers import DPOptimizer

from fdit.lora import inject
from fdit.private import wrap, reference_gradients, reference_clipped, private_step, poisson
from test_lora_weights import TinyAttention


def loss(model, x):
    return model(x).square().flatten(1).mean(1)


def test_per_record_hooks_against_reference_token_axes():
    torch.manual_seed(7)
    model = inject(TinyAttention(), 1, 1)
    # Make A and B gradients both nontrivial (PEFT initializes B to zero).
    for n, p in model.named_parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=.1)
    x = torch.randn(3, 5, 4)
    expected = reference_gradients(model, x, loss)
    wrapped = wrap(model)
    loss(wrapped, x).sum().backward()
    actual = [p.grad_sample for p in model.parameters() if p.requires_grad]
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-6)
    wrapped.remove_hooks()


def test_clipping_microbatch_once_and_empty():
    torch.manual_seed(9)
    model = inject(TinyAttention(), 1, 1)
    x = torch.randn(4, 2, 4) * 100
    expected = reference_clipped(reference_gradients(model, x, loss), .2, 8)
    params = [p for p in model.parameters() if p.requires_grad]
    initial = [p.clone() for p in params]
    wrapped = wrap(model)
    opt = DPOptimizer(torch.optim.SGD(params, lr=1.), noise_multiplier=0., max_grad_norm=.2,
                       expected_batch_size=8, loss_reduction="mean")
    acc = RDPAccountant()
    private_step(wrapped, opt, list(x.split(1)), loss, acc, .1)
    for p, old, grad in zip(params, initial, expected):
        torch.testing.assert_close(p, old - grad, rtol=1e-4, atol=1e-6)
    assert sum(h[2] for h in acc.history) == 1
    private_step(wrapped, opt, [], loss, acc, .1)
    assert sum(h[2] for h in acc.history) == 2
    wrapped.remove_hooks()


def test_noise_std_and_once_per_logical_step():
    p = torch.nn.Parameter(torch.zeros(10000))
    opt = DPOptimizer(torch.optim.SGD([p], lr=1.), noise_multiplier=2., max_grad_norm=3.,
                      expected_batch_size=10, generator=torch.Generator().manual_seed(19))
    p.grad_sample = torch.zeros(0, 10000)
    p.grad = torch.zeros_like(p)
    opt.step()
    assert abs(p.std().item() - .6) < .025


def test_poisson_empty_independent_inclusion():
    batches = list(poisson(20, .1, torch.Generator().manual_seed(1), 4000))
    sizes = torch.tensor(list(map(len, batches)), dtype=torch.float32)
    assert abs(sizes.mean().item() - 2.) < .1
    assert abs(sizes.var().item() - 1.8) < .2
    assert any(not batch for batch in batches)
    assert all(len(b) == len(set(b)) for b in batches)


def test_actual_diffusers_peft_gradients_against_reference():
    from test_diffusion_shapes import tiny_components
    from fdit.diffusion import epsilon_channels
    torch.manual_seed(12)
    model, _, _ = tiny_components()
    model = inject(model, 1, 1)
    for p in model.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=.1)
    x = torch.randn(2, 4, 8, 8)
    def objective(m, z):
        output = m(z, timestep=torch.full((len(z),), 7),
                   class_labels=torch.full((len(z),), 1000)).sample
        return epsilon_channels(output, z).square().flatten(1).mean(1)
    expected = reference_gradients(model, x, objective)
    wrapped = wrap(model)
    objective(wrapped, x).sum().backward()
    for p, e in zip([p for p in model.parameters() if p.requires_grad], expected):
        torch.testing.assert_close(p.grad_sample, e, rtol=2e-4, atol=1e-6)
    wrapped.remove_hooks()


@pytest.mark.parametrize("physical_batch", [1, 2, 8])
def test_larger_private_chunks_match_clipping_and_one_noise_draw(physical_batch):
    torch.manual_seed(81)
    model = inject(TinyAttention(), 1, 1)
    x = torch.randn(6, 2, 4) * 10
    params = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in params]
    expected = reference_clipped(reference_gradients(model, x, loss), .2, 8)
    sigma = .7
    oracle_rng = torch.Generator().manual_seed(99)
    noise = [torch.normal(0., sigma * .2, size=p.shape, generator=oracle_rng) / 8 for p in params]
    wrapped = wrap(model)
    opt = DPOptimizer(torch.optim.SGD(params, lr=1.), noise_multiplier=sigma, max_grad_norm=.2,
                       expected_batch_size=8, generator=torch.Generator().manual_seed(99))
    acc = RDPAccountant()
    private_step(wrapped, opt, list(x.split(physical_batch)), loss, acc, .1)
    for p, old, clipped, gaussian in zip(params, before, expected, noise):
        torch.testing.assert_close(p, old - clipped - gaussian, rtol=1e-4, atol=1e-6)
    assert sum(h[2] for h in acc.history) == 1
    wrapped.remove_hooks()
