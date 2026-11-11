import torch
import pytest
from opacus.accountants import RDPAccountant

from fdit.checkpoints import save_checkpoint, load_checkpoint, rng_state, restore_rng
from fdit.private import poisson


def test_partial_checkpoint_recovers_rng_accountant_sampler_optimizer(tmp_path):
    torch.manual_seed(42)
    gen = torch.Generator().manual_seed(18)
    p = torch.nn.Parameter(torch.ones(2))
    opt = torch.optim.AdamW([p], lr=.01)
    p.square().sum().backward()
    opt.step()
    acc = RDPAccountant()
    acc.step(noise_multiplier=2., sample_rate=.1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, {"identity": "correct", "round": 1, "client": 1, "local_step": 2,
                           "local": p.detach().clone(), "optimizer": opt.state_dict(),
                           "accountant": acc.state_dict(), "sampler": gen.get_state(), "rng": rng_state()})
    expected_random = torch.randn(8)
    expected_sample = next(iter(poisson(50, .2, gen, 1)))
    state = load_checkpoint(path, "correct")
    restore_rng(state["rng"])
    gen.set_state(state["sampler"])
    torch.testing.assert_close(torch.randn(8), expected_random)
    assert next(iter(poisson(50, .2, gen, 1))) == expected_sample
    restored = RDPAccountant()
    restored.load_state_dict(state["accountant"])
    assert restored.get_epsilon(1e-6) == acc.get_epsilon(1e-6)
    q = torch.nn.Parameter(state["local"].clone())
    other = torch.optim.AdamW([q], lr=.01)
    other.load_state_dict(state["optimizer"])
    for value, optimizer in ((p, opt), (q, other)):
        optimizer.zero_grad()
        value.square().sum().backward()
        optimizer.step()
    torch.testing.assert_close(p, q)
    with pytest.raises(ValueError, match="fingerprints"):
        load_checkpoint(path, "wrong")


@pytest.mark.parametrize("private", [False, True])
def test_federated_mid_client_resume_matches_uninterrupted(tmp_path, monkeypatch, private):
    import fdit.federated as fed
    from fdit.lora import inject, adapter_state
    from test_lora_weights import TinyAttention
    from test_dp_reference import loss
    torch.manual_seed(5)
    model = inject(TinyAttention(), 1, 1)
    initial = adapter_state(model)
    data = torch.randn(30, 2, 4)
    class Cache:
        def batch(self, ids, device="cpu"):
            return data[ids].to(device)
    monkeypatch.setattr(fed, "loss_vector", lambda m, s, x: loss(m, x))
    config = {"rounds": 2, "local_steps_per_round": 3, "expected_logical_batch_size": 2,
              "physical_batch_size": 1, "learning_rate": .001, "max_grad_norm": 1.,
              "delta": 1e-6, "seed": 42, "sampler_seed": 43, "noise_seed": 44, "precision": "bf16"}
    clients = [list(range(k * 10, (k + 1) * 10)) for k in range(3)]
    condition = {"id": "toy", "dp": private, "epsilon": 3.}
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    full = fed.train(model, None, Cache(), clients, config, condition, initial, a, "same", lambda: None)
    calls = 0
    def guard():
        nonlocal calls
        calls += 1
        if calls == 6:
            raise InterruptedError("intentional")
    with pytest.raises(InterruptedError):
        fed.train(model, None, Cache(), clients, config, condition, initial, b, "same", guard)
    partial = load_checkpoint(b / "checkpoint.pt", "same")
    assert partial["client"] == 1 and partial["local_step"] == 2
    resumed = fed.train(model, None, Cache(), clients, config, condition, initial, b, "same", lambda: None)
    first = load_checkpoint(a / "checkpoint.pt", "same")
    second = load_checkpoint(b / "checkpoint.pt", "same")
    for name in initial:
        torch.testing.assert_close(first["global"][name], second["global"][name], rtol=0, atol=0)
    assert first["steps"] == second["steps"] == [6, 6, 6]
    assert full["privacy"] == resumed["privacy"]
