import torch
import pytest

import fdit.centralized as central
from fdit.checkpoints import load_checkpoint
from fdit.lora import inject, adapter_state
from test_lora_weights import TinyAttention


def settings():
    return {"seed": 42, "sampler_seed": 45, "noise_seed": 44, "learning_rate": .001,
            "batch_size": 4, "physical_batch_size": 2, "precision": "fp32",
            "milestones": [3, 6], "checkpoint_interval": 2, "grid_interval": 3, "log_interval": 1}


def test_index_stream_saved_order_is_exact():
    stream = central.IndexStream(list(range(7)), 2)
    assert len(set(stream.draw(7))) == 7
    state = stream.state_dict()
    expected = stream.draw(13)
    other = central.IndexStream(list(range(7)), 2)
    other.load_state_dict(state)
    assert other.draw(13) == expected


def test_central_nonprivate_resume_and_callback_rng_isolation(tmp_path, monkeypatch):
    torch.manual_seed(6)
    model = inject(TinyAttention(), 1, 1)
    initial = adapter_state(model)
    x = torch.randn(30, 2, 4)
    class Cache:
        def batch(self, ids, device="cpu"):
            return x[ids].to(device)
    def objective(m, scheduler, z, rng):
        noisy = z + torch.randn(z.shape, generator=rng)
        return m(noisy).square().flatten(1).mean(1)
    monkeypatch.setattr(central, "loss_vector", objective)
    milestones = []
    def callback(step, m):
        milestones.append(step)
        torch.manual_seed(987)
        torch.rand(200)  # Must not change the subsequent training trajectory.
    a, b = tmp_path / "a", tmp_path / "b"
    config = settings()
    central.train_centralized(model, None, Cache(), list(range(30)), config, initial, a, "id", 6, lambda: None, callback)
    calls = 0
    def guard():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise InterruptedError("intentional")
    with pytest.raises(InterruptedError):
        central.train_centralized(model, None, Cache(), list(range(30)), config, initial, b, "id", 6, guard, callback)
    assert load_checkpoint(b / "checkpoint.pt", "id")["step"] == 3
    central.train_centralized(model, None, Cache(), list(range(30)), config, initial, b, "id", 6, lambda: None, callback)
    left, right = [load_checkpoint(p / "checkpoint.pt", "id") for p in (a, b)]
    assert left["step"] == right["step"] == 6
    for name in initial:
        torch.testing.assert_close(left["global"][name], right["global"][name], rtol=0, atol=0)
    assert all(h["gradient_l2"] > 0 and h["adapter_update_l2"] > 0 for h in left["history"])
    assert len(left["history"]) == len(right["history"]) == 6
