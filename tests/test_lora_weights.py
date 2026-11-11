import torch
from torch import nn
import pytest

from fdit.lora import inject, adapter_state, load_adapter


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4)
        self.to_k = nn.Linear(4, 4)
        self.to_v = nn.Linear(4, 4)
        self.to_out = nn.Sequential(nn.Linear(4, 4))
        self.base = nn.Linear(4, 512)

    def forward(self, x):
        return self.base(self.to_out(self.to_q(x) + self.to_k(x) + self.to_v(x)))


def test_adapter_changes_only():
    model = inject(TinyAttention(), rank=1, alpha=1)
    original = {n: p.clone() for n, p in model.named_parameters()}
    initial = adapter_state(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model(torch.randn(3, 4)).square().mean().backward()
    opt.step()
    assert any(not torch.equal(initial[n], p) for n, p in adapter_state(model).items())
    assert all(torch.equal(p, original[n]) for n, p in model.named_parameters() if not p.requires_grad)
    load_adapter(model, initial)
    assert all(torch.equal(p, initial[n]) for n, p in adapter_state(model).items())


def test_missing_targets_fail():
    with pytest.raises(RuntimeError, match="missing LoRA"):
        inject(nn.Linear(4, 4))
