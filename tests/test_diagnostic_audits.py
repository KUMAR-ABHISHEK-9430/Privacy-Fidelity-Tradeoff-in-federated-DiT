import torch
import pytest
from safetensors.torch import load_file, save_file

from fdit.audit import (activation_audit, adapters_disabled, aggregation_artifacts,
                        exact_dense_mean_inference, frozen_fingerprint,
                        gradient_update_audit, schedule_audit)
from fdit.lora import inject, adapter_state, load_adapter
from fdit.federated import fedavg
from test_lora_weights import TinyAttention
from test_diffusion_shapes import tiny_components


def test_named_safetensors_reload_and_enabled_disabled_predictions(tmp_path):
    model = inject(TinyAttention(), 1, 1)
    initial = adapter_state(model)
    x = torch.randn(3, 4)
    predict = lambda m, z: m(z)
    assert not activation_audit(model, x, False, predict)["effect_detected"]
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.fill_(.1)
    model.eval()
    expected = model(x).detach()
    path = tmp_path / "adapter.safetensors"
    save_file(adapter_state(model), str(path))
    load_adapter(model, initial)
    # A safetensors mapping need not have the named_parameters insertion order.
    loaded = dict(reversed(list(load_file(str(path)).items())))
    load_adapter(model, loaded)
    torch.testing.assert_close(model(x), expected)
    audit = activation_audit(model, x, True, predict)
    assert audit["effect_detected"] and audit["adapter_layers_executed"] == 4
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    with pytest.raises(RuntimeError, match="intentional"):
        with adapters_disabled(model):
            assert all(not p.requires_grad for p in model.parameters())
            raise RuntimeError("intentional")
    assert flags == {n: p.requires_grad for n, p in model.named_parameters()}


def test_actual_dit_gradients_updates_frozen_weights_and_scheduler():
    torch.manual_seed(4)
    model, vae, scheduler = tiny_components()
    model = inject(model, 1, 1)
    z = torch.randn(2, 4, 8, 8)
    before, initial = frozen_fingerprint(model), adapter_state(model)
    audit = gradient_update_audit(model, scheduler, z, .001)
    assert audit["steps"][0]["A_gradient_l2"] == 0
    assert audit["steps"][0]["B_gradient_l2"] > 0
    assert audit["steps"][1]["A_gradient_l2"] > 0
    assert frozen_fingerprint(model) == before
    for n, p in adapter_state(model).items():
        torch.testing.assert_close(p, initial[n], rtol=0, atol=0)
    assert schedule_audit(model, vae, scheduler, z)["training_and_sampling_betas_identical"]


def test_factor_average_counterexample_and_common_a_equivalence():
    # Two representations of the SAME dense update cancel under separate A/B averaging.
    a = {"layer.lora_A.default.weight": torch.ones(1, 2), "layer.lora_B.default.weight": torch.ones(2, 1)}
    b = {k: -v for k, v in a.items()}
    result = aggregation_artifacts([a, b], [1, 1])
    assert result["relative_frobenius_error"] == pytest.approx(1.)
    same_a = dict(b, **{"layer.lora_A.default.weight": a["layer.lora_A.default.weight"]})
    same_a["layer.lora_B.default.weight"] = torch.ones(2, 1) * 3
    result = aggregation_artifacts([a, same_a], [1, 3])
    assert result["factor_average_error_frobenius"] < 1e-6


def test_dense_mean_hooks_match_manual_dense_weights_and_restore():
    model = inject(TinyAttention(), 1, 1)
    states = []
    for sign in (1., -1.):
        states.append({n: torch.ones_like(p) * sign for n, p in adapter_state(model).items()})
    load_adapter(model, fedavg(states, [1, 1]))
    x = torch.randn(2, 4)
    factor = model(x).detach()
    frozen = frozen_fingerprint(model)
    def dense(layer, z):
        base = layer.base_layer
        update = torch.ones_like(base.weight)
        return torch.nn.functional.linear(z, base.weight + update, base.bias)
    hidden = dense(model.to_q, x) + dense(model.to_k, x) + dense(model.to_v, x)
    expected = model.base(dense(model.to_out[0], hidden))
    with torch.no_grad(), exact_dense_mean_inference(model, states, [1, 1]):
        torch.testing.assert_close(model(x), expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(model(x), factor)
    assert frozen_fingerprint(model) == frozen
