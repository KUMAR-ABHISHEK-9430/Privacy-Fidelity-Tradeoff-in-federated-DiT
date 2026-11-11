import torch
from peft import LoraConfig, inject_adapter_in_model

TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


def inject(model, rank=4, alpha=4):
    model.requires_grad_(False)
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
             and any(n.endswith(t) for t in TARGETS)]
    if not names or any(not any(n.endswith(t) for n in names) for t in TARGETS):
        raise RuntimeError("Incompatible attention modules: missing LoRA targets")
    model = inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.,
                                              target_modules=names, bias="none"), model)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if any("lora_" not in n for n in trainable):
        raise RuntimeError("Non-LoRA parameter is trainable")
    count = sum(p.numel() for p in trainable.values())
    total = sum(p.numel() for p in model.parameters())
    if not 0 < count < total * .05:
        raise RuntimeError("Unexpected trainable fraction")
    for p in trainable.values():
        p.data = p.data.float()
    print(f"Trainable LoRA parameters: {count:,} / {total:,}", flush=True)
    print(f"Discovered {len(names)} LoRA projections; first: {names[0]}", flush=True)
    return model


def adapter_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_adapter(model, state):
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    # Safetensors serializes keys in a different order from named_parameters.
    # Alignment is by exact name, not dictionary insertion order.
    if set(params) != set(state) or any(params[n].shape != state[n].shape for n in params):
        raise ValueError("Adapter keys/shapes differ")
    if any(not torch.isfinite(state[n]).all() for n in params):
        raise ValueError("Non-finite adapter weights")
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(state[n])
