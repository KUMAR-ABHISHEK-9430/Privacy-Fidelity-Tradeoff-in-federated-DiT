import csv
import time
from pathlib import Path

import torch
from opacus.accountants import RDPAccountant

from .checkpoints import save_checkpoint, load_checkpoint, rng_state, restore_rng, atomic_json
from .diffusion import loss_vector
from .lora import adapter_state, load_adapter
from .private import calibrate, poisson, wrap, optimizer, private_step, privacy_record


def fedavg(states, sizes):
    if not states or len(states) != len(sizes) or any(s <= 0 for s in sizes):
        raise ValueError("Invalid aggregation input")
    keys = list(states[0])
    if any(list(s) != keys for s in states):
        raise ValueError("Adapter order differs")
    if any(s[k].shape != states[0][k].shape for s in states for k in keys):
        raise ValueError("Adapter shape differs")
    return {k: sum(s[k].float() * (n / sum(sizes)) for s, n in zip(states, sizes)) for k in keys}


def train(model, scheduler, cache, clients, config, condition, initial, folder, identity, guard, resume=True):
    folder = Path(folder)
    device = next(model.parameters()).device
    is_cuda = device.type == "cuda"
    synchronize = torch.cuda.synchronize if is_cuda else lambda: None
    peak_memory = torch.cuda.max_memory_allocated if is_cuda else lambda: 0
    path = folder / "checkpoint.pt"
    rates = [config["expected_logical_batch_size"] / len(c) for c in clients]
    if any(q > 1 for q in rates):
        raise ValueError("Logical expected batch exceeds client size")
    total_steps = config["rounds"] * config["local_steps_per_round"]
    sigmas = [calibrate(condition["epsilon"], config["delta"], q, total_steps) for q in rates] if condition["dp"] else [0.] * 3
    accountants = [RDPAccountant() for _ in clients]
    sampler = [torch.Generator().manual_seed(config["sampler_seed"] + k) for k in range(3)]
    noise = [torch.Generator(device=device).manual_seed(config["noise_seed"] + k) for k in range(3)]
    torch.manual_seed(config["seed"])
    if is_cuda:
        torch.cuda.manual_seed_all(config["seed"])
    state = {"identity": identity, "round": 0, "client": 0, "local_step": 0,
             "global": initial, "updates": [], "steps": [0] * 3, "history": [],
             "local": None, "optimizer": None, "elapsed": 0., "peak_memory_bytes": 0}
    if path.exists():
        if not resume:
            raise FileExistsError("Existing condition; use --resume")
        state = load_checkpoint(path, identity)
        for a, saved in zip(accountants, state["accountants"]):
            a.load_state_dict(saved)
        for g, saved in zip(sampler, state["sampler_rng"]):
            g.set_state(saved)
        for g, saved in zip(noise, state["noise_rng"]):
            g.set_state(saved)
        restore_rng(state["rng"])
    started = time.monotonic()
    previous_elapsed = state["elapsed"]
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()

    def save():
        state.update(accountants=[a.state_dict() for a in accountants], rng=rng_state(),
                     sampler_rng=[g.get_state() for g in sampler], noise_rng=[g.get_state() for g in noise],
                     elapsed=previous_elapsed + time.monotonic() - started,
                     peak_memory_bytes=max(state["peak_memory_bytes"], peak_memory()))
        save_checkpoint(path, state)

    save()
    wrapped = wrap(model) if condition["dp"] else None
    try:
        while state["round"] < config["rounds"]:
            while state["client"] < 3:
                k = state["client"]
                load_adapter(model, state["local"] if state["local"] is not None else state["global"])
                model.train()
                opt = (optimizer(wrapped, config["learning_rate"], sigmas[k], config["max_grad_norm"],
                                 config["expected_logical_batch_size"], noise[k]) if condition["dp"] else
                       torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"]))
                if state["optimizer"] is not None:
                    opt.load_state_dict(state["optimizer"])
                while state["local_step"] < config["local_steps_per_round"]:
                    guard()  # Latest durable checkpoint already contains the previous logical step.
                    indices = next(iter(poisson(len(clients[k]), rates[k], sampler[k], 1)))
                    ids = [clients[k][i] for i in indices]
                    chunks = [cache.batch(ids[i:i + config["physical_batch_size"]], device=device)
                              for i in range(0, len(ids), config["physical_batch_size"])]
                    synchronize()
                    tick = time.monotonic()
                    if condition["dp"]:
                        # FP32 forward/backward for private hooks; no reduced precision DP math.
                        private_step(wrapped, opt, chunks, lambda m, b: loss_vector(m, scheduler, b), accountants[k], rates[k])
                    else:
                        opt.zero_grad()
                        for chunk in chunks:
                            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=is_cuda and config["precision"] == "bf16"):
                                loss = loss_vector(model, scheduler, chunk).sum() / config["expected_logical_batch_size"]
                            loss.backward()
                        if not chunks:
                            for p in model.parameters():
                                if p.requires_grad:
                                    p.grad = torch.zeros_like(p)
                        opt.step()
                        opt.zero_grad()
                    synchronize()
                    duration = time.monotonic() - tick
                    state["steps"][k] += 1
                    state["local_step"] += 1
                    # Operational timing only; no private losses or sampled counts logged.
                    state["history"].append({"round": state["round"], "client": k,
                                             "step": state["steps"][k], "seconds": duration})
                    state["local"] = adapter_state(model)
                    state["optimizer"] = opt.state_dict()
                    save()
                state["updates"].append(adapter_state(model))
                state.update(client=k + 1, local_step=0, local=None, optimizer=None)
                save()
                del opt
            state["global"] = fedavg(state["updates"], list(map(len, clients)))
            state.update(round=state["round"] + 1, client=0, updates=[])
            save()
            if condition["dp"]:
                atomic_json(folder / "privacy.json", privacy_record(accountants, state["steps"], sigmas, rates, config, condition["epsilon"]))
            print(f"{condition['id']}: communication round {state['round']}/{config['rounds']}")
        load_adapter(model, state["global"])
        with (folder / "training_history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["round", "client", "step", "seconds"])
            writer.writeheader()
            writer.writerows(state["history"])
        privacy = privacy_record(accountants, state["steps"], sigmas, rates, config, condition["epsilon"]) if condition["dp"] else {"dp": False}
        atomic_json(folder / "privacy.json", privacy)
        durations = [r["seconds"] for r in state["history"] if r["step"] > 1]
        return {"training_seconds": state["elapsed"], "warmup_adjusted_steps_per_second": len(durations) / sum(durations) if durations else None,
                "maximum_gpu_memory_bytes": state["peak_memory_bytes"], "privacy": privacy}
    finally:
        if wrapped is not None:
            wrapped.remove_hooks()
        # Partial reports describe the LAST DURABLE checkpoint, including interrupts.
        durable = load_checkpoint(path, identity)
        with (folder / "training_history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["round", "client", "step", "seconds"])
            writer.writeheader()
            writer.writerows(durable["history"])
        if condition["dp"]:
            saved_accountants = [RDPAccountant() for _ in clients]
            for a, saved in zip(saved_accountants, durable["accountants"]):
                a.load_state_dict(saved)
            atomic_json(folder / "privacy.json", privacy_record(saved_accountants, durable["steps"], sigmas, rates, config, condition["epsilon"]))
