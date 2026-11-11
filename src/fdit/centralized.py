"""Standalone NON-PRIVATE LoRA adaptation, with exact local checkpoint continuation."""
import csv
import math
from pathlib import Path
import time

import torch

from .audit import (assert_frozen_versions, frozen_versions, gradient_norm,
                    preserve_rng, state_distance)
from .checkpoints import load_checkpoint, save_checkpoint, rng_state, restore_rng
from .diffusion import loss_vector
from .lora import adapter_state, load_adapter


class IndexStream:
    """Seeded shuffled passes; exclusively for the non-private centralized diagnostic."""
    def __init__(self, indices, seed):
        self.indices = list(indices)
        if not self.indices:
            raise ValueError("No centralized training records")
        self.generator = torch.Generator().manual_seed(seed)
        self.permutation = []
        self.offset = 0

    def draw(self, count):
        result = []
        while len(result) < count:
            if self.offset == len(self.permutation):
                self.permutation = torch.randperm(len(self.indices), generator=self.generator).tolist()
                self.offset = 0
            size = min(count - len(result), len(self.permutation) - self.offset)
            result.extend(self.indices[i] for i in self.permutation[self.offset:self.offset + size])
            self.offset += size
        return result

    def state_dict(self):
        return {"generator": self.generator.get_state(), "permutation": self.permutation, "offset": self.offset}

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"])
        self.permutation, self.offset = state["permutation"], state["offset"]


def write_history(folder, rows):
    path = folder / "training_history.csv"
    fields = ["step", "nonprivate_train_mean_loss", "gradient_l2", "adapter_update_l2",
              "seconds", "peak_allocated_bytes"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_centralized(model, scheduler, cache, indices, config, initial, folder,
                      identity, requested_steps, guard, callback, resume=True):
    """callback(step, model) runs previews/evaluation while preserving training RNG.

    No DP optimizer/accountant is created, touched, or claimed by this experiment.
    On graceful interruption, save the last fully completed optimizer step even
    between configured checkpoint intervals. Hard kills replay from the last save.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "checkpoint.pt"
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    versions = frozen_versions(model)
    sampler = IndexStream(indices, config["sampler_seed"])
    noise = torch.Generator(device=device).manual_seed(config["noise_seed"])
    load_adapter(model, initial)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"])
    state = {"identity": identity, "step": 0, "global": initial, "history": [], "elapsed_seconds": 0.}
    torch.manual_seed(config["seed"])
    if path.exists():
        if not resume:
            raise FileExistsError("Centralized diagnostic exists; use --resume")
        state = load_checkpoint(path, identity)
        load_adapter(model, state["global"])
        opt.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        noise.set_state(state["noise_rng"])
        restore_rng(state["rng"])
    write_history(folder, state["history"])
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    previous_elapsed = state["elapsed_seconds"]
    in_step = False

    def save():
        # No full frozen model is ever serialized.
        state.update({"global": adapter_state(model), "optimizer": opt.state_dict(),
                      "sampler": sampler.state_dict(), "noise_rng": noise.get_state(),
                      "rng": rng_state(), "elapsed_seconds": previous_elapsed + time.monotonic() - started})
        save_checkpoint(path, state)
        write_history(folder, state["history"])
        print(f"[central] Durable adapter/optimizer checkpoint at step {state['step']}", flush=True)

    milestones = set(config["milestones"]) | {requested_steps}
    try:
        save()
        # Recover interrupted callback work at the last saved milestone too.
        if state["step"] and (state["step"] in milestones or state["step"] % config["grid_interval"] == 0):
            with preserve_rng():
                callback(state["step"], model)
        while state["step"] < requested_steps:
            guard()
            in_step = True
            step = state["step"] + 1
            ids = sampler.draw(config["batch_size"])
            before = torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.requires_grad]).clone()
            if cuda:
                torch.cuda.synchronize()
            tick = time.perf_counter()
            model.train()
            opt.zero_grad()
            loss_total = 0.
            for start in range(0, len(ids), config["physical_batch_size"]):
                z = cache.batch(ids[start:start + config["physical_batch_size"]], device=device)
                with torch.autocast(device.type, dtype=torch.bfloat16,
                                    enabled=cuda and config["precision"] == "bf16"):
                    losses = loss_vector(model, scheduler, z, noise)
                    objective = losses.sum() / len(ids)
                loss_total += float(losses.detach().sum())
                objective.backward()
            norm = gradient_norm(model)
            if not norm > 0:
                raise RuntimeError("Centralized LoRA gradients are all zero")
            opt.step()
            opt.zero_grad()
            assert_frozen_versions(model, versions)
            after = torch.cat([p.detach().reshape(-1) for p in model.parameters() if p.requires_grad])
            delta = float(torch.linalg.vector_norm(after - before))
            if not math.isfinite(delta) or delta == 0:
                raise RuntimeError("Centralized optimizer made no adapter update")
            if cuda:
                torch.cuda.synchronize()
            duration = time.perf_counter() - tick
            state["step"] = step
            state["history"].append({"step": step, "nonprivate_train_mean_loss": loss_total / len(ids),
                                      "gradient_l2": norm, "adapter_update_l2": delta, "seconds": duration,
                                      "peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else 0})
            in_step = False
            # Aggregate losses are deliberately logged ONLY in this non-private workflow.
            if step == 1 or step % config["log_interval"] == 0:
                write_history(folder, state["history"])
                print(f"[central] step {step}/{requested_steps}: non-private mean MSE={loss_total / len(ids):.6f}, "
                      f"grad L2={norm:.5g}, update L2={delta:.5g}, {duration:.2f}s", flush=True)
            if step % config["checkpoint_interval"] == 0 or step in milestones:
                save()
            if step % config["grid_interval"] == 0 or step in milestones:
                with preserve_rng():
                    callback(step, model)
    except (InterruptedError, KeyboardInterrupt):
        # Only persist a fully completed step. Failures inside a step do not
        # overwrite the older durable checkpoint with partially updated RNG/state.
        if not in_step:
            save()
        raise
    return {"steps": state["step"], "nonprivate": True, "history": state["history"],
            "checkpoint_interval": config["checkpoint_interval"], "no_privacy_claim": True}
