import os
from pathlib import Path

import numpy as np
import torch

from .checkpoints import atomic_json
from .config import fingerprint
from .dataset import PREPROCESS


@torch.no_grad()
def build_cache(dataset, indices, vae, metadata, root, split_hash, guard=lambda: None, shard_size=512):
    identity = {"preprocess": PREPROCESS, "vae_id": metadata["vae_id"],
                "revision": metadata["model_revision"], "scaling_factor": vae.config.scaling_factor,
                "versions": metadata.get("hardware", {}).get("versions", {}),
                "split": split_hash, "indices": indices}
    key = fingerprint(identity)
    root = Path(root) / key
    root.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(indices), shard_size):
        guard()
        path = root / f"{start:06d}.npy"
        count = min(shard_size, len(indices) - start)
        print(f"[cache] Checking/building training latents {start + 1}-{start + count}/{len(indices)}", flush=True)
        if path.exists():
            arr = np.load(path, mmap_mode="r")
            if arr.shape != (count, 4, 32, 32) or arr.dtype != np.float32:
                raise RuntimeError("Corrupt latent shard")
            continue
        tmp = path.with_suffix(".tmp.npy")
        slab = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(count, 4, 32, 32))
        for offset in range(0, count, 8):
            guard()
            ids = indices[start + offset:start + min(count, offset + 8)]
            images = torch.stack([dataset[i][0] for i in ids]).to(next(vae.parameters()).device)
            # Fixed posterior mean: reproducible, unlike resampling the posterior each epoch.
            z = vae.encode(images).latent_dist.mean * vae.config.scaling_factor
            if z.shape[1:] != (4, 32, 32):
                raise RuntimeError("Unexpected encoded latent shape")
            slab[offset:offset + len(ids)] = z.float().cpu().numpy()
        slab.flush()
        del slab
        os.replace(tmp, path)
        print(f"[cache] Saved {start + count}/{len(indices)} training latents", flush=True)
    atomic_json(root / "manifest.json", {"fingerprint": key, "identity": identity})
    return LatentCache(root, indices, shard_size), key


class LatentCache:
    def __init__(self, root, indices, shard_size=512):
        self.root = Path(root)
        self.positions = {idx: i for i, idx in enumerate(indices)}
        self.shard_size = shard_size
        self.shards = {}

    def batch(self, ids, device="cuda"):
        values = []
        for idx in ids:
            pos = self.positions[int(idx)]
            start = (pos // self.shard_size) * self.shard_size
            if start not in self.shards:
                self.shards[start] = np.load(self.root / f"{start:06d}.npy", mmap_mode="r")
            values.append(self.shards[start][pos - start])
        return torch.from_numpy(np.stack(values)).to(device)
