from pathlib import Path

import numpy as np

from .config import fingerprint

PREPROCESS = {"rgb": True, "resize": [256, 256], "interpolation": "bilinear",
              "antialias": True, "normalization": "[-1,1]", "posterior": "mean"}


def stratified_split(labels, seed=42):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    splits = {k: [] for k in ("train", "val", "test")}
    for label in np.unique(labels):
        ids = rng.permutation(np.flatnonzero(labels == label))
        a, b = int(.8 * len(ids)), int(.9 * len(ids))
        for key, part in zip(splits, (ids[:a], ids[a:b], ids[b:])):
            splits[key].extend(part.tolist())
    for key in splits:
        splits[key] = sorted(splits[key])
    flat = sum(splits.values(), [])
    if len(flat) != len(set(flat)) or set(flat) != set(range(len(labels))):
        raise ValueError("Invalid or overlapping split")
    return splits


def load_dataset(root):
    from torchvision.datasets import EuroSAT
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
    transform = Compose([Resize((256, 256), interpolation=InterpolationMode.BILINEAR, antialias=True),
                         ToTensor(), Normalize([.5] * 3, [.5] * 3)])
    return EuroSAT(root=root, download=True, transform=transform)


def split_manifest(dataset, seed):
    ids = [Path(p).parent.name + "/" + Path(p).name for p, _ in dataset.samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Sample IDs are not unique")
    splits = stratified_split(dataset.targets, seed)
    return {"sample_ids": ids, "indices": splits, "hash": fingerprint({"ids": ids, "splits": splits}),
            "preprocess": PREPROCESS, "source": "https://github.com/phelber/EuroSAT",
            "attribution": "Helber, Bischke, Dengel, Borth; EuroSAT, IEEE JSTARS 2019; Copernicus Sentinel-2",
            "license": "Canonical EuroSAT repository: MIT (Copyright Patrick Helber 2023); retain upstream imagery terms",
            "simulation_only": True}
