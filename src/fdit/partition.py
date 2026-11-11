import numpy as np


def partition(indices, labels, mode, seed=43, alpha=.1):
    """Class targets allocated under exact capacities. Selection is seeded and greedy."""
    indices = np.asarray(indices)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    capacities = np.full(3, len(indices) // 3)
    capacities[:len(indices) % 3] += 1
    clients = [[] for _ in range(3)]
    classes = np.unique(labels)
    if mode not in ("iid", "non_iid"):
        raise ValueError(mode)
    if mode == "iid":
        # Cycle across shuffled classes with a continuous round-robin pointer.
        cursor = 0
        for label in classes:
            for idx in rng.permutation(indices[labels[indices] == label]):
                while capacities[cursor] == 0:
                    cursor = (cursor + 1) % 3
                clients[cursor].append(int(idx))
                capacities[cursor] -= 1
                cursor = (cursor + 1) % 3
    else:
        targets = rng.dirichlet(np.full(3, alpha), size=len(classes))
        # Interleave records from classes to avoid a last-class capacity artifact.
        records = rng.permutation(indices)
        lookup = {label: i for i, label in enumerate(classes)}
        for idx in records:
            weights = targets[lookup[labels[idx]]].copy()
            weights[capacities == 0] = 0
            weights = weights / weights.sum()
            k = rng.choice(3, p=weights)
            clients[k].append(int(idx))
            capacities[k] -= 1
    flat = sum(clients, [])
    if sorted(flat) != sorted(indices.tolist()) or len(set(flat)) != len(flat):
        raise ValueError("Partition duplicated or omitted training records")
    if max(map(len, clients)) - min(map(len, clients)) > 1:
        raise ValueError("Unbalanced partition")
    return clients


def statistics(clients, labels):
    labels = np.asarray(labels)
    counts = np.array([np.bincount(labels[c], minlength=10) for c in clients])
    freq = counts / counts.sum(axis=1, keepdims=True)
    base = counts.sum(axis=0) / counts.sum()
    def kl(p, q):
        mask = p > 0
        return float((p[mask] * np.log2(p[mask] / q[mask])).sum())
    js = []
    for p in freq:
        m = (p + base) / 2
        js.append((kl(p, m) + kl(base, m)) / 2)
    return {"counts": counts.tolist(), "frequencies": freq.tolist(),
            "js_divergence_bits": js, "mean_js": float(np.mean(js))}
