import hashlib
import json
from pathlib import Path

import yaml


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_config(path, preset="fast"):
    config = yaml.safe_load(Path(path).read_text())
    if preset not in ("fast", "full"):
        raise ValueError("preset must be fast or full")
    config.update(config[preset])
    config["preset"] = preset
    if config["num_clients"] != 3 or len(config["conditions"]) != 6:
        raise ValueError("The core protocol requires three clients and six conditions")
    for key in ("rounds", "local_steps_per_round", "physical_batch_size", "expected_logical_batch_size"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    return config
