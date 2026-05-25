import json
import os
import random
from copy import deepcopy

import numpy as np
import torch


def load_config(config_path):
    if not os.path.exists(config_path) and config_path.startswith("configs/"):
        relocated = os.path.join("training", config_path)
        if os.path.exists(relocated):
            config_path = relocated
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_dict(base, override):
    """Recursively merge override into base and return a new dict."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic mode is optional; we keep benchmark enabled for speed.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
