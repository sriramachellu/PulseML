import hashlib
import json
import logging
import os
import random
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from src import config


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed=config.SEED):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs():
    for d in (config.RAW_DIR, config.PARQUET_DIR, config.MODELS_DIR, config.FIGURES_DIR, config.REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def utc_now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def to_jsonable(obj):
    # numpy scalars/arrays -> python, NaN -> null
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path, length=12):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:length]


def array_hash(values, length=16):
    """Hash of an input array, so requests can be logged without storing the signal."""
    return hashlib.sha256(np.asarray(values, dtype=np.float32).tobytes()).hexdigest()[:length]


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT_DIR, capture_output=True, text=True
        )
        return out.stdout.strip() or None
    except OSError:
        return None
