"""
The one place where experiment configuration is loaded, overridden and validated,
plus run bookkeeping (seeding, config hash, git commit).

Every hyperparameter lives in configs/<task>_clients.yaml. Command-line overrides
use dotted keys, e.g. `--set aggregation.beta=1.0 ga.generations=3`.
"""
import copy
import hashlib
import json
import os
import random
import subprocess

import numpy as np
import torch
import yaml

# Configs are resolved relative to the CODE root (this file's repo), not the working directory, so a
# frozen worktree runs with its own frozen configs while data, cache and results come from the cwd.
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK_CONFIGS = {"breast": os.path.join(CODE_ROOT, "configs", "breast_clients.yaml"),
                "prostate": os.path.join(CODE_ROOT, "configs", "prostate_clients.yaml"),
                "prostate_ms": os.path.join(CODE_ROOT, "configs", "prostate_ms_clients.yaml"),
                "prostate_ms_runmc": os.path.join(CODE_ROOT, "configs", "prostate_ms_runmc_clients.yaml")}
REQUIRED = ["image_size", "seed", "clients", "server", "train", "aggregation", "ga", "fitness",
            "saliency", "evaluation", "splits", "loader", "method_options", "scenario"]


def _parse_value(text):
    return yaml.safe_load(text)


def apply_overrides(cfg, overrides):
    """Copy of `cfg` with dotted `key=value` overrides applied (values parsed as YAML)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"override must be key=value, got {item!r}")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node:
                raise KeyError(f"unknown config section {p!r} in {key!r}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[parts[-1]] = _parse_value(value)
    return cfg


def validate(cfg):
    """Check the config for missing sections and invalid settings; returns it unchanged or raises ValueError."""
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config is missing sections: {missing}")
    w = cfg["fitness"]["weights"]
    if len(w) != 3 or any(x < 0 for x in w) or abs(sum(w) - 1.0) > 1e-9:
        raise ValueError(f"fitness weights (w1, w2, w3) must be 3 non-negative numbers summing to 1, got {w}")
    if cfg["aggregation"]["beta"] < 0:
        raise ValueError("beta must be >= 0")
    if cfg["fitness"]["hd95_normalizer"] != cfg["image_size"][0] or cfg["image_size"][0] != cfg["image_size"][1]:
        raise ValueError("hd95_normalizer must equal the (square) image side: images are resized before metrics")
    sc = cfg["scenario"]
    if sc["partition"] not in ("random", "acquisition"):
        raise ValueError(f"scenario.partition must be random|acquisition, got {sc['partition']!r}")
    if sc["partition"] == "acquisition" and cfg.get("task") != "prostate_segmentation":
        raise ValueError("scenario.partition=acquisition is defined for the prostate task only")
    n_clients = len(cfg["clients"])
    for key in ("noisy_qi_client", "corrupted_label_client"):
        if sc[key] is not None and not (0 <= sc[key] < n_clients):
            raise ValueError(f"scenario.{key} must be null or a client index in [0, {n_clients})")
    if not (0.0 < sc.get("corrupted_label_fraction", 1.0) <= 1.0):
        raise ValueError("scenario.corrupted_label_fraction must be in (0, 1]")
    if sc["dp"]["enabled"] and (sc["dp"]["clip_norm"] <= 0 or sc["dp"]["noise_multiplier"] < 0):
        raise ValueError("dp.clip_norm must be > 0 and dp.noise_multiplier >= 0")
    if cfg["fitness"].get("cam_target") not in ("all", "pred"):
        raise ValueError("fitness.cam_target must be all|pred")
    if cfg["fitness"].get("cam_layer") not in ("fullres", "halfres"):
        raise ValueError("fitness.cam_layer must be fullres|halfres")
    pp = cfg.get("preprocessing") or {}
    if pp.get("intensity", "minmax") not in ("minmax", "zscore"):
        raise ValueError("preprocessing.intensity must be minmax|zscore")
    if pp.get("fov_crop_mm") is not None and pp["fov_crop_mm"] <= 0:
        raise ValueError("preprocessing.fov_crop_mm must be null or > 0")
    if pp and not str(cfg.get("task", "")).startswith("prostate_segmentation"):
        raise ValueError("preprocessing (FOV crop / z-score) is defined for the prostate task only")
    if cfg["method_options"]["fedprox_mu"] < 0:
        raise ValueError("fedprox_mu must be >= 0")
    g = cfg["ga"]
    if g["population"] < 1 or g["generations"] < 0 or not (0 <= g["pc"] <= 1) or not (0 <= g["pm"] <= 1):
        raise ValueError(f"invalid GA settings {g}")
    if not isinstance(g.get("acceptance_check", False), bool):
        raise ValueError("ga.acceptance_check must be true or false")
    if g.get("kappa", 1.0) <= 0:
        raise ValueError("ga.kappa (XFedGA+ Dirichlet concentration) must be > 0")
    return cfg


def load_config(task, overrides=None, path=None):
    """Load a task config (or `path`), apply the overrides and validate it."""
    with open(path or TASK_CONFIGS[task], encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return validate(apply_overrides(cfg, overrides))


def config_hash(cfg):
    """Short SHA-256 of the canonical JSON of `cfg` (part of every run id)."""
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def git_commit():
    """Commit of the code that is running (CODE_ROOT), with -dirty if tracked files changed."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=CODE_ROOT,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True,
                                        cwd=CODE_ROOT, stderr=subprocess.DEVNULL).strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def seed_everything(seed):
    """Seed python, numpy and torch (CPU + CUDA) and request deterministic cuDNN kernels.
    Must run before any CUDA work: CUBLAS_WORKSPACE_CONFIG is read when cuBLAS initialises."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def torch_generator(seed):
    """torch.Generator seeded with `seed` (deterministic data-loader shuffling)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
