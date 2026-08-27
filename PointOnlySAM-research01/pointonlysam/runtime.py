from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str) -> dict:
    cfg = json.loads(Path(path).read_text())
    for key in ("train_manifest", "val_manifest", "sam_source", "sam_checkpoint", "output_dir", "geometry_cache"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ema_update(student: torch.nn.Module, teacher: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for s, t in zip(student.parameters(), teacher.parameters()):
            t.mul_(decay).add_(s, alpha=1.0 - decay)


def make_teacher(decoder: torch.nn.Module) -> torch.nn.Module:
    teacher = copy.deepcopy(decoder)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher
