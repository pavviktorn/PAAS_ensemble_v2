"""Score-level fusion of the two models' per-frame fake-scores.

Both models emit a per-frame P(fake) in [0,1]:
  * 9-class ensemble : mean-marginal fake-score (forgery_score).
  * FFAA MLLM+MIDS   : make_decision's forgery_score (== match if pred==fake else 1-match).

This module merges them into a single fused fake-score. Every "combination" experiment is one
``method`` here. All ops are vectorised over a frame batch (numpy), with a scalar convenience path.
The empirically strong operating points (see docs/COMBINATION_FINDINGS.md): frame-level
``weighted`` with ensemble_weight~0.2 (0.2*ens+0.8*ffaa), or ``mean`` at strict real-recall.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

METHODS = ("ffaa_only", "ensemble_only", "mean", "weighted", "max", "min", "per_model_or", "learned")


def _as_arr(x):
    return None if x is None else np.asarray(x, dtype=np.float64).reshape(-1)


def fuse(ensemble_fake, ffaa_fake, cfg) -> np.ndarray:
    """Return the fused fake-score array. ``cfg`` is a FusionCfg.

    ``ensemble_fake`` / ``ffaa_fake`` are per-frame arrays (or None when that model wasn't run).
    """
    e = _as_arr(ensemble_fake)
    f = _as_arr(ffaa_fake)
    m = cfg.method

    if m == "ffaa_only":
        _require(f, "ffaa")
        return f
    if m == "ensemble_only":
        _require(e, "ensemble")
        return e

    _require(e, "ensemble"); _require(f, "ffaa")
    if e.shape != f.shape:
        raise ValueError(f"ensemble/ffaa score length mismatch: {e.shape} vs {f.shape}")

    if m == "mean":
        return (e + f) / 2.0
    if m == "weighted":
        w = float(cfg.ensemble_weight)
        return w * e + (1.0 - w) * f
    if m == "max":
        return np.maximum(e, f)
    if m == "min":
        return np.minimum(e, f)
    if m == "per_model_or":
        # OR-rule: fake if ensemble>=te OR ffaa>=tf. Encoded as a {0,1} pseudo-score so the
        # downstream decision threshold (any tau in (0,1]) reproduces the OR decision.
        fired = (e >= float(cfg.per_model_or_te)) | (f >= float(cfg.per_model_or_tf))
        return fired.astype(np.float64)
    if m == "learned":
        return _apply_learned(e, f, cfg.learned_path)
    raise ValueError(f"unknown fusion method {m!r}; expected one of {METHODS}")


def _require(x, who):
    if x is None:
        raise ValueError(f"fusion needs the {who} model's scores but they are None "
                         f"(is that model enabled / did it run?)")


# ----------------------------------------------------------------------------------------------
# Learned combiner: a tiny, dependency-free calibrator fitted by train/train_combiner.py.
# JSON forms:  {"type":"logistic","w":[w_ens,w_ffaa],"b":bias}
#              {"type":"weighted","ensemble_weight":0.2}
# ----------------------------------------------------------------------------------------------
def _apply_learned(e: np.ndarray, f: np.ndarray, path: Optional[str]) -> np.ndarray:
    if not path:
        raise ValueError("fusion.method=='learned' requires fusion.learned_path")
    import json
    with open(path) as fh:
        spec = json.load(fh)
    t = spec.get("type", "logistic")
    if t == "weighted":
        w = float(spec["ensemble_weight"])
        return w * e + (1.0 - w) * f
    if t == "logistic":
        w_e, w_f = spec["w"]
        b = float(spec.get("b", 0.0))
        z = w_e * e + w_f * f + b
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError(f"unknown learned combiner type {t!r}")


def fuse_scalar(ensemble_fake: Optional[float], ffaa_fake: Optional[float], cfg) -> float:
    """Single-frame convenience wrapper around :func:`fuse`."""
    e = None if ensemble_fake is None else [ensemble_fake]
    f = None if ffaa_fake is None else [ffaa_fake]
    return float(fuse(e, f, cfg)[0])
