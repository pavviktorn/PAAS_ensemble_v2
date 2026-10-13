"""PAAS experiment configuration.

A single ``PaasConfig`` describes ONE experiment: which models to run (the MLLM+MIDS FFAA model
and/or the MLLM-free 9-class ensemble), how to FUSE their per-frame fake-scores, and the decision
rule (threshold + ambiguity). All "various combinations" the project supports are expressed by
varying this object -- usually loaded from a small JSON in ``config/experiments/``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from . import env


@dataclass
class FFAACfg:
    enabled: bool = True
    llava_dir: str = env.LLAVA_DIR
    mids_path: str = env.MIDS_PATH
    clip_path: str = env.BASE_CLIP
    t5_path: str = env.BASE_T5
    prompt: str = "The image is a human face image. Is it real or fake? Why?"
    temperature: float = 0.0
    top_p: Optional[float] = None
    num_beams: int = 1
    max_new_tokens: int = 256
    generate_num: int = 3          # 3 -> N=1,M=1 ; 1 -> single-answer (faster, less accurate)
    conv_mode: str = "v1"
    # Optional MIDS-format JSON of pre-generated LLaVA answers ({image, answers:[{content,result}]}).
    # When set, frames whose path is in the cache skip LLaVA generation (MIDS scoring still runs);
    # frames NOT in the cache fall back to the normal generate-then-score path.
    cache_path: Optional[str] = None


@dataclass
class Ensemble9Cfg:
    enabled: bool = True
    config_path: str = env.ENSEMBLE9_CONFIG


@dataclass
class FusionCfg:
    # method: ffaa_only | ensemble_only | mean | weighted | max | min | per_model_or | learned
    method: str = "mean"
    ensemble_weight: float = 0.5   # weight on the ensemble in `weighted` (FFAA gets 1-w)
    per_model_or_te: float = 0.70  # `per_model_or`: fake if ensemble>=te OR ffaa>=tf
    per_model_or_tf: float = 0.02
    learned_path: Optional[str] = None  # joblib/json calibrator from train/train_combiner.py


@dataclass
class DecisionCfg:
    threshold: float = 0.5343            # decision tau on the fused fake-score
    real_ambiguous_match_min: float = 0.9  # decision==real & match<this -> "ambiguous"
    treat_likely_fake_as_ambiguous: bool = True


@dataclass
class PaasConfig:
    name: str = "default_mean"
    device: str = "cuda:0"
    ffaa: FFAACfg = field(default_factory=FFAACfg)
    ensemble9: Ensemble9Cfg = field(default_factory=Ensemble9Cfg)
    fusion: FusionCfg = field(default_factory=FusionCfg)
    decision: DecisionCfg = field(default_factory=DecisionCfg)

    # ---- validation ----
    def validate(self) -> "PaasConfig":
        m = self.fusion.method
        need_ffaa = m in ("ffaa_only", "mean", "weighted", "max", "min", "per_model_or", "learned")
        need_ens = m in ("ensemble_only", "mean", "weighted", "max", "min", "per_model_or", "learned")
        if m == "ffaa_only":
            need_ens = False
        if m == "ensemble_only":
            need_ffaa = False
        if need_ffaa and not self.ffaa.enabled:
            raise ValueError(f"fusion '{m}' needs the FFAA model but ffaa.enabled=False")
        if need_ens and not self.ensemble9.enabled:
            raise ValueError(f"fusion '{m}' needs the 9-class ensemble but ensemble9.enabled=False")
        return self

    # ---- (de)serialise ----
    @classmethod
    def from_dict(cls, d: dict) -> "PaasConfig":
        d = dict(d)
        sub = {
            "ffaa": (FFAACfg, d.pop("ffaa", {})),
            "ensemble9": (Ensemble9Cfg, d.pop("ensemble9", {})),
            "fusion": (FusionCfg, d.pop("fusion", {})),
            "decision": (DecisionCfg, d.pop("decision", {})),
        }
        kw = {k: klass(**(vals or {})) for k, (klass, vals) in sub.items()}
        return cls(**d, **kw).validate()

    @classmethod
    def from_file(cls, path: str) -> "PaasConfig":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
