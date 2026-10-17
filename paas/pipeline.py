"""PAAS unified inference pipeline.

Loads exactly the models a given experiment needs, scores frames with each, fuses the per-frame
fake-scores, and applies the decision rule. This is the single entry point every experiment / script
goes through, so swapping the combination is a config change, not a code change.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from . import env, fusion as F
from .config import PaasConfig
from .decision import decide


def _needs(method: str):
    """(need_ffaa, need_ensemble) for a fusion method."""
    if method == "ffaa_only":
        return True, False
    if method == "ensemble_only":
        return False, True
    return True, True  # mean/weighted/max/min/per_model_or/learned


class PaasPipeline:
    def __init__(self, cfg: PaasConfig):
        cfg.validate()
        env.setup(cfg.device)
        self.cfg = cfg
        need_ffaa, need_ens = _needs(cfg.fusion.method)
        self.ens = None
        self.ffaa = None
        if need_ens and cfg.ensemble9.enabled:
            from .models.ensemble9_model import Ensemble9Model
            self.ens = Ensemble9Model(cfg.ensemble9.config_path, device="cuda:0")
        if need_ffaa and cfg.ffaa.enabled:
            from .models.ffaa_model import FFAAModel
            cache = self._load_ffaa_cache(cfg.ffaa.cache_path)
            self.ffaa = FFAAModel(cfg.ffaa, device="cuda:0", cache=cache)
        self.need_ffaa, self.need_ens = need_ffaa, need_ens

    @staticmethod
    def _load_ffaa_cache(path: Optional[str]) -> Optional[dict]:
        """Load a MIDS-format JSON [{image, answers:[...]}] into {abspath(image): answers}."""
        if not path:
            return None
        import json
        import os
        with open(path) as fh:
            recs = json.load(fh)
        cache = {os.path.abspath(r["image"]): r["answers"] for r in recs if r.get("answers")}
        print(f"[paas] FFAA answer cache: {len(cache)} entries from {path}")
        return cache

    # ------------------------------------------------------------------ scoring ----------------
    def predict_frames(self, rgb_list: List[np.ndarray], keys: Optional[List[str]] = None,
                       ens_batch_size: int = 32, ffaa_batch_size: int = 8) -> List[dict]:
        n = len(rgb_list)
        ens = self.ens.score_frames(rgb_list, batch_size=ens_batch_size) if self.ens else [None] * n
        ffaa = (self.ffaa.score_frames(rgb_list, batch_size=ffaa_batch_size, keys=keys)
                if self.ffaa else [None] * n)

        results = []
        for i in range(n):
            er = ens[i] if ens[i] is not None else {}
            fr = ffaa[i] if ffaa[i] is not None else {}
            e_fake = er.get("fake") if self.need_ens else None
            f_fake = fr.get("fake") if self.need_ffaa else None

            missing = ((self.need_ens and e_fake is None) or (self.need_ffaa and f_fake is None))
            if missing:
                results.append({"decision": "error",
                                "error": er.get("error") or fr.get("error") or "missing score",
                                "ensemble_fake": e_fake, "ffaa_fake": f_fake})
                continue

            fused = F.fuse_scalar(e_fake, f_fake, self.cfg.fusion)
            d = decide(fused, self.cfg.decision,
                       type_probs=er.get("type_probs"),
                       ffaa_forgery_type=fr.get("forgery_type"))
            d.update({"ensemble_fake": e_fake, "ffaa_fake": f_fake,
                      "ensemble_per_model": er.get("per_model"),
                      "ffaa_analysis": fr.get("analysis"), "ffaa_match": fr.get("match"),
                      "ffaa_answer": fr.get("answer")})
            results.append(d)
        return results

    def predict_images(self, paths: List[str], **kw) -> List[dict]:
        """Score image files on disk (RGB-loaded). Unreadable files yield decision='error'."""
        import cv2
        rgb, ok_idx = [], []
        out = [None] * len(paths)
        for i, p in enumerate(paths):
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                out[i] = {"decision": "error", "error": "unreadable", "image": p}
            else:
                rgb.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ok_idx.append(i)
        if rgb:
            scored = self.predict_frames(rgb, keys=[paths[i] for i in ok_idx], **kw)
            for j, i in enumerate(ok_idx):
                out[i] = {**scored[j], "image": paths[i]}
        return out
