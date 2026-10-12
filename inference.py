#!/usr/bin/env python3.12
"""PAAS_ensemble_v2 inference CLI.

Run the combined (or single-model) face real/fake detector on one or more images. The model set and
fusion are taken from an experiment config; convenience flags let you override the fusion method,
weight, threshold, and device without editing the config -- handy for quick combination experiments.

Examples:
  python3.12 inference.py face.jpg                              # default config (config/experiments/mean.json)
  python3.12 inference.py a.jpg b.png --json
  python3.12 inference.py --dir /path/to/folder --config config/experiments/ensemble_only.json
  python3.12 inference.py face.jpg --fusion weighted --ensemble-weight 0.2 --threshold 0.134
  python3.12 inference.py face.jpg --fusion ffaa_only            # MLLM+MIDS only
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from paas.config import PaasConfig
from paas.pipeline import PaasPipeline

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
DEFAULT_CONFIG = os.path.join(ROOT, "config", "experiments", "weighted_ffaa.json")


def collect_images(images, directory):
    # accept a single path string, a list of paths, or None
    if images is None:
        images = []
    elif isinstance(images, str):
        images = [images] if images else []
    files = [p for p in images if p]
    if directory and os.path.isdir(directory):
        for dp, _, fns in os.walk(directory):
            for fn in sorted(fns):
                if os.path.splitext(fn)[1].lower() in IMG_EXT:
                    files.append(os.path.join(dp, fn))
    # de-duplicate, preserving order (handles --images overlapping --dir)
    seen, out = set(), []
    for f in files:
        k = os.path.abspath(f)
        if k not in seen:
            seen.add(k); out.append(f)
    return out


def build_config(args) -> PaasConfig:
    cfg = PaasConfig.from_file(args.config)
    if args.device:
        cfg.device = args.device
    if args.fusion:
        cfg.fusion.method = args.fusion
        # auto-enable the models the chosen fusion needs
        cfg.ffaa.enabled = args.fusion not in ("ensemble_only",)
        cfg.ensemble9.enabled = args.fusion not in ("ffaa_only",)
    if args.ensemble_weight is not None:
        cfg.fusion.ensemble_weight = args.ensemble_weight
    if args.learned_path:
        cfg.fusion.learned_path = args.learned_path
    if args.threshold is not None:
        cfg.decision.threshold = args.threshold
    if args.ambiguous_match_min is not None:
        cfg.decision.real_ambiguous_match_min = args.ambiguous_match_min
    return cfg.validate()


def main():
    ap = argparse.ArgumentParser(description="PAAS_ensemble_v2 combined inference")
    ap.add_argument("--images", default="/datasets/work/vLLM/temp/PAAS_ensemble_v2/images/0034.jpg", help="image file paths")
    ap.add_argument("--dir", default="/datasets/work/vLLM/temp/PAAS_ensemble_v2/images", help="recurse a folder and score every image in it")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help=f"experiment config (default: {DEFAULT_CONFIG})")
    ap.add_argument("--fusion", choices=("ffaa_only", "ensemble_only", "mean", "weighted",
                                          "max", "min", "per_model_or", "learned"),
                    help="override fusion method (auto-enables the needed models)")
    ap.add_argument("--ensemble-weight", type=float, default=None, help="weight on ensemble for 'weighted'")
    ap.add_argument("--learned-path", default=None, help="combiner json for 'learned'")
    ap.add_argument("--threshold", type=float, default=None, help="override decision threshold")
    ap.add_argument("--ambiguous-match-min", type=float, default=None,
                    help="override: decision==real & match<this -> 'ambiguous' (default 0.9)")
    ap.add_argument("--device", default="", help="cuda:0 / cuda:2 / cpu (default from config)")
    ap.add_argument("--ens-batch", type=int, default=32)
    ap.add_argument("--ffaa-batch", type=int, default=8)
    ap.add_argument("--json", action="store_true", help="emit a JSON array instead of human-readable lines")
    args = ap.parse_args()

    files = collect_images(args.images, args.dir)
    if not files:
        ap.error("no images given (pass image paths and/or --dir)")

    cfg = build_config(args)
    print(f"# loading models for config='{cfg.name}' fusion={cfg.fusion.method} "
          f"(ffaa={cfg.ffaa.enabled}, ensemble9={cfg.ensemble9.enabled}) on {cfg.device} ... "
          f"[{len(files)} image(s); the 7B MLLM load can take a few minutes]", flush=True)
    t0 = time.time()
    pipe = PaasPipeline(cfg)
    load_t = time.time() - t0
    print(f"# models loaded in {load_t:.1f}s; scoring ...", flush=True)

    t1 = time.time()
    results = pipe.predict_images(files, ens_batch_size=args.ens_batch, ffaa_batch_size=args.ffaa_batch)
    dt = (time.time() - t1) / max(len(files), 1)
    for r in results:
        r["processing_time_sec"] = round(dt, 4)

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"# PAAS config='{cfg.name}' fusion={cfg.fusion.method} threshold={cfg.decision.threshold} "
          f"| models: ffaa={pipe.ffaa is not None} ensemble9={pipe.ens is not None} "
          f"| load={load_t:.1f}s, {dt*1000:.0f}ms/img")
    for r in results:
        img = r.get("image", "?")
        if r.get("decision") == "error":
            print(f"  ERROR  {img}  ({r.get('error')})")
            continue
        extra = []
        if r.get("ensemble_fake") is not None:
            extra.append(f"ens={r['ensemble_fake']:.4f}")
        if r.get("ffaa_fake") is not None:
            extra.append(f"ffaa={r['ffaa_fake']:.4f}")
        extra = ("  [" + " ".join(extra) + "]") if extra else ""
        print(f"  {r['decision']:9s} {r['forgery_type']:8s} score={r['forgery_score']:.4f} "
              f"match={r['match_score']:.4f}{extra}  {img}")


if __name__ == "__main__":
    main()
