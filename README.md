# PAAS_ensemble_v2

A single, self-contained project that combines the two face real/fake detectors:

- **FFAA** — MLLM+MIDS (LLaVA-Mistral-7B reasons about the face, a T5+CLIP MIDS head turns the
  answers into a 4-class decision). Strong but heavy.
- **9-class ensemble** — MLLM-free A1+A2+A3 MIDS++ models (Effort-SVD + GenD + artifact head),
  scored on fixed candidate answers (no text generation). Cheap, frame-based.

Both emit a per-frame **fake-score** P(fake)∈[0,1]; PAAS **fuses** them and decides. Every
"combination" is a config change, so one project covers all experiments.

> Runs on the **global `python3.12`** interpreter with **`transformers==4.37.2`** (the FFAA/LLaVA
> stack). Do **not** use a venv. All model assets are bundled under `weights/` and `base_models/`.

## The loading fault — fixed
The 9-class checkpoints were trained under transformers 5.x, where `CLIPVisionModel` keys are
`image_encoder.encoder.layers…`. Under 4.37.2 the tower is nested as
`image_encoder.vision_model.encoder.layers…`, so the old loader **silently dropped 18–118 trained
tensors per model** (all Effort-SVD residuals + GenD LayerNorms), leaving un-adapted CLIP — the
"fault" seen on python3.12. PAAS's loader (`ensemble9/mids9lib/model.py:load_trainable_state_dict`)
tries both spellings, so **100% of trained tensors load**. (Loading is ~5s/member; the historical
">4.5 min" was cold network-FS reads, not SVD.)

## Layout
```
paas/                 unified layer
  env.py              path/device/offline setup (call paas.env.setup first)
  config.py           PaasConfig: which models, fusion method, decision rule
  pipeline.py         PaasPipeline: models -> fuse -> decide
  fusion.py           ffaa_only|ensemble_only|mean|weighted|max|min|per_model_or|learned
  decision.py         threshold + ambiguity (real & match<0.9 -> "ambiguous")
  io_results.py       unified results line format + offline combination frontier
  models/             ffaa_model.py (reuses FFAA code), ensemble9_model.py
ffaa/                 vendored FFAA code (llava/, mids/, utils/, models.py, train_mids.py, …)
ensemble9/            vendored 9-class code
  mids9lib/           model + training + inference (the load FIX lives here)
  configs/            mids_pp.yaml / mids_original.yaml / …
  scripts/            build_mids9.py, build_mids9_val_heldout3.py, …
weights/
  ffaa_llava_mids/    active LLaVA-Mistral-7B + mids.pth (bundled)
  ensemble9/          A1_svd_9c.pt, A2_svdgend_9c.pt, A3_midspp_9c.pt
base_models/          clip-vit-large-patch14-336/, t5-base/  (shared encoders)
config/
  ensemble9.json      bundled-path config for the 9-class ensemble
  experiments/        ready-made experiment configs (one per combination)
scripts/              run_dataset.py, combine_eval.py
train/                train_ensemble9.sh, train_mids_head.sh, train_mllm_lora.sh, train_combiner.py
```

## Quick start — inference
```python
from paas.config import PaasConfig
from paas.pipeline import PaasPipeline
pipe = PaasPipeline(PaasConfig.from_file("config/experiments/mean.json"))
for r in pipe.predict_images(["a.jpg", "b.jpg"]):
    print(r["decision"], r["forgery_score"], r["forgery_type"])
```
`config/experiments/` ships: `ensemble_only` (fast, no MLLM), `ffaa_only`, `mean`,
`weighted_ffaa` (0.2·ens+0.8·ffaa), `per_model_or`. Copy + tweak `fusion.method` /
`decision.threshold` for new combinations.

## The efficient experiment workflow (explore ALL combinations cheaply)
Running the 7B MLLM per combination is wasteful. Instead, score each model **once**, then fuse
offline:
```bash
# 1) one pass over a dataset -> per-model + fused result files
python3.12 scripts/run_dataset.py --config config/experiments/mean.json \
    --input-dir /path/with/real_and_fake_subdirs --out-dir runs/exp1 --frame-stride 10 --filter-real 1

# 2) explore every fusion + threshold offline from the two per-model files (no model re-run)
python3.12 scripts/combine_eval.py --ensemble runs/exp1/results_ensemble.txt --ffaa runs/exp1/results_ffaa.txt

# 3) optionally fit a learned combiner and use fusion.method='learned'
python3.12 train/train_combiner.py --ensemble runs/exp1/results_ensemble.txt \
    --ffaa runs/exp1/results_ffaa.txt --out config/learned_combiner.json
```
Result files use the established line format
`OK/XX/SK/ER  truth=  pred=  type=  fake=  match=  <path>`; `--filter-real` skips low-quality reals
(`SK`, excluded from accuracy); ambiguous counts as fake for accuracy.

## Training (all three targets)
- **9-class ensemble** on the new `mids9.json` (val from heldout3, filtered reals):
  `train/train_ensemble9.sh` → copy `runs/ensemble9_train/*/best.pt` into `weights/ensemble9/`.
- **FFAA MIDS head**: `DATA=… VAL=… train/train_mids_head.sh` (DeepSpeed).
- **FFAA MLLM (LLaVA LoRA)**: `train/train_mllm_lora.sh` (wraps the vendored LoRA script; heavy).

## Findings
See `docs/COMBINATION_FINDINGS.md`. Headline (frame-level, paired on 614k frames): FFAA alone is
already 99.68% fake @ 90% real; `weighted 0.2·ens+0.8·ffaa` ≈ 99.73%; the equal `mean` wins at
strict real-recall (99.18% @ 98% real). The ensemble alone is weak frame-level (95.5%) but rescues
FFAA's few confident misses.
```
