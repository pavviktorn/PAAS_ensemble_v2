"""PAAS_ensemble_v2 -- combined FFAA (MLLM+MIDS) + 9-class MLLM-free ensemble.

Runs on the global python3.12 / transformers==4.37.2 interpreter. Typical use:

    from paas.config import PaasConfig
    from paas.pipeline import PaasPipeline
    pipe = PaasPipeline(PaasConfig.from_file("config/experiments/mean.json"))
    results = pipe.predict_images(["a.jpg", "b.jpg"])
"""
from .config import PaasConfig            # noqa: F401
