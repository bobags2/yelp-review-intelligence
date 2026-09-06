"""Real-time scoring service: ONNX Runtime behind FastAPI.

Two endpoints that do different jobs:

    POST /score     review text -> category probabilities (the supervised head)
    POST /anomaly   reviewer aggregates -> queue score (the unsupervised head,
                    using the scaler persisted by the batch job so an online
                    score is identical to the offline one)

Latency is measured and exposed rather than assumed. /metrics reports p50, p95
and p99 over a rolling window, split into tokenisation and inference, because
when a text service is slow it is tokenisation about a third of the time and
you cannot tell without separating them.

Run:
    uvicorn serving.app:app --host 0.0.0.0 --port 8080 --workers 1

Use one worker. ONNX Runtime already parallelises across cores internally, so
multiple workers on one box means N sessions fighting for the same 12 cores
and each one thinking it owns them.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

ARTIFACTS = Path(os.environ.get("YELP_ARTIFACTS", Path(__file__).resolve().parents[1] / "artifacts"))
ENCODER_DIR = ARTIFACTS / "encoder"
MAX_BATCH = int(os.environ.get("MAX_BATCH", "64"))
WINDOW = int(os.environ.get("LATENCY_WINDOW", "2000"))


# --------------------------------------------------------------------------
# Latency tracking
# --------------------------------------------------------------------------


class LatencyWindow:
    """Fixed-size rolling window of observations.

    Percentiles from a rolling window, not a running mean. A mean latency of
    12 ms tells you nothing about the request that took 400 ms, and the tail is
    the only part anyone downstream actually feels.
    """

    def __init__(self, size: int = WINDOW):
        self._d: deque[float] = deque(maxlen=size)
        self._lock = threading.Lock()

    def observe(self, ms: float) -> None:
        with self._lock:
            self._d.append(ms)

    def snapshot(self) -> dict:
        with self._lock:
            xs = np.array(self._d, dtype=np.float64)
        if xs.size == 0:
            return {"n": 0}
        return {
            "n": int(xs.size),
            "mean_ms": round(float(xs.mean()), 3),
            "p50_ms": round(float(np.percentile(xs, 50)), 3),
            "p95_ms": round(float(np.percentile(xs, 95)), 3),
            "p99_ms": round(float(np.percentile(xs, 99)), 3),
            "max_ms": round(float(xs.max()), 3),
        }


LAT = {"total": LatencyWindow(), "tokenize": LatencyWindow(), "infer": LatencyWindow()}
COUNTERS = {"requests": 0, "reviews_scored": 0, "errors": 0}


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------


class Scorer:
    def __init__(self) -> None:
        manifest_path = ENCODER_DIR / "serving_manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"No serving manifest at {manifest_path}. Run `python -m src.export_onnx` first."
            )
        self.manifest = json.loads(manifest_path.read_text())
        self.categories: list[str] = self.manifest["categories"]
        self.max_len: int = self.manifest["max_len"]

        self.tokenizer = AutoTokenizer.from_pretrained(str(ENCODER_DIR))

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Pinned to physical cores. Left at the default, ORT spawns a thread per
        # logical core, and on a 12700 the E-cores drag the batch out because
        # ORT splits work evenly across threads that are not evenly fast.
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "8"))

        available = ort.get_available_providers()
        wanted = os.environ.get("ORT_PROVIDER", "CPUExecutionProvider")
        providers = [wanted] if wanted in available else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            self.manifest["onnx_path"], sess_options=opts, providers=providers
        )
        self.provider = self.session.get_providers()[0]
        print(f"[serving] provider={self.provider} labels={len(self.categories)} "
              f"max_len={self.max_len}")

    def score(self, texts: list[str]) -> tuple[np.ndarray, float, float]:
        t0 = time.perf_counter()
        enc = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="np"
        )
        t1 = time.perf_counter()
        logits = self.session.run(
            ["logits"],
            {"input_ids": enc["input_ids"].astype(np.int64),
             "attention_mask": enc["attention_mask"].astype(np.int64)},
        )[0]
        t2 = time.perf_counter()
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        return probs, (t1 - t0) * 1000, (t2 - t1) * 1000


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF.

    Inlined rather than importing scipy: the serving image should not carry
    scipy for one function, and this is Acklam's rational approximation, whose
    error is under 1.15e-9 -- four orders below the 1e-6 tolerance the parity
    check enforces.
    """
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


class AnomalyScorer:
    def __init__(self) -> None:
        path = ARTIFACTS / "anomaly_scaler.json"
        if not path.exists():
            raise RuntimeError(f"No scaler at {path}. Run `python -m src.anomaly` first.")
        cfg = json.loads(path.read_text())
        self.signals: list[str] = cfg["signals"]
        # Only the robust_z path uses these; rank_normal has no scale estimate.
        self.median: dict = cfg.get("median", {})
        self.method: str = cfg.get("method", "robust_z")
        self.scale: dict = cfg.get("scale", {})
        self.breakpoints: dict = {k: np.asarray(v, dtype=np.float64)
                                  for k, v in (cfg.get("cdf_breakpoints") or {}).items()}
        self.lo: float = cfg["clip"]["low"]
        self.hi: float = cfg["clip"]["high"]

        # The batch job shrinks per-review ratios toward the population mean by
        # sample size before scaling them, so the service must do the same
        # arithmetic from the raw signals and n_reviews or the two paths
        # diverge. Absent block = a scaler written before shrinkage existed.
        shrink = cfg.get("shrinkage") or {}
        self.pseudo_count: float = float(shrink.get("pseudo_count", 0.0))
        self.shrunk_signals: list[str] = list(shrink.get("shrunk_signals", []))
        self.prior_mean: dict = shrink.get("prior_mean", {})

        # Refuse a scaler that cannot reproduce the batch score. Defaulting a
        # missing scale to 1.0 here is what made the served `duplication`
        # component differ from the offline one by the ratio of its true scale,
        # with nothing erroring. A disabled endpoint gets noticed; a wrong
        # enforcement score does not.
        if self.method == "rank_normal":
            missing = [s for s in self.signals if s not in self.breakpoints]
        else:
            missing = [s for s in self.signals if not self.scale.get(s)]
        if missing:
            raise RuntimeError(
                f"scaler cannot reproduce the batch score for {missing}; "
                f"regenerate with `python -m src.anomaly`"
            )

    def score(self, signals: dict[str, float], n_reviews: int) -> dict:
        missing = [s for s in self.signals if s not in signals]
        if missing:
            raise ValueError(f"missing signals: {missing}")
        if self.shrunk_signals and n_reviews is None:
            raise ValueError("n_reviews is required to reproduce the batch shrinkage")

        components, total = {}, 0.0
        n = float(n_reviews or 0)
        k = self.pseudo_count
        for s in self.signals:
            raw = float(signals[s])
            if s in self.shrunk_signals:
                raw = (n * raw + k * float(self.prior_mean[s])) / (n + k)
            if self.method == "rank_normal":
                # Same breakpoint table the batch job wrote, same interpolation:
                # position in the empirical CDF, then the normal quantile. No
                # scale estimated here, so there is nothing to diverge.
                bp = self.breakpoints[s]
                eps = 0.5 / len(bp)
                cdf = float(np.searchsorted(bp, raw, side="right")) / len(bp)
                z = float(_norm_ppf(min(max(cdf, eps), 1.0 - eps)))
            else:
                z = (raw - self.median[s]) / self.scale[s]
            z = max(self.lo, min(self.hi, z))
            components[s] = round(z, 4)
            total += z
        # queue_score is what the batch job ranks on; anomaly_score is retained
        # so a caller can still see raw weirdness separately from harm.
        return {
            "anomaly_score": round(total, 4),
            "queue_score": round(total * math.log1p(n), 4),
            "components": components,
        }


app = FastAPI(title="Yelp Content & Contributor Intelligence", version="1.0")
SCORER: Scorer | None = None
ANOMALY: AnomalyScorer | None = None


@app.on_event("startup")
def _startup() -> None:
    global SCORER, ANOMALY
    SCORER = Scorer()
    try:
        ANOMALY = AnomalyScorer()
    except RuntimeError as exc:
        # The category head is the hard dependency; the anomaly head is
        # optional so the service still comes up on a fresh clone.
        print(f"[serving] anomaly head disabled: {exc}")


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="review bodies")
    top_k: int = Field(5, ge=1, le=50, description="categories returned per review")
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="drop labels below this")


class ScoreResponse(BaseModel):
    results: list[list[dict]]
    latency_ms: dict
    provider: str


class AnomalyRequest(BaseModel):
    # Required: the batch scorer shrinks ratios by sample size, so an account's
    # score is not defined by its signals alone.
    n_reviews: int = Field(..., ge=1)
    burstiness: float
    duplication: float
    rating_extremity: float
    deviation: float
    precocity: float


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------


@app.get("/healthz")
def healthz() -> dict:
    return {
        "status": "ok" if SCORER else "loading",
        "provider": SCORER.provider if SCORER else None,
        "anomaly_head": ANOMALY is not None,
    }


@app.get("/metrics")
def metrics() -> dict:
    return {"counters": dict(COUNTERS), "latency": {k: v.snapshot() for k, v in LAT.items()}}


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    if SCORER is None:
        raise HTTPException(503, "model not loaded")
    if len(req.texts) > MAX_BATCH:
        raise HTTPException(413, f"batch of {len(req.texts)} exceeds MAX_BATCH={MAX_BATCH}")

    t0 = time.perf_counter()
    try:
        probs, tok_ms, inf_ms = SCORER.score(req.texts)
    except Exception as exc:
        COUNTERS["errors"] += 1
        raise HTTPException(500, f"scoring failed: {exc}") from exc
    total_ms = (time.perf_counter() - t0) * 1000

    LAT["total"].observe(total_ms)
    LAT["tokenize"].observe(tok_ms)
    LAT["infer"].observe(inf_ms)
    COUNTERS["requests"] += 1
    COUNTERS["reviews_scored"] += len(req.texts)

    cats = SCORER.categories
    results = []
    for row in probs:
        order = np.argsort(-row)[: req.top_k]
        results.append([
            {"category": cats[j], "probability": round(float(row[j]), 5)}
            for j in order if row[j] >= req.threshold
        ])

    return ScoreResponse(
        results=results,
        latency_ms={"total": round(total_ms, 3),
                    "tokenize": round(tok_ms, 3),
                    "inference": round(inf_ms, 3),
                    "per_review": round(total_ms / len(req.texts), 3)},
        provider=SCORER.provider,
    )


@app.post("/anomaly")
def anomaly(req: AnomalyRequest) -> dict:
    if ANOMALY is None:
        raise HTTPException(503, "anomaly head not loaded; run `python -m src.anomaly`")
    try:
        payload = req.model_dump()
        return ANOMALY.score(payload, n_reviews=payload.pop("n_reviews"))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
