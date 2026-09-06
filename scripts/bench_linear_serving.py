"""What does the linear model actually cost to serve?

The transformer buys +0.057 macro AP (0.6335 vs 0.5760) and serves at ~11
reviews/s on CPU EP. That trade is only a decision if the other side of it is
measured, and "the linear model is obviously cheaper" is an assertion until
someone puts it through the same runtime on the same box.

So: export TF-IDF -> one-vs-rest logistic regression to ONNX via skl2onnx,
including the tokenisation, and time it through onnxruntime on the same corpus
and the same batch sizes `bench_latency.py` uses.

Serving cost is set by the vocabulary size, the label count and the graph, not
by how many rows trained the weights, so this fits on a subsample to keep the
run short. The numbers it reports are inference cost, not accuracy -- the
accuracy of this configuration is the 0.5760 already recorded in
baseline_metrics.json.

Usage:
    python scripts/bench_linear_serving.py
    python scripts/bench_linear_serving.py --fit-rows 100000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ARTIFACTS_DIR, RANDOM_SEED, get_spark  # noqa: E402
from src.train_baseline import chi2_scores_multilabel, load_split  # noqa: E402
from scripts.bench_latency import CORPUS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fit-rows", type=int, default=60_000,
                   help="training rows; affects weights, not inference cost")
    p.add_argument("--select-k", type=int, default=20_000)
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    spark = get_spark("yelp-linear-serving")
    try:
        tr_text, _num, ytr = load_split(spark, "train", args.fit_rows)
    finally:
        spark.stop()
    print(f"[linear-serve] fitting on {len(tr_text):,} rows, {ytr.shape[1]} labels")

    # Restrict the vocabulary to the chi2-selected terms so the exported graph is
    # the 20k-feature configuration, in one vectoriser that skl2onnx can convert.
    t0 = time.time()
    # strip_accents=None, not "unicode": skl2onnx cannot convert accent
    # stripping. It is a preprocessing detail with negligible compute cost, so
    # it does not affect what this script measures, but it does mean the graph
    # here is not byte-identical to the configuration that scored 0.5760.
    wide = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3,
                           max_features=200_000, sublinear_tf=True,
                           strip_accents=None)
    Xw = wide.fit_transform(tr_text)
    keep = np.argsort(chi2_scores_multilabel(Xw, ytr))[::-1][:args.select_k]
    terms = [t for t, i in sorted(wide.vocabulary_.items(), key=lambda kv: kv[1])]
    vocab_sel = {terms[i]: j for j, i in enumerate(sorted(keep))}
    print(f"[linear-serve] vocabulary {len(vocab_sel):,} terms in {time.time() - t0:.1f}s")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                  sublinear_tf=True, strip_accents=None,
                                  vocabulary=vocab_sel)),
        ("clf", OneVsRestClassifier(
            LogisticRegression(solver="liblinear", C=4.0, max_iter=400,
                               random_state=RANDOM_SEED), n_jobs=-1)),
    ])
    t0 = time.time()
    pipe.fit(tr_text, ytr)
    print(f"[linear-serve] fitted in {time.time() - t0:.1f}s")

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType
    import onnxruntime as ort

    t0 = time.time()
    onx = convert_sklearn(pipe, "yelp-linear",
                          [("input", StringTensorType([None]))],
                          options={id(pipe.named_steps["clf"]): {"zipmap": False}},
                          target_opset=17)
    out = ARTIFACTS_DIR / "linear" / "model.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(onx.SerializeToString())
    print(f"[linear-serve] exported {out} "
          f"({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8   # same as serving.Scorer
    sess = ort.InferenceSession(str(out), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    rows = []
    print(f"\n{'batch':>6}{'p50 ms':>10}{'p95 ms':>10}{'ms/review':>12}{'reviews/s':>12}")
    print("-" * 50)
    for n in args.sizes:
        rng = random.Random(0)
        for _ in range(args.warmup):
            sess.run(None, {iname: np.array([rng.choice(CORPUS) for _ in range(n)])})
        ts = []
        for _ in range(args.iters):
            batch = np.array([rng.choice(CORPUS) for _ in range(n)])
            t0 = time.perf_counter()
            sess.run(None, {iname: batch})
            ts.append((time.perf_counter() - t0) * 1000.0)
        a = np.asarray(ts)
        row = {"batch_size": n,
               "batch_p50_ms": round(float(np.percentile(a, 50)), 3),
               "batch_p95_ms": round(float(np.percentile(a, 95)), 3),
               "per_review_ms": round(float(np.percentile(a, 50)) / n, 4),
               "throughput_per_sec": round(n / (float(np.mean(a)) / 1000.0), 1)}
        rows.append(row)
        print(f"{n:>6}{row['batch_p50_ms']:>10.2f}{row['batch_p95_ms']:>10.2f}"
              f"{row['per_review_ms']:>12.4f}{row['throughput_per_sec']:>12.1f}")

    best = max(rows, key=lambda r: r["throughput_per_sec"])
    path = ARTIFACTS_DIR / "linear_serving_bench.json"
    path.write_text(json.dumps({"provider": "CPUExecutionProvider",
                                "n_features": len(vocab_sel),
                                "n_labels": int(ytr.shape[1]),
                                "points": rows}, indent=2))
    print(f"\n[linear-serve] peak {best['throughput_per_sec']:,.0f} reviews/s "
          f"at batch {best['batch_size']}")
    print(f"[linear-serve] -> {path}")


if __name__ == "__main__":
    main()
