"""How much of this task is vocabulary?

The lr-svd ablation showed the 256-dim LSA projection costs 0.139 macro AP,
but it conflates two things: *few* dimensions, and a *dense semantic*
projection. This sweep separates them. Chi-squared selection keeps the
representation lexical and sparse and varies only how many features survive,
so the curve answers a question LSA cannot:

  If a few thousand selected terms recover most of the full-vocabulary score,
  the signal is concentrated in a modest set of distinctive words, and a
  contextual model -- whose advantage is disambiguating words by context --
  has little left to work with.

  If the score keeps climbing out to the full 200k, the signal is diffuse
  across a long tail of rare terms, which is a different picture and rather
  more favourable to a learned representation.

Either way the curve is the artifact: "how much of this task is vocabulary"
is a better question than "did the transformer win".

The estimator and its hyperparameters are held fixed across every point --
this reuses train_baseline.fit_linear -- so the only variable is how many
features the model sees.

Usage:
    python scripts/sweep_vocab_size.py
    python scripts/sweep_vocab_size.py --sizes 1000 20000 200000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ARTIFACTS_DIR, BASELINE_SAMPLE_ROWS, get_spark  # noqa: E402
from src.train_baseline import (  # noqa: E402
    chi2_scores_multilabel,
    evaluate,
    fit_linear,
    load_split,
)

DEFAULT_SIZES = [256, 1_000, 5_000, 20_000, 100_000, 200_000]


def main() -> None:
    p = argparse.ArgumentParser(description="Vocabulary-size sweep with chi2 selection")
    p.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    p.add_argument("--max-features", type=int, default=200_000)
    p.add_argument("--train-rows", type=int, default=BASELINE_SAMPLE_ROWS)
    p.add_argument("--test-rows", type=int, default=None)
    args = p.parse_args()

    vocab = json.loads((ARTIFACTS_DIR / "category_vocab.json").read_text())["categories"]

    spark = get_spark("yelp-vocab-sweep")
    try:
        tr_text, _tr_num, ytr = load_split(spark, "train", args.train_rows)
        te_text, _te_num, yte = load_split(spark, "test", args.test_rows)
    finally:
        spark.stop()
    print(f"[sweep] train={len(tr_text):,}  test={len(te_text):,}  labels={ytr.shape[1]}")

    # Vectorise once at full width; every point below is a subset of these
    # columns, so the sweep varies selection only, never the tokenisation.
    t0 = time.time()
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_features=args.max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    Xtr_full = tfidf.fit_transform(tr_text)
    Xte_full = tfidf.transform(te_text)
    n_features = Xtr_full.shape[1]
    print(f"[sweep] tfidf {n_features:,} features in {time.time() - t0:.1f}s")

    results = []
    scores_chi2 = None   # computed once, reused for every k
    for k in args.sizes:
        if k >= n_features:
            Xtr, Xte, k_eff = Xtr_full, Xte_full, n_features
            print(f"\n[sweep] k={k:,} -> using all {n_features:,} features")
        else:
            t0 = time.time()
            # chi2 needs non-negative input; TF-IDF is non-negative by
            # construction. Scored on train only -- scoring on the full matrix
            # would leak test labels into the feature choice.
            if scores_chi2 is None:
                scores_chi2 = chi2_scores_multilabel(Xtr_full, ytr)
            keep = np.argsort(scores_chi2)[::-1][:k]
            keep.sort()
            Xtr, Xte, k_eff = Xtr_full[:, keep], Xte_full[:, keep], k
            print(f"\n[sweep] k={k:,} selected in {time.time() - t0:.1f}s "
                  f"(chi2 {scores_chi2[keep].min():.1f}-{scores_chi2[keep].max():.1f})")

        t0 = time.time()
        scores = fit_linear(Xtr, ytr, Xte)
        m = evaluate(yte, scores, vocab)
        row = {
            "k": k_eff,
            "macro_average_precision": m["macro_average_precision"],
            "micro_average_precision": m["micro_average_precision"],
            "macro_lift": m["macro_lift"],
            "fit_seconds": time.time() - t0,
        }
        results.append(row)
        print(f"[sweep] k={k_eff:,}  macro AP {row['macro_average_precision']:.4f}  "
              f"micro AP {row['micro_average_precision']:.4f}  "
              f"({row['fit_seconds']:.0f}s)")

    out = ARTIFACTS_DIR / "vocab_sweep.json"
    out.write_text(json.dumps({"floor": None, "points": results}, indent=2))

    best = max(r["macro_average_precision"] for r in results)
    print(f"\n{'k':>10}{'macro AP':>11}{'micro AP':>11}{'% of best':>11}")
    print("-" * 43)
    for r in results:
        print(f"{r['k']:>10,}{r['macro_average_precision']:>11.4f}"
              f"{r['micro_average_precision']:>11.4f}"
              f"{r['macro_average_precision'] / best * 100:>10.1f}%")
    print(f"\n[sweep] -> {out}")


if __name__ == "__main__":
    main()
