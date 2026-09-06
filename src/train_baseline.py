"""Stage 4 - multi-label baselines for content-type classification.

Two models, because one number in isolation says nothing:

  linear  TF-IDF -> one-vs-rest logistic regression. The strong, boring text
          baseline. If a deep model cannot beat this, the deep model is
          decoration.
  xgb     TF-IDF -> truncated SVD (256 dims) + reviewer/metadata features ->
          one-vs-rest gradient boosting. Runs on the 3070 Ti with
          XGB_DEVICE=cuda.

Every metric is reported against the prevalence floor. Average precision for a
label that appears in 2% of rows is 0.02 for a model that has learned nothing,
so a headline "AP 0.31" is meaningless without the floor printed beside it.
That column is the difference between a portfolio project and a plot.

Usage:
    python -m src.train_baseline --model both
    XGB_DEVICE=cuda python -m src.train_baseline --model xgb
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
from pyspark.sql import functions as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import (
    ARTIFACTS_DIR,
    BASELINE_SAMPLE_ROWS,
    FEATURES_DIR,
    RANDOM_SEED,
    get_spark,
)

NUMERIC_COLS = [
    "review_stars",
    "text_len",
    "word_count",
    "user_review_count",
    "user_average_stars",
    "user_fans",
    "friend_count",
    "elite_years",
    "account_age_days_at_review",
    "star_deviation_from_user_mean",
]


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_split(spark, split: str, limit: int | None):
    path = FEATURES_DIR / "review_labeled"
    df = spark.read.parquet(str(path)).filter(F.col("split") == split)

    if limit is not None:
        total = df.count()
        if total > limit:
            # Sample rather than limit(): limit() takes whichever partitions
            # come back first, which on year-partitioned data means a biased
            # slice of the earliest businesses.
            df = df.sample(withReplacement=False, fraction=limit / total, seed=RANDOM_SEED)

    cols = ["text", "labels"] + NUMERIC_COLS
    pdf = df.select(*cols).toPandas()

    y = np.vstack(pdf["labels"].apply(lambda a: np.asarray(a, dtype=np.int8)).values)
    text = pdf["text"].fillna("").tolist()
    num = pdf[NUMERIC_COLS].astype("float64").fillna(0.0).to_numpy()
    num = np.nan_to_num(num, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return text, num, y


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def evaluate(y_true: np.ndarray, y_score: np.ndarray, vocab: list[str]) -> dict:
    per_label = []
    for j, name in enumerate(vocab):
        yt, ys = y_true[:, j], y_score[:, j]
        prevalence = float(yt.mean())
        if yt.sum() == 0 or yt.sum() == len(yt):
            per_label.append({"category": name, "prevalence": prevalence,
                              "average_precision": None, "roc_auc": None, "lift": None})
            continue
        ap = float(average_precision_score(yt, ys))
        per_label.append({
            "category": name,
            "prevalence": prevalence,
            "average_precision": ap,
            "roc_auc": float(roc_auc_score(yt, ys)),
            # How many times better than guessing the base rate.
            "lift": ap / prevalence if prevalence > 0 else None,
        })

    scored = [r for r in per_label if r["average_precision"] is not None]
    macro_ap = float(np.mean([r["average_precision"] for r in scored])) if scored else None
    macro_prev = float(np.mean([r["prevalence"] for r in scored])) if scored else None
    micro_ap = float(average_precision_score(y_true.ravel(), y_score.ravel()))

    return {
        "macro_average_precision": macro_ap,
        "macro_prevalence_floor": macro_prev,
        "macro_lift": (macro_ap / macro_prev) if macro_ap and macro_prev else None,
        "micro_average_precision": micro_ap,
        "n_labels_scored": len(scored),
        "per_label": sorted(per_label, key=lambda r: -(r["lift"] or 0)),
    }


def print_report(name: str, m: dict, top_n: int = 12) -> None:
    print(f"\n=== {name} ===")
    print(f"macro AP  {m['macro_average_precision']:.4f}   "
          f"floor {m['macro_prevalence_floor']:.4f}   "
          f"lift {m['macro_lift']:.2f}x")
    print(f"micro AP  {m['micro_average_precision']:.4f}")
    print(f"\n{'category':<32}{'prev':>8}{'AP':>9}{'ROC':>8}{'lift':>8}")
    print("-" * 65)
    for r in m["per_label"][:top_n]:
        if r["average_precision"] is None:
            continue
        print(f"{r['category'][:31]:<32}{r['prevalence']:>8.4f}"
              f"{r['average_precision']:>9.4f}{r['roc_auc']:>8.4f}{r['lift']:>8.2f}")


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def fit_linear(Xtr, ytr, Xte):
    n_labels = ytr.shape[1]
    scores = np.zeros((Xte.shape[0], n_labels), dtype=np.float32)
    for j in range(n_labels):
        col = ytr[:, j]
        if col.sum() == 0:
            continue
        clf = LogisticRegression(
            solver="liblinear", C=4.0, max_iter=400, random_state=RANDOM_SEED,
        )
        clf.fit(Xtr, col)
        scores[:, j] = clf.predict_proba(Xte)[:, 1]
        if (j + 1) % 10 == 0:
            print(f"  [linear] {j + 1}/{n_labels} labels")
    return scores


def fit_xgb(Xtr, ytr, Xte):
    from xgboost import XGBClassifier

    device = os.environ.get("XGB_DEVICE", "cpu")
    n_labels = ytr.shape[1]
    scores = np.zeros((Xte.shape[0], n_labels), dtype=np.float32)
    for j in range(n_labels):
        col = ytr[:, j]
        pos = int(col.sum())
        if pos == 0:
            continue
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.7,
            min_child_weight=4,
            reg_lambda=1.5,
            tree_method="hist",
            device=device,
            # Rebalance rather than resample: at 50 labels most are rare, and
            # undersampling the negatives 50 separate times throws away most of
            # the corpus.
            scale_pos_weight=max((len(col) - pos) / pos, 1.0),
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbosity=0,
        )
        clf.fit(Xtr, col)
        scores[:, j] = clf.predict_proba(Xte)[:, 1]
        if (j + 1) % 10 == 0:
            print(f"  [xgb] {j + 1}/{n_labels} labels (device={device})")
    return scores


# --------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["linear", "xgb", "both"], default="both")
    p.add_argument("--max-features", type=int, default=200_000)
    p.add_argument("--svd-dims", type=int, default=256)
    p.add_argument("--train-rows", type=int, default=BASELINE_SAMPLE_ROWS)
    p.add_argument("--test-rows", type=int, default=None)
    args = p.parse_args()

    vocab = json.loads((ARTIFACTS_DIR / "category_vocab.json").read_text())["categories"]

    spark = get_spark("yelp-baseline")
    try:
        t0 = time.time()
        tr_text, tr_num, ytr = load_split(spark, "train", args.train_rows)
        te_text, te_num, yte = load_split(spark, "test", args.test_rows)
    finally:
        spark.stop()
    print(f"[baseline] train={len(tr_text):,}  test={len(te_text):,}  "
          f"labels={ytr.shape[1]}  load {time.time() - t0:.1f}s")

    t0 = time.time()
    tfidf = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_features=args.max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    Xtr_txt = tfidf.fit_transform(tr_text)
    Xte_txt = tfidf.transform(te_text)
    print(f"[baseline] tfidf {Xtr_txt.shape[1]:,} features in {time.time() - t0:.1f}s")

    results = {}

    if args.model in ("linear", "both"):
        t0 = time.time()
        s = fit_linear(Xtr_txt, ytr, Xte_txt)
        results["linear_tfidf"] = evaluate(yte, s, vocab)
        results["linear_tfidf"]["fit_seconds"] = time.time() - t0
        print_report("linear / tf-idf", results["linear_tfidf"])

    if args.model in ("xgb", "both"):
        t0 = time.time()
        dims = min(args.svd_dims, min(Xtr_txt.shape) - 1)
        svd = TruncatedSVD(n_components=dims, random_state=RANDOM_SEED)
        Xtr = np.hstack([svd.fit_transform(Xtr_txt).astype(np.float32), tr_num])
        Xte = np.hstack([svd.transform(Xte_txt).astype(np.float32), te_num])
        print(f"[baseline] svd {dims} dims, "
              f"explained variance {svd.explained_variance_ratio_.sum():.3f}")
        s = fit_xgb(Xtr, ytr, Xte)
        results["xgb_svd_meta"] = evaluate(yte, s, vocab)
        results["xgb_svd_meta"]["fit_seconds"] = time.time() - t0
        print_report("xgboost / svd + metadata", results["xgb_svd_meta"])

    out = ARTIFACTS_DIR / "baseline_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[baseline] metrics -> {out}")


if __name__ == "__main__":
    main()
