"""Assert the serving anomaly path reproduces the batch anomaly path exactly.

The batch job scores accounts in Spark and persists its scaler; the service
rebuilds the same score in Python from that artifact. Nothing forces the two
to agree, and when they disagreed the failure was silent: `duplication` is zero
for most accounts, so its MAD is zero on every run, so it always took the
stddev fallback -- which the batch job used but did not persist. The service
defaulted the missing scale to 1.0 and understated that component by the ratio
of the two. No exception, no log line, just a queue ordered differently online
than offline.

This is the same check `src/export_onnx.py` runs against the ONNX graph, on the
path that lacked one: sample rows the batch job already scored, push their raw
signals through the serving scorer, and require the totals to match.

Usage:
    python scripts/check_scoring_parity.py
    python scripts/check_scoring_parity.py --sample 1000 --tolerance 1e-6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serving.app import AnomalyScorer  # noqa: E402
from src.anomaly import SIGNALS  # noqa: E402
from src.config import FEATURES_DIR  # noqa: E402

# The serving scorer rounds its output to 4 decimals for the HTTP response, so
# the batch score is rounded the same way before comparison. Both sides compute
# in float64 from the same inputs, so this is a formatting alignment, not a
# tolerance: the divergence this script exists to catch was ~3.4x.
ROUND_DP = 4


def main() -> int:
    p = argparse.ArgumentParser(description="Batch vs serving anomaly-score parity")
    p.add_argument("--sample", type=int, default=200, help="rows to check")
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--show", type=int, default=5, help="worst mismatches to print")
    args = p.parse_args()

    table_path = FEATURES_DIR / "reviewer_anomaly"
    if not table_path.exists():
        print(f"[parity] no {table_path}; run `python -m src.anomaly` first", file=sys.stderr)
        return 1

    # pyarrow rather than Spark: this reads a few hundred rows, and paying 20s
    # of JVM startup inside the smoke test to do it would be silly.
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(table_path), format="parquet")
    # The per-signal z columns are what let a failure name the guilty signal
    # instead of only reporting that the totals differ. They are always written
    # by robust_zscore; tolerate their absence rather than refusing to run.
    available = set(dataset.schema.names)
    z_cols = [f"z_{s}" for s in SIGNALS if f"z_{s}" in available]
    # n_reviews is part of the input now: the batch job shrinks per-review
    # ratios by sample size, so an account's score is not a function of its
    # signals alone.
    cols = ["user_id", "n_reviews", *SIGNALS, "anomaly_score", *z_cols]
    table = dataset.to_table(columns=cols)
    n_total = table.num_rows
    if n_total == 0:
        print("[parity] reviewer_anomaly is empty", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n_total, size=min(args.sample, n_total), replace=False)
    rows = table.take(idx).to_pylist()

    scorer = AnomalyScorer()  # raises if the scaler cannot reproduce the batch score

    mismatches: list[tuple[float, dict]] = []
    skipped = 0
    checked = 0

    for row in rows:
        signals = {s: row[s] for s in SIGNALS}
        if (any(v is None for v in signals.values())
                or row["anomaly_score"] is None or row["n_reviews"] is None):
            skipped += 1
            continue

        served = scorer.score(signals, n_reviews=row["n_reviews"])
        expected = round(float(row["anomaly_score"]), ROUND_DP)
        delta = abs(served["anomaly_score"] - expected)
        checked += 1

        if delta > args.tolerance:
            # Attribute the gap by comparing each served component against the
            # z the batch job actually wrote. Recomputing from the scaler here
            # would compare the serving path against itself and always agree.
            per_signal = {
                s: (
                    served["components"][s]
                    - round(max(scorer.lo, min(scorer.hi, float(row[f"z_{s}"]))), ROUND_DP)
                )
                for s in SIGNALS
                if f"z_{s}" in row and row[f"z_{s}"] is not None
            }
            worst = max(per_signal, key=lambda s: abs(per_signal[s])) if per_signal else "?"
            mismatches.append((delta, {
                "user_id": row["user_id"],
                "batch": expected,
                "served": served["anomaly_score"],
                "delta": delta,
                "per_signal": {s: round(d, 4) for s, d in per_signal.items()},
                "worst_component": worst,
            }))

    if skipped:
        print(f"[parity] skipped {skipped} rows with null signals")

    if mismatches:
        mismatches.sort(key=lambda m: -m[0])
        print(
            f"[parity] FAIL {len(mismatches)}/{checked} rows diverge "
            f"beyond {args.tolerance:.0e}",
            file=sys.stderr,
        )
        for _, m in mismatches[: args.show]:
            print(
                f"  {m['user_id']}  batch={m['batch']:.4f}  served={m['served']:.4f}  "
                f"delta={m['delta']:.4f}  suspect={m['worst_component']}",
                file=sys.stderr,
            )
            print(f"    served - batch, per signal: {m['per_signal']}", file=sys.stderr)
        print(
            "[parity] the batch and serving scorers disagree; an account can rank "
            "into the queue offline and below threshold online",
            file=sys.stderr,
        )
        return 1

    print(
        f"[parity] OK  {checked} sampled accounts match the batch score within "
        f"{args.tolerance:.0e}"
    )
    print(f"[parity] scales in use: " + ", ".join(f"{s}={scorer.scale[s]:.4f}" for s in SIGNALS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
