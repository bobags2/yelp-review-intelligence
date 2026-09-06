"""Sweep batch size against the ONNX model and report the latency/throughput curve.

The single most useful plot for a serving system, and the one people skip.
Throughput rises with batch size until the model saturates; per-review latency
falls, then flattens; but tail latency for the *first* review in a batch keeps
rising, because it waits for the whole batch. The right batch size is the knee,
not the maximum.

Run this before you pick --batch-size for the Kafka consumer. Guessing 32
because 32 is a nice number is how services end up with a p99 nobody can
explain.

Usage:
    python scripts/bench_latency.py
    ORT_PROVIDER=CUDAExecutionProvider python scripts/bench_latency.py
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serving.app import Scorer  # noqa: E402

# Length-varied so padding behaviour is exercised. A benchmark built from
# uniform-length strings measures a workload that does not exist.
CORPUS = [
    "Good.",
    "Solid breakfast, decent coffee, quick service.",
    "We booked a table for six on a Saturday and were seated within five minutes, "
    "which never happens anywhere else downtown.",
    "Brought my car in for what I thought was a wheel bearing. They diagnosed a "
    "worn CV joint instead, showed me the torn boot on the lift, and charged less "
    "than the original quote. Explained what to watch for and did not try to upsell "
    "me on anything I did not need. This is the third shop I have tried in town and "
    "the first one I will go back to.",
    "Overpriced and loud. The server forgot our drinks twice.",
    "Clean, friendly, and the stylist actually listened to what I asked for instead "
    "of doing whatever she felt like. Booked my next appointment on the way out.",
]


def make_batch(n: int, rng: random.Random) -> list[str]:
    """Draw n texts so the length distribution does not depend on n.

    Cycling the corpus (CORPUS[i % len]) made batch *composition* a function of
    batch size: n=1 got only the 5-char entry, n=2 the two shortest, and n>=4
    was the first to include the 342-char one. Under per-batch padding that
    means sequence length grows with batch size, so the sweep measured padding
    rather than batching -- ms/review went 8.16 -> 5.17 -> 19.03 between n=1, 2
    and 4, and the harness "suggested" --batch-size 2, which is three times
    worse than the truth at the sizes that matter.

    Sampling with replacement, resampled every iteration, holds the length
    distribution constant across sizes while still exercising ragged padding.
    """
    return [rng.choice(CORPUS) for _ in range(n)]


def bench_one(scorer: Scorer, batch_size: int, iters: int, warmup: int,
              seed: int = 0) -> dict:
    # Fixed seed so every batch size sees the same draw sequence, resampled per
    # iteration so the mean length converges rather than depending on one draw.
    rng = random.Random(seed)

    for _ in range(warmup):
        scorer.score(make_batch(batch_size, rng))

    totals, toks, infs = [], [], []
    for _ in range(iters):
        texts = make_batch(batch_size, rng)
        t0 = time.perf_counter()
        _, tok_ms, inf_ms = scorer.score(texts)
        totals.append((time.perf_counter() - t0) * 1000.0)
        toks.append(tok_ms)
        infs.append(inf_ms)

    a = np.asarray(totals)
    return {
        "batch_size": batch_size,
        "iters": iters,
        "batch_p50_ms": round(float(np.percentile(a, 50)), 3),
        "batch_p95_ms": round(float(np.percentile(a, 95)), 3),
        "batch_p99_ms": round(float(np.percentile(a, 99)), 3),
        "per_review_p50_ms": round(float(np.percentile(a, 50)) / batch_size, 4),
        "throughput_per_sec": round(batch_size / (float(np.mean(a)) / 1000.0), 1),
        "tokenize_share": round(statistics.mean(toks) / statistics.mean(totals), 3),
        "inference_share": round(statistics.mean(infs) / statistics.mean(totals), 3),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    scorer = Scorer()
    print(f"[bench] provider={scorer.provider}\n")

    header = (f"{'batch':>6}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}"
              f"{'ms/review':>12}{'reviews/s':>12}{'tok%':>8}{'infer%':>8}")
    print(header)
    print("-" * len(header))

    rows = []
    for n in args.sizes:
        r = bench_one(scorer, n, args.iters, args.warmup)
        rows.append(r)
        print(f"{r['batch_size']:>6}{r['batch_p50_ms']:>10.2f}{r['batch_p95_ms']:>10.2f}"
              f"{r['batch_p99_ms']:>10.2f}{r['per_review_p50_ms']:>12.3f}"
              f"{r['throughput_per_sec']:>12.1f}"
              f"{r['tokenize_share'] * 100:>7.0f}%{r['inference_share'] * 100:>7.0f}%")

    # The knee: the last size that still buys at least 10% throughput over the
    # previous one. Past that you are paying latency for nothing.
    knee = rows[0]["batch_size"]
    for prev, cur in zip(rows, rows[1:]):
        if cur["throughput_per_sec"] >= prev["throughput_per_sec"] * 1.10:
            knee = cur["batch_size"]
        else:
            break

    print(f"\n[bench] suggested consumer --batch-size {knee}")
    print(f"[bench] at that size, a review waits up to "
          f"{[r for r in rows if r['batch_size'] == knee][0]['batch_p95_ms']:.1f} ms "
          f"(p95) for its batch to complete, plus the fill wait set by --max-wait-ms")

    out = Path(__file__).resolve().parents[1] / "artifacts" / "latency_bench.json"
    out.write_text(json.dumps({"provider": scorer.provider, "suggested_batch_size": knee,
                               "rows": rows}, indent=2))
    print(f"[bench] -> {out}")


if __name__ == "__main__":
    main()
