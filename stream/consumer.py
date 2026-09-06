"""Consume review events, score them in-process with ONNX Runtime, emit results.

In-process rather than calling the HTTP service. An HTTP hop per review adds
serialisation and a round trip to a model that takes single-digit milliseconds,
so the network would be most of the measured latency and the numbers would tell
you about your loopback interface instead of your model.

Micro-batching is the whole trick. Scoring 32 reviews in one ONNX call is
roughly 10x cheaper per review than 32 separate calls, because the fixed
per-call overhead dominates at these sizes. So the consumer accumulates until
either the batch is full or a deadline expires -- the same latency/throughput
trade every streaming inference system makes, made explicit here with
--batch-size and --max-wait-ms.

Reported at the end:
    e2e latency   produce -> scored, the number a downstream consumer feels
    score latency time inside ONNX, the number you can actually optimise
    lag           messages behind the head of the partition

Usage:
    python -m stream.consumer --batch-size 32 --max-wait-ms 50
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from confluent_kafka import Consumer, KafkaError, Producer, TopicPartition  # noqa: E402

from serving.app import Scorer  # noqa: E402

IN_TOPIC = "yelp.reviews.raw"
OUT_TOPIC = "yelp.reviews.scored"
DEFAULT_BROKER = "localhost:9092"

_running = True


def _stop(signum, frame):  # noqa: ARG001
    global _running
    _running = False
    print("\n[consumer] draining and shutting down")


def percentiles(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs)
    return {
        "n": int(a.size),
        "mean_ms": round(float(a.mean()), 2),
        "p50_ms": round(float(np.percentile(a, 50)), 2),
        "p95_ms": round(float(np.percentile(a, 95)), 2),
        "p99_ms": round(float(np.percentile(a, 99)), 2),
        "max_ms": round(float(a.max()), 2),
    }


def current_lag(consumer: Consumer) -> int:
    """Messages between our committed position and the partition head."""
    total = 0
    for tp in consumer.assignment():
        try:
            position = consumer.position([tp])[0].offset
            _, high = consumer.get_watermark_offsets(tp, timeout=5, cached=False)
            if position is not None and position >= 0:
                total += max(high - position, 0)
        except Exception:  # noqa: BLE001 - lag is diagnostic, never fatal
            continue
    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--broker", default=DEFAULT_BROKER)
    p.add_argument("--in-topic", default=IN_TOPIC)
    p.add_argument("--out-topic", default=OUT_TOPIC)
    p.add_argument("--group", default="yelp-cci-scorer")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-wait-ms", type=int, default=50)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--max-events", type=int, default=0, help="0 = run until interrupted")
    p.add_argument("--from-beginning", action="store_true")
    args = p.parse_args()

    signal.signal(signal.SIGINT, _stop)

    scorer = Scorer()
    print(f"[consumer] scorer ready on {scorer.provider}")

    consumer = Consumer({
        "bootstrap.servers": args.broker,
        "group.id": args.group,
        "auto.offset.reset": "earliest" if args.from_beginning else "latest",
        # Manual commit, after the batch is scored and emitted. Auto-commit
        # would acknowledge messages that were read but not yet scored, so a
        # crash mid-batch silently loses them.
        "enable.auto.commit": False,
        "max.poll.interval.ms": 300_000,
        "fetch.min.bytes": 1,
    })
    consumer.subscribe([args.in_topic])

    producer = Producer({
        "bootstrap.servers": args.broker,
        "linger.ms": 10,
        "compression.type": "lz4",
    })

    e2e_ms: list[float] = []
    score_ms: list[float] = []
    batch_sizes: list[int] = []
    processed = 0
    t_start = time.perf_counter()
    next_report = t_start + 5.0

    buffer: list[dict] = []
    deadline = time.perf_counter() + args.max_wait_ms / 1000.0

    def flush_batch() -> None:
        nonlocal buffer, processed, deadline
        if not buffer:
            deadline = time.perf_counter() + args.max_wait_ms / 1000.0
            return

        texts = [e["text"] or "" for e in buffer]
        t0 = time.perf_counter()
        probs, _tok_ms, _inf_ms = scorer.score(texts)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        now_ms = time.time() * 1000.0
        cats = scorer.categories
        for event, row in zip(buffer, probs):
            order = np.argsort(-row)[: args.top_k]
            out = {
                "review_id": event["review_id"],
                "business_id": event["business_id"],
                "user_id": event["user_id"],
                "predictions": [
                    {"category": cats[j], "probability": round(float(row[j]), 5)} for j in order
                ],
                "produced_at_ms": event["produced_at_ms"],
                "scored_at_ms": now_ms,
                "e2e_ms": round(now_ms - event["produced_at_ms"], 2),
            }
            e2e_ms.append(out["e2e_ms"])
            producer.produce(
                args.out_topic,
                key=event["business_id"].encode(),
                value=json.dumps(out).encode(),
            )

        # Per-review cost, so the number is comparable across batch sizes.
        score_ms.append(elapsed_ms / len(buffer))
        batch_sizes.append(len(buffer))
        processed += len(buffer)
        producer.poll(0)
        consumer.commit(asynchronous=True)

        buffer = []
        deadline = time.perf_counter() + args.max_wait_ms / 1000.0

    while _running:
        msg = consumer.poll(timeout=0.05)

        if msg is not None:
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"[consumer] kafka error: {msg.error()}")
            else:
                try:
                    buffer.append(json.loads(msg.value()))
                except json.JSONDecodeError:
                    print("[consumer] skipped malformed event")

        if len(buffer) >= args.batch_size or time.perf_counter() >= deadline:
            flush_batch()

        now = time.perf_counter()
        if now >= next_report:
            rate = processed / (now - t_start) if processed else 0.0
            avg_batch = float(np.mean(batch_sizes)) if batch_sizes else 0.0
            print(f"[consumer] scored={processed:,} rate={rate:,.0f}/s "
                  f"lag={current_lag(consumer):,} avg_batch={avg_batch:.1f}")
            next_report = now + 5.0

        if args.max_events and processed >= args.max_events:
            break

    flush_batch()
    producer.flush(30)

    elapsed = time.perf_counter() - t_start
    report = {
        "events_scored": processed,
        "wall_seconds": round(elapsed, 2),
        "throughput_per_sec": round(processed / elapsed, 1) if elapsed > 0 else 0,
        "batch_size_target": args.batch_size,
        "batch_size_actual_mean": round(float(np.mean(batch_sizes)), 2) if batch_sizes else 0,
        "max_wait_ms": args.max_wait_ms,
        "provider": scorer.provider,
        "score_latency_per_review": percentiles(score_ms),
        "end_to_end_latency": percentiles(e2e_ms),
        "final_lag": current_lag(consumer),
    }
    consumer.close()

    print("\n" + json.dumps(report, indent=2))
    out = Path(__file__).resolve().parents[1] / "artifacts" / "stream_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[consumer] report -> {out}")


if __name__ == "__main__":
    main()
