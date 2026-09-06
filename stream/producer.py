"""Replay reviews from Parquet into Kafka as a rate-controlled event stream.

The point is not to move data. It is to create back-pressure you can measure:
run the producer faster than the consumer can score, watch lag build, then find
the batch size and thread count where lag stops growing. That number -- the
sustainable throughput -- is the only capacity claim worth making, and you
cannot get it from a benchmark script that feeds the model as fast as it will
go with nothing else happening.

Each event carries `produced_at_ms` so the consumer can compute true end-to-end
latency (produce -> score), not just its own processing time.

Usage:
    python -m stream.producer --rate 500 --limit 50000
    python -m stream.producer --rate 0            # unthrottled, to find the ceiling
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from confluent_kafka import Producer  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from src.config import FEATURES_DIR, get_spark  # noqa: E402

DEFAULT_TOPIC = "yelp.reviews.raw"
DEFAULT_BROKER = "localhost:9092"

_running = True


def _stop(signum, frame):  # noqa: ARG001
    global _running
    _running = False
    print("\n[producer] stopping after current batch")


def load_reviews(limit: int, split: str) -> list[dict]:
    spark = get_spark("yelp-stream-producer")
    try:
        df = (
            spark.read.parquet(str(FEATURES_DIR / "review_labeled"))
            .filter(F.col("split") == split)
            .select("review_id", "business_id", "user_id", "text", "review_stars")
            .limit(limit)
        )
        return [r.asDict() for r in df.collect()]
    finally:
        spark.stop()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--broker", default=DEFAULT_BROKER)
    p.add_argument("--topic", default=DEFAULT_TOPIC)
    p.add_argument("--limit", type=int, default=50_000)
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--rate", type=float, default=500.0,
                   help="events per second; 0 means unthrottled")
    p.add_argument("--loop", action="store_true", help="replay forever")
    args = p.parse_args()

    signal.signal(signal.SIGINT, _stop)

    print(f"[producer] loading up to {args.limit:,} reviews from the {args.split} split")
    rows = load_reviews(args.limit, args.split)
    if not rows:
        raise SystemExit("no rows loaded; run `make features` first")
    print(f"[producer] loaded {len(rows):,} reviews")

    producer = Producer({
        "bootstrap.servers": args.broker,
        "linger.ms": 5,
        "batch.size": 64 * 1024,
        "compression.type": "lz4",
        "acks": "1",
        "queue.buffering.max.messages": 200_000,
    })

    delivered = {"ok": 0, "failed": 0}

    def on_delivery(err, msg):  # noqa: ARG001
        if err is None:
            delivered["ok"] += 1
        else:
            delivered["failed"] += 1

    interval = (1.0 / args.rate) if args.rate > 0 else 0.0
    sent, t_start, next_report = 0, time.perf_counter(), time.perf_counter() + 5.0

    while _running:
        for row in rows:
            if not _running:
                break
            event = {
                "review_id": row["review_id"],
                "business_id": row["business_id"],
                "user_id": row["user_id"],
                "text": row["text"],
                "stars": row["review_stars"],
                "produced_at_ms": time.time() * 1000.0,
            }
            while True:
                try:
                    producer.produce(
                        args.topic,
                        # Key by business so all reviews for one business land
                        # on the same partition. Any per-business aggregation
                        # downstream then needs no shuffle.
                        key=row["business_id"].encode(),
                        value=json.dumps(event).encode(),
                        callback=on_delivery,
                    )
                    break
                except BufferError:
                    # Local queue full: the broker is the bottleneck. Serve
                    # callbacks and retry rather than dropping the event.
                    producer.poll(0.1)

            sent += 1
            producer.poll(0)

            if interval:
                time.sleep(interval)

            now = time.perf_counter()
            if now >= next_report:
                rate = sent / (now - t_start)
                print(f"[producer] sent={sent:,} delivered={delivered['ok']:,} "
                      f"failed={delivered['failed']:,} rate={rate:,.0f}/s "
                      f"queue={len(producer)}")
                next_report = now + 5.0

        if not args.loop:
            break

    # flush() returns what it could not deliver in the timeout. Ignoring that
    # return value makes an unreachable broker look like a clean run: events
    # queue locally, no callback ever fires, and the summary prints
    # delivered=0 alongside a zero exit status.
    remaining = producer.flush(30)
    elapsed = time.perf_counter() - t_start
    print(f"[producer] done: {sent:,} events in {elapsed:.1f}s "
          f"({sent / elapsed:,.0f}/s), delivered={delivered['ok']:,}, "
          f"failed={delivered['failed']:,}")

    undelivered = sent - delivered["ok"]
    if remaining or undelivered:
        print(
            f"[producer] FAILED: {undelivered:,} of {sent:,} events were never "
            f"acknowledged ({remaining:,} still queued after flush). "
            f"Is the broker reachable at {args.broker}?",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
