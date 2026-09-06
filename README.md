# Yelp Content & Contributor Intelligence

An end-to-end machine learning pipeline over the [Yelp Open Dataset](https://www.yelp.com/dataset)
(~7M reviews, 150,346 businesses, 11 metro areas): Spark ingestion and feature
engineering, a supervised multi-label content classifier, and an unsupervised
reviewer-anomaly ranker that produces enforcement queue candidates.

## Why these two tasks

**The Yelp Open Dataset contains no fake-review labels.** Yelp's filtered
reviews are not published. Any project claiming a supervised fake-review
detector on this data is either training on heuristic labels it invented — a
model that learns to reproduce its own rules — or quietly leaking. So the work
is split by what the data can honestly support:

| Head | Task | Labels | Maps to |
|---|---|---|---|
| Supervised | Multi-label business-category classification from review text | Real, from `business.categories` | "content type classification" |
| Unsupervised | Behavioural reviewer-anomaly ranking | None — produces a review queue | "sophisticated bot detection" |

The second head is framed as candidate generation for human adjudication, which
is how enforcement systems are actually built: a cheap signal orders the queue,
humans work the top of it, and those decisions eventually become the labels a
supervised model can train on.

## Design decisions worth defending

**Grouped split by `business_id`, not random and not temporal.** Reviews
frequently name the business they are about. A random row split lets the model
memorise "Joe's Diner → Restaurants" from training rows and score inflated on
test rows about the same business. A temporal split has the same defect,
because a business straddles the cutoff. Holding out whole businesses measures
the thing being claimed: generalisation to a business never seen before.
Assignment is by stable hash, so the split is reproducible and does not drift
when data is appended.

**No business-derived features in the supervised head.** `attributes`,
`name`, and the business star average all encode the label almost directly.
Inputs are restricted to the review and its author.

**Every metric is reported against its prevalence floor.** Average precision
for a label appearing in 2% of rows is 0.02 for a model that has learned
nothing. The `lift` column is AP divided by that floor. A headline AP without
the floor beside it is not a result.

**Median/MAD scaling in the anomaly ranker, not mean/stdev.** The tail being
hunted is exactly what would inflate a standard deviation and hide itself.

**Explicit Spark schemas.** Inference costs an extra full pass over 8.65 GB and,
on `business.attributes`, produces a different struct depending on which rows
Spark samples first — which makes output non-reproducible run to run.

## Running it

Requires **Java 17+** (Spark 4.x) and Python 3.10+.

```bash
pip install -r requirements.txt

# Prove the pipeline works before the 4.35 GB download finishes.
make smoke
```

`make smoke` generates Yelp-shaped synthetic data — identical field names and
types — and runs all four stages against it in a couple of minutes. It also
plants burst-posting, copy-paste reviewer accounts, which the anomaly ranker
should surface at the top of the queue.

Expect the classifier to show ~1.0x lift on synthetic data. The generator draws
review text from a shared phrase pool regardless of category, so there is no
signal to find, and the harness correctly reports none. That is the smoke test
passing, not failing.

For the real thing: download from <https://www.yelp.com/dataset>, unpack the
tarball **twice** (it is a gzipped tar), drop the five `.json` files into
`data/raw/`, then:

```bash
make all                    # ingest -> features -> anomaly -> baseline
make baseline-gpu           # XGBoost on CUDA
```

## Pipeline

```
data/raw/*.json
   │  src/ingest.py      explicit schemas, JSON -> Parquet (reviews partitioned by year)
   ▼
data/parquet/{business,review,user,checkin,tip}
   │  src/features.py    top-K multi-hot labels, reviewer features, grouped split
   │  src/anomaly.py     5 behavioural signals -> robust z -> ranked queue
   ▼
data/features/{review_labeled,reviewer_anomaly}
   │  src/train_baseline.py
   ▼
artifacts/{category_vocab.json,baseline_metrics.json,review_queue_top.csv}
```

## Anomaly signals

| Signal | Definition |
|---|---|
| `burstiness` | peak single-day review volume ÷ distinct active days |
| `duplication` | share of the author's reviews sharing a normalised text fingerprint with another of their own |
| `rating_extremity` | share of reviews at 1 or 5 stars |
| `deviation` | mean absolute gap from business consensus (businesses with ≥10 reviews) |
| `precocity` | inverse log of days between account creation and first review |

Components are robust-z scored, clipped to [-5, 8] so one saturated signal
cannot carry an account into the queue alone, then summed. Every component is
emitted alongside the total, so a flagged account arrives with its reason —
the minimum bar for anything a human is expected to action.

## Tuning

All knobs are environment variables (see `src/config.py`):

| Variable | Default | Notes |
|---|---|---|
| `SPARK_DRIVER_MEMORY` | `40g` | local mode runs the executor in the driver JVM |
| `SPARK_SHUFFLE_PARTITIONS` | `96` | ~8× physical cores |
| `TOP_K_CATEGORIES` | `50` | of ~1,300 raw categories |
| `TEST_FRACTION` | `0.20` | grouped by `business_id` |
| `BASELINE_SAMPLE_ROWS` | `400000` | train-side sample |
| `XGB_DEVICE` | `cpu` | set `cuda` for GPU |

## Known data quirks

`account_age_days_at_review` is occasionally negative — reviews timestamped
before their author's `yelping_since`. Kept rather than clipped: it is a real
data-quality signal, and silently repairing it would hide a property of the
source.

## Deep model

`src/train_encoder.py` fine-tunes a transformer encoder (DistilBERT by default)
with mean pooling and a linear multi-label head. Sized for 8 GB of VRAM:
per-batch padding rather than padding to `max_len`, fp16 autocast, and
discriminative learning rates (small for the pretrained encoder, large for the
randomly initialised head).

If it OOMs, reach for the knobs in this order — `--grad-accum 2`, then
`--max-len 192`, then `--batch-size 16`, then a smaller `--model-name`.
Gradient checkpointing is last: it costs ~30% throughput for memory you
probably do not need at these sizes.

`pos_weight` for rare labels is **capped at 20x**. Uncapped, a label at 0.2%
prevalence gets 500x, and the loss for that one label dominates the gradient
and destabilises the encoder.

The point of this stage is not to beat the linear baseline by a lot. It is to
find out whether it beats it *at all*, and by how much. If the answer is two
points of macro AP, the honest engineering call is to ship the TF-IDF model and
skip the serving complexity entirely.

## Export

`src/export_onnx.py` exports to ONNX and then **verifies the export is
faithful**. A graph that loads is not the same as a graph that computes the
same function — operator fallbacks and opset differences silently change
outputs, and the failure mode is a model serving subtly wrong scores for months
without erroring once. The script re-runs both graphs on real review text and
exits non-zero if they diverge beyond tolerance. It also checks the dynamic
axes actually work at a different batch shape than the one used for tracing.

## Serving

```bash
make serve          # uvicorn on :8080, CPU execution provider
make serve-gpu      # CUDAExecutionProvider (needs onnxruntime-gpu)
make bench          # batch-size sweep
```

| Endpoint | Purpose |
|---|---|
| `POST /score` | review text → category probabilities |
| `POST /anomaly` | reviewer aggregates → queue score, using the persisted scaler |
| `GET /metrics` | p50/p95/p99 over a rolling window, split tokenise vs inference |
| `GET /healthz` | liveness and which provider loaded |

**One uvicorn worker.** ONNX Runtime already parallelises internally, so N
workers on one box means N sessions each believing they own all 12 cores.

**Latency is split into tokenisation and inference** because when a text
service is slow it is tokenisation about a third of the time, and a single
total number cannot tell you which half to optimise.

**The anomaly scaler is loaded from disk, not recomputed.** Recomputing medians
online would make an account's score depend on *when* it was scored, which
makes an enforcement decision impossible to audit after the fact.

## Streaming

```bash
make kafka-up       # single-broker KRaft cluster + UI on :8081
make produce        # replay reviews at 500/s
make consume        # micro-batched ONNX scoring, writes stream_report.json
make kafka-down
```

The producer stamps `produced_at_ms` on every event so the consumer measures
true end-to-end latency (produce → scored), not just its own processing time.
Events are keyed by `business_id`, so all reviews for a business land on one
partition and any per-business aggregation downstream needs no shuffle.

The consumer micro-batches, and the right size is a trade between two cost
curves that run in opposite directions — measured, not assumed.

The **model** gets worse per review as the batch grows, because batches pad to
their longest member and review text is ragged. The **pipeline** gets better,
because an offset commit and an out-topic produce are paid once per batch
regardless of size. On the DistilBERT graph over CPU EP:

| batch | score/review | end-to-end throughput |
|---|---|---|
| 2 | 31.2 ms | 4.8 reviews/s |
| 8 | 44.8 ms | **11.4 reviews/s** |
| 32 | 65.8 ms | 10.4 reviews/s |

The sum minimises near batch 8, which is the default.

Two earlier claims here were wrong, and the way they were wrong is the
interesting part. The first was that batching 32 is "roughly 10x cheaper per
review than 32 separate calls, because fixed per-call overhead dominates" —
true when per-call overhead *does* dominate, and false here, where ONNX Runtime
already parallelises one inference across 8 threads and padding waste grows
with batch size. `make bench` measures that and refutes it.

The second was the correction: taking `make bench` at its word and dropping the
default to 2. But the bench times `Scorer.score()` in isolation, so it sees only
the model's curve. At batch 2 the per-batch commit and produce are paid 16x more
often, and end-to-end throughput falls to 4.8/s — less than half. A correct
measurement, applied outside the scope it measured.

Note the absolute numbers: ~11 reviews/s for DistilBERT on CPU EP. An earlier
figure of 140/s in this repo was measured against a MiniLM graph trained on
synthetic data, not the model you would ship. Serving a transformer on CPU is
expensive, and that cost belongs beside the +0.057 macro AP it buys.

Offsets are committed **after** scoring, never auto-committed. Auto-commit
acknowledges messages that were read but not yet scored, so a crash mid-batch
loses them silently.

The useful experiment is to run the producer faster than the consumer can
score, watch lag build in the UI, and find where it stops growing. That number
is the sustainable throughput, and it is the only capacity claim worth making.

## S3 + Athena

```bash
python scripts/land_s3_athena.py --bucket my-yelp-cci --dry-run
make s3 BUCKET=my-yelp-cci
```

Uploads the Parquet layer preserving the `year=NNNN/` Hive layout — flatten it
and partition pruning breaks silently — then prints the external-table DDL and
sample queries.

Athena bills per byte scanned, at $5/TB. Two things follow: query the partition
column directly (`WHERE year = 2019` prunes; `WHERE YEAR(review_date) = 2019`
cannot, because Athena must read rows to evaluate it), and never `SELECT *` on
a columnar format — on the review table, `text` is roughly 90% of the bytes.

## Not yet built

Model monitoring (score-distribution drift against a reference window),
retraining trigger, per-category threshold calibration for the enforcement
queue.
