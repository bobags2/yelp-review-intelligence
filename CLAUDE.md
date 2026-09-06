# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Everything is driven through the `Makefile`; every target is a thin wrapper over `python -m <module>` with a flag or two, so run modules directly when you need different arguments.

```bash
make smoke          # synthetic data -> ingest -> features -> anomaly -> baseline, ~2 min, no GPU/Kafka/network
make all            # ingest -> features -> anomaly -> baseline on real data
make clean          # wipes data/parquet, data/features, data/spark-tmp, artifacts/*.json|csv, artifacts/encoder

make encoder        # fine-tune DistilBERT (needs GPU); encoder-small for the 22M MiniLM variant
make export         # ONNX export + faithfulness verification (exits non-zero on divergence)
make serve          # uvicorn :8080; serve-gpu for CUDAExecutionProvider
make bench          # batch-size latency sweep against the in-process Scorer
make parity         # batch vs serving anomaly-score parity (also runs inside make smoke)

make kafka-up       # KRaft broker :9092 + UI :8081; then produce / consume; kafka-down tears down with -v
make s3 BUCKET=...  # upload parquet layer, print Athena DDL
```

Override anything via env var: `make baseline XGB_DEVICE=cuda`, `SPARK_DRIVER_MEMORY=8g make features`.

**There is no test suite, linter config, or CI.** `make smoke` is the regression check — it now ends with `scripts/check_scoring_parity.py` — run it after any change to `src/`, and it must complete all four stages. Expect ~1.0x classifier lift on synthetic data (the generator draws review text from a shared phrase pool, so there is no signal to find); that is the smoke test passing.

Requires Java 17+ (Spark 4.x) and Python 3.10+.

### Running on Windows

Spark's `RawLocalFileSystem.setPermission` needs `winutils.exe` to write Parquet, so `make ingest` fails natively on Windows with `HADOOP_HOME and hadoop.home.dir are unset` unless a winutils build is installed. The pipeline runs clean under WSL, which needs none of that.

Under WSL, run a whole streaming session in **one** `wsl.exe` invocation. WSL tears the distro down between separate invocations, which takes dockerd with it and SIGTERMs the broker mid-test (containers show `Exited (143)`). Set `PYTHONUNBUFFERED=1` when capturing consumer output to a file, or its readiness line sits in a block buffer.

Two traps when invoking make here: pass `PY` as an **absolute** path (`make smoke PY=/abs/path/python`) — GNU make on Windows resolves a relative `./venv/...` command against `PATH`, not the working directory, and can silently launch a different interpreter — and never pipe make through `tail` when you care about the result, since the pipeline exit code hides make's.

## Architecture

Four batch stages, each reading the previous stage's output off disk. There is no orchestrator and no in-process handoff — the contract between stages is the file layout, so stages can be re-run independently.

```
data/raw/*.json          5 Yelp Open Dataset files (or scripts/make_synthetic.py output)
  src/ingest.py          explicit schemas -> data/parquet/{business,review,user,checkin,tip}
                         review is partitioned by year=NNNN/ (Hive layout; S3 upload depends on it)
  src/features.py        -> data/features/review_labeled (partitioned by split)
                         + artifacts/category_vocab.json
  src/anomaly.py         -> data/features/reviewer_anomaly
                         + artifacts/anomaly_scaler.json, artifacts/review_queue_top.csv
  src/train_baseline.py  -> artifacts/baseline_metrics.json
  src/train_encoder.py   -> artifacts/encoder/{model.pt,tokenizer,config}, encoder_history.json
  src/export_onnx.py     -> artifacts/encoder/{model.onnx,serving_manifest.json}
```

`serving/app.py` is the only runtime component and everything downstream imports from it: `scripts/bench_latency.py` and `stream/consumer.py` both `from serving.app import Scorer`, so the offline benchmark and the Kafka consumer exercise the exact code path the HTTP service does. Changing `Scorer` changes all three.

### Artifact contract

`artifacts/` is the coupling surface between stages and between offline and online.

- **`category_vocab.json`** pins label ordering. Column *j* of the label array means the same category in `features`, `train_baseline`, `train_encoder`, `export_onnx`, and `/score`. Regenerating the vocab (changing `TOP_K_CATEGORIES` or `CATEGORY_STOPLIST`) invalidates every trained model and exported graph downstream.
- **`anomaly_scaler.json`** carries the median/MAD parameters so `/anomaly` reproduces the batch score exactly. `serving.AnomalyScorer` reads it at startup and the anomaly head is *optional* — the service still boots if the file is missing or unusable, printing `anomaly head disabled`. The category head is a hard dependency.
- **`encoder/serving_manifest.json`** is what `Scorer` loads to find the graph and its input shapes. It records an **absolute** `onnx_path`, so a manifest exported under WSL will not load from Windows-native serving. The streaming path depends on this too: `stream/consumer.py` constructs a `Scorer`, so `make consume` cannot run until `train_encoder` and `export_onnx` have produced `artifacts/encoder/` — Kafka alone is not enough.

`artifacts/*.json` and `*.csv` are gitignored, as is all of `data/`. A fresh clone has no artifacts; `make smoke` is the fastest way to populate them.

### Configuration

`src/config.py` is the single source of paths, Spark tuning, and task parameters, and **every value is an env-var override** — nothing is edited in place to change a run. It also owns `get_spark()`, which every Spark module calls; Spark session config (adaptive execution, Arrow, zstd, datetime rebase mode) lives there and nowhere else. Serving and streaming read their own env vars directly (`YELP_ARTIFACTS`, `ORT_PROVIDER`, `ORT_THREADS`, `MAX_BATCH`, `LATENCY_WINDOW`).

### Invariants worth not breaking

These are load-bearing decisions, each with a comment at its site explaining why:

- **Grouped split by `business_id`, stable-hashed** (`config.SPLIT_SALT`, `features.assign_split`). Not random, not temporal — reviews name their business, so any other split leaks. Whole businesses land on one side only.
- **No business-derived features in the supervised head.** `attributes`, `name`, and business star average encode the label. Only the review and its author are legal inputs. Adding a business column to `features.build` silently invalidates every reported metric.
- **Explicit Spark schemas in `ingest.py`.** Inference on `business.attributes` produces a different struct depending on sampling, making output non-reproducible.
- **The queue ranks harm, not weirdness.** `anomaly_score` is how unusual an account is; `queue_score = anomaly_score * log1p(n_reviews)` is what the queue orders on, because a 3-review burst account puts three bad reviews in front of users while a 200-review account at 0.3 duplication puts sixty. `--rank-by anomaly_score` restores pure-weirdness ordering.
- **Per-review ratios are shrunk toward the population mean before scaling** (`shrink_signals`, pseudo-count `SHRINKAGE_PSEUDOCOUNT=10`). Without it the queue is entirely 3-review accounts, whose ratios saturate every signal at once on two or three events. `precocity` is excluded: it is a single event, not an average. The priors are persisted in the scaler and the serving path redoes the arithmetic, which is why `/anomaly` now requires `n_reviews` — an account's score is not a function of its signals alone.
- **Median/MAD, never mean/stdev, in `anomaly.robust_zscore`.** The tail being hunted would inflate a standard deviation and hide itself. Components are clipped to [-5, 8] before summing so one saturated signal cannot carry an account into the queue alone, and every component is emitted alongside the total.
- **The linear/xgb gap is representation, not model class — measured, not assumed.** Three runs on the real data, `--model all`:

  | model | features | macro AP | micro AP |
  |---|---|---|---|
  | `linear_tfidf` | 200k sparse TF-IDF | 0.5680 | 0.5795 |
  | `linear_svd_meta` | 256 dense LSA + metadata | 0.4291 | 0.4932 |
  | `xgb_svd_meta` | 256 dense LSA + metadata | 0.4190 | 0.4479 |

  Holding the estimator fixed and changing only the features costs 0.1389 macro AP; holding the features fixed and changing only the model class costs 0.0101. The representation accounts for **93%** of the gap, the model class for 7%. Gradient boosting is not meaningfully worse here — it was handed a worse input. Never quote the first and third rows without the second.
- **The task is lexical, and a few thousand terms carry it** (`scripts/sweep_vocab_size.py`, chi2 selection with the estimator held fixed):

  | k features | macro AP | % of peak |
  |---|---|---|
  | 256 | 0.3909 | 67.8% |
  | 1,000 | 0.4789 | 83.1% |
  | 5,000 | 0.5516 | 95.7% |
  | 20,000 | **0.5762** | 100% |
  | 100,000 | 0.5735 | 99.5% |
  | 200,000 | 0.5680 | 98.6% |

  5k selected terms recover 95.7%; the curve peaks at 20k and the full 200k vocabulary is *worse* than a selected 20k subset while taking 4x as long to fit (915s vs 224s). The last 180k terms are net noise — at k=100k the weakest selected feature scores chi2 3.0. Two consequences: `--max-features 200000` is not the right default, and since the signal is concentrated in a modest set of distinctive terms, a contextual model's advantage (disambiguating words by context) has little left to work with. Note also that 256 chi2-selected *terms* (0.3909) score below 256 *LSA components* (0.4291) — LSA components are combinations of all 200k terms, so at equal width the projection carries more than the best individual terms. Dimensionality, not the density of the representation, is what the lr-svd ablation was measuring.
- **Low SVD explained variance is not a defect.** 0.098 at 256 components is what TF-IDF does — the matrix is near full rank, and recovering most of the variance would take thousands of components, abandoning the reduction. Do not "fix" it by raising `--svd-dims`.
- **Two label slots are the same label.** `Beer` and `Wine & Spirits` both cover exactly 2413 businesses and score identically to four decimals — Yelp's "Beer, Wine & Spirits" split into perfectly co-occurring labels. They consume two of the 50 slots and are double-counted in macro AP.
- **Metrics are reported against their prevalence floor** (`train_baseline.evaluate` emits a `lift` column). An AP without its floor is not a result.
- **Kafka offsets are committed after scoring**, never auto-committed (`stream/consumer.py`).
- **The producer checks delivery.** `flush()` returns what it could not deliver; that count and `delivered` vs `sent` are both asserted before exit, because an unreachable broker otherwise yields `delivered=0, failed=0` and exit 0.
- **One uvicorn worker.** ONNX Runtime parallelises internally; N workers means N sessions each assuming they own all cores.

### Offline/online parity

Two independent implementations compute the anomaly score — Spark in `src/anomaly.py`, Python in `serving.AnomalyScorer` — coupled only by `anomaly_scaler.json`. Nothing in the type system forces them to agree, and they once did not: `duplication` is zero for most accounts, so its MAD is zero on *every* run, so the batch job always took the stddev fallback but persisted `null` for that scale; serving defaulted it to `1.0` and understated the component by several multiples, silently.

Three guards now exist, and all of them matter when touching either scorer:

- `robust_zscore` resolves every scale *before* writing any column and persists exactly what it used, plus a `scale_method` field (`mad` / `stddev` / `unit`) so a degenerate signal is visible in the artifact. Never re-derive a scale at the persist site.
- `AnomalyScorer.__init__` refuses a scaler with an unusable scale rather than defaulting. The startup handler catches that and disables the anomaly head — a disabled endpoint gets noticed, a wrong enforcement score does not.
- `scripts/check_scoring_parity.py` samples scored rows, pushes their raw signals back through the serving scorer, and requires the totals to match. It uses the persisted `z_<signal>` columns to name the guilty component on failure — attributing via the scaler instead would compare the serving path against itself and always agree.

This mirrors the ONNX faithfulness check in `export_onnx.py`. Any new offline/online pair of implementations should get the same treatment.
