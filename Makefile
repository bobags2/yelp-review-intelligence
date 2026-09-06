# Yelp Content & Contributor Intelligence
#
# Batch:      ingest -> features -> anomaly -> baseline
# Deep:       encoder -> export
# Serving:    serve, bench
# Streaming:  kafka-up, produce, consume, kafka-down
#
# Override anything: make baseline XGB_DEVICE=cuda

PY := python

.PHONY: all synthetic ingest features anomaly baseline baseline-gpu \
        encoder export serve bench kafka-up kafka-down produce consume \
        s3 smoke clean

all: ingest features anomaly baseline

# ---- batch ----------------------------------------------------------------
synthetic:
	$(PY) scripts/make_synthetic.py --businesses 400 --users 800 --reviews 20000

ingest:
	$(PY) -m src.ingest

features:
	$(PY) -m src.features

anomaly:
	$(PY) -m src.anomaly --top 500

baseline:
	$(PY) -m src.train_baseline --model both

baseline-gpu:
	XGB_DEVICE=cuda $(PY) -m src.train_baseline --model xgb

# ---- deep model -----------------------------------------------------------
encoder:
	$(PY) -m src.train_encoder --epochs 3

# If this OOMs on the 3070 Ti, in this order:
#   --grad-accum 2  /  --max-len 192  /  --batch-size 16  /  smaller --model-name
encoder-small:
	$(PY) -m src.train_encoder --epochs 3 --grad-accum 2 --max-len 192 \
		--model-name sentence-transformers/all-MiniLM-L6-v2

export:
	$(PY) -m src.export_onnx

# ---- serving --------------------------------------------------------------
serve:
	uvicorn serving.app:app --host 0.0.0.0 --port 8080 --workers 1

serve-gpu:
	ORT_PROVIDER=CUDAExecutionProvider uvicorn serving.app:app --host 0.0.0.0 --port 8080 --workers 1

bench:
	$(PY) scripts/bench_latency.py

# Batch and serving must produce the same anomaly score from the same inputs.
# Nothing else enforces that -- they are separate implementations reading one
# artifact -- and when they diverged it was silent.
parity:
	$(PY) scripts/check_scoring_parity.py

# ---- streaming ------------------------------------------------------------
kafka-up:
	docker compose up -d
	@echo "waiting for broker..."
	@until docker compose exec -T kafka /opt/kafka/bin/kafka-topics.sh \
		--bootstrap-server localhost:19092 --list >/dev/null 2>&1; do sleep 2; done
	@echo "kafka ready on localhost:9092, UI on http://localhost:8081"

kafka-down:
	docker compose down -v

produce:
	$(PY) -m stream.producer --rate 500 --limit 50000

consume:
	$(PY) -m stream.consumer --batch-size 8 --max-wait-ms 50 --from-beginning

# ---- cloud ----------------------------------------------------------------
s3:
	@test -n "$(BUCKET)" || (echo "usage: make s3 BUCKET=my-bucket" && exit 1)
	$(PY) scripts/land_s3_athena.py --bucket $(BUCKET)

# ---- smoke test -----------------------------------------------------------
# Full batch pipeline on synthetic data, small memory footprint. No GPU, no
# Kafka, no network. Should finish in a couple of minutes.
smoke:
	SPARK_DRIVER_MEMORY=2g SPARK_MASTER='local[2]' SPARK_SHUFFLE_PARTITIONS=8 \
	TOP_K_CATEGORIES=20 $(MAKE) synthetic ingest features anomaly
	SPARK_DRIVER_MEMORY=2g SPARK_MASTER='local[2]' SPARK_SHUFFLE_PARTITIONS=8 \
	$(PY) -m src.train_baseline --model both --max-features 20000 --svd-dims 64 --train-rows 9000
	$(MAKE) parity

clean:
	rm -rf data/parquet data/features data/spark-tmp \
	       artifacts/*.json artifacts/*.csv artifacts/encoder
