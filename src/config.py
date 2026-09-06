"""Central configuration: paths, Spark tuning, and task parameters.

Every knob here is overridable by environment variable so the same code runs
on the 64 GB workstation and inside a 3 GB CI container without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

from pyspark.sql import SparkSession

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("YELP_DATA_ROOT", PROJECT_ROOT / "data"))

RAW_DIR = DATA_ROOT / "raw"          # the 5 unpacked .json files land here
PARQUET_DIR = DATA_ROOT / "parquet"  # normalised columnar copy
FEATURES_DIR = DATA_ROOT / "features"
ARTIFACTS_DIR = Path(os.environ.get("YELP_ARTIFACTS", PROJECT_ROOT / "artifacts"))

for _d in (RAW_DIR, PARQUET_DIR, FEATURES_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_FILES = {
    "business": "yelp_academic_dataset_business.json",
    "review": "yelp_academic_dataset_review.json",
    "user": "yelp_academic_dataset_user.json",
    "checkin": "yelp_academic_dataset_checkin.json",
    "tip": "yelp_academic_dataset_tip.json",
}

# --------------------------------------------------------------------------
# Spark tuning
# --------------------------------------------------------------------------
# Local mode runs the executor inside the driver JVM, so driver.memory is the
# only heap setting that matters. 40g on a 64 GB box leaves room for the OS,
# the page cache Spark leans on for Parquet reads, and a CUDA process later.

SPARK_DRIVER_MEMORY = os.environ.get("SPARK_DRIVER_MEMORY", "40g")
SPARK_MASTER = os.environ.get("SPARK_MASTER", "local[*]")
SPARK_SHUFFLE_PARTITIONS = int(os.environ.get("SPARK_SHUFFLE_PARTITIONS", "96"))
SPARK_LOCAL_DIR = os.environ.get("SPARK_LOCAL_DIR", str(DATA_ROOT / "spark-tmp"))


def get_spark(app_name: str) -> SparkSession:
    """Build a SparkSession tuned for a single fat local machine.

    Requires Java 17+ for Spark 4.x. Check with `java -version`.
    """
    Path(SPARK_LOCAL_DIR).mkdir(parents=True, exist_ok=True)

    builder = (
        SparkSession.builder.appName(app_name)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.local.dir", SPARK_LOCAL_DIR)
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        # Adaptive execution collapses the 96 shuffle partitions back down when
        # a stage turns out to be small, which matters a lot on skewed joins
        # like review -> user.
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        # Arrow makes toPandas() on the sampled training slice ~10x faster.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "zstd")
        # The progress bar writes bare carriage returns into stdout, which
        # mangles piped logs and breaks line-oriented greps over run output.
        .config("spark.ui.showConsoleProgress", "false")
        # Timestamps in the Yelp dump predate the Gregorian cutover rules Spark
        # 3+ enforces; CORRECTED avoids a rebase exception on write.
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
    )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(os.environ.get("SPARK_LOG_LEVEL", "WARN"))
    return spark


# --------------------------------------------------------------------------
# Task parameters
# --------------------------------------------------------------------------

# Multi-label head: how many of the ~1,300 raw Yelp categories to model.
# Top 50 covers the overwhelming majority of businesses while keeping the
# label matrix dense enough that per-label PR-AUC is meaningful.
TOP_K_CATEGORIES = int(os.environ.get("TOP_K_CATEGORIES", "50"))

# Categories this generic get dropped: they carry no signal and dominate the
# head of the frequency distribution.
CATEGORY_STOPLIST = {"Restaurants", "Food", "Shopping", "Nightlife", "Event Planning & Services"}

# Split strategy for the classification head: GROUP, by business_id.
#
# A random row split is wrong here and it is the single most common flaw in
# public Yelp notebooks. Reviews frequently name the business, so a random
# split lets the model memorise "Joe's Diner -> Restaurants" from the training
# rows and score inflated on test rows about the same business. A temporal
# split has the same defect for this task, because a business straddles the
# cutoff. Holding out whole businesses is the only split that measures what we
# claim to measure: generalisation to a business the model has never seen.
#
# Businesses are assigned by a stable hash, so the split is reproducible and
# does not drift when new data is appended.
TEST_FRACTION = float(os.environ.get("TEST_FRACTION", "0.20"))
SPLIT_SALT = os.environ.get("SPLIT_SALT", "cci-v1")

# Temporal bounds, used by the behavioural/anomaly head where time ordering is
# the point rather than a leak.
ANOMALY_START_DATE = os.environ.get("ANOMALY_START_DATE", "2005-01-01")
ANOMALY_END_DATE = os.environ.get("ANOMALY_END_DATE", "2100-01-01")

MIN_REVIEW_CHARS = int(os.environ.get("MIN_REVIEW_CHARS", "40"))

# Baseline sampling: XGBoost one-vs-rest over 50 labels on the full 7M rows is
# hours of CPU for no extra insight. Sample, then scale up once the pipeline is
# proven end to end.
BASELINE_SAMPLE_ROWS = int(os.environ.get("BASELINE_SAMPLE_ROWS", "400000"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "17"))
