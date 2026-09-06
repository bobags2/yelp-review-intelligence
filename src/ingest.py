"""Stage 1 - ingest the raw Yelp Open Dataset JSON into partitioned Parquet.

Schemas are declared explicitly rather than inferred. Inference costs a full
extra pass over 8.65 GB of JSON, and on `business.attributes` it produces a
different struct depending on which rows Spark happens to sample first, which
makes the output non-reproducible run to run.

Usage:
    python -m src.ingest
    python -m src.ingest --only review user
"""

from __future__ import annotations

import argparse
import time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from src.config import PARQUET_DIR, RAW_DIR, RAW_FILES, get_spark

# --------------------------------------------------------------------------
# Schemas -- these match the Yelp Open Dataset documentation PDF shipped in the
# tarball. `attributes` and `hours` are read as string->string maps: the raw
# values are a mix of bare strings, "True"/"False", and Python-repr dicts, so a
# map keeps them addressable without a 40-field struct that half the rows null.
# --------------------------------------------------------------------------

BUSINESS_SCHEMA = StructType([
    StructField("business_id", StringType(), False),
    StructField("name", StringType(), True),
    StructField("address", StringType(), True),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("postal_code", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("stars", DoubleType(), True),
    StructField("review_count", LongType(), True),
    StructField("is_open", IntegerType(), True),
    StructField("attributes", MapType(StringType(), StringType()), True),
    StructField("categories", StringType(), True),
    StructField("hours", MapType(StringType(), StringType()), True),
])

REVIEW_SCHEMA = StructType([
    StructField("review_id", StringType(), False),
    StructField("user_id", StringType(), True),
    StructField("business_id", StringType(), True),
    StructField("stars", DoubleType(), True),
    StructField("useful", LongType(), True),
    StructField("funny", LongType(), True),
    StructField("cool", LongType(), True),
    StructField("text", StringType(), True),
    StructField("date", StringType(), True),
])

USER_SCHEMA = StructType([
    StructField("user_id", StringType(), False),
    StructField("name", StringType(), True),
    StructField("review_count", LongType(), True),
    StructField("yelping_since", StringType(), True),
    StructField("friends", StringType(), True),
    StructField("useful", LongType(), True),
    StructField("funny", LongType(), True),
    StructField("cool", LongType(), True),
    StructField("fans", LongType(), True),
    StructField("elite", StringType(), True),
    StructField("average_stars", DoubleType(), True),
    StructField("compliment_hot", LongType(), True),
    StructField("compliment_more", LongType(), True),
    StructField("compliment_profile", LongType(), True),
    StructField("compliment_cute", LongType(), True),
    StructField("compliment_list", LongType(), True),
    StructField("compliment_note", LongType(), True),
    StructField("compliment_plain", LongType(), True),
    StructField("compliment_cool", LongType(), True),
    StructField("compliment_funny", LongType(), True),
    StructField("compliment_writer", LongType(), True),
    StructField("compliment_photos", LongType(), True),
])

CHECKIN_SCHEMA = StructType([
    StructField("business_id", StringType(), False),
    StructField("date", StringType(), True),
])

TIP_SCHEMA = StructType([
    StructField("user_id", StringType(), True),
    StructField("business_id", StringType(), True),
    StructField("text", StringType(), True),
    StructField("date", StringType(), True),
    StructField("compliment_count", LongType(), True),
])

SCHEMAS = {
    "business": BUSINESS_SCHEMA,
    "review": REVIEW_SCHEMA,
    "user": USER_SCHEMA,
    "checkin": CHECKIN_SCHEMA,
    "tip": TIP_SCHEMA,
}


def _read_raw(spark: SparkSession, entity: str) -> DataFrame:
    path = RAW_DIR / RAW_FILES[entity]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download the Yelp Open Dataset from "
            f"https://www.yelp.com/dataset, unpack the tarball twice, and put "
            f"the five .json files in {RAW_DIR}."
        )
    return spark.read.schema(SCHEMAS[entity]).json(str(path))


# --------------------------------------------------------------------------
# Per-entity normalisation
# --------------------------------------------------------------------------


def transform_business(df: DataFrame) -> DataFrame:
    """Split the comma-separated `categories` string into a clean array."""
    return (
        df.withColumn(
            "category_list",
            F.when(
                F.col("categories").isNotNull(),
                F.expr("transform(split(categories, ','), x -> trim(x))"),
            ).otherwise(F.array().cast(ArrayType(StringType()))),
        )
        .withColumn(
            "category_list",
            F.expr("filter(category_list, x -> x is not null and length(x) > 0)"),
        )
        .withColumn("n_categories", F.size("category_list"))
        .withColumn("is_open", F.col("is_open").cast(BooleanType()))
        .drop("categories")
    )


def transform_review(df: DataFrame) -> DataFrame:
    """Parse the timestamp and derive the partition column plus cheap text stats."""
    return (
        df.withColumn("ts", F.to_timestamp("date", "yyyy-MM-dd HH:mm:ss"))
        .withColumn("review_date", F.to_date("ts"))
        .withColumn("year", F.year("ts"))
        .withColumn("text_len", F.length("text"))
        .withColumn("word_count", F.size(F.split(F.trim(F.col("text")), r"\s+")))
        # Stable 64-bit fingerprint of the normalised text. Used downstream to
        # find copy-paste reviewers without a second pass over the text column.
        .withColumn(
            "text_hash",
            F.xxhash64(F.lower(F.regexp_replace(F.col("text"), r"[^a-z0-9]+", ""))),
        )
        .drop("date")
    )


def transform_user(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("yelping_since_ts", F.to_timestamp("yelping_since", "yyyy-MM-dd HH:mm:ss"))
        .withColumn(
            "friend_count",
            F.when(
                (F.col("friends").isNull()) | (F.col("friends") == "None"),
                F.lit(0),
            ).otherwise(F.size(F.split(F.col("friends"), ","))),
        )
        .withColumn(
            "elite_years",
            F.when(
                (F.col("elite").isNull()) | (F.col("elite") == ""),
                F.lit(0),
            ).otherwise(F.size(F.split(F.col("elite"), ","))),
        )
        # The friends list is a comma-separated wall of 22-char ids and is by far
        # the largest column in user.json. We keep the count, not the payload.
        .drop("friends", "elite", "yelping_since")
    )


def transform_checkin(df: DataFrame) -> DataFrame:
    return (
        df.withColumn(
            "checkin_count",
            F.when(
                (F.col("date").isNull()) | (F.col("date") == ""),
                F.lit(0),
            ).otherwise(F.size(F.split(F.col("date"), ","))),
        ).drop("date")
    )


def transform_tip(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("ts", F.to_timestamp("date", "yyyy-MM-dd HH:mm:ss"))
        .withColumn("tip_date", F.to_date("ts"))
        .drop("date")
    )


TRANSFORMS = {
    "business": transform_business,
    "review": transform_review,
    "user": transform_user,
    "checkin": transform_checkin,
    "tip": transform_tip,
}

# Only reviews are large enough to justify partitioning. 20 years x ~350k rows
# gives partitions that land near the 128 MB Parquet sweet spot.
PARTITION_BY = {"review": ["year"]}


def ingest_entity(spark: SparkSession, entity: str) -> int:
    start = time.time()
    df = TRANSFORMS[entity](_read_raw(spark, entity))

    out = PARQUET_DIR / entity
    writer = df.write.mode("overwrite")
    if entity in PARTITION_BY:
        writer = writer.partitionBy(*PARTITION_BY[entity])
    writer.parquet(str(out))

    n = spark.read.parquet(str(out)).count()
    print(f"[ingest] {entity:9s} -> {out}  rows={n:,}  {time.time() - start:.1f}s")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Yelp JSON -> Parquet")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(RAW_FILES),
        default=sorted(RAW_FILES),
        help="subset of entities to ingest",
    )
    args = parser.parse_args()

    spark = get_spark("yelp-ingest")
    try:
        for entity in args.only:
            ingest_entity(spark, entity)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
