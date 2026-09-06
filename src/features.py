"""Stage 2 - build the supervised modelling table for content-type classification.

Task: given a single review (text + who wrote it), predict the multi-label
category set of the business it is about.

Two things this module is deliberately careful about, because they are what
separate a portfolio project from a portfolio project that survives questions:

1. No business-derived features. Business attributes, name, and star average
   all encode the label almost directly. Only the review and its author are
   allowed as inputs.
2. Grouped split by business_id. See the note in config.py.

Usage:
    python -m src.features
"""

from __future__ import annotations

import json
import time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import (
    ARTIFACTS_DIR,
    CATEGORY_STOPLIST,
    FEATURES_DIR,
    MIN_REVIEW_CHARS,
    PARQUET_DIR,
    SPLIT_SALT,
    TEST_FRACTION,
    TOP_K_CATEGORIES,
    get_spark,
)


def build_label_vocabulary(business: DataFrame) -> list[str]:
    """Pick the top-K categories by business frequency, minus the stoplist."""
    counts = (
        business.select(F.explode("category_list").alias("category"))
        .filter(~F.col("category").isin(list(CATEGORY_STOPLIST)))
        .groupBy("category")
        .count()
        .orderBy(F.desc("count"))
        .limit(TOP_K_CATEGORIES)
        .collect()
    )
    vocab = [r["category"] for r in counts]

    path = ARTIFACTS_DIR / "category_vocab.json"
    path.write_text(json.dumps(
        {"k": len(vocab), "categories": vocab,
         "counts": {r["category"]: r["count"] for r in counts}},
        indent=2,
    ))
    print(f"[features] label vocabulary: {len(vocab)} categories -> {path}")
    return vocab


def attach_labels(business: DataFrame, vocab: list[str]) -> DataFrame:
    """Multi-hot encode each business against the fixed vocabulary.

    Order is pinned to `vocab`, so column j of the label array always means the
    same category across training, evaluation, and serving.
    """
    vocab_col = F.array(*[F.lit(c) for c in vocab])
    return (
        business.select("business_id", "category_list")
        .withColumn(
            "labels",
            F.transform(
                vocab_col,
                lambda c: F.array_contains(F.col("category_list"), c).cast("int"),
            ),
        )
        .withColumn("n_labels", F.aggregate("labels", F.lit(0), lambda acc, x: acc + x))
        # A business matching none of the top-K carries no supervisory signal
        # for this head, so it is dropped rather than taught as all-zeros.
        .filter(F.col("n_labels") > 0)
        .select("business_id", "labels", "n_labels")
    )


def reviewer_features(user: DataFrame) -> DataFrame:
    """Author-side features. Nothing here touches the business or its category."""
    return user.select(
        "user_id",
        F.col("review_count").alias("user_review_count"),
        F.col("average_stars").alias("user_average_stars"),
        F.col("fans").alias("user_fans"),
        F.col("useful").alias("user_useful_votes"),
        "friend_count",
        "elite_years",
        "yelping_since_ts",
    )


def assign_split(df: DataFrame) -> DataFrame:
    """Deterministic grouped split. Whole businesses land on one side only."""
    bucket = F.pmod(F.xxhash64(F.concat_ws("|", F.lit(SPLIT_SALT), F.col("business_id"))), F.lit(1000))
    cutoff = int(round(TEST_FRACTION * 1000))
    return df.withColumn("split", F.when(bucket < cutoff, F.lit("test")).otherwise(F.lit("train")))


def build(spark: SparkSession) -> DataFrame:
    business = spark.read.parquet(str(PARQUET_DIR / "business"))
    review = spark.read.parquet(str(PARQUET_DIR / "review"))
    user = spark.read.parquet(str(PARQUET_DIR / "user"))

    vocab = build_label_vocabulary(business)
    labels = attach_labels(business, vocab)
    users = reviewer_features(user)

    reviews = review.filter(
        F.col("text").isNotNull()
        & (F.col("text_len") >= MIN_REVIEW_CHARS)
        & F.col("business_id").isNotNull()
        & F.col("user_id").isNotNull()
    ).select(
        "review_id", "business_id", "user_id", "text", "text_len",
        "word_count", "review_date", "text_hash",
        F.col("stars").alias("review_stars"),
    )

    # Broadcast both dimension tables: business is ~150k rows and user ~2M,
    # both far below the driver heap, so this avoids two full shuffles of the
    # 7M-row fact table.
    df = (
        reviews.join(F.broadcast(labels), on="business_id", how="inner")
        .join(F.broadcast(users), on="user_id", how="left")
    )

    df = (
        df.withColumn(
            "account_age_days_at_review",
            F.datediff(F.col("review_date"), F.to_date("yelping_since_ts")),
        )
        .withColumn(
            "star_deviation_from_user_mean",
            F.col("review_stars") - F.coalesce(F.col("user_average_stars"), F.col("review_stars")),
        )
        .drop("yelping_since_ts")
    )

    return assign_split(df)


def main() -> None:
    spark = get_spark("yelp-features")
    start = time.time()
    try:
        df = build(spark)
        out = FEATURES_DIR / "review_labeled"
        df.write.mode("overwrite").partitionBy("split").parquet(str(out))

        summary = (
            spark.read.parquet(str(out))
            .groupBy("split")
            .agg(
                F.count("*").alias("rows"),
                F.countDistinct("business_id").alias("businesses"),
                F.countDistinct("user_id").alias("users"),
                F.round(F.avg("n_labels"), 3).alias("avg_labels"),
            )
            .collect()
        )
        for r in summary:
            print(
                f"[features] {r['split']:5s} rows={r['rows']:>10,}  "
                f"businesses={r['businesses']:>8,}  users={r['users']:>9,}  "
                f"avg_labels={r['avg_labels']}"
            )
        print(f"[features] wrote {out} in {time.time() - start:.1f}s")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
