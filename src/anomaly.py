"""Stage 3 - unsupervised anomalous-reviewer scoring.

The Yelp Open Dataset ships no fake-review labels, so this head does not
pretend to be a classifier. It is a ranker that produces review-queue
candidates, which is how enforcement systems are actually built: a cheap signal
orders the queue, humans adjudicate the top of it, and their decisions
eventually become the labels a supervised model can train on.

Five behavioural signals, each computed over the full review history:

  burstiness              most reviews written by this account in a single day
  duplication             fraction of their reviews sharing a normalised text
                          fingerprint with another of their own reviews
  rating_extremity        fraction of reviews at 1 or 5 stars
  deviation               mean absolute gap between their rating and the
                          business consensus, on businesses with enough
                          reviews for a consensus to exist
  precocity               how quickly after account creation they started
                          posting, inverted -- accounts that post immediately
                          and then stop look different from organic ones

Each is converted to a robust z-score (median / MAD, so a single extreme
account cannot flatten the scale) and summed. Every component is emitted
alongside the total so a flagged account comes with its reason, which is the
minimum bar for anything a human is expected to action.

Usage:
    python -m src.anomaly --top 200
"""

from __future__ import annotations

import argparse
import json
import time

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from src.config import (
    ANOMALY_END_DATE,
    ANOMALY_START_DATE,
    ARTIFACTS_DIR,
    FEATURES_DIR,
    PARQUET_DIR,
    get_spark,
)

MIN_REVIEWS_PER_USER = 3        # below this, every ratio is noise
MIN_REVIEWS_PER_BUSINESS = 10   # below this, there is no consensus to deviate from
SIGNALS = ["burstiness", "duplication", "rating_extremity", "deviation", "precocity"]


def _business_consensus(review: DataFrame) -> DataFrame:
    return (
        review.groupBy("business_id")
        .agg(
            F.avg("stars").alias("business_mean_stars"),
            F.count("*").alias("business_n_reviews"),
        )
        .filter(F.col("business_n_reviews") >= MIN_REVIEWS_PER_BUSINESS)
    )


def compute_signals(spark: SparkSession) -> DataFrame:
    review = spark.read.parquet(str(PARQUET_DIR / "review")).filter(
        (F.col("review_date") >= F.lit(ANOMALY_START_DATE))
        & (F.col("review_date") < F.lit(ANOMALY_END_DATE))
        & F.col("user_id").isNotNull()
    )
    user = spark.read.parquet(str(PARQUET_DIR / "user"))

    consensus = _business_consensus(review)

    # --- burstiness: peak single-day volume, and how concentrated the account is
    per_day = (
        review.groupBy("user_id", "review_date")
        .count()
        .groupBy("user_id")
        .agg(
            F.max("count").alias("max_reviews_per_day"),
            F.countDistinct("review_date").alias("active_days"),
        )
    )

    # --- duplication: reviews sharing a text fingerprint with the same author's
    #     other reviews. Window over the author, not globally, so two unrelated
    #     people writing "Great food!" do not implicate each other.
    dup_window = Window.partitionBy("user_id", "text_hash")
    duplication = (
        review.select("user_id", "review_id", "text_hash")
        .withColumn("hash_repeats", F.count("*").over(dup_window))
        .groupBy("user_id")
        .agg(
            F.sum(F.when(F.col("hash_repeats") > 1, 1).otherwise(0)).alias("duplicate_reviews"),
            F.count("*").alias("n_reviews_for_dup"),
        )
        .withColumn("duplication", F.col("duplicate_reviews") / F.col("n_reviews_for_dup"))
        .select("user_id", "duplication", "duplicate_reviews")
    )

    # --- extremity and consensus deviation
    with_consensus = review.join(F.broadcast(consensus), on="business_id", how="left")
    per_user = (
        with_consensus.groupBy("user_id")
        .agg(
            F.count("*").alias("n_reviews"),
            F.avg(F.when(F.col("stars").isin(1.0, 5.0), 1.0).otherwise(0.0)).alias("rating_extremity"),
            F.avg(F.abs(F.col("stars") - F.col("business_mean_stars"))).alias("deviation"),
            F.min("review_date").alias("first_review_date"),
            F.max("review_date").alias("last_review_date"),
            F.avg("text_len").alias("avg_text_len"),
        )
        .filter(F.col("n_reviews") >= MIN_REVIEWS_PER_USER)
    )

    df = (
        per_user.join(per_day, on="user_id", how="left")
        .join(duplication, on="user_id", how="left")
        .join(
            user.select("user_id", "yelping_since_ts", F.col("fans").alias("user_fans"),
                        F.col("useful").alias("user_useful_votes"), "friend_count", "elite_years"),
            on="user_id",
            how="left",
        )
    )

    df = (
        df.withColumn(
            "burstiness",
            F.col("max_reviews_per_day").cast("double")
            / F.greatest(F.col("active_days").cast("double"), F.lit(1.0)),
        )
        .withColumn(
            "days_to_first_review",
            F.datediff(F.col("first_review_date"), F.to_date("yelping_since_ts")),
        )
        # Inverted and log-damped: 0 days -> high precocity, years -> ~0.
        .withColumn(
            "precocity",
            F.lit(1.0) / F.log1p(F.greatest(F.coalesce(F.col("days_to_first_review"), F.lit(0)), F.lit(0)).cast("double") + F.lit(1.0)),
        )
        .withColumn("duplication", F.coalesce(F.col("duplication"), F.lit(0.0)))
        .withColumn("deviation", F.coalesce(F.col("deviation"), F.lit(0.0)))
    )

    return df


def robust_zscore(df: DataFrame, columns: list[str]) -> DataFrame:
    """Median/MAD standardisation.

    Mean/stdev is the wrong scaler here: the tail we are hunting is exactly the
    thing that would inflate the standard deviation and hide itself. MAD is
    scaled by 1.4826 so it estimates sigma for normally distributed data.
    """
    medians = df.approxQuantile(columns, [0.5], 0.001)
    med = {c: (m[0] if m else 0.0) for c, m in zip(columns, medians)}

    tmp = df
    for c in columns:
        tmp = tmp.withColumn(f"__ad_{c}", F.abs(F.col(c) - F.lit(med[c])))

    mad_rows = tmp.approxQuantile([f"__ad_{c}" for c in columns], [0.5], 0.001)
    mad = {c: (m[0] if m else 0.0) for c, m in zip(columns, mad_rows)}

    # Resolve every scale before writing any column, so the value used for
    # scoring and the value persisted for serving are the same object. Deriving
    # them separately is how the batch and serving paths silently diverged:
    # `duplication` is zero for most accounts, so its MAD is zero on every run,
    # and the fallback below is its normal path, not an edge case.
    resolved: dict[str, float] = {}
    method: dict[str, str] = {}
    for c in columns:
        scale = 1.4826 * mad[c]
        if scale > 1e-9:
            method[c] = "mad"
        else:
            # Degenerate spread (more than half of accounts sit exactly at the
            # median). Fall back to standard deviation rather than dividing by
            # ~zero and manufacturing infinite z-scores.
            stats = df.select(F.stddev_samp(c).alias("s")).collect()[0]["s"]
            if stats and stats > 1e-9:
                scale, method[c] = float(stats), "stddev"
            else:
                scale, method[c] = 1.0, "unit"
        resolved[c] = float(scale)

    out = df
    for c in columns:
        out = out.withColumn(f"z_{c}", (F.col(c) - F.lit(med[c])) / F.lit(resolved[c]))

    print("[anomaly] robust scaling: " + ", ".join(
        f"{c} median={med[c]:.4f} scale={resolved[c]:.4f} ({method[c]})" for c in columns
    ))

    # Persist so the serving path scores a single account identically to the
    # batch job. Recomputing medians online would make an account's score
    # depend on when it was scored, which makes an enforcement decision
    # impossible to audit after the fact.
    scaler_path = ARTIFACTS_DIR / "anomaly_scaler.json"
    scaler_path.write_text(json.dumps({
        "signals": columns,
        "median": med,
        "scale": resolved,
        # Which estimator produced each scale. A signal that has silently
        # fallen back to `unit` is something you want visible in the artifact
        # rather than buried in a run log.
        "scale_method": method,
        "clip": {"low": -5.0, "high": 8.0},
        "min_reviews_per_user": MIN_REVIEWS_PER_USER,
        "min_reviews_per_business": MIN_REVIEWS_PER_BUSINESS,
    }, indent=2))
    print(f"[anomaly] scaler -> {scaler_path}")
    return out


def score(df: DataFrame) -> DataFrame:
    scored = robust_zscore(df, SIGNALS)
    z_cols = [F.col(f"z_{c}") for c in SIGNALS]
    # Clip each component before summing so one saturated signal cannot carry an
    # account into the queue on its own.
    clipped = [F.least(F.greatest(c, F.lit(-5.0)), F.lit(8.0)) for c in z_cols]
    total = clipped[0]
    for c in clipped[1:]:
        total = total + c
    return scored.withColumn("anomaly_score", total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank reviewer accounts by behavioural anomaly")
    parser.add_argument("--top", type=int, default=200, help="rows to print and export")
    args = parser.parse_args()

    spark = get_spark("yelp-anomaly")
    start = time.time()
    try:
        scored = score(compute_signals(spark))
        out = FEATURES_DIR / "reviewer_anomaly"
        scored.write.mode("overwrite").parquet(str(out))

        top = (
            spark.read.parquet(str(out))
            .orderBy(F.desc("anomaly_score"))
            .limit(args.top)
            .select(
                "user_id", "n_reviews", "anomaly_score",
                *[F.round(F.col(c), 4).alias(c) for c in SIGNALS],
                "max_reviews_per_day", "duplicate_reviews", "avg_text_len",
            )
        )
        csv_path = ARTIFACTS_DIR / "review_queue_top.csv"
        top.toPandas().to_csv(csv_path, index=False)

        print(f"[anomaly] wrote {out}")
        print(f"[anomaly] top {args.top} queue candidates -> {csv_path}")
        top.show(15, truncate=False)
        print(f"[anomaly] done in {time.time() - start:.1f}s")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
