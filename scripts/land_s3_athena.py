"""Land the Parquet layer in S3 and register it as Athena external tables.

Two things to understand before running this, because Athena bills by bytes
scanned and it is easy to spend real money on a dataset you already have on
local disk:

  1. Partition pruning only works if you query the partition column. The review
     table is partitioned by `year`, so `WHERE year = 2019` scans one year and
     `WHERE YEAR(review_date) = 2019` scans all of them -- the second form
     cannot be pruned because Athena has to read the rows to evaluate it.
  2. SELECT * on a columnar format defeats the point of a columnar format.
     Athena charges for the columns you touch, so naming them is a cost
     decision, not a style preference.

The free tier covers the storage here comfortably. Athena has no free tier --
it is $5 per TB scanned, so the whole 8.65 GB dataset scanned end to end costs
about four cents. Cheap, but not zero, and a bad query in a loop is how people
get surprised.

Usage:
    python scripts/land_s3_athena.py --bucket my-yelp-review-intelligence --dry-run
    python scripts/land_s3_athena.py --bucket my-yelp-review-intelligence
    python scripts/land_s3_athena.py --bucket my-yelp-review-intelligence --sql-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PARQUET_DIR  # noqa: E402

DATABASE = "yelp_cci"

TABLE_DDL = {
    "business": """
CREATE EXTERNAL TABLE IF NOT EXISTS {db}.business (
  business_id   string,
  name          string,
  address       string,
  city          string,
  state         string,
  postal_code   string,
  latitude      double,
  longitude     double,
  stars         double,
  review_count  bigint,
  is_open       boolean,
  attributes    map<string,string>,
  hours         map<string,string>,
  category_list array<string>,
  n_categories  int
)
STORED AS PARQUET
LOCATION 's3://{bucket}/{prefix}/business/'
TBLPROPERTIES ('parquet.compression'='ZSTD');
""",
    "review": """
CREATE EXTERNAL TABLE IF NOT EXISTS {db}.review (
  review_id   string,
  user_id     string,
  business_id string,
  stars       double,
  useful      bigint,
  funny       bigint,
  cool        bigint,
  text        string,
  ts          timestamp,
  review_date date,
  text_len    int,
  word_count  int,
  text_hash   bigint
)
PARTITIONED BY (year int)
STORED AS PARQUET
LOCATION 's3://{bucket}/{prefix}/review/'
TBLPROPERTIES ('parquet.compression'='ZSTD');
""",
    "user": """
CREATE EXTERNAL TABLE IF NOT EXISTS {db}.user_profile (
  user_id           string,
  name              string,
  review_count      bigint,
  useful            bigint,
  funny             bigint,
  cool              bigint,
  fans              bigint,
  average_stars     double,
  yelping_since_ts  timestamp,
  friend_count      int,
  elite_years       int
)
STORED AS PARQUET
LOCATION 's3://{bucket}/{prefix}/user/'
TBLPROPERTIES ('parquet.compression'='ZSTD');
""",
}

SAMPLE_QUERIES = """
-- Partition pruning: scans one year, not twenty.
SELECT count(*) AS reviews, round(avg(stars), 3) AS avg_stars
FROM {db}.review
WHERE year = 2019;

-- Named columns only. SELECT * here would scan the `text` column, which is
-- roughly 90% of the bytes in the table and is not needed for this answer.
SELECT b.state, count(*) AS n, round(avg(r.stars), 3) AS avg_stars
FROM {db}.review r
JOIN {db}.business b ON r.business_id = b.business_id
WHERE r.year BETWEEN 2018 AND 2021
GROUP BY b.state
ORDER BY n DESC
LIMIT 20;

-- Burst detection pushed into Athena: accounts posting more than 10 reviews in
-- a single day. The Spark job computes this over all history; this is the
-- ad-hoc version for one window.
SELECT user_id, review_date, count(*) AS same_day_reviews
FROM {db}.review
WHERE year = 2021
GROUP BY user_id, review_date
HAVING count(*) > 10
ORDER BY same_day_reviews DESC
LIMIT 100;
"""


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def upload(bucket: str, prefix: str, dry_run: bool) -> None:
    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3")

    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as exc:
        raise SystemExit(
            f"Cannot access s3://{bucket} ({exc.response['Error']['Code']}). "
            f"Create it first and check your credentials."
        ) from exc

    files = [p for p in PARQUET_DIR.rglob("*.parquet") if p.is_file()]
    if not files:
        raise SystemExit(f"No parquet under {PARQUET_DIR}. Run `make ingest` first.")

    total = sum(p.stat().st_size for p in files)
    print(f"[s3] {len(files):,} files, {human(total)} -> s3://{bucket}/{prefix}/")
    if dry_run:
        for p in files[:10]:
            print(f"  would upload {p.relative_to(PARQUET_DIR)}  ({human(p.stat().st_size)})")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10:,} more")
        return

    done = 0
    for p in files:
        # Preserve the Hive-style `year=NNNN/` directories exactly: Athena
        # discovers partitions from the key path, so flattening the layout
        # breaks partition pruning silently.
        key = f"{prefix}/{p.relative_to(PARQUET_DIR).as_posix()}"
        s3.upload_file(str(p), bucket, key)
        done += 1
        if done % 25 == 0 or done == len(files):
            print(f"[s3] {done:,}/{len(files):,}")
    print(f"[s3] upload complete: {human(total)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default="parquet")
    p.add_argument("--database", default=DATABASE)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sql-only", action="store_true", help="print DDL without touching S3")
    args = p.parse_args()

    if not args.sql_only:
        upload(args.bucket, args.prefix, args.dry_run)

    fmt = {"db": args.database, "bucket": args.bucket, "prefix": args.prefix}
    print("\n" + "=" * 72)
    print("Run in the Athena query editor:")
    print("=" * 72)
    print(f"\nCREATE DATABASE IF NOT EXISTS {args.database};")
    for name, ddl in TABLE_DDL.items():
        print(ddl.format(**fmt).rstrip())
    print(f"\n-- Register the year partitions. Without this the review table\n"
          f"-- returns zero rows even though the files are there.\n"
          f"MSCK REPAIR TABLE {args.database}.review;")
    print("\n-- Sample queries")
    print(SAMPLE_QUERIES.format(db=args.database))


if __name__ == "__main__":
    main()
