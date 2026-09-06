"""Generate small Yelp-shaped JSON files for smoke-testing the pipeline.

Field names and types mirror the real dataset exactly, so anything that runs
here runs against the real 8.65 GB unchanged. It also plants a handful of
obviously anomalous accounts (burst posters, copy-paste reviewers) so the
anomaly ranker has something it should visibly catch.

Usage:
    python scripts/make_synthetic.py --businesses 400 --users 800 --reviews 20000
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import RAW_DIR, RAW_FILES  # noqa: E402

CATEGORIES = [
    "Restaurants", "Food", "Shopping", "Nightlife", "Bars", "Coffee & Tea",
    "Sandwiches", "Pizza", "American (Traditional)", "Mexican", "Italian",
    "Chinese", "Breakfast & Brunch", "Beauty & Spas", "Hair Salons", "Nail Salons",
    "Automotive", "Auto Repair", "Home Services", "Health & Medical", "Dentists",
    "Fast Food", "Burgers", "Seafood", "Bakeries", "Desserts", "Grocery",
    "Hotels & Travel", "Fitness & Instruction", "Active Life", "Event Planning & Services",
]

STATES = ["PA", "FL", "LA", "AZ", "NV", "CA", "MO", "TN", "IN", "AB", "ID"]

PHRASES = [
    "the service was quick and the staff were friendly",
    "portions were generous and the price was fair",
    "waited almost an hour and nobody checked on us",
    "atmosphere is great for a weeknight with friends",
    "parking is a nightmare but the food makes up for it",
    "everything came out cold and the order was wrong",
    "best in the neighbourhood by a wide margin",
    "clean, well run, and easy to book an appointment",
    "would not come back after the way we were treated",
    "the owner came out to say hello which was a nice touch",
]

SPAM_TEXT = "Amazing experience! Highly recommend to everyone! Five stars all the way!"


def rand_date(rng: random.Random, start: datetime, end: datetime) -> str:
    delta = int((end - start).total_seconds())
    return (start + timedelta(seconds=rng.randrange(delta))).strftime("%Y-%m-%d %H:%M:%S")


def write_jsonl(path: Path, rows) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
            n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--businesses", type=int, default=400)
    p.add_argument("--users", type=int, default=800)
    p.add_argument("--reviews", type=int, default=20000)
    p.add_argument("--spam-users", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = random.Random(args.seed)
    t0, t1 = datetime(2007, 1, 1), datetime(2022, 1, 1)

    businesses = []
    for i in range(args.businesses):
        k = rng.randint(1, 4)
        cats = rng.sample(CATEGORIES, k)
        businesses.append({
            "business_id": f"biz{i:06d}",
            "name": f"Business {i}",
            "address": f"{rng.randint(1, 9999)} Main St",
            "city": rng.choice(["Philadelphia", "Tampa", "Tucson", "Reno", "Boise"]),
            "state": rng.choice(STATES),
            "postal_code": f"{rng.randint(10000, 99999)}",
            "latitude": round(rng.uniform(25.0, 50.0), 6),
            "longitude": round(rng.uniform(-123.0, -75.0), 6),
            "stars": round(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]), 1),
            "review_count": rng.randint(5, 900),
            "is_open": rng.choice([0, 1]),
            "attributes": {"RestaurantsTakeOut": rng.choice(["True", "False"]),
                           "BusinessParking": "{'garage': False, 'street': True}"},
            "categories": ", ".join(cats),
            "hours": {"Monday": "9:0-22:0", "Tuesday": "9:0-22:0"},
        })

    n_spam = min(args.spam_users, args.users)
    users = []
    for i in range(args.users):
        users.append({
            "user_id": f"usr{i:06d}",
            "name": f"User{i}",
            "review_count": rng.randint(1, 400),
            "yelping_since": rand_date(rng, datetime(2005, 1, 1), datetime(2019, 1, 1)),
            "friends": ",".join(f"usr{rng.randrange(args.users):06d}" for _ in range(rng.randint(0, 30))) or "None",
            "useful": rng.randint(0, 500), "funny": rng.randint(0, 300), "cool": rng.randint(0, 300),
            "fans": rng.randint(0, 80),
            "elite": ",".join(str(y) for y in rng.sample(range(2010, 2021), rng.randint(0, 3))),
            "average_stars": round(rng.uniform(1.5, 5.0), 2),
            **{f"compliment_{c}": rng.randint(0, 50) for c in
               ["hot", "more", "profile", "cute", "list", "note", "plain", "cool", "funny", "writer", "photos"]},
        })

    biz_ids = [b["business_id"] for b in businesses]
    reviews = []
    rid = 0

    # Organic reviews.
    for _ in range(args.reviews):
        text = ". ".join(rng.sample(PHRASES, rng.randint(2, 4))) + "."
        reviews.append({
            "review_id": f"rev{rid:08d}",
            "user_id": f"usr{rng.randrange(n_spam, args.users):06d}",
            "business_id": rng.choice(biz_ids),
            "stars": float(rng.randint(1, 5)),
            "useful": rng.randint(0, 20), "funny": rng.randint(0, 8), "cool": rng.randint(0, 8),
            "text": text,
            "date": rand_date(rng, t0, t1),
        })
        rid += 1

    # Planted anomalies: burst posting, duplicate text, all 5 stars, day-one
    # account activity. The ranker should surface these at the top.
    for u in range(n_spam):
        burst_day = rand_date(rng, t0, t1)[:10]
        for j in range(rng.randint(15, 40)):
            reviews.append({
                "review_id": f"rev{rid:08d}",
                "user_id": f"usr{u:06d}",
                "business_id": rng.choice(biz_ids),
                "stars": 5.0,
                "useful": 0, "funny": 0, "cool": 0,
                "text": SPAM_TEXT,
                "date": f"{burst_day} {j % 24:02d}:{(j * 7) % 60:02d}:00",
            })
            rid += 1
        users[u]["yelping_since"] = f"{burst_day} 00:00:00"

    checkins = [{"business_id": b["business_id"],
                 "date": ", ".join(rand_date(rng, t0, t1) for _ in range(rng.randint(0, 40)))}
                for b in businesses]

    tips = [{"user_id": f"usr{rng.randrange(args.users):06d}",
             "business_id": rng.choice(biz_ids),
             "text": rng.choice(PHRASES),
             "date": rand_date(rng, t0, t1),
             "compliment_count": rng.randint(0, 5)}
            for _ in range(args.reviews // 10)]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for entity, rows in [("business", businesses), ("review", reviews),
                         ("user", users), ("checkin", checkins), ("tip", tips)]:
        path = RAW_DIR / RAW_FILES[entity]
        n = write_jsonl(path, rows)
        print(f"[synthetic] {entity:9s} {n:>7,} rows -> {path}")


if __name__ == "__main__":
    main()
