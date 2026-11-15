"""Data ingestion: generate documented synthetic recommendation datasets."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import ensure_dirs, load_config, save_json


def _generate_synthetic(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create users, items, and interactions for an online movie platform."""
    rng = np.random.default_rng(config["data"]["random_seed"])
    n_users = config["data"]["n_users"]
    n_items = config["data"]["n_items"]
    n_interactions = config["data"]["n_interactions"]
    categories = config["data"]["categories"]

    cities = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Kolkata"]
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    genders = ["M", "F", "Other"]

    users = pd.DataFrame(
        {
            "user_id": [f"U{i:04d}" for i in range(1, n_users + 1)],
            "age_group": rng.choice(age_groups, n_users),
            "gender": rng.choice(genders, n_users, p=[0.48, 0.48, 0.04]),
            "city": rng.choice(cities, n_users),
            "signup_date": [
                (datetime(2023, 1, 1) + timedelta(days=int(x))).strftime("%Y-%m-%d")
                for x in rng.integers(0, 700, n_users)
            ],
        }
    )

    items = pd.DataFrame(
        {
            "item_id": [f"M{i:04d}" for i in range(1, n_items + 1)],
            "title": [f"Movie_{i:04d}" for i in range(1, n_items + 1)],
            "category": rng.choice(categories, n_items),
            "release_year": rng.integers(1990, 2025, n_items),
            "duration_min": rng.integers(80, 180, n_items),
            "price": np.round(rng.uniform(49, 499, n_items), 2),
        }
    )

    # Preferential attachment style interactions for realism
    user_ids = users["user_id"].to_numpy()
    item_ids = items["item_id"].to_numpy()
    item_popularity = rng.dirichlet(np.ones(n_items) * 0.7)
    user_activity = rng.dirichlet(np.ones(n_users) * 0.8)

    start = datetime(2024, 1, 1)
    rows = []
    seen = set()
    attempts = 0
    while len(rows) < n_interactions and attempts < n_interactions * 20:
        attempts += 1
        u = rng.choice(user_ids, p=user_activity)
        i = rng.choice(item_ids, p=item_popularity)
        key = (u, i)
        if key in seen:
            continue
        seen.add(key)
        ts = start + timedelta(minutes=int(rng.integers(0, 500 * 24 * 60)))
        rating = float(rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.25, 0.35, 0.25]))
        event = rng.choice(["view", "click", "purchase", "rating"], p=[0.35, 0.25, 0.15, 0.25])
        rows.append(
            {
                "user_id": u,
                "item_id": i,
                "rating": rating,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "event_type": event,
            }
        )

    interactions = pd.DataFrame(rows)

    # Inject a small number of dirty records so validation can demonstrate catching them
    dirty = pd.DataFrame(
        [
            {
                "user_id": "U9999",  # unknown user
                "item_id": item_ids[0],
                "rating": 4.0,
                "timestamp": "2024-06-01 10:00:00",
                "event_type": "rating",
            },
            {
                "user_id": user_ids[0],
                "item_id": "M9999",  # unknown item
                "rating": 3.0,
                "timestamp": "2024-06-02 10:00:00",
                "event_type": "view",
            },
            {
                "user_id": user_ids[1],
                "item_id": item_ids[1],
                "rating": 9.5,  # invalid rating
                "timestamp": "2024-06-03 10:00:00",
                "event_type": "rating",
            },
            {
                "user_id": user_ids[2],
                "item_id": item_ids[2],
                "rating": 4.0,
                "timestamp": "not-a-date",  # invalid timestamp
                "event_type": "click",
            },
            {
                "user_id": user_ids[3],
                "item_id": item_ids[3],
                "rating": np.nan,  # missing rating
                "timestamp": "2024-06-04 10:00:00",
                "event_type": "rating",
            },
        ]
    )
    # Duplicate a valid row
    dup = interactions.iloc[[0]].copy()
    interactions = pd.concat([interactions, dirty, dup], ignore_index=True)

    return users, items, interactions


def ingest_data(config_path: str | None = None) -> dict[str, Path]:
    """Ingest synthetic recommendation data and write CSV + Parquet to data/raw."""
    config = load_config(config_path)
    paths = ensure_dirs(config)
    raw_dir = paths["raw"]

    users, items, interactions = _generate_synthetic(config)

    outputs = {
        "users": raw_dir / "users.csv",
        "items": raw_dir / "items.csv",
        "interactions": raw_dir / "interactions.csv",
    }
    users.to_csv(outputs["users"], index=False)
    items.to_csv(outputs["items"], index=False)
    interactions.to_csv(outputs["interactions"], index=False)

    # Also store parquet copies for modern lakehouse-style storage
    users.to_parquet(raw_dir / "users.parquet", index=False)
    items.to_parquet(raw_dir / "items.parquet", index=False)
    interactions.to_parquet(raw_dir / "interactions.parquet", index=False)

    meta = {
        "source": "synthetic_movie_platform",
        "description": (
            "Documented synthetic dataset mimicking an online movie platform. "
            "Includes intentional dirty records for validation demos."
        ),
        "n_users": int(len(users)),
        "n_items": int(len(items)),
        "n_interactions": int(len(interactions)),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "student_id": config["project"]["student_id"],
        "files": {k: str(v.relative_to(raw_dir.parent.parent)) for k, v in outputs.items()},
    }
    save_json(meta, raw_dir / "ingestion_metadata.json")
    print(f"[ingestion] Wrote users={len(users)}, items={len(items)}, interactions={len(interactions)}")
    return outputs
