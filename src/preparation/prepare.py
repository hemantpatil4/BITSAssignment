"""Data preparation: cleaning, standardization, time-based train/test split."""

from __future__ import annotations

import pandas as pd

from src.utils import ensure_dirs, load_config, save_json


def prepare_data(config_path: str | None = None) -> dict:
    config = load_config(config_path)
    paths = ensure_dirs(config)
    processed = paths["processed"]

    users = pd.read_csv(processed / "users_validated.csv")
    items = pd.read_csv(processed / "items_validated.csv")
    interactions = pd.read_csv(processed / "interactions_validated.csv")

    # Standardize formats
    users["user_id"] = users["user_id"].astype(str).str.strip()
    items["item_id"] = items["item_id"].astype(str).str.strip()
    items["category"] = items["category"].astype(str).str.strip().str.title()
    interactions["user_id"] = interactions["user_id"].astype(str).str.strip()
    interactions["item_id"] = interactions["item_id"].astype(str).str.strip()
    interactions["rating"] = pd.to_numeric(interactions["rating"], errors="coerce")
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], errors="coerce")

    # Drop residual nulls after coercion
    before = len(interactions)
    interactions = interactions.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
    interactions = interactions.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    interactions = interactions.sort_values("timestamp").reset_index(drop=True)

    # Time-based split
    cutoff_idx = int(len(interactions) * (1 - config["split"]["test_ratio"]))
    cutoff_ts = interactions.iloc[cutoff_idx]["timestamp"]
    train = interactions[interactions["timestamp"] <= cutoff_ts].copy()
    test = interactions[interactions["timestamp"] > cutoff_ts].copy()

    # Ensure test users/items also appear in train when possible for CF
    train_users = set(train["user_id"])
    train_items = set(train["item_id"])
    test = test[test["user_id"].isin(train_users) & test["item_id"].isin(train_items)].copy()

    users.to_parquet(processed / "users_clean.parquet", index=False)
    items.to_parquet(processed / "items_clean.parquet", index=False)
    train.to_parquet(processed / "interactions_train.parquet", index=False)
    test.to_parquet(processed / "interactions_test.parquet", index=False)
    train.to_csv(processed / "interactions_train.csv", index=False)
    test.to_csv(processed / "interactions_test.csv", index=False)

    meta = {
        "rows_before_clean": before,
        "rows_after_clean": int(len(interactions)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "cutoff_timestamp": str(cutoff_ts),
        "split_method": "time_based",
        "test_ratio": config["split"]["test_ratio"],
    }
    save_json(meta, processed / "preparation_metadata.json")
    print(f"[preparation] train={len(train)} test={len(test)} cutoff={cutoff_ts}")
    return meta
