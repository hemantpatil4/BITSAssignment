"""Derived datasets: activity summaries, popularity, user-item matrix."""

from __future__ import annotations

import pandas as pd

from src.utils import ensure_dirs, load_config


def transform_data(config_path: str | None = None) -> dict:
    config = load_config(config_path)
    paths = ensure_dirs(config)
    processed = paths["processed"]

    users = pd.read_parquet(processed / "users_clean.parquet")
    items = pd.read_parquet(processed / "items_clean.parquet")
    train = pd.read_parquet(processed / "interactions_train.parquet")

    # User activity summary
    user_activity = (
        train.groupby("user_id")
        .agg(
            interaction_count=("item_id", "count"),
            avg_rating=("rating", "mean"),
            std_rating=("rating", "std"),
            first_interaction=("timestamp", "min"),
            last_interaction=("timestamp", "max"),
        )
        .reset_index()
    )
    user_activity["std_rating"] = user_activity["std_rating"].fillna(0.0)
    user_activity = users.merge(user_activity, on="user_id", how="left")
    user_activity["interaction_count"] = user_activity["interaction_count"].fillna(0).astype(int)
    user_activity["avg_rating"] = user_activity["avg_rating"].fillna(0.0)

    # Item popularity
    item_popularity = (
        train.groupby("item_id")
        .agg(
            popularity_count=("user_id", "count"),
            avg_rating=("rating", "mean"),
            rating_std=("rating", "std"),
        )
        .reset_index()
    )
    item_popularity["rating_std"] = item_popularity["rating_std"].fillna(0.0)
    item_popularity = items.merge(item_popularity, on="item_id", how="left")
    item_popularity["popularity_count"] = item_popularity["popularity_count"].fillna(0).astype(int)
    item_popularity["avg_rating"] = item_popularity["avg_rating"].fillna(0.0)
    item_popularity = item_popularity.sort_values("popularity_count", ascending=False)

    # User-item interaction matrix (ratings)
    matrix = train.pivot_table(index="user_id", columns="item_id", values="rating", aggfunc="mean")
    matrix = matrix.fillna(0.0)

    user_activity.to_parquet(processed / "user_activity_summary.parquet", index=False)
    item_popularity.to_parquet(processed / "item_popularity.parquet", index=False)
    matrix.to_parquet(processed / "user_item_matrix.parquet")
    user_activity.to_csv(processed / "user_activity_summary.csv", index=False)
    item_popularity.to_csv(processed / "item_popularity.csv", index=False)

    print(
        f"[transformation] user_activity={len(user_activity)} "
        f"item_popularity={len(item_popularity)} matrix={matrix.shape}"
    )
    return {
        "user_activity_rows": int(len(user_activity)),
        "item_popularity_rows": int(len(item_popularity)),
        "matrix_shape": list(matrix.shape),
    }
