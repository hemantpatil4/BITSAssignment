"""Feature management layer (custom reusable feature store)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.utils import ensure_dirs, load_config, save_json


FEATURE_CATALOG = [
    {
        "feature_name": "user_avg_rating",
        "entity": "user",
        "dtype": "float",
        "description": "Average rating given by the user on training interactions",
        "source": "interactions_train",
    },
    {
        "feature_name": "user_activity_count",
        "entity": "user",
        "dtype": "int",
        "description": "Number of training interactions for the user",
        "source": "interactions_train",
    },
    {
        "feature_name": "user_rating_std",
        "entity": "user",
        "dtype": "float",
        "description": "Standard deviation of user ratings",
        "source": "interactions_train",
    },
    {
        "feature_name": "preferred_category",
        "entity": "user",
        "dtype": "string",
        "description": "Most frequently rated item category for the user",
        "source": "interactions_train + items",
    },
    {
        "feature_name": "days_since_last_interaction",
        "entity": "user",
        "dtype": "float",
        "description": "Days between reference date and user's last interaction",
        "source": "interactions_train",
    },
    {
        "feature_name": "item_popularity",
        "entity": "item",
        "dtype": "int",
        "description": "Number of users who interacted with the item in training",
        "source": "interactions_train",
    },
    {
        "feature_name": "item_avg_rating",
        "entity": "item",
        "dtype": "float",
        "description": "Average rating received by the item",
        "source": "interactions_train",
    },
    {
        "feature_name": "item_category",
        "entity": "item",
        "dtype": "string",
        "description": "Canonical item category",
        "source": "items",
    },
    {
        "feature_name": "item_release_year",
        "entity": "item",
        "dtype": "int",
        "description": "Release year of the item",
        "source": "items",
    },
    {
        "feature_name": "interaction_recency_days",
        "entity": "interaction",
        "dtype": "float",
        "description": "Days between interaction timestamp and max training timestamp",
        "source": "interactions_train",
    },
]


def build_features(config_path: str | None = None) -> dict:
    config = load_config(config_path)
    paths = ensure_dirs(config)
    processed = paths["processed"]
    feat_dir = paths["features"]
    reports = paths["reports"]

    users = pd.read_parquet(processed / "users_clean.parquet")
    items = pd.read_parquet(processed / "items_clean.parquet")
    train = pd.read_parquet(processed / "interactions_train.parquet")
    train["timestamp"] = pd.to_datetime(train["timestamp"])

    ref_ts = train["timestamp"].max()

    # User features
    user_stats = (
        train.groupby("user_id")
        .agg(
            user_avg_rating=("rating", "mean"),
            user_activity_count=("item_id", "count"),
            user_rating_std=("rating", "std"),
            last_interaction=("timestamp", "max"),
        )
        .reset_index()
    )
    user_stats["user_rating_std"] = user_stats["user_rating_std"].fillna(0.0)
    user_stats["days_since_last_interaction"] = (
        (ref_ts - user_stats["last_interaction"]).dt.total_seconds() / 86400.0
    )

    train_with_cat = train.merge(items[["item_id", "category"]], on="item_id", how="left")
    preferred = (
        train_with_cat.groupby(["user_id", "category"])
        .size()
        .reset_index(name="cnt")
        .sort_values(["user_id", "cnt"], ascending=[True, False])
        .drop_duplicates("user_id")
        .rename(columns={"category": "preferred_category"})
    )[["user_id", "preferred_category"]]

    user_features = users[["user_id", "age_group", "gender", "city"]].merge(
        user_stats.drop(columns=["last_interaction"]), on="user_id", how="left"
    )
    user_features = user_features.merge(preferred, on="user_id", how="left")
    user_features["user_avg_rating"] = user_features["user_avg_rating"].fillna(0.0)
    user_features["user_activity_count"] = user_features["user_activity_count"].fillna(0).astype(int)
    user_features["user_rating_std"] = user_features["user_rating_std"].fillna(0.0)
    user_features["days_since_last_interaction"] = user_features["days_since_last_interaction"].fillna(-1.0)
    user_features["preferred_category"] = user_features["preferred_category"].fillna("Unknown")

    # Item features
    item_stats = (
        train.groupby("item_id")
        .agg(item_popularity=("user_id", "count"), item_avg_rating=("rating", "mean"))
        .reset_index()
    )
    item_features = items[["item_id", "category", "release_year", "duration_min", "price"]].rename(
        columns={"category": "item_category", "release_year": "item_release_year"}
    )
    item_features = item_features.merge(item_stats, on="item_id", how="left")
    item_features["item_popularity"] = item_features["item_popularity"].fillna(0).astype(int)
    item_features["item_avg_rating"] = item_features["item_avg_rating"].fillna(0.0)

    # Interaction features
    interaction_features = train.copy()
    interaction_features["interaction_recency_days"] = (
        (ref_ts - interaction_features["timestamp"]).dt.total_seconds() / 86400.0
    )
    interaction_features = interaction_features.merge(
        user_features[["user_id", "user_avg_rating", "user_activity_count", "preferred_category"]],
        on="user_id",
        how="left",
    )
    interaction_features = interaction_features.merge(
        item_features[["item_id", "item_popularity", "item_avg_rating", "item_category"]],
        on="item_id",
        how="left",
    )

    user_features.to_parquet(feat_dir / "user_features.parquet", index=False)
    item_features.to_parquet(feat_dir / "item_features.parquet", index=False)
    interaction_features.to_parquet(feat_dir / "interaction_features.parquet", index=False)
    user_features.to_csv(feat_dir / "user_features.csv", index=False)
    item_features.to_csv(feat_dir / "item_features.csv", index=False)

    # Feature catalog table
    catalog_df = pd.DataFrame(FEATURE_CATALOG)
    catalog_df.to_csv(feat_dir / "feature_catalog.csv", index=False)
    catalog_df.to_csv(reports / "feature_catalog.csv", index=False)

    md = [
        "# Feature Catalog",
        "",
        f"Generated at: {datetime.utcnow().isoformat()}Z",
        "",
        "| Feature | Entity | Type | Description | Source |",
        "|---|---|---|---|---|",
    ]
    for row in FEATURE_CATALOG:
        md.append(
            f"| `{row['feature_name']}` | {row['entity']} | {row['dtype']} | "
            f"{row['description']} | {row['source']} |"
        )
    (reports / "feature_catalog.md").write_text("\n".join(md), encoding="utf-8")
    save_json(
        {
            "n_user_features_rows": int(len(user_features)),
            "n_item_features_rows": int(len(item_features)),
            "n_interaction_features_rows": int(len(interaction_features)),
            "reference_timestamp": str(ref_ts),
            "catalog_size": len(FEATURE_CATALOG),
        },
        feat_dir / "feature_metadata.json",
    )

    print(
        f"[features] users={len(user_features)} items={len(item_features)} "
        f"interactions={len(interaction_features)}"
    )
    return {"catalog_size": len(FEATURE_CATALOG)}
