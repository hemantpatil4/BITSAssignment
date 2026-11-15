"""Automated data quality validation for recommendation datasets."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.utils import ensure_dirs, load_config, save_json


REQUIRED_USER_COLS = ["user_id", "age_group", "gender", "city", "signup_date"]
REQUIRED_ITEM_COLS = ["item_id", "title", "category", "release_year", "duration_min", "price"]
REQUIRED_INT_COLS = ["user_id", "item_id", "rating", "timestamp", "event_type"]


def _append_error(errors: list[dict], df: pd.DataFrame, mask: pd.Series, reason: str) -> pd.Series:
    bad = df.loc[mask].copy()
    if not bad.empty:
        bad["error_reason"] = reason
        errors.append(bad)
    return ~mask


def validate_data(config_path: str | None = None) -> dict:
    config = load_config(config_path)
    paths = ensure_dirs(config)
    raw = paths["raw"]
    err_dir = paths["errors"]
    reports = paths["reports"]

    users = pd.read_csv(raw / "users.csv")
    items = pd.read_csv(raw / "items.csv")
    interactions = pd.read_csv(raw / "interactions.csv")

    checks = []
    error_frames: list[pd.DataFrame] = []

    # Schema / dtype checks
    missing_user_cols = [c for c in REQUIRED_USER_COLS if c not in users.columns]
    missing_item_cols = [c for c in REQUIRED_ITEM_COLS if c not in items.columns]
    missing_int_cols = [c for c in REQUIRED_INT_COLS if c not in interactions.columns]
    checks.append({"check": "users_schema", "passed": not missing_user_cols, "detail": missing_user_cols})
    checks.append({"check": "items_schema", "passed": not missing_item_cols, "detail": missing_item_cols})
    checks.append({"check": "interactions_schema", "passed": not missing_int_cols, "detail": missing_int_cols})

    # Users / items duplicates and null keys
    user_dup = users["user_id"].duplicated()
    item_dup = items["item_id"].duplicated()
    checks.append({"check": "users_duplicate_ids", "passed": int(user_dup.sum()) == 0, "count": int(user_dup.sum())})
    checks.append({"check": "items_duplicate_ids", "passed": int(item_dup.sum()) == 0, "count": int(item_dup.sum())})

    keep = pd.Series(True, index=interactions.index)

    # Missing values in interactions
    missing_mask = interactions[REQUIRED_INT_COLS].isna().any(axis=1)
    keep &= _append_error(error_frames, interactions, missing_mask & keep, "missing_values")
    checks.append({"check": "interactions_missing_values", "passed": int(missing_mask.sum()) == 0, "count": int(missing_mask.sum())})

    # Duplicates
    dup_mask = interactions.duplicated(subset=["user_id", "item_id", "timestamp"], keep="first")
    keep &= _append_error(error_frames, interactions, dup_mask & keep, "duplicate_interaction")
    checks.append({"check": "interactions_duplicates", "passed": int(dup_mask.sum()) == 0, "count": int(dup_mask.sum())})

    # Rating range
    rmin, rmax = config["validation"]["rating_min"], config["validation"]["rating_max"]
    rating_num = pd.to_numeric(interactions["rating"], errors="coerce")
    invalid_rating = keep & ((rating_num < rmin) | (rating_num > rmax) | rating_num.isna())
    # only count as invalid rating if rating was present but wrong, or non-numeric
    keep &= _append_error(error_frames, interactions, invalid_rating, "invalid_rating")
    checks.append({"check": "invalid_ratings", "passed": int(invalid_rating.sum()) == 0, "count": int(invalid_rating.sum())})

    # Timestamp validity
    parsed_ts = pd.to_datetime(interactions["timestamp"], errors="coerce")
    invalid_ts = keep & parsed_ts.isna()
    keep &= _append_error(error_frames, interactions, invalid_ts, "invalid_timestamp")
    checks.append({"check": "invalid_timestamps", "passed": int(invalid_ts.sum()) == 0, "count": int(invalid_ts.sum())})

    # Referential integrity
    valid_users = set(users["user_id"])
    valid_items = set(items["item_id"])
    bad_user = keep & ~interactions["user_id"].isin(valid_users)
    bad_item = keep & ~interactions["item_id"].isin(valid_items)
    keep &= _append_error(error_frames, interactions, bad_user, "unknown_user_id")
    keep &= _append_error(error_frames, interactions, bad_item, "unknown_item_id")
    checks.append({"check": "referential_integrity_users", "passed": int(bad_user.sum()) == 0, "count": int(bad_user.sum())})
    checks.append({"check": "referential_integrity_items", "passed": int(bad_item.sum()) == 0, "count": int(bad_item.sum())})

    valid = interactions.loc[keep].copy()
    errors = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()

    valid_path = paths["processed"] / "interactions_validated.csv"
    error_path = err_dir / "invalid_interactions.csv"
    valid.to_csv(valid_path, index=False)
    errors.to_csv(error_path, index=False)
    users.to_csv(paths["processed"] / "users_validated.csv", index=False)
    items.to_csv(paths["processed"] / "items_validated.csv", index=False)

    summary = {
        "validated_at": datetime.utcnow().isoformat() + "Z",
        "input_interactions": int(len(interactions)),
        "valid_interactions": int(len(valid)),
        "invalid_interactions": int(len(errors)),
        "checks": checks,
        "error_breakdown": errors["error_reason"].value_counts().to_dict() if not errors.empty else {},
        "artifacts": {
            "valid_interactions": str(valid_path),
            "error_dataset": str(error_path),
        },
    }
    save_json(summary, reports / "data_validation_report.json")

    # Human-readable markdown report
    lines = [
        "# Data Validation Report",
        "",
        f"- Student ID: `{config['project']['student_id']}`",
        f"- Validated at: {summary['validated_at']}",
        f"- Input interactions: **{summary['input_interactions']}**",
        f"- Valid interactions: **{summary['valid_interactions']}**",
        f"- Invalid interactions (quarantined): **{summary['invalid_interactions']}**",
        "",
        "## Checks",
        "",
        "| Check | Passed | Count/Detail |",
        "|---|---|---|",
    ]
    for c in checks:
        detail = c.get("count", c.get("detail", ""))
        lines.append(f"| {c['check']} | {'PASS' if c['passed'] else 'FAIL'} | {detail} |")
    lines += ["", "## Error Breakdown", ""]
    if summary["error_breakdown"]:
        for k, v in summary["error_breakdown"].items():
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("- No invalid records")
    (reports / "data_validation_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        f"[validation] valid={len(valid)} invalid={len(errors)} "
        f"(errors logged to {error_path.name})"
    )
    return summary
