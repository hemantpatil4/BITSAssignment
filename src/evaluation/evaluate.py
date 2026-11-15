"""Model evaluation: ranking metrics + rating error metrics."""

from __future__ import annotations

from collections import defaultdict

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.utils import ensure_dirs, load_config, project_root, save_json


def _dcg(relevances: list[float]) -> float:
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    top = recommended[:k]
    return len([i for i in top if i in relevant]) / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = recommended[:k]
    return len([i for i in top if i in relevant]) / len(relevant)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    top = recommended[:k]
    gains = [1.0 if i in relevant else 0.0 for i in top]
    ideal = sorted(gains, reverse=True)
    idcg = _dcg(ideal)
    if idcg == 0:
        return 0.0
    return _dcg(gains) / idcg


def coverage(all_recs: list[list[str]], catalog_size: int) -> float:
    unique = set(i for rec in all_recs for i in rec)
    return len(unique) / catalog_size if catalog_size else 0.0


def diversity(all_recs: list[list[str]], item_category: dict[str, str]) -> float:
    """Average intra-list category diversity (unique categories / list length)."""
    scores = []
    for rec in all_recs:
        if not rec:
            continue
        cats = {item_category.get(i, "Unknown") for i in rec}
        scores.append(len(cats) / len(rec))
    return float(np.mean(scores)) if scores else 0.0


def rmse_mae(model, test: pd.DataFrame) -> tuple[float, float]:
    preds = [model.predict_rating(u, i) for u, i in zip(test["user_id"], test["item_id"])]
    y = test["rating"].to_numpy(dtype=float)
    p = np.asarray(preds, dtype=float)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    return rmse, mae


def _evaluate_one(model, train: pd.DataFrame, test: pd.DataFrame, items: pd.DataFrame, k: int, threshold: float) -> dict:
    seen_map = train.groupby("user_id")["item_id"].apply(set).to_dict()
    relevant_map: dict[str, set[str]] = defaultdict(set)
    for _, row in test.iterrows():
        if row["rating"] >= threshold:
            relevant_map[row["user_id"]].add(row["item_id"])

    users = [u for u in relevant_map if u in seen_map]
    all_recs = []
    precs, recalls, ndcgs = [], [], []

    for user in users:
        recs = model.recommend(user, k=k, seen=seen_map.get(user, set()))
        all_recs.append(recs)
        rel = relevant_map[user]
        precs.append(precision_at_k(recs, rel, k))
        recalls.append(recall_at_k(recs, rel, k))
        ndcgs.append(ndcg_at_k(recs, rel, k))

    item_category = dict(zip(items["item_id"], items["category"]))
    rmse, mae = rmse_mae(model, test)

    return {
        f"Precision@{k}": float(np.mean(precs)) if precs else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "Coverage": float(coverage(all_recs, items["item_id"].nunique())),
        "Diversity": float(diversity(all_recs, item_category)),
        "RMSE": rmse,
        "MAE": mae,
        "n_eval_users": int(len(users)),
    }


def evaluate_models(config_path: str | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    paths = ensure_dirs(config)
    processed = paths["processed"]
    models_dir = paths["models"]
    reports = paths["reports"]
    mlruns = paths["mlruns"]

    train = pd.read_parquet(processed / "interactions_train.parquet")
    test = pd.read_parquet(processed / "interactions_test.parquet")
    items = pd.read_parquet(processed / "items_clean.parquet")
    k = config["evaluation"]["k"]
    threshold = config["evaluation"]["relevance_threshold"]

    pop = joblib.load(models_dir / "popularity_model.joblib")
    mf = joblib.load(models_dir / "mf_model.joblib")

    pop_metrics = _evaluate_one(pop, train, test, items, k, threshold)
    mf_metrics = _evaluate_one(mf, train, test, items, k, threshold)

    results = pd.DataFrame(
        [
            {"model": "Popularity Baseline", **pop_metrics},
            {"model": "Matrix Factorization (CF)", **mf_metrics},
        ]
    )
    results.to_csv(reports / "model_evaluation_results.csv", index=False)

    # Markdown comparison table
    cols = ["model", f"Precision@{k}", f"Recall@{k}", f"NDCG@{k}", "Coverage", "Diversity", "RMSE", "MAE"]
    md = [
        "# Model Evaluation Results",
        "",
        f"K = {k}, relevance threshold = {threshold}",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for _, row in results.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        md.append("| " + " | ".join(vals) + " |")
    winner = (
        "Matrix Factorization"
        if mf_metrics[f"NDCG@{k}"] >= pop_metrics[f"NDCG@{k}"]
        else "Popularity Baseline"
    )
    md += ["", f"**Best model by NDCG@{k}:** {winner}", ""]
    (reports / "model_evaluation_results.md").write_text("\n".join(md), encoding="utf-8")
    save_json(results.to_dict(orient="records"), reports / "model_evaluation_results.json")

    tracking_db = project_root() / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    mlflow.set_experiment("recommendation-pipeline")
    with mlflow.start_run(run_name="evaluation_comparison"):
        for prefix, metrics in [("popularity", pop_metrics), ("mf", mf_metrics)]:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"{prefix}_{name.replace('@', '_at_')}", float(value))
        mlflow.log_artifact(str(reports / "model_evaluation_results.csv"))

    print("[evaluation] Wrote comparison table to reports/model_evaluation_results.csv")
    print(results[cols].to_string(index=False))
    return results
