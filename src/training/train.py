"""Model training: popularity baseline + matrix factorization (personalized)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.utils import ensure_dirs, load_config, project_root, save_json


class PopularityRecommender:
    """Baseline: recommend globally most popular items by interaction count."""

    def __init__(self, top_n: int = 100):
        self.top_n = top_n
        self.ranking_: list[str] = []
        self.item_avg_rating_: dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        counts = train.groupby("item_id").size().sort_values(ascending=False)
        self.ranking_ = counts.index.tolist()[: self.top_n]
        self.item_avg_rating_ = train.groupby("item_id")["rating"].mean().to_dict()
        self.global_mean_ = float(train["rating"].mean())
        return self

    def recommend(self, user_id: str, k: int = 10, seen: set[str] | None = None) -> list[str]:
        seen = seen or set()
        recs = [i for i in self.ranking_ if i not in seen]
        return recs[:k]

    def predict_rating(self, user_id: str, item_id: str) -> float:
        return float(self.item_avg_rating_.get(item_id, self.global_mean_))


class MatrixFactorizationRecommender:
    """Personalized collaborative filtering via Truncated SVD on the rating matrix."""

    def __init__(self, n_factors: int = 32, random_seed: int = 42):
        self.n_factors = n_factors
        self.random_seed = random_seed
        self.svd: TruncatedSVD | None = None
        self.user_index: dict[str, int] = {}
        self.item_index: dict[str, int] = {}
        self.index_item: dict[int, str] = {}
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.global_mean: float = 0.0

    def fit(self, train: pd.DataFrame) -> "MatrixFactorizationRecommender":
        users = sorted(train["user_id"].unique())
        items = sorted(train["item_id"].unique())
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {it: i for i, it in enumerate(items)}
        self.index_item = {i: it for it, i in self.item_index.items()}
        self.global_mean = float(train["rating"].mean())

        rows = train["user_id"].map(self.user_index).to_numpy()
        cols = train["item_id"].map(self.item_index).to_numpy()
        data = train["rating"].to_numpy(dtype=float)
        mat = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

        n_comp = min(self.n_factors, min(mat.shape) - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_seed)
        self.user_factors = self.svd.fit_transform(mat)
        self.item_factors = self.svd.components_.T
        return self

    def predict_rating(self, user_id: str, item_id: str) -> float:
        if user_id not in self.user_index or item_id not in self.item_index:
            return self.global_mean
        u = self.user_index[user_id]
        i = self.item_index[item_id]
        pred = float(self.user_factors[u] @ self.item_factors[i])
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(self, user_id: str, k: int = 10, seen: set[str] | None = None) -> list[str]:
        seen = seen or set()
        if user_id not in self.user_index or self.user_factors is None:
            return []
        u = self.user_index[user_id]
        scores = self.user_factors[u] @ self.item_factors.T
        ranked_idx = np.argsort(-scores)
        recs = []
        for idx in ranked_idx:
            item = self.index_item[int(idx)]
            if item in seen:
                continue
            recs.append(item)
            if len(recs) >= k:
                break
        return recs


def train_models(config_path: str | None = None) -> dict:
    config = load_config(config_path)
    paths = ensure_dirs(config)
    processed = paths["processed"]
    models_dir = paths["models"]
    mlruns = paths["mlruns"]

    train = pd.read_parquet(processed / "interactions_train.parquet")
    k = config["training"]["k"]
    n_factors = config["training"]["n_factors"]
    seed = config["training"]["random_seed"]

    # Prefer SQLite tracking (MLflow file store is deprecated/disabled by default)
    tracking_db = project_root() / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    mlflow.set_experiment("recommendation-pipeline")

    results = {}

    with mlflow.start_run(run_name="popularity_baseline") as run:
        pop = PopularityRecommender(top_n=config["training"]["popularity_top_n"])
        pop.fit(train)
        pop_path = models_dir / "popularity_model.joblib"
        joblib.dump(pop, pop_path)
        mlflow.log_params({"model_type": "popularity", "top_n": pop.top_n, "k": k})
        mlflow.log_metric("train_interactions", len(train))
        mlflow.log_metric("unique_items_ranked", len(pop.ranking_))
        mlflow.log_artifact(str(pop_path))
        results["popularity"] = {
            "run_id": run.info.run_id,
            "model_path": str(pop_path),
            "params": {"top_n": pop.top_n},
        }

    with mlflow.start_run(run_name="matrix_factorization") as run:
        mf = MatrixFactorizationRecommender(n_factors=n_factors, random_seed=seed)
        mf.fit(train)
        mf_path = models_dir / "mf_model.joblib"
        joblib.dump(mf, mf_path)
        mlflow.log_params(
            {
                "model_type": "matrix_factorization",
                "n_factors": n_factors,
                "random_seed": seed,
                "k": k,
            }
        )
        mlflow.log_metric("train_interactions", len(train))
        mlflow.log_metric("n_users", len(mf.user_index))
        mlflow.log_metric("n_items", len(mf.item_index))
        mlflow.log_metric("explained_variance_sum", float(mf.svd.explained_variance_ratio_.sum()))
        mlflow.log_artifact(str(mf_path))
        results["matrix_factorization"] = {
            "run_id": run.info.run_id,
            "model_path": str(mf_path),
            "params": {"n_factors": n_factors, "random_seed": seed},
            "explained_variance_sum": float(mf.svd.explained_variance_ratio_.sum()),
        }

    # Also dump a small mapping of seen items per user for inference reuse
    seen_map = train.groupby("user_id")["item_id"].apply(set).to_dict()
    seen_serializable = {u: sorted(list(v)) for u, v in seen_map.items()}
    save_json(seen_serializable, models_dir / "user_seen_items.json")
    save_json(results, models_dir / "training_metadata.json")

    print("[training] Saved popularity + matrix factorization models with MLflow tracking")
    return results
