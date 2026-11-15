# End-to-End Data Management Pipeline for a Recommendation System

**Course:** Data Management for Machine Learning — Assignment 2  
**Student ID:** `2025ag05734`  
**Focus:** Modular, reproducible recommendation data pipeline (not only model accuracy)

## Pipeline Stages

```
Data Ingestion → Validation → Preparation → Transformation
→ Feature Management → Model Training → Evaluation → Orchestration
```

## Project Structure

```text
Assignment2_2025ag05734/
├── configs/config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── errors/
├── src/
│   ├── ingestion/
│   ├── validation/
│   ├── preparation/
│   ├── transformation/
│   ├── features/
│   ├── training/
│   └── evaluation/
├── dags/recommendation_pipeline_dag.py
├── models/
├── reports/
├── mlruns/
├── run_pipeline.py
├── requirements.txt
└── README.md
```

## Setup

```bash
cd Assignment2_2025ag05734
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run End-to-End

```bash
python run_pipeline.py
```

This executes the DAG:

`ingest → validate → prepare → transform → features → train → evaluate`

Optional helpers:

```bash
python scripts/generate_architecture_diagram.py
python scripts/generate_final_report.py
```

## Dataset

Documented **synthetic movie-platform data** (allowed by the assignment):

- 500 users, 200 movies, ~12,000 interactions
- Intentional dirty records for validation demonstration
- Stored as CSV + Parquet under `data/raw/`

## Models

1. **Popularity baseline** — global item frequency ranking  
2. **Matrix Factorization (Truncated SVD)** — personalized collaborative filtering  

Tracked with **MLflow** (`mlruns/`).

## Evaluation Metrics

- Precision@10, Recall@10, NDCG@10  
- Coverage, Diversity  
- RMSE, MAE (explicit ratings)

Results: `reports/model_evaluation_results.csv`

## Key Deliverables in `reports/`

| Artifact | File |
|---|---|
| Architecture diagram | `architecture_diagram.png` |
| Validation report | `data_validation_report.md` / `.json` |
| Feature catalog | `feature_catalog.md` / `.csv` |
| Model comparison | `model_evaluation_results.md` / `.csv` |
| Orchestration run log | `orchestration_run_log.json` |
| Successful run screenshot | `orchestration_successful_run.png` |
| Final report (PDF) | `Final_Report_2025ag05734.pdf` |

## Orchestration

`dags/recommendation_pipeline_dag.py` defines:

- Explicit task dependency graph
- Runnable scheduled Python workflow (`run_workflow()`)
- Optional Apache Airflow DAG (if Airflow is installed)

## Notes

- Random seed is fixed in `configs/config.yaml` for reproducibility.
- Invalid records are quarantined in `data/errors/invalid_interactions.csv`.
- Features are reusable for training and prediction via `data/features/`.
