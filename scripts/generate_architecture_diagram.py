"""Generate pipeline architecture diagram for the report."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    stages = [
        ("1. Ingestion", "CSV / Parquet\nsynthetic movie data"),
        ("2. Validation", "Missing / dups /\nratings / FK checks"),
        ("3. Preparation", "Clean + standardize\ntime-based split"),
        ("4. Transformation", "Activity / popularity\nuser-item matrix"),
        ("5. Features", "User / item /\ninteraction features"),
        ("6. Training", "Popularity +\nMatrix Factorization"),
        ("7. Evaluation", "Precision/Recall/NDCG\nCoverage/Diversity"),
        ("8. Orchestration", "DAG dependencies\nMLflow tracking"),
    ]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(
        "Architecture: End-to-End Recommendation Data Management Pipeline\n"
        "Student: 2025ag05734 | DM4ML Assignment 2",
        fontsize=13,
        fontweight="bold",
        pad=16,
    )

    # Two rows of 4 boxes
    positions = []
    for i, (title, detail) in enumerate(stages):
        row = 0 if i < 4 else 1
        col = i if i < 4 else i - 4
        x = 0.7 + col * 3.3
        y = 6.2 if row == 0 else 2.0
        positions.append((x, y))
        box = FancyBboxPatch(
            (x, y),
            2.8,
            2.0,
            boxstyle="round,pad=0.05,rounding_size=0.25",
            facecolor="#e3f2fd" if row == 0 else "#e8f5e9",
            edgecolor="#1565c0" if row == 0 else "#2e7d32",
            linewidth=1.8,
        )
        ax.add_patch(box)
        ax.text(x + 1.4, y + 1.35, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + 1.4, y + 0.7, detail, ha="center", va="center", fontsize=8, color="#333333")

    # Horizontal arrows row 1
    for i in range(3):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 2.8, y1 + 1.0),
                (x2, y2 + 1.0),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.5,
                color="#455a64",
            )
        )
    # Connector from 4 -> 5
    ax.add_patch(
        FancyArrowPatch(
            (positions[3][0] + 1.4, positions[3][1]),
            (positions[4][0] + 1.4, positions[4][1] + 2.0),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.5,
            color="#455a64",
            connectionstyle="arc3,rad=0.0",
        )
    )
    for i in range(4, 7):
        x1, y1 = positions[i]
        x2, y2 = positions[i + 1]
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 2.8, y1 + 1.0),
                (x2, y2 + 1.0),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.5,
                color="#455a64",
            )
        )

    # Storage lane
    storage = FancyBboxPatch(
        (0.7, 0.35),
        12.6,
        1.1,
        boxstyle="round,pad=0.04,rounding_size=0.2",
        facecolor="#fff8e1",
        edgecolor="#ef6c00",
        linewidth=1.5,
    )
    ax.add_patch(storage)
    ax.text(
        7.0,
        0.9,
        "Storage & Artifacts: data/raw → data/processed → data/features → models/ → reports/ + mlruns/",
        ha="center",
        va="center",
        fontsize=9,
    )

    out = ROOT / "reports" / "architecture_diagram.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

    # Mermaid source for GitHub README
    mermaid = """```mermaid
flowchart LR
  A[Data Ingestion] --> B[Data Validation]
  B --> C[Data Preparation]
  C --> D[Transformation]
  D --> E[Feature Management]
  E --> F[Model Training]
  F --> G[Model Evaluation]
  G --> H[Orchestration / MLflow]
  B --> E1[errors/]
  C --> P[processed/]
  E --> F1[features/]
  F --> M[models/]
```
"""
    (ROOT / "reports" / "architecture_diagram.md").write_text(
        "# Pipeline Architecture\n\n![Architecture](architecture_diagram.png)\n\n" + mermaid,
        encoding="utf-8",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
