"""Generate the 2–3 page final report PDF."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    ListFlowable,
    ListItem,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    pdf_path = reports / "Final_Report_2025ag05734.pdf"

    eval_csv = reports / "model_evaluation_results.csv"
    eval_rows = []
    if eval_csv.exists():
        import pandas as pd

        df = pd.read_csv(eval_csv)
        eval_rows = [list(df.columns)] + df.round(4).astype(str).values.tolist()

    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=16, spaceAfter=8)
    h1 = ParagraphStyle("H1Custom", parent=styles["Heading1"], fontSize=13, spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("BodyCustom", parent=styles["BodyText"], fontSize=10, leading=14, spaceAfter=6)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )
    story = []
    story.append(Paragraph("Final Report: End-to-End Data Management Pipeline for a Recommendation System", title))
    story.append(Paragraph("Data Management for Machine Learning — Assignment 2", body))
    story.append(Paragraph("Student ID: <b>2025ag05734</b> &nbsp;&nbsp;|&nbsp;&nbsp; Name: Hemant Patil", body))
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Objective and Problem Setting", h1))
    story.append(
        Paragraph(
            "This project designs and implements a reproducible recommendation data pipeline for an "
            "online movie platform. The system ingests user, item, and interaction data; validates quality; "
            "prepares and transforms datasets; builds reusable features; trains two recommenders; evaluates "
            "them with ranking metrics; and orchestrates the workflow through an explicit task DAG.",
            body,
        )
    )

    story.append(Paragraph("2. Design Decisions", h1))
    story.append(
        Paragraph(
            "<b>Dataset:</b> Documented synthetic movie-platform data (500 users, 200 items, ~12k interactions) "
            "with intentional dirty records so validation can quarantine invalid rows. This keeps the pipeline "
            "fully reproducible offline while matching MovieLens-style schemas.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Validation:</b> Custom automated checks for missing values, duplicates, invalid ratings, "
            "invalid timestamps, and referential integrity. Invalid rows are written to <i>data/errors/</i> "
            "and summarized in a validation report.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Split:</b> Time-based train/test split (80/20 by timestamp) to avoid leakage from future interactions.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Models:</b> (1) Popularity baseline for a strong non-personalized reference; "
            "(2) Truncated-SVD matrix factorization for personalized collaborative filtering. "
            "Artifacts and metrics are tracked with MLflow.",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Orchestration:</b> Airflow-style DAG with hard task dependencies, executable via "
            "<i>run_pipeline.py</i>. The same module optionally registers an Airflow DAG if Airflow is installed.",
            body,
        )
    )

    story.append(Paragraph("3. Tools Used", h1))
    story.append(
        ListFlowable(
            [
                ListItem(Paragraph("Python, Pandas, NumPy, PyArrow/Parquet", body)),
                ListItem(Paragraph("Scikit-learn TruncatedSVD for collaborative filtering", body)),
                ListItem(Paragraph("MLflow for experiment tracking and artifact logging", body)),
                ListItem(Paragraph("Custom feature layer + feature catalog (CSV/Markdown)", body)),
                ListItem(Paragraph("Matplotlib for architecture/orchestration visuals", body)),
                ListItem(Paragraph("Structured Python DAG (Airflow-compatible)", body)),
            ],
            bulletType="bullet",
        )
    )

    story.append(Paragraph("4. Results", h1))
    if eval_rows:
        table = Table(eval_rows, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565c0")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#e3f2fd")),
                    ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#e8f5e9")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 8))
        story.append(
            Paragraph(
                "Personalized matrix factorization is expected to improve ranking quality over pure popularity "
                "for users with sufficient history, while popularity often retains strong coverage of globally "
                "liked titles. Exact numeric winners depend on the generated split and are reported in "
                "<i>reports/model_evaluation_results.csv</i>.",
                body,
            )
        )
    else:
        story.append(Paragraph("Evaluation results will appear after running the pipeline.", body))

    arch = reports / "architecture_diagram.png"
    if arch.exists():
        story.append(Paragraph("5. Pipeline Architecture", h1))
        story.append(Image(str(arch), width=6.5 * inch, height=3.7 * inch))

    story.append(Paragraph("6. Limitations and Future Work", h1))
    story.append(
        Paragraph(
            "Limitations include: synthetic data may under-represent real sparsity and popularity skew; "
            "SVD collaborative filtering is a linear latent-factor baseline rather than a deep model; "
            "cold-start users/items fall back to global statistics; and orchestration uses a lightweight "
            "Airflow-compatible DAG runner rather than a full managed Airflow deployment.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Future improvements: integrate Feast for online feature serving, add Great Expectations suites, "
            "support incremental ingestion from PostgreSQL, and evaluate hybrid content + collaborative models.",
            body,
        )
    )

    story.append(Paragraph("7. Reproducibility", h1))
    story.append(
        Paragraph(
            "Install dependencies from <i>requirements.txt</i>, then run <b>python run_pipeline.py</b>. "
            "All stages write deterministic artifacts under <i>data/</i>, <i>models/</i>, <i>reports/</i>, and <i>mlruns/</i>.",
            body,
        )
    )

    doc.build(story)
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
