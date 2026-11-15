#!/usr/bin/env python3
"""Entry point: run the full recommendation data management pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dags.recommendation_pipeline_dag import run_workflow


def main() -> int:
    summary = run_workflow()
    return 0 if summary["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
