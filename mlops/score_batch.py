"""Score transactions from a CSV file with the trained fraud model."""

import argparse
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from app.main import FEATURE_NAMES


def score_dataframe(
    frame: pd.DataFrame, model, threshold: float = 0.5
) -> pd.DataFrame:
    """Return a copy of the input with fraud probabilities and decisions."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    missing_features = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing_features:
        raise ValueError(f"Input is missing required features: {', '.join(missing_features)}")
    if frame.empty:
        raise ValueError("Input CSV contains no rows to score.")

    probabilities = np.asarray(
        model.predict_proba(frame.loc[:, FEATURE_NAMES]), dtype=float
    )
    if probabilities.ndim != 2 or probabilities.shape != (len(frame), 2):
        raise ValueError("Model returned an unexpected probability array shape.")

    scored = frame.copy()
    scored["fraud_probability"] = probabilities[:, 1]
    scored["predicted_fraud"] = scored["fraud_probability"] >= threshold
    return scored


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Score transactions in a CSV using the saved fraud model."
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="CSV file containing transaction features."
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Path for the scored CSV."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "models" / "fraud_model_pipeline.joblib",
        help="Model pipeline joblib file (default: models/fraud_model_pipeline.joblib).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fraud probability threshold from 0 to 1 (default: 0.5).",
    )
    args = parser.parse_args(argv)

    output_path = args.output.resolve()
    if output_path in {args.input.resolve(), args.model.resolve()}:
        parser.error("--output must not overwrite the input CSV or model artifact.")
    if not args.input.is_file():
        parser.error(f"Input CSV does not exist: {args.input}")
    if not args.model.is_file():
        parser.error(f"Model artifact does not exist: {args.model}")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")

    frame = pd.read_csv(args.input)
    model = joblib.load(args.model)
    scored = score_dataframe(frame, model, threshold=args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(f"Scored {len(scored):,} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
