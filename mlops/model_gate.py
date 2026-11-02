import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train import train_fraud_model


def reference_summary(data: pd.DataFrame) -> dict:
    features = data.drop(columns=["target"], errors="ignore")
    summary = {}
    for column in features:
        series = features[column]
        is_datetime = column.endswith(("_time", "_datetime"))
        if is_datetime:
            parsed = pd.to_datetime(series, errors="coerce", utc=True)
            numeric_series = parsed.astype("int64").where(parsed.notna(), np.nan) / 1e9
        else:
            numeric_series = pd.to_numeric(series, errors="coerce")
        if is_datetime or pd.api.types.is_numeric_dtype(series):
            values = numeric_series.dropna().to_numpy(dtype=float)
            if values.size:
                bins = np.unique(np.quantile(values, np.linspace(0, 1, 11))).tolist()
                if len(bins) == 1:
                    bins = [bins[0] - 0.5, bins[0] + 0.5]
                edges = np.asarray(bins, dtype=float)
                edges[0], edges[-1] = -np.inf, np.inf
                counts, _ = np.histogram(values, bins=edges)
                proportions = (counts / counts.sum()).tolist()
                summary[column] = {
                    "type": "numeric",
                    "bins": bins,
                    "proportions": proportions,
                    "missing_proportion": float(numeric_series.isna().mean()),
                }
        else:
            counts = series.fillna("__MISSING__").astype(str).value_counts(normalize=True)
            summary[column] = {"type": "categorical", "distribution": counts.to_dict()}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a model and enforce the baseline AUC gate.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "fraud_dataset.csv")
    parser.add_argument("--baseline", type=Path, default=ROOT / "models" / "baseline_auc.json")
    parser.add_argument("--model-output", type=Path, default=ROOT / "models" / "fraud_model_pipeline.joblib")
    parser.add_argument("--metrics-output", type=Path, default=ROOT / "models" / "model_metrics.json")
    parser.add_argument("--reference-output", type=Path, default=ROOT / "models" / "reference_stats.json")
    args = parser.parse_args()

    if not args.data.is_file() or not args.baseline.is_file():
        parser.error("Training data and baseline metric files must exist.")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))["roc_auc"]
    dataset = pd.read_csv(args.data)
    if "target" not in dataset or dataset["target"].isna().any():
        raise ValueError("Training data must contain a complete 'target' column.")

    results = train_fraud_model(
        dataset,
        save_artifacts=False,
        log_mlflow_run=False,
    )
    candidate_auc = float(results["metrics"]["roc_auc"])
    if not np.isfinite(candidate_auc):
        raise ValueError("Candidate ROC-AUC is not finite.")

    metrics = {
        "baseline_roc_auc": float(baseline),
        "candidate_roc_auc": candidate_auc,
        "passed": candidate_auc > float(baseline),
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if not metrics["passed"]:
        print(
            f"Model promotion blocked: candidate ROC-AUC {candidate_auc:.6f} "
            f"must exceed baseline {baseline:.6f}.",
            file=sys.stderr,
        )
        return 1

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results["pipeline"], args.model_output)
    args.reference_output.write_text(
        json.dumps(reference_summary(dataset), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        f"Model validation passed: candidate ROC-AUC {candidate_auc:.6f} "
        f"> baseline {baseline:.6f}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
