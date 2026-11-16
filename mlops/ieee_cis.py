"""IEEE-CIS Fraud Detection: loading, feature preparation, and training.

The dataset is distributed through Kaggle only (competition rules), so the
CSVs are not in this repo. Download them once into ``data/ieee_cis/``:

    kaggle competitions download -c ieee-fraud-detection -p data/ieee_cis
    unzip data/ieee_cis/ieee-fraud-detection.zip -d data/ieee_cis

Expected files: ``train_transaction.csv`` and ``train_identity.csv``.
Then train with:

    python mlops/ieee_cis.py --data-dir data/ieee_cis

Training uses a chronological split on TransactionDT (random splits leak in
fraud), logs the run to MLflow, and reports ROC-AUC / PR-AUC. The retail
demo pipeline in ``models/train.py`` is unaffected.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# TransactionDT is seconds since a reference point at the start of the
# dataset window (Dec 2017); anchoring it gives a real datetime column.
REFERENCE_START = pd.Timestamp("2017-12-01")

CATEGORICAL_FEATURES = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "DeviceType",
    "id_12",
    "id_13",
    "id_14",
    "id_15",
    "id_16",
    "id_17",
    "id_18",
    "id_19",
    "id_20",
    "id_21",
    "id_22",
    "id_23",
    "id_24",
    "id_25",
    "id_26",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_32",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
]

BASE_NUMERIC_FEATURES = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
] + [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]


def load_ieee_cis(data_dir: Path) -> pd.DataFrame:
    """Load and merge the IEEE-CIS transaction and identity tables."""
    transaction_path = data_dir / "train_transaction.csv"
    identity_path = data_dir / "train_identity.csv"
    if not transaction_path.is_file():
        raise FileNotFoundError(
            f"{transaction_path} not found. Download the IEEE-CIS dataset from "
            "Kaggle into data/ieee_cis/ (see this module's docstring)."
        )
    transactions = pd.read_csv(transaction_path)
    if identity_path.is_file():
        identity = pd.read_csv(identity_path)
        frame = transactions.merge(identity, on="TransactionID", how="left")
    else:
        frame = transactions
    frame = frame.rename(columns={"isFraud": "target"})
    frame["transaction_datetime"] = REFERENCE_START + pd.to_timedelta(
        frame["TransactionDT"], unit="s"
    )
    return frame.sort_values("TransactionDT").reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric, categorical) feature lists present in the frame."""
    vesta = [c for c in frame.columns if c.startswith("V") and c[1:].isdigit()]
    numeric = [c for c in BASE_NUMERIC_FEATURES + vesta if c in frame.columns]
    categorical = [c for c in CATEGORICAL_FEATURES if c in frame.columns]
    return numeric, categorical


def time_based_split(
    frame: pd.DataFrame, train_ratio: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the earliest rows train, the latest rows test."""
    ordered = frame.sort_values("TransactionDT").reset_index(drop=True)
    split_index = int(len(ordered) * train_ratio)
    return ordered.iloc[:split_index], ordered.iloc[split_index:]


def build_pipeline(numeric: list[str], categorical: list[str]):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import XGBClassifier

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ]
    )
    classifier = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.6,
        eval_metric="auc",
        n_jobs=-1,
        seed=0,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_ieee_cis(frame: pd.DataFrame, log_mlflow_run: bool = True) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    numeric, categorical = feature_columns(frame)
    train_df, test_df = time_based_split(frame)
    features = numeric + categorical
    pipeline = build_pipeline(numeric, categorical)
    pipeline.fit(train_df[features], train_df["target"])
    probabilities = pipeline.predict_proba(test_df[features])[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(test_df["target"], probabilities)),
        "pr_auc": float(average_precision_score(test_df["target"], probabilities)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
    }
    print(
        f"IEEE-CIS split: Train={metrics['train_rows']:,} rows, "
        f"Test={metrics['test_rows']:,} rows"
    )
    print(
        f"IEEE-CIS test metrics -> ROC-AUC: {metrics['roc_auc']:.4f} | "
        f"PR-AUC: {metrics['pr_auc']:.4f}"
    )

    if log_mlflow_run:
        import mlflow

        tracking_uri = f"sqlite:///{(Path(__file__).resolve().parents[1] / 'mlflow.db').as_posix()}"
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("fraud-detection-ieee-cis")
        with mlflow.start_run(run_name="ieee-cis-training"):
            mlflow.log_params(
                {
                    "dataset": "ieee-cis",
                    "split_strategy": "chronological",
                    "numeric_features": len(numeric),
                    "categorical_features": len(categorical),
                }
            )
            mlflow.log_metrics(
                {"test_roc_auc": metrics["roc_auc"], "test_pr_auc": metrics["pr_auc"]}
            )
    return {"pipeline": pipeline, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a fraud model on IEEE-CIS data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ieee_cis"))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    frame = load_ieee_cis(args.data_dir)
    train_ieee_cis(frame, log_mlflow_run=not args.no_mlflow)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
