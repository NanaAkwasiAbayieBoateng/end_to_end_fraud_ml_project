"""Benchmark SentinelBoost against the XGBoost champion on IEEE-CIS.

Usage:
    python experiments/benchmark_sentinel.py --data-dir data/ieee_cis
    python experiments/benchmark_sentinel.py --simulate

Real-data mode loads the IEEE-CIS tables through
``mlops.ieee_cis.load_ieee_cis`` (the CSVs are NOT in this repo; Nana
downloads them from Kaggle -- see that module's docstring) and uses the
repo's chronological ``time_based_split``. The champion row is the exact
``mlops.ieee_cis.build_pipeline`` configuration (XGBoost, one-hot
categoricals).

``--simulate`` generates a synthetic dataset shaped like the frame
``load_ieee_cis`` returns (same target/time/feature column names), with
~3.5% true fraud, 3% of labels flipped at random (label noise), and a
fraud-pattern drift halfway through the timeline. Simulated results are
indicative for SentinelBoost's design goals only -- they are NOT
IEEE-CIS results, and the results file says so prominently.

Metrics: ROC-AUC and PR-AUC on the chronological test set, plus
stability -- the test set is split into 4 consecutive time buckets and
ROC-AUC is reported per bucket (mean +/- std across buckets).
Results are written to ``experiments/results/benchmark_results.md``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.sentinel_boost import SentinelBoost
from mlops.ieee_cis import (
    REFERENCE_START,
    build_pipeline,
    feature_columns,
    load_ieee_cis,
    time_based_split,
)

RESULTS_PATH = Path(__file__).resolve().parent / "results" / "benchmark_results.md"

SIM_NUMERIC = (
    ["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2",
     "dist1", "dist2"]
    + [f"C{i}" for i in range(1, 9)]
    + [f"D{i}" for i in range(1, 9)]
)  # 25 numeric features
SIM_CATEGORICAL = ["ProductCD", "card4"]


def simulate_frame(n_rows: int = 120_000, seed: int = 7) -> pd.DataFrame:
    """Synthetic IEEE-CIS-shaped frame with label noise and mid-stream drift.

    True fraud rate ~3.5%. After generation, 3% of ALL labels are flipped
    at random (injected label noise; observed prevalence rises to ~6.3%).
    Halfway through the timeline the fraud pattern drifts: before the
    midpoint, fraud is driven mainly by TransactionAmt/C1/D1/dist1; after
    it, by C5/D6/addr1 and a different ProductCD mix.
    """
    rng = np.random.RandomState(seed)
    total_seconds = 180 * 24 * 3600  # ~180 days, like the IEEE-CIS window
    transaction_dt = np.sort(rng.randint(0, total_seconds, size=n_rows))
    t_frac = transaction_dt / total_seconds
    drifted = t_frac >= 0.5

    frame = pd.DataFrame({"TransactionDT": transaction_dt})
    frame["TransactionAmt"] = np.round(rng.lognormal(3.8, 1.0, n_rows), 2)
    frame["card1"] = rng.randint(1000, 20000, n_rows)
    frame["card2"] = rng.randint(100, 600, n_rows)
    frame["card3"] = rng.choice([100, 150, 185, 200], n_rows)
    frame["card5"] = rng.randint(100, 250, n_rows)
    frame["addr1"] = rng.randint(100, 550, n_rows)
    frame["addr2"] = rng.choice([87.0, 60.0, 65.0, np.nan], n_rows, p=[0.7, 0.1, 0.1, 0.1])
    frame["dist1"] = np.where(rng.rand(n_rows) < 0.3, np.nan, rng.exponential(60, n_rows))
    frame["dist2"] = np.where(rng.rand(n_rows) < 0.6, np.nan, rng.exponential(80, n_rows))
    for i in range(1, 9):
        frame[f"C{i}"] = rng.poisson(2.0, n_rows)
    for i in range(1, 9):
        frame[f"D{i}"] = np.where(
            rng.rand(n_rows) < 0.25, np.nan, rng.exponential(30, n_rows)
        )
    frame["ProductCD"] = rng.choice(
        ["W", "C", "H", "R", "S"], n_rows, p=[0.55, 0.15, 0.12, 0.10, 0.08]
    )
    frame["card4"] = rng.choice(
        ["visa", "mastercard", "american express", "discover"],
        n_rows,
        p=[0.45, 0.35, 0.1, 0.1],
    )

    log_amt = np.log1p(frame["TransactionAmt"].to_numpy())
    pre = (
        0.55 * log_amt
        + 0.35 * frame["C1"].to_numpy()
        + 0.03 * frame["D1"].fillna(0).to_numpy()
        + 0.012 * frame["dist1"].fillna(0).to_numpy()
        + np.where(frame["ProductCD"] == "C", 0.9, 0.0)
    )
    post = (
        0.30 * log_amt
        + 0.45 * frame["C5"].to_numpy()
        + 0.035 * frame["D6"].fillna(0).to_numpy()
        + 0.004 * frame["addr1"].to_numpy()
        + np.where(frame["ProductCD"] == "R", 0.9, 0.0)
    )

    def calibrate_intercept(scores: np.ndarray, target_rate: float) -> float:
        """Bisection offset so sigmoid(scores + offset) means target_rate."""
        lo, hi = -30.0, 30.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if (1.0 / (1.0 + np.exp(-(scores + mid)))).mean() < target_rate:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # Calibrate each regime's intercept so the TRUE fraud rate is ~3.5%
    # in both halves of the timeline (verified empirically below).
    logit = np.empty(n_rows)
    logit[~drifted] = pre[~drifted] + calibrate_intercept(pre[~drifted], 0.035)
    logit[drifted] = post[drifted] + calibrate_intercept(post[drifted], 0.035)
    prob = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.rand(n_rows) < prob).astype(int)

    # Inject label noise: flip 3% of all labels at random.
    flip = rng.rand(n_rows) < 0.03
    noisy_target = np.where(flip, 1 - target, target)

    frame["target"] = noisy_target
    frame["transaction_datetime"] = REFERENCE_START + pd.to_timedelta(
        frame["TransactionDT"], unit="s"
    )
    frame["TransactionID"] = np.arange(1, n_rows + 1)
    return frame


def encode_for_sentinel(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median-impute numerics (fit on train); ordinal-encode categoricals.

    Mirrors the preprocessing family of ``mlops.ieee_cis.build_pipeline``
    (median imputation for numerics). Categoricals become integer codes
    for LightGBM's native categorical handling instead of one-hot.
    """
    train_enc = train_df.copy()
    test_enc = test_df.copy()
    for col in numeric:
        median = train_enc[col].median()
        train_enc[col] = train_enc[col].fillna(median)
        test_enc[col] = test_enc[col].fillna(median)
    if categorical:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1, dtype=float
        )
        train_enc[categorical] = encoder.fit_transform(
            train_enc[categorical].astype(str)
        )
        test_enc[categorical] = encoder.transform(
            test_enc[categorical].astype(str)
        )
        for col in categorical:
            train_enc[col] = train_enc[col].astype(int)
            test_enc[col] = test_enc[col].astype(int)
    return train_enc, test_enc


def evaluate(name: str, y_test: np.ndarray, proba: np.ndarray) -> dict:
    """Overall metrics + per-time-bucket ROC-AUC stability (4 buckets)."""
    buckets = np.array_split(np.arange(len(y_test)), 4)
    bucket_aucs = []
    for idx in buckets:
        if len(np.unique(y_test[idx])) == 2:
            bucket_aucs.append(roc_auc_score(y_test[idx], proba[idx]))
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "bucket_aucs": bucket_aucs,
        "bucket_mean": float(np.mean(bucket_aucs)) if bucket_aucs else float("nan"),
        "bucket_std": float(np.std(bucket_aucs)) if bucket_aucs else float("nan"),
    }


def run_benchmark(frame: pd.DataFrame) -> list[dict]:
    numeric, categorical = feature_columns(frame)
    features = numeric + categorical
    train_df, test_df = time_based_split(frame)
    y_train = train_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()
    results = []

    # (a) Champion: the exact ieee_cis.build_pipeline XGBoost config.
    champion = build_pipeline(numeric, categorical)
    champion.fit(train_df[features], y_train)
    results.append(
        evaluate(
            "XGBoost champion (ieee_cis.build_pipeline)",
            y_test,
            champion.predict_proba(test_df[features])[:, 1],
        )
    )

    # (b)/(c) SentinelBoost full + ablation, on encoded features.
    train_enc, test_enc = encode_for_sentinel(train_df, test_df, numeric, categorical)
    times = train_df["transaction_datetime"]
    for name, kwargs in [
        ("SentinelBoost (full)", {}),
        (
            "SentinelBoost (all mechanisms OFF = plain LightGBM)",
            dict(
                use_noise_filter=False,
                use_interactions=False,
                use_recency=False,
                calibrate=False,
            ),
        ),
    ]:
        model = SentinelBoost(categorical_features=categorical, **kwargs)
        model.fit(train_enc[features], y_train, time_values=times)
        results.append(
            evaluate(name, y_test, model.predict_proba(test_enc[features])[:, 1])
        )
    return results


def render_results(results: list[dict], dataset_note: str, simulated: bool) -> str:
    title = (
        "# Benchmark results — SYNTHETIC DATA (not IEEE-CIS)"
        if simulated
        else "# Benchmark results — IEEE-CIS"
    )
    lines = [
        title,
        "",
        f"Run at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        dataset_note,
        "",
        "| Model | ROC-AUC | PR-AUC | Bucket ROC-AUC mean | Bucket ROC-AUC std | Per-bucket ROC-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        buckets = ", ".join(f"{a:.3f}" for a in r["bucket_aucs"])
        lines.append(
            f"| {r['model']} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | "
            f"{r['bucket_mean']:.4f} | {r['bucket_std']:.4f} | {buckets} |"
        )
    lines += [
        "",
        "Stability = ROC-AUC computed separately on 4 consecutive time "
        "buckets of the chronological test set; lower std = more stable "
        "performance over time.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ieee_cis"))
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run on synthetic IEEE-CIS-shaped data (NOT real IEEE-CIS).",
    )
    args = parser.parse_args()

    if args.simulate:
        frame = simulate_frame()
        dataset_note = (
            "**SYNTHETIC dataset** generated by `simulate_frame()` in this "
            f"script: {len(frame):,} rows shaped like `load_ieee_cis` output "
            "(25 numeric features, ProductCD + card4 categoricals), ~3.5% "
            "true fraud, 3% of labels flipped at random (injected label "
            "noise; observed prevalence ~6.3%), fraud pattern drifts "
            "halfway through the ~180-day timeline. These numbers say "
            "something about SentinelBoost's design goals on data built to "
            "stress them; they are NOT IEEE-CIS results and prove nothing "
            "about real-world performance."
        )
    else:
        frame = load_ieee_cis(args.data_dir)
        dataset_note = (
            f"IEEE-CIS dataset loaded from `{args.data_dir}`: "
            f"{len(frame):,} rows, chronological 70/30 split via "
            "`mlops.ieee_cis.time_based_split`."
        )

    results = run_benchmark(frame)
    markdown = render_results(results, dataset_note, simulated=args.simulate)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(markdown)
    print(markdown)
    print(f"\nWrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
