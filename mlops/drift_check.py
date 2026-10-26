import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, edges: list[float]) -> float:
    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_pct = np.maximum(expected_counts / max(expected_counts.sum(), 1), 1e-6)
    actual_pct = np.maximum(actual_counts / max(actual_counts.sum(), 1), 1e-6)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def psi_from_reference(
    expected_proportions: list[float],
    actual: np.ndarray,
    edges: list[float],
    expected_missing: float,
    actual_missing: float,
) -> float:
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_pct = np.append(
        np.asarray(expected_proportions, dtype=float) * (1 - expected_missing),
        expected_missing,
    )
    valid_ratio = 1 - actual_missing
    actual_pct = np.append(
        actual_counts / max(actual_counts.sum(), 1) * valid_ratio,
        actual_missing,
    )
    expected_pct = np.maximum(expected_pct, 1e-6)
    actual_pct = np.maximum(actual_pct, 1e-6)
    expected_pct /= expected_pct.sum()
    actual_pct /= actual_pct.sum()
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def records_from_cloudwatch(log_group: str, hours: float) -> Iterable[dict]:
    import boto3

    client = boto3.client("logs")
    start_ms = int((time.time() - hours * 3600) * 1000)
    token = None
    while True:
        params = {
            "logGroupName": log_group,
            "startTime": start_ms,
            "filterPattern": '{ $.event = "fraud_prediction" }',
        }
        if token:
            params["nextToken"] = token
        result = client.filter_log_events(**params)
        for event in result.get("events", []):
            try:
                yield json.loads(event["message"])
            except json.JSONDecodeError:
                continue
        next_token = result.get("nextToken")
        if not next_token or next_token == token:
            return
        token = next_token


def records_from_file(log_path: Path) -> Iterable[dict]:
    with log_path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("event") == "fraud_prediction":
                yield item


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute PSI on recent prediction features.")
    parser.add_argument("--reference", type=Path, default=Path("models/reference_stats.json"))
    parser.add_argument("--prediction-log", type=Path)
    parser.add_argument("--cloudwatch-log-group")
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--sns-topic-arn", default=os.getenv("DRIFT_SNS_TOPIC_ARN"))
    parser.add_argument("--github-repository", default=os.getenv("GITHUB_REPOSITORY"))
    parser.add_argument("--github-token", default=os.getenv("DRIFT_DISPATCH_TOKEN"))
    parser.add_argument("--min-predictions", type=int, default=100)
    args = parser.parse_args()
    if not args.reference.is_file():
        parser.error(f"Reference summary not found: {args.reference}")
    if bool(args.prediction_log) == bool(args.cloudwatch_log_group):
        parser.error("Specify exactly one of --prediction-log or --cloudwatch-log-group.")
    if bool(args.github_repository) != bool(args.github_token):
        parser.error("--github-repository and --github-token must be configured together.")

    if args.prediction_log:
        records = records_from_file(args.prediction_log)
    else:
        records = records_from_cloudwatch(args.cloudwatch_log_group, args.hours)
    samples = list(records)
    if len(samples) < args.min_predictions:
        print(f"Drift check skipped: only {len(samples)} recent predictions.")
        return 0

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    current = pd.DataFrame([row.get("features", {}) for row in samples])
    drifted = {}
    for feature, stats in reference.items():
        if feature not in current:
            continue
        values = current[feature]
        if stats["type"] == "numeric":
            if feature.endswith(("_time", "_datetime")):
                parsed = pd.to_datetime(values, errors="coerce", utc=True)
                numeric_series = parsed.astype("int64").where(parsed.notna(), np.nan) / 1e9
            else:
                numeric_series = pd.to_numeric(values, errors="coerce")
            numeric = numeric_series.dropna().to_numpy(dtype=float)
            edges = np.asarray(stats["bins"], dtype=float)
            edges[0] = -np.inf
            edges[-1] = np.inf
            score = psi_from_reference(
                stats["proportions"],
                numeric,
                edges.tolist(),
                float(stats["missing_proportion"]),
                float(numeric_series.isna().mean()),
            )
        else:
            baseline = stats["distribution"]
            observed = values.fillna("__MISSING__").astype(str).value_counts(normalize=True).to_dict()
            categories = set(baseline) | set(observed)
            expected_pct = np.asarray([baseline.get(key, 0.0) for key in categories]) + 1e-6
            observed_pct = np.asarray([observed.get(key, 0.0) for key in categories]) + 1e-6
            expected_pct /= expected_pct.sum()
            observed_pct /= observed_pct.sum()
            score = float(np.sum((observed_pct - expected_pct) * np.log(observed_pct / expected_pct)))
        if score >= args.threshold:
            drifted[feature] = round(score, 6)

    report = {
        "event": "fraud_drift_check",
        "prediction_count": len(samples),
        "threshold": args.threshold,
        "drifted_features": drifted,
    }
    print(json.dumps(report, sort_keys=True))
    if not drifted:
        return 0

    message = f"Fraud model drift detected (PSI >= {args.threshold}): {drifted}"
    if args.sns_topic_arn:
        import boto3

        boto3.client("sns").publish(
            TopicArn=args.sns_topic_arn,
            Subject="Fraud model input drift detected",
            Message=message,
        )
    if args.github_repository and args.github_token:
        import urllib.request

        request = urllib.request.Request(
            f"https://api.github.com/repos/{args.github_repository}/dispatches",
            data=json.dumps({"event_type": "model-drift-detected", "client_payload": report}).encode(),
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {args.github_token}",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status not in (200, 204):
                raise RuntimeError(f"GitHub retraining dispatch failed: HTTP {response.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
