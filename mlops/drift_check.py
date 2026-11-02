import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

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


def records_from_file(log_path: Path, hours: float) -> Iterable[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    paths = [log_path, *sorted(log_path.parent.glob(f"{log_path.name}.*"))]
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                raw_timestamp = item.get("timestamp")
                if item.get("event") != "fraud_prediction" or not isinstance(
                    raw_timestamp, str
                ):
                    continue
                try:
                    timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                if timestamp >= cutoff:
                    yield item


def post_json(url: str, payload: dict, token: str | None = None) -> None:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["Accept"] = "application/vnd.github+json"
        headers["X-GitHub-Api-Version"] = "2022-11-28"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urlopen(request, timeout=15) as response:
        if response.status not in (200, 201, 202, 204):
            raise RuntimeError(f"Notification request failed: HTTP {response.status}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute PSI on recent prediction features.")
    parser.add_argument("--reference", type=Path, default=Path("models/reference_stats.json"))
    parser.add_argument(
        "--prediction-log",
        type=Path,
        default=Path(os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl")),
    )
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--github-repository", default=os.getenv("GITHUB_REPOSITORY"))
    parser.add_argument("--github-token", default=os.getenv("DRIFT_DISPATCH_TOKEN"))
    parser.add_argument("--alert-webhook-url", default=os.getenv("DRIFT_ALERT_WEBHOOK_URL"))
    parser.add_argument("--min-predictions", type=int, default=100)
    args = parser.parse_args()
    if not args.reference.is_file():
        parser.error(f"Reference summary not found: {args.reference}")
    if bool(args.github_repository) != bool(args.github_token):
        parser.error("--github-repository and --github-token must be configured together.")

    if args.hours <= 0 or args.threshold <= 0 or args.min_predictions < 1:
        parser.error("--hours, --threshold, and --min-predictions must be positive.")
    if not args.prediction_log.is_file():
        print(f"Drift check skipped: prediction log not found: {args.prediction_log}")
        return 0
    samples = list(records_from_file(args.prediction_log, args.hours))
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
    if args.alert_webhook_url:
        post_json(
            args.alert_webhook_url,
            {"text": message, "message": message, "report": report},
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
        print("Retraining candidate dispatched; the AUC promotion gate still applies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
