import json
from datetime import datetime, timedelta, timezone

import numpy as np

from mlops.drift_check import psi, psi_from_reference, records_from_file


def test_psi_is_zero_for_identical_distributions():
    values = np.array([1, 2, 3, 4])
    assert psi(values, values, [0, 2, 4, 5]) == 0


def test_psi_increases_when_distribution_shifts():
    baseline = np.array([0.1, 0.2, 0.3, 0.4])
    shifted = np.array([10, 11, 12, 13])
    assert psi(baseline, shifted, [0, 1, 2, 14]) > 0.2


def test_reference_psi_is_near_zero_for_matching_population():
    assert psi_from_reference([0.5, 0.5], np.array([0.2, 0.4, 1.2, 1.4]), [0, 1, 2], 0, 0) < 1e-8


def test_records_from_file_keeps_only_recent_prediction_events(tmp_path):
    now = datetime.now(timezone.utc)
    records = [
        {"event": "fraud_prediction", "timestamp": now.isoformat(), "features": {}},
        {
            "event": "fraud_prediction",
            "timestamp": (now - timedelta(hours=25)).isoformat(),
            "features": {},
        },
        {"event": "fraud_prediction", "timestamp": "not-a-timestamp", "features": {}},
    ]
    log_path = tmp_path / "predictions.jsonl"
    log_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    recent = list(records_from_file(log_path, hours=24))

    assert len(recent) == 1
    assert recent[0]["features"] == {}
