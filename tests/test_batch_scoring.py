import numpy as np
import pandas as pd
import pytest

from app.main import FEATURE_NAMES
from mlops.score_batch import score_dataframe


class FakeModel:
    def predict_proba(self, frame):
        assert tuple(frame.columns) == FEATURE_NAMES
        return np.array([[0.8, 0.2], [0.1, 0.9]])


def test_score_dataframe_preserves_input_and_adds_prediction_columns():
    frame = pd.DataFrame(
        [
            [2, 24.5, 18, 8, "Retail_A", "Clothing", "2023-01-01", "2023-01-02", "row-a"],
            [1, 10.0, 7, 4, "Service_C", "Other", "2023-02-01", "2023-02-02", "row-b"],
        ],
        columns=[*FEATURE_NAMES, "transaction_id"],
    )

    scored = score_dataframe(frame, FakeModel(), threshold=0.8)

    assert scored["transaction_id"].tolist() == ["row-a", "row-b"]
    assert scored["fraud_probability"].tolist() == [0.2, 0.9]
    assert scored["predicted_fraud"].tolist() == [False, True]
    assert "fraud_probability" not in frame


def test_score_dataframe_rejects_missing_features_and_invalid_threshold():
    with pytest.raises(ValueError, match="missing required features"):
        score_dataframe(pd.DataFrame(columns=FEATURE_NAMES[:-1]), FakeModel())
    with pytest.raises(ValueError, match="threshold"):
        score_dataframe(pd.DataFrame(columns=FEATURE_NAMES), FakeModel(), threshold=1.1)
