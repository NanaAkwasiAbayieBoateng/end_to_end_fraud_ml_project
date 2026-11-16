import numpy as np
import pandas as pd
import pytest

from mlops.ieee_cis import feature_columns, load_ieee_cis, time_based_split


def write_fixture(tmp_path):
    transactions = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "isFraud": [0, 1, 0, 1],
            "TransactionDT": [400, 100, 300, 200],
            "TransactionAmt": [10.0, 250.0, 35.0, 99.0],
            "ProductCD": ["W", "C", "W", "H"],
            "card1": [1000, 2000, 3000, 4000],
            "C1": [1.0, 2.0, 1.0, 0.0],
            "V1": [0.5, np.nan, 1.5, 2.5],
        }
    )
    identity = pd.DataFrame(
        {
            "TransactionID": [2, 4],
            "DeviceType": ["mobile", "desktop"],
            "id_12": ["Found", "NotFound"],
        }
    )
    transactions.to_csv(tmp_path / "train_transaction.csv", index=False)
    identity.to_csv(tmp_path / "train_identity.csv", index=False)
    return tmp_path


def test_load_merges_identity_and_maps_target(tmp_path):
    frame = load_ieee_cis(write_fixture(tmp_path))

    assert len(frame) == 4
    assert frame["target"].tolist() == [1, 1, 0, 0]  # sorted by TransactionDT
    assert frame.loc[frame["TransactionID"] == 2, "DeviceType"].iloc[0] == "mobile"
    assert frame["transaction_datetime"].is_monotonic_increasing


def test_load_requires_transaction_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_ieee_cis(tmp_path)


def test_feature_columns_picks_present_features(tmp_path):
    frame = load_ieee_cis(write_fixture(tmp_path))
    numeric, categorical = feature_columns(frame)

    assert "TransactionAmt" in numeric and "V1" in numeric
    assert "ProductCD" in categorical and "DeviceType" in categorical
    assert "target" not in numeric + categorical


def test_time_based_split_is_chronological(tmp_path):
    frame = load_ieee_cis(write_fixture(tmp_path))
    train, test = time_based_split(frame, train_ratio=0.5)

    assert len(train) == 2 and len(test) == 2
    assert train["TransactionDT"].max() < test["TransactionDT"].min()
