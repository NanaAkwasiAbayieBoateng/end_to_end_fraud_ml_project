"""Tests for the SentinelBoost experiment estimator."""

import numpy as np
import pandas as pd
import pytest

from experiments.sentinel_boost import SentinelBoost


def make_frame(n: int = 800, seed: int = 3):
    """Small, strongly separable dataset with a time column."""
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        {
            "amt": rng.lognormal(3.5, 1.0, n),
            "cnt": rng.poisson(2.0, n),
            "days": rng.exponential(30, n),
            "score": rng.normal(0, 1, n),
        }
    )
    logit = -4.0 + 0.8 * np.log1p(X["amt"]) + 0.9 * X["cnt"] + 0.05 * X["days"]
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.rand(n) < prob).astype(int)
    times = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        np.sort(rng.randint(0, 120 * 24 * 3600, n)), unit="s"
    )
    return X, y, times


def test_fit_predict_proba_valid():
    X, y, times = make_frame()
    model = SentinelBoost(n_estimators=60, pilot_estimators=40)
    model.fit(X, y, time_values=times)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_noise_filter_downweights_flipped_labels():
    # Two well-separated clusters; labels follow the cluster. Flip a
    # random scatter of points in each cluster -- unambiguous label
    # noise that a pilot model's OOF probabilities should dispute.
    rng = np.random.RandomState(11)
    n_per = 600
    X0 = rng.normal(0.0, 0.5, size=(n_per, 3))
    X1 = rng.normal(4.0, 0.5, size=(n_per, 3))
    X = pd.DataFrame(np.vstack([X0, X1]), columns=["f1", "f2", "f3"])
    y = np.array([0] * n_per + [1] * n_per)
    y_noisy = y.copy()
    # Flip randomly scattered points in each cluster (clustered flips
    # could be carved out as their own region by the trees; scattered
    # label noise is what the filter is designed to catch).
    flip0 = rng.choice(n_per, size=40, replace=False)
    flip1 = rng.choice(n_per, size=40, replace=False) + n_per
    flip_idx = np.concatenate([flip0, flip1])
    y_noisy[flip_idx] = 1 - y_noisy[flip_idx]

    model = SentinelBoost()
    weights = model._noise_weights(X, y_noisy)
    assert model.n_downweighted_ > 0
    # The injected flips should be downweighted far more often than not.
    flipped_weights = weights[flip_idx]
    assert (flipped_weights < 1.0).mean() > 0.5
    assert set(np.unique(weights)) <= {1.0, model.noise_downweight}


def test_interaction_features_added():
    X, y, times = make_frame()
    model = SentinelBoost(
        n_estimators=40, pilot_estimators=40, top_k_interactions=3,
        use_noise_filter=False, use_recency=False, calibrate=False,
    )
    model.fit(X, y, time_values=times)
    # 3 top features -> 3 pairwise products appended.
    assert len(model.interaction_pairs_) == 3
    transformed = model._add_interactions(X)
    assert transformed.shape[1] == X.shape[1] + 3
    assert model.n_features_out_ == X.shape[1] + 3


def test_calibration_path_runs():
    X, y, times = make_frame()
    model = SentinelBoost(n_estimators=60, pilot_estimators=40, calibrate=True)
    model.fit(X, y, time_values=times)
    assert model.calibration_size_ > 0
    assert model.calibrator_ is not None
    proba = model.predict_proba(X)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_recency_weights_decay_with_age():
    model = SentinelBoost(half_life_days=30.0)
    times = pd.to_datetime(
        ["2024-01-01", "2024-01-31", "2024-03-01", "2024-03-31"]
    )
    weights = model._recency_weights(times)
    # Oldest sample weighs least; weights normalized to mean 1.
    assert weights[0] < weights[1] < weights[2] < weights[3]
    assert weights.mean() == pytest.approx(1.0)
    # exp(-age/half_life): 30 days at half_life=30 -> factor e^-1.
    assert weights[2] / weights[3] == pytest.approx(np.exp(-1.0), rel=0.05)
