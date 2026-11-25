"""SentinelBoost: noise-robust, drift-adaptive gradient boosting (experiment).

SentinelBoost wraps LightGBM with four mechanisms aimed at the known
weaknesses of plain gradient boosting on fraud data. Each mechanism can be
toggled independently (constructor flags) so the benchmark harness can run
ablations; with all four off, the estimator reduces to plain LightGBM.

1. Label-noise filtering (``use_noise_filter``)
   Fraud labels are noisy and delayed (late chargebacks, friendly fraud
   filed as legitimate, investigator error). Plain boosting chases its
   highest-loss samples, which are disproportionately the mislabeled ones.
   SentinelBoost first fits a pilot LightGBM and computes 3-fold
   out-of-fold (OOF) predicted probabilities on the training set. Samples
   whose label *strongly* disagrees with their OOF probability
   (``y=1`` with prob < 0.05, or ``y=0`` with prob > 0.95) are
   downweighted (x0.2 by default) in the main fit rather than deleted.

   This is a **confident-learning-style approximation** of the design
   sketched in the SentinelBoost proposal: it uses a single pilot model's
   OOF probabilities as the confidence signal instead of a full
   confident-joint / neighborhood-consensus procedure. It is simpler and
   deterministic, but cruder -- it can downweight genuinely rare-but-real
   fraud that the pilot model cannot yet see, which is why the thresholds
   are extreme (0.05 / 0.95) and the downweight is mild (0.2, not 0).

2. Interaction features (``use_interactions``)
   Axis-aligned trees need many splits to express products/ratios such as
   amount x velocity. The pilot model's top-K numeric features by
   importance (K=8) are crossed pairwise into product features (at most
   28), appended to the inputs for both training and prediction.

3. Recency weighting (``use_recency``)
   When transaction times are passed to :meth:`fit` via ``time_values``,
   each sample is weighted by ``exp(-age_days / half_life_days)``
   (half-life 60 days by default), normalized to mean 1, and combined
   multiplicatively with the noise-filter weights. This is the
   "soft" half of drift adaptation: recent behaviour counts more.
   (Hard drift-triggered tree retirement from the original design is not
   implemented in this experiment; recency weighting plus the benchmark's
   per-time-bucket stability metrics are the measurable proxy.)

4. Isotonic calibration (``calibrate``)
   The last 15% of the training data (by time when ``time_values`` is
   given, else a random split) is held out from the main fit and used to
   fit an isotonic regression on the main model's raw probabilities, so
   ``predict_proba`` returns calibrated probabilities. The calibration
   holdout is never used to train the main model (no leakage); the pilot
   model and its OOF/noise/interaction steps also see only the
   non-holdout portion.

Input contract: features must be pre-encoded numeric (this estimator does
no imputation or categorical encoding itself). The benchmark script
(``experiments/benchmark_sentinel.py``) prepares inputs with the same
preprocessing family as ``mlops/ieee_cis.build_pipeline``: median
imputation for numeric columns, and ordinal integer codes for
categoricals (passed to LightGBM as native categorical features instead
of the champion pipeline's one-hot encoding -- LightGBM handles integer
categoricals natively, and this keeps the feature space compact).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted


class SentinelBoost(BaseEstimator, ClassifierMixin):
    """LightGBM boosting with noise filtering, interactions, recency, calibration.

    Parameters mirror a modest LightGBM configuration sized for the
    IEEE-CIS benchmark; see module docstring for the four mechanisms.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 40,
        use_noise_filter: bool = True,
        use_interactions: bool = True,
        use_recency: bool = True,
        calibrate: bool = True,
        half_life_days: float = 60.0,
        noise_downweight: float = 0.2,
        noise_low: float = 0.05,
        noise_high: float = 0.95,
        top_k_interactions: int = 8,
        calibration_fraction: float = 0.15,
        pilot_estimators: int = 120,
        categorical_features: list[str] | None = None,
        random_state: int = 0,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_samples = min_child_samples
        self.use_noise_filter = use_noise_filter
        self.use_interactions = use_interactions
        self.use_recency = use_recency
        self.calibrate = calibrate
        self.half_life_days = half_life_days
        self.noise_downweight = noise_downweight
        self.noise_low = noise_low
        self.noise_high = noise_high
        self.top_k_interactions = top_k_interactions
        self.calibration_fraction = calibration_fraction
        self.pilot_estimators = pilot_estimators
        self.categorical_features = categorical_features
        self.random_state = random_state

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _lgbm_params(self) -> dict:
        return dict(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_samples=self.min_child_samples,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def _make_lgbm(self, n_estimators: int):
        from lightgbm import LGBMClassifier

        params = self._lgbm_params()
        params["n_estimators"] = n_estimators
        return LGBMClassifier(**params)

    def _categorical_indices(self, columns: list[str]) -> list[int] | str:
        if not self.categorical_features:
            return "auto"
        return [columns.index(c) for c in self.categorical_features if c in columns]

    def _noise_weights(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """Confident-learning-style weights from pilot-model OOF probabilities.

        Returns an array of 1.0 weights, with ``noise_downweight`` for
        samples whose label strongly disagrees with the pilot's
        out-of-fold probability. Exposed (as a private method) so tests
        can unit-test the weight computation directly.
        """
        weights = np.ones(len(y), dtype=float)
        if len(np.unique(y)) < 2 or len(y) < 30:
            return weights
        oof = np.zeros(len(y), dtype=float)
        splitter = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=self.random_state
        )
        cat_idx = self._categorical_indices(list(X.columns))
        for train_idx, valid_idx in splitter.split(X, y):
            pilot = self._make_lgbm(self.pilot_estimators)
            pilot.fit(
                X.iloc[train_idx],
                y[train_idx],
                categorical_feature=cat_idx,
            )
            oof[valid_idx] = pilot.predict_proba(X.iloc[valid_idx])[:, 1]
        suspicious = ((y == 1) & (oof < self.noise_low)) | (
            (y == 0) & (oof > self.noise_high)
        )
        weights[suspicious] = self.noise_downweight
        self.n_downweighted_ = int(suspicious.sum())
        self.oof_probabilities_ = oof
        return weights

    def _recency_weights(
        self, time_values: pd.Series | np.ndarray
    ) -> np.ndarray:
        """Exponential-decay weights by age, normalized to mean 1."""
        times = pd.to_datetime(pd.Series(np.asarray(time_values))).reset_index(
            drop=True
        )
        age_days = (times.max() - times).dt.total_seconds().to_numpy() / 86400.0
        weights = np.exp(-age_days / float(self.half_life_days))
        return weights / weights.mean()

    def _add_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append the stored pairwise product features to ``X``."""
        if not getattr(self, "interaction_pairs_", None):
            return X
        out = X.copy()
        for left, right in self.interaction_pairs_:
            out[f"{left}__x__{right}"] = (
                X[left].astype(float) * X[right].astype(float)
            )
        return out

    def _split_calibration(
        self, X: pd.DataFrame, y: np.ndarray, time_values
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (train_idx, calib_idx). Last fraction by time, else random."""
        n = len(y)
        n_calib = max(int(n * self.calibration_fraction), 1)
        if time_values is not None:
            order = np.argsort(
                pd.to_datetime(pd.Series(np.asarray(time_values)))
                .to_numpy()
                .astype("datetime64[ns]")
            )
            return order[:-n_calib], order[-n_calib:]
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n)
        return perm[n_calib:], perm[:n_calib]

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------
    def fit(self, X, y, time_values=None):
        if isinstance(X, pd.DataFrame):
            frame = X.reset_index(drop=True).copy()
        else:
            frame = pd.DataFrame(np.asarray(X))
        y_arr = np.asarray(y).astype(int)
        check_classification_targets(pd.Series(y_arr))
        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = list(frame.columns)
        self.n_downweighted_ = 0

        # --- calibration holdout split (before anything trains) --------
        if self.calibrate and len(y_arr) > 50:
            train_idx, calib_idx = self._split_calibration(
                frame, y_arr, time_values
            )
        else:
            train_idx, calib_idx = np.arange(len(y_arr)), np.array([], dtype=int)
        self.calibration_size_ = int(len(calib_idx))

        X_train = frame.iloc[train_idx].reset_index(drop=True)
        y_train = y_arr[train_idx]
        times_train = (
            None
            if time_values is None
            else pd.Series(np.asarray(time_values)).iloc[train_idx].to_numpy()
        )

        # --- sample weights: noise filter x recency --------------------
        weights = np.ones(len(y_train), dtype=float)
        if self.use_noise_filter:
            weights = weights * self._noise_weights(X_train, y_train)
        if self.use_recency and times_train is not None:
            weights = weights * self._recency_weights(times_train)
        self.sample_weights_ = weights

        # --- interaction screening from a pilot model ------------------
        self.interaction_pairs_ = []
        if self.use_interactions:
            pilot = self._make_lgbm(self.pilot_estimators)
            pilot.fit(
                X_train,
                y_train,
                categorical_feature=self._categorical_indices(
                    list(X_train.columns)
                ),
            )
            importances = pd.Series(
                pilot.feature_importances_, index=X_train.columns
            )
            cat = set(self.categorical_features or [])
            numeric_top = [
                c
                for c in importances.sort_values(ascending=False).index
                if c not in cat
            ][: self.top_k_interactions]
            self.interaction_pairs_ = [
                (numeric_top[i], numeric_top[j])
                for i in range(len(numeric_top))
                for j in range(i + 1, len(numeric_top))
            ]
        X_train_full = self._add_interactions(X_train)
        self.n_features_out_ = X_train_full.shape[1]

        # --- main model -------------------------------------------------
        self.model_ = self._make_lgbm(self.n_estimators)
        self.model_.fit(
            X_train_full,
            y_train,
            sample_weight=weights,
            categorical_feature=self._categorical_indices(
                list(X_train_full.columns)
            ),
        )

        # --- isotonic calibration on the untouched holdout --------------
        self.calibrator_ = None
        if self.calibrate and len(calib_idx) > 0:
            X_calib = self._add_interactions(
                frame.iloc[calib_idx].reset_index(drop=True)
            )
            raw = self.model_.predict_proba(X_calib)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw, y_arr[calib_idx])
            self.calibrator_ = iso
        return self

    def predict_proba(self, X) -> np.ndarray:
        check_is_fitted(self, "model_")
        if isinstance(X, pd.DataFrame):
            frame = X.reset_index(drop=True).copy()
            frame = frame.reindex(columns=self.feature_names_in_)
        else:
            frame = pd.DataFrame(np.asarray(X), columns=self.feature_names_in_)
        frame = self._add_interactions(frame)
        raw = self.model_.predict_proba(frame)[:, 1]
        if self.calibrator_ is not None:
            positive = self.calibrator_.predict(raw)
        else:
            positive = raw
        positive = np.clip(np.asarray(positive, dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X) -> np.ndarray:
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
