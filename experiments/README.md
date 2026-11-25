# Experiments: SentinelBoost

SentinelBoost is an experimental gradient-boosting variant built on
LightGBM, designed against four known weaknesses of plain
XGBoost/LightGBM/Random Forest on fraud data. It lives in
`experiments/sentinel_boost.py` as a sklearn-compatible estimator
(`fit` / `predict_proba`) and is benchmarked against this repo's
current XGBoost champion configuration (the exact pipeline built by
`mlops.ieee_cis.build_pipeline`) by
`experiments/benchmark_sentinel.py`.

## The four mechanisms

Each is toggleable via a constructor flag (`use_noise_filter`,
`use_interactions`, `use_recency`, `calibrate`) for ablations; with
all four off, SentinelBoost is plain LightGBM.

1. **Label-noise filtering** — Fraud labels are noisy and delayed
   (late chargebacks, friendly fraud). A pilot LightGBM produces
   3-fold out-of-fold probabilities on the training set; samples whose
   label strongly disagrees (fraud with OOF prob < 0.05, legitimate
   with OOF prob > 0.95) are downweighted x0.2 in the main fit instead
   of being chased as "hard examples". This is a
   confident-learning-style approximation — see the module docstring
   for what it simplifies.
2. **Interaction features** — The pilot model's top-8 numeric features
   by importance are crossed pairwise (28 product features max), so
   shallow trees can express amount x velocity-style structure that
   axis-aligned splits need many levels for.
3. **Recency weighting** — With transaction times passed to `fit`,
   samples are weighted `exp(-age_days / 60)` (60-day half-life
   parameter), normalized to mean 1 and combined multiplicatively with
   the noise weights, so recent behaviour counts more as patterns
   drift.
4. **Isotonic calibration** — The last 15% of training data (by time)
   is held out from the main fit and used to fit isotonic regression
   on the model's outputs, so predicted probabilities are calibrated.
   The holdout never trains the main model.

Input contract: SentinelBoost takes pre-encoded numeric features. The
benchmark script prepares them with the same preprocessing family as
the champion pipeline (median imputation for numerics) except that
categoricals are ordinal-encoded and passed to LightGBM as native
categorical features instead of one-hot.

## Running the benchmark

### Real IEEE-CIS data

The IEEE-CIS CSVs are **not** in this repo (Kaggle competition rules).
Download them once, exactly as described in `mlops/ieee_cis.py`:

```bash
kaggle competitions download -c ieee-fraud-detection -p data/ieee_cis
unzip data/ieee_cis/ieee-fraud-detection.zip -d data/ieee_cis
python experiments/benchmark_sentinel.py --data-dir data/ieee_cis
```

### Simulated data (no download needed)

```bash
python experiments/benchmark_sentinel.py --simulate
```

This generates 120,000 synthetic rows shaped like the IEEE-CIS frame
(25 numeric features, `ProductCD` + `card4` categoricals, ~3.5% true
fraud), injects label noise (3% of labels flipped at random) and a
fraud-pattern drift halfway through the timeline — the two conditions
SentinelBoost is designed for.

## Results

Latest run: [results/benchmark_results.md](results/benchmark_results.md)
— ROC-AUC, PR-AUC, and stability (ROC-AUC mean/std across 4
consecutive time buckets of the chronological test set) for the
XGBoost champion, full SentinelBoost, and the all-mechanisms-off
ablation (plain LightGBM).

**Honest note on the simulated run:** the synthetic data was built to
stress exactly the failure modes SentinelBoost targets (label noise +
drift), so a SentinelBoost win there shows the mechanisms work as
designed on data that contains those pathologies — nothing more. It
does **not** prove SentinelBoost beats XGBoost on real fraud data;
only the IEEE-CIS run (above) can speak to that, and even that is one
dataset with one split. Treat the simulated numbers as a smoke test
with a pulse, not evidence.

### Findings from the first simulated run (Nov 24, 2026)

Full-model result: ROC-AUC 0.6052 / PR-AUC 0.0866 — slightly *behind*
plain LightGBM (0.6183 / 0.1009) and the XGBoost champion (0.6083 /
0.0965). Per-mechanism ablations on the same data explain why:

| Configuration | ROC-AUC | PR-AUC | Bucket std |
|---|---|---|---|
| Plain LightGBM (all off) | 0.6183 | 0.1009 | 0.0194 |
| Noise filter only | 0.6219 | 0.1006 | 0.0115 |
| Interactions only | 0.6143 | 0.1012 | 0.0155 |
| Recency only | 0.6359 | 0.1377 | 0.0188 |
| Calibration only | 0.5844 | 0.0782 | 0.0181 |
| Full minus calibration | 0.6352 | 0.1308 | 0.0201 |
| **Full (all on)** | **0.6052** | **0.0866** | **0.0150** |

- **Recency weighting is the clear win** (+0.018 ROC-AUC, +0.037
  PR-AUC over plain LightGBM): under drift, emphasizing recent data
  works exactly as designed.
- **The noise filter helps modestly** (small AUC gain, best stability
  std 0.0115; it downweighted ~2.5% of training samples).
- **Interactions are roughly neutral** on this data.
- **Calibration is the problem — specifically its holdout, not the
  isotonic mapping** (which is rank-preserving). Holding out the
  *last* 15% by time removes the newest, most post-drift training
  data (12,600 rows), starving the main model exactly where the test
  period lives. That cost (~-0.03 ROC-AUC) wipes out the recency gain
  in the full model. On non-drifting data this tradeoff should be much
  cheaper, and calibration buys probability quality that AUC doesn't
  measure — but under drift, a random or interleaved calibration split
  would likely be the better design. Flagged here rather than silently
  changed, because the time-ordered holdout was the specified design.
- Evaluation caveat: test labels carry the same 3% flip noise, which
  compresses all AUCs toward ~0.6 and can penalize a model that
  predicts the *true* pattern. Absolute values here are not comparable
  to clean-label benchmarks.

## Tests

```bash
pytest tests/test_sentinel_boost.py
```

Covers: valid probability outputs, noise-filter downweighting of
injected label flips, interaction-feature construction, the
calibration path, and recency-weight decay.
