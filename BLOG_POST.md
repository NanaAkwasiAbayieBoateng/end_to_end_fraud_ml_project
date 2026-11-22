# From transaction data to a running fraud model: a file-by-file walkthrough

This project is a complete, small fraud-scoring system. It trains an XGBoost
classifier on transaction data, refuses to promote the new model unless it
beats the current baseline, serves the winner through a FastAPI service,
watches the live traffic for drift, and can kick off retraining when the
inputs change. Everything runs locally — the service is brought up on one
machine, from time to time, for learning. There is no cloud hosting.

Two honesty notes before anything else. The checked-in dataset
(`data/fraud_dataset.csv`) is a small demo set: 10,000 rows, 3% fraud. It is
not representative production data. And the promotion threshold
(`models/baseline_auc.json`, ROC-AUC 0.70) is a starter gate, not proof that
a model is safe to make real decisions with. The point of the repo is the
engineering path around the model — the same path a real system needs:
time-aware evaluation, one shared feature pipeline for training and serving,
gated promotion, and observable behaviour after deployment.

If you only read one other file, read `README.md` for how to run everything.
This post explains what every file does and why it exists.

## Project structure

The tracked repository is organized around the path from training data to
serving and monitoring:

```text
.
├── .github/workflows/model-ci-cd.yml
├── app/                  # FastAPI service and demo form
├── data/                 # Demo training data and example scoring input
├── mlops/                # Promotion gate, batch scoring, drift, ONNX, IEEE-CIS
├── models/               # Training code and checked-in baseline
├── tests/                # API, batch, drift, and IEEE-CIS tests
├── BLOG_POST.md
├── Dockerfile
├── README.md
├── compose.yaml
├── requirements-ci.txt
└── requirements-serving.txt
```

The trained pipeline, metrics, and feature-reference statistics are generated
under `models/` and are intentionally not tracked. The IEEE-CIS CSVs are also
downloaded separately and are not committed.

## Set up the Python environment

The project targets Python 3.12. In Windows PowerShell, run these commands
from the repository root to create and activate a project-local virtual
environment and install the development and test dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-ci.txt
```

The prompt should show `(.venv)` after activation. In a new terminal, activate
the environment again before running project commands. `requirements-ci.txt`
includes the serving requirements plus pytest, ONNX conversion, and MLflow;
Docker installs the smaller `requirements-serving.txt` set. From a WSL shell
at the repository root, use a separate Linux virtual environment—Windows and
Linux environments cannot be shared:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ci.txt
```

Do not commit the `.venv` directory.

## Dataset and prediction contract

The demo CSV and the online API share eight input features and a binary
`target` label:

| Feature | Meaning |
|---|---|
| `items_purchased` | Number of items in the transaction |
| `transaction_amount` | Transaction value |
| `avg_7day_spend` | Recent average spend |
| `account_age_months` | Account age |
| `mcc_code` | Merchant category code |
| `product_category` | Product category |
| `last_login_time` | Most recent login time |
| `transaction_datetime` | Transaction time |
| `target` | Label: 1 for fraud, 0 otherwise |

Dates supplied to the API use ISO-8601 strings. Numeric and categorical
inputs may be null or omitted because the feature pipeline handles missing
values; unknown fields and non-finite numbers are rejected. Before using a
real dataset, validate its schema, label completeness and provenance, time
availability, and potential leakage. In particular, inputs must be available
at decision time, and related transactions or customers may require
group-aware evaluation in addition to chronological splitting.

## The tools, and why each one

Every tool here solves a specific problem in the pipeline. Nothing is
included "because it's popular."

- **Python + pandas** — the lingua franca of tabular ML. All data wrangling
  (loading CSVs, building feature frames, reading prediction logs) is pandas.
- **scikit-learn `Pipeline`, `FunctionTransformer`, `ColumnTransformer`** —
  the feature engineering is packaged *inside* the model artifact as a
  pipeline. Training and serving then run literally the same code on raw
  inputs, which removes the classic training/serving skew bug where the API
  re-implements preprocessing slightly differently from training.
- **XGBoost (`XGBClassifier`)** — gradient-boosted trees are a strong,
  well-understood choice for tabular fraud features, and it outputs scores
  used for threshold decisions. Those scores are not automatically calibrated;
  calibration and threshold choice need separate evaluation. Alternatives
  (LightGBM, CatBoost) would also work; XGBoost is what's wired in.
- **FastAPI + Pydantic + Uvicorn** — FastAPI gives a small HTTP API with
  automatic OpenAPI docs at `/docs`; Pydantic models define and enforce the
  request contract (unknown fields and `Infinity`/`NaN` numbers are
  rejected); Uvicorn is the ASGI server that runs it.
- **joblib** — the fitted scikit-learn pipeline is saved and loaded with
  joblib (`models/fraud_model_pipeline.joblib`). Simple and exact, but a
  pickle-format caveat applies: only ever load artifacts you produced
  yourself, and expect version sensitivity between scikit-learn releases.
- **MLflow (tracking + Model Registry)** — `models/train.py` logs every
  training run (parameters, ROC-AUC/PR-AUC, tags, the model itself) to a
  local SQLite store (`mlflow.db`). With `--register`, the gate also
  registers a winning model in the MLflow Model Registry under the name
  `fraud-scorer` and promotes it to the `champion` alias, so "which model is
  current, and what did it score?" is a query, not an archaeology project.
- **skops, with cloudpickle as fallback** — when MLflow saves the sklearn
  model, `train.py` prefers the **skops** format, which only loads trusted
  types (the list is `SKOPS_TRUSTED_TYPES` in `train.py`) and is safer than
  unpickling arbitrary code. If skops isn't installed, the code falls back
  to cloudpickle and logs a warning. Gotcha: skops is *not* pinned in the
  requirements files, so a bare CI environment silently takes the
  cloudpickle path; the local `mlruns/` artifacts in this repo are skops
  (you can see `model.skops`) because the local environment had it.
- **ONNX (`onnxmltools` + `onnxruntime`)** — `mlops/export_onnx.py` converts
  *only the XGBoost classifier* to the portable ONNX format and verifies its
  probabilities match the sklearn classifier (max error must be ≤ 1e-5). If
  `ONNX_MODEL_PATH` is set, the API runs the sklearn preprocessor/selector
  and then the ONNX classifier. Feature engineering stays in Python — ONNX
  doesn't parse the API's ISO-8601 date strings — so this is a portability
  and benchmarking option, not a full export.
- **Docker + Docker Compose** — a multi-stage `Dockerfile` builds a small
  runtime image (dependencies in a builder stage, non-root user, health
  check). Compose runs two services from that image — the API and a
  one-shot drift check — sharing a named volume for prediction logs. Local
  and single-host by design; Kubernetes would be ceremony at this scale.
- **GitHub Actions** — `.github/workflows/model-ci-cd.yml` runs the tests,
  trains and gates a candidate, exports ONNX, and deploys. Automation is
  what makes the gate real: a model that fails tests or the AUC check
  cannot reach deployment.
- **A self-hosted GitHub Actions runner** — deployment targets a dedicated virtual 
  machine similar to a local machine (label `fraud-model-server`), because the service is hosted
  locally and intermittently. The trade-off is explicit: that runner can
  control Docker on its host, so only reviewed, trusted code should ever
  reach the deploy job.
- **pytest + FastAPI's `TestClient`** — the tests exercise the API, batch
  scorer, drift maths, and IEEE-CIS loading with small fake models, so they
  run fast and need no trained artifact. `TestClient` is why
  `requirements-ci.txt` contains both `httpx` and `httpx2`: newer Starlette
  versions moved to the renamed `httpx2` package, and keeping both means
  the test client works either way.
- **PSI drift detection, implemented directly** — there is no Evidently (or
  any drift library) dependency. `mlops/drift_check.py` implements the
  Population Stability Index by hand with numpy: bin the live features the
  same way as the training reference, compare proportions, flag anything at
  PSI ≥ 0.25. The approach is the standard one Evidently popularised; here
  it's ~30 lines, which is all a single-service monitor justifies.

## How it was built, step by step

The build order follows the data. Each stage only exists because the
previous one produced something it needs.

1. **Data and training first** (`data/fraud_dataset.csv`, `models/train.py`).
   Sort transactions by time, train on the earliest 70%, evaluate on the
   latest 30% — never a random split, because in fraud the future must not
   leak into evaluation of the past. All feature engineering lives in the
   pipeline built here.
2. **The API** (`app/main.py`, `app/index.html`). Load the saved pipeline,
   validate requests against a strict schema, return a probability and a
   decision, and log every prediction as JSON. The prediction log is a core system artifact, enabling auditability, performance tracking, and future analysis.
3. **The promotion gate** (`mlops/model_gate.py`). Wrap training in a check:
   candidate ROC-AUC must *strictly* beat `models/baseline_auc.json`, or
   nothing deployable is written. On success it also writes the feature
   reference statistics the drift check will compare against.
4. **Batch scoring and ONNX** (`mlops/score_batch.py`,
   `mlops/export_onnx.py`). The same pipeline scores CSV files offline, and
   the classifier can be exported to ONNX with a parity check. Both reuse
   the artifact from step 3 — no second implementation of anything.
5. **Packaging** (`Dockerfile`, `compose.yaml`). Multi-stage image,
   localhost-only port binding, prediction logs in a Docker volume.
6. **Drift monitoring** (`mlops/drift_check.py`). Compare the last 24 hours
   of logged prediction features with the reference statistics from step 3
   using PSI. On drift: open a GitHub issue, POST to a webhook, and dispatch
   a retraining run. Drift is a reason to investigate, not proof the model
   got worse — the monitor sees inputs, not ground-truth labels.
7. **CI/CD** (`.github/workflows/model-ci-cd.yml`). Tests → gated training →
   ONNX export → artifact upload on GitHub-hosted runners; deployment to the
   self-hosted runner on pushes to `main` only.
8. **The MLflow registry** (`model_gate.py --register`). A passing candidate
   is registered as `fraud-scorer` and aliased `champion`, with its baseline
   and candidate AUCs tagged on the version. The joblib file is still what
   the API serves; the registry is the record of what's current and why.
9. **Real data** (`mlops/ieee_cis.py`). A separate pipeline for the
   IEEE-CIS Fraud Detection dataset — the industry-standard benchmark —
   with the same discipline (chronological split, MLflow logging) on much
   wider, messier data. Only the Kaggle download is missing (see the end).

## Every file, explained

Read the repo in this order and it will make sense top to bottom.

### Root files

- **`README.md`** — the operating manual: architecture diagram, run
  commands, CI/CD and drift configuration, measured performance. Start
  there to *do* things; this post is for understanding them.
- **`BLOG_POST.md`** — this file.
- **`conftest.py`** — the smallest file with the best story. It contains
  only a docstring — no code. Its mere presence at the repo root makes
  pytest add the root to `sys.path`, so `app`, `mlops`, and `models`
  import as top-level packages. It exists because CI runs the bare `pytest`
  console script, which (unlike `python -m pytest`) does not put the
  working directory on `sys.path`; the test job failed at collection with
  `ModuleNotFoundError: No module named 'app'` until this file landed.
- **`requirements-serving.txt`** — the pinned runtime dependencies (FastAPI,
  Uvicorn, Pydantic, pandas, numpy, scikit-learn, XGBoost, joblib,
  onnxruntime). This is all the Docker image installs.
- **`requirements-ci.txt`** — starts with `-r requirements-serving.txt`,
  then adds what CI and local development need: pytest, httpx + httpx2
  (see the tools section), onnxmltools (ONNX export), and mlflow (training
  logs and the registry).
- **`Dockerfile`** — two stages. The builder creates a virtualenv at
  `/opt/venv` from `requirements-serving.txt`; the runtime stage copies
  that venv plus `app/`, `models/`, and `mlops/`, installs `libgomp1`
  (XGBoost's OpenMP runtime), creates a non-root `appuser`, and runs
  Uvicorn via a small `python -c` command that honours the `PORT`
  environment variable. A Docker `HEALTHCHECK` polls `/ready`. Defaults
  baked in: `MODEL_PATH=/app/models/fraud_model_pipeline.joblib`,
  `MODEL_VERSION=candidate`.
- **`compose.yaml`** — two services from the same image.
  `fraud-api` runs the server, binds to `127.0.0.1:8000` by default
  (override with `API_BIND_ADDRESS`/`API_PORT` only behind real network
  controls), and writes prediction logs to the shared `prediction-logs`
  volume. `drift-check` overrides the entrypoint to run
  `mlops/drift_check.py` once against that volume (mounted read-only) with
  `--hours 24 --threshold 0.2`, passing the GitHub repository, dispatch
  token, and webhook URL through from the environment.
- **`.gitignore`** — an *allowlist*, which is unusual and deliberate: it
  ignores everything (`*`) and then un-ignores the source files one by one.
  Generated files — the trained pipeline, metrics, reference stats,
  `mlflow.db`, IEEE-CIS CSVs — can never be committed by accident. Only
  `data/fraud_dataset.csv` and `data/transactions.csv` are allowed out of
  `data/`, and only `train.py`, `baseline_auc.json`, and `__init__.py` out
  of `models/`.
- **`.dockerignore`** — the same idea for the Docker build context: keeps
  `.git`, data, tests, and MLflow state out of the image, but explicitly
  allows the built artifacts the image needs (`fraud_model_pipeline.joblib`,
  `reference_stats.json`, `fraud_classifier.onnx`). In the deploy job those
  files arrive from the CI artifact download just before the build.

### `app/` — the API service

- **`app/__init__.py`** — one docstring ("Fraud prediction API"). Marks the
  directory as a package.
- **`app/main.py`** — the whole service, ~170 lines. Key pieces:
  - `FEATURE_NAMES` — the eight input features, in the exact column order
    the pipeline expects. `mlops/score_batch.py` imports this tuple so the
    batch path can never drift from the API's contract.
  - `PredictionRequest` / `PredictionResponse` — Pydantic models. Requests
    forbid unknown fields and non-finite numbers; every field is optional
    because the pipeline imputes missing values itself. Responses carry
    `request_id`, `fraud_probability`, `predicted_fraud`, `latency_ms`,
    and `model_version`.
  - `create_app(model=None, model_version=None)` — an app factory. Tests
    inject a fake model through it; production calls it with no arguments
    and the FastAPI *lifespan* handler loads the joblib pipeline from
    `MODEL_PATH` at startup (failing fast if the artifact is missing), plus
    an ONNX session if `ONNX_MODEL_PATH` is set.
  - Routes: `GET /` serves the HTML form; `GET /health` and `/ping` are
    liveness; `GET /ready` returns 503 until a model is loaded;
    `POST /predict` and its alias `POST /invocations` score one
    transaction. The decision rule is fixed: probability ≥ 0.5 means fraud.
  - Every prediction is timed, given the caller's `x-request-id` (or a new
    UUID), and written as one JSON line to stdout and — when
    `PREDICTION_LOG_PATH` is set — to a rotating file (10 MB × 5 backups).
    That JSONL log is the drift check's input, which is why it records the
    raw input features.
- **`app/index.html`** — a single self-contained page (inline CSS and
  JavaScript, no build step) with a form for the eight features. It POSTs to
  `/predict` and shows the probability and decision. It's a demo and smoke
  test, not a product UI.

### `models/` — training and the gate's inputs/outputs

- **`models/__init__.py`** — one docstring; package marker.
- **`models/train.py`** — where the model is defined. Read it in its four
  numbered sections:
  1. *Transformer callables.* Five plain functions —
     `impute_missing_values` (sentinel substitutions: `-9999.01` for
     numbers, `"MISS"` for categories, `1900-01-01` for dates),
     `rename_columns_with_suffix` (adds `_MISS` to every column),
     `cat_to_bool_transformer` (category-membership flags, e.g. "is this
     MCC one of the risky ones"), `create_ratio_features` (amount ÷ average
     spend, account age ÷ items, with zero-division protection), and
     `calculate_days_since` (days between last login and the transaction).
     They live at module level for a hard reason: joblib/pickle serialises
     functions *by reference*, so anything nested inside another function
     would make the saved pipeline unloadable.
  2. *`build_feature_engineering_pipeline(...)`* — assembles those five
     functions into a sklearn `Pipeline` of `FunctionTransformer`s, in the
     order above.
  3. *`train_fraud_model(...)`* — the main entry point. Loads the data,
     sorts by `transaction_datetime`, splits chronologically (70/30),
     builds the full pipeline — preprocessor → `ColumnTransformer`
     "selector" that keeps the nine final numeric features → `XGBClassifier`
     (120 trees, depth 8, and friends; overridable via `model_params`) —
     fits it, and reports ROC-AUC and PR-AUC on the holdout. It can save
     artifacts (joblib pipeline + preprocessor, the XGBoost model in its
     native JSON, and a PMML export) and logs the run to MLflow with
     parameters, metrics, tags, an inferred signature, and the chosen
     serialization format (skops preferred, cloudpickle fallback — see the
     tools section). Two gotchas: the PMML path imports `sklearn2pmml`,
     which is not in the requirements files, so that branch only works in an
     environment where it's been installed separately; and `SKOPS_TRUSTED_TYPES`
     is defined twice, identically — harmless, but it's there.
  4. *`__main__`* — running `python models/train.py` trains on
     `data/fraud_dataset.csv` and saves everything into `models/`.
- **`models/baseline_auc.json`** — `{ "roc_auc": 0.70 }`. The number a
  candidate must strictly beat. It is tracked in git on purpose: changing
  the bar is a reviewed decision, and the README is blunt about never
  lowering it just to make a candidate pass.
- **Generated files you'll see here but not in git** —
  `fraud_model_pipeline.joblib` (the deployable pipeline the API loads),
  `model_metrics.json` (the last gate result: baseline, candidate, pass/fail,
  and — when registered — the MLflow run ID and `fraud-scorer` version),
  `reference_stats.json` (per-feature training distributions for drift:
  decile bins + proportions + missing-rate for numeric and datetime
  features, a proportion table for categoricals), and `fraud_classifier.onnx`
  (the exported classifier).

### `mlops/` — the operational scripts

Note there is no `mlops/__init__.py`: these are scripts first. `model_gate.py`
and `export_onnx.py` insert the repo root into `sys.path` themselves so they
run from anywhere; the tests import them as a namespace package, which works
because of the root `conftest.py`.

- **`mlops/model_gate.py`** — the promotion checkpoint, run as
  `python mlops/model_gate.py`. It loads the dataset (validating that
  `target` exists and is complete), calls `train_fraud_model` with
  `save_artifacts=False` — nothing is written for a candidate that might
  fail — then compares ROC-AUC against the baseline. Fail: write the metrics
  file, print why, exit 1. Pass: dump the pipeline to
  `models/fraud_model_pipeline.joblib` and write `reference_stats.json` via
  its `reference_summary(...)` helper (the binning logic the drift check
  later mirrors). With `--register` it also logs the MLflow run and, on a
  pass, registers the model as `fraud-scorer`, sets the `champion` alias,
  and tags the version with both AUCs. CI runs it exactly this way.
- **`mlops/score_batch.py`** — offline scoring. `score_dataframe(frame,
  model, threshold)` is the testable core: it checks all eight
  `FEATURE_NAMES` are present, rejects empty input and out-of-range
  thresholds, and returns a copy of the frame with `fraud_probability` and
  `predicted_fraud` appended (extra columns like `transaction_id` pass
  through untouched). The CLI (`python -m mlops.score_batch --input ...
  --output ...`) refuses to overwrite its own input or the model file.
- **`mlops/drift_check.py`** — the monitor. `records_from_file(...)` reads
  the prediction log *and its rotated backups*, keeping only
  `fraud_prediction` events from the last `--hours` hours (default 24).
  Fewer than `--min-predictions` (default 100) records and the check skips
  quietly — no drift claims on thin data. `psi(...)` and
  `psi_from_reference(...)` implement the Population Stability Index; the
  latter compares live values against the stored reference proportions and
  accounts for each side's missing-value rate. Datetime features are
  compared as epoch seconds, categoricals as proportion tables. Any feature
  at PSI ≥ `--threshold` (default 0.2) is reported. On drift it does three
  things, in order: opens a GitHub issue (`build_issue_payload(...)` +
  `create_github_issue(...)`, via `post_json(...)`), POSTs the report to
  `DRIFT_ALERT_WEBHOOK_URL` if configured, and sends a
  `model-drift-detected` repository dispatch that re-runs CI training.
  Gotcha: the issue text still says "scheduled drift check" — a leftover
  from when the workflow ran on a schedule; it's manual now (below).
  Repository and token must be configured together or the script refuses to
  start.
- **`mlops/export_onnx.py`** — loads the gated pipeline, converts only the
  `classifier` step with `onnxmltools.convert_xgboost` (opset 15, input
  width taken from the selector step), writes `models/fraud_classifier.onnx`,
  then proves parity: it pushes the first 100 training rows through the
  sklearn preprocessor/selector and both classifiers, and fails if any
  probability differs by more than 1e-5. Measured locally, the error is
  around 1.5e-07.
- **`mlops/ieee_cis.py`** — the real-data pipeline, deliberately separate
  from the demo one. Its module docstring is the download instructions.
  `load_ieee_cis(...)` merges `train_transaction.csv` with
  `train_identity.csv` on `TransactionID`, renames `isFraud` to `target`,
  and anchors `TransactionDT` (seconds) to 2017-12-01 to get a real
  datetime. `feature_columns(...)` picks the numeric set (amount/card/count
  columns, the C and D series, plus every Vesta `V*` column present) and
  the categorical list. `build_pipeline(...)` is a more standard sklearn
  setup — median imputation for numbers, constant-impute + one-hot
  (`min_frequency=50`) for categoricals, a 400-tree XGBoost.
  `time_based_split(...)` keeps the chronological discipline, and
  `train_ieee_cis(...)` reports ROC-AUC/PR-AUC and logs to the MLflow
  experiment `fraud-detection-ieee-cis`. sklearn imports are lazy (inside
  functions) so the module — and its tests — load fast.

### `tests/` — what is actually verified

16 tests, all using fakes instead of the real model, so the suite runs in
about a minute and proves wiring, not model quality.

- **`tests/test_api.py`** (5 tests) — a `FakeModel` returning a fixed
  0.75 probability verifies: a valid request returns probability, decision,
  latency, and the caller's `x-request-id`; the prediction log line contains
  the features and timestamp; `"Infinity"` amounts and unknown fields get a
  422; an empty JSON body still scores (the pipeline imputes); and `/`
  serves the form.
- **`tests/test_batch_scoring.py`** (2 tests) — `score_dataframe` preserves
  input columns and the input frame itself, appends the two result columns,
  honours a custom threshold, and raises on missing features or a bad
  threshold.
- **`tests/test_drift.py`** (5 tests) — PSI is 0 for identical
  distributions and > 0.2 for a shifted one; reference-based PSI is ~0 for a
  matching population; the log reader keeps only recent, parseable
  prediction events; and the GitHub issue payload names the drifted features
  and the prediction count.
- **`tests/test_ieee_cis.py`** (4 tests) — on tiny synthetic CSV fixtures:
  the identity table merges, `isFraud` becomes `target`, rows come out
  sorted by time, a missing transaction file raises `FileNotFoundError`,
  feature selection picks what's present, and the split is chronological.

### `.github/workflows/model-ci-cd.yml` — the automation

One workflow, four jobs, triggered by pushes to any branch, pull requests to
`main`, manual `workflow_dispatch`, and `repository_dispatch` events of type
`model-drift-detected` (that's the drift check's retraining trigger arriving
back here).

- **`tests`** (GitHub-hosted Ubuntu, Python 3.12) — installs
  `requirements-ci.txt`, runs `pytest -q tests`. This is the job the root
  `conftest.py` rescued.
- **`model-validation`** (needs `tests`) — runs the gate with `--register`,
  exports ONNX, and uploads the pipeline, reference stats, metrics, and ONNX
  file as the `validated-model` artifact, failing if any file is missing.
  A candidate that doesn't beat the baseline stops the pipeline here.
- **`deploy-local`** (needs both jobs above; only on pushes to `main`) —
  runs on the self-hosted `fraud-model-server` runner: downloads the
  validated artifact into `models/`, then `docker compose build --pull` and
  `up --force-recreate`, with `MODEL_VERSION` set to the commit SHA so every
  prediction log line names the exact code that served it.
- **`drift-check`** (only on `workflow_dispatch`) — runs on the self-hosted
  runner and executes the Compose `drift-check` service with the
  `DRIFT_DISPATCH_TOKEN` and `DRIFT_ALERT_WEBHOOK_URL` secrets. It used to
  run on a daily schedule; it's manual now because the host machine is only
  on intermittently — a scheduled job would just queue for a machine that's
  off. (The `if: github.event_name != 'schedule'` conditions on the first
  two jobs are leftovers from that era.)

### `data/` — the inputs

- **`data/fraud_dataset.csv`** — the demo training set: 10,000 transactions,
  the eight features plus `target`, 3% fraud, deliberately messy (missing
  amounts, missing logins) so the imputation code has something to do.
- **`data/transactions.csv`** — five example rows with a `transaction_id`,
  the sample input for `score_batch.py` in the README.
- **`data/ieee_cis/`** — not in git. Where the Kaggle download goes; see
  what's left, below.

## A request's journey

**One `POST /predict`.** A client (or the form in `app/index.html`) sends the
eight features as JSON to Uvicorn, which hands it to the FastAPI app built by
`create_app()` in `app/main.py`. Pydantic validates the body against
`PredictionRequest` — an unknown field or an `Infinity` amount never reaches
the model. The handler builds a one-row pandas DataFrame in `FEATURE_NAMES`
order and calls the loaded pipeline: the same `FunctionTransformer` steps
from `models/train.py` impute, rename, flag, ratio, and date-diff the row;
the selector keeps the nine numeric features; XGBoost returns a probability
(or, with `ONNX_MODEL_PATH` set, the first two steps run in sklearn and the
ONNX session scores). Probability ≥ 0.5 becomes `predicted_fraud: true`. The
handler stamps the record with a UTC timestamp, the request ID, the model
version, and the measured latency, writes it as a JSON line to stdout and
the rotating log file in the Compose volume, and returns the
`PredictionResponse`.

**One drift-check run.** Someone triggers the workflow manually
(`workflow_dispatch`). The `drift-check` job starts on the self-hosted
runner and runs the Compose `drift-check` service, which is
`mlops/drift_check.py` with the shared log volume mounted read-only. The
script loads `models/reference_stats.json` (written by the gate when the
current model was promoted) and pulls the last 24 hours of
`fraud_prediction` records from `predictions.jsonl` and its rotated backups.
Under 100 records: it prints "skipped" and exits 0. Otherwise it computes
PSI for every feature against the reference — binned proportions for numeric
and datetime features, proportion tables for categoricals — and prints a
JSON report. If every feature is under 0.2, that's the end. If any feature
drifts, it opens a GitHub issue listing the drifted features and their PSI
scores, POSTs the report to the alert webhook, and fires a
`model-drift-detected` dispatch. That dispatch re-enters this same workflow
at the top — tests, then gated training — and the new candidate still has
to beat the baseline and land on `main` before `deploy-local` will touch
the running service. Drift can start the process; it cannot shortcut it.

## How to run it, what it scores, and what's left

**Running it** starts in the activated environment above. From the repository
root, run the focused tests, train and validate a model, then start the API
with Docker Compose:

```powershell
python -m pytest -q tests
python mlops/model_gate.py
docker compose --project-name fraud-model up --build -d
```

The API is at `http://127.0.0.1:8000` (form) and
`http://127.0.0.1:8000/docs` (Swagger). The README has the batch-scoring,
drift-check, and deployment commands. For Python-only local serving instead
of Docker, run `python -m uvicorn app.main:app --reload` after the model gate
has produced the model artifact.

**Measured numbers** (from the README, measured 2026-11-15 on a small shared
VM, model trained by the gate on the demo data):

- Candidate ROC-AUC **0.840** against the 0.70 baseline — the gate passed,
  and that run is registered in MLflow as `fraud-scorer` version 1, alias
  `champion`.
- `POST /predict` latency over 100 requests: **p50 13.0 ms, p95 17.3 ms**,
  as reported by the API itself.
- ONNX parity: max probability error ~1.5e-07 versus the sklearn pipeline.

**What remains** is the real-data step. `mlops/ieee_cis.py` is written and
tested, but the IEEE-CIS CSVs can't be committed (Kaggle competition rules),
so the one manual step left is yours:

```bash
kaggle competitions download -c ieee-fraud-detection -p data/ieee_cis
# unzip so data/ieee_cis contains train_transaction.csv and train_identity.csv
python mlops/ieee_cis.py --data-dir data/ieee_cis
```

After that, the honest upgrades are the unglamorous ones: replace the 0.70
starter baseline with the champion's measured score on the same data and
split, choose a decision threshold from the actual costs of missed fraud
versus false alarms instead of assuming 0.5 is right, and assess PR-AUC,
calibration, and performance across relevant time periods and transaction
segments. Keep remembering the principle the whole repo is built around:
preserve chronology, keep feature engineering inside the artifact, promote
only what passes the gate, and never mistake input drift for measured model
performance.
