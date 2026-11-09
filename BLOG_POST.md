# From transaction data to a running fraud model: a project walkthrough

Fraud detection is not just a matter of fitting a classifier. A useful system
must preserve time order during evaluation, apply the same feature logic during
training and prediction, prevent weak models from being promoted, and make
changes observable after deployment. This project demonstrates that complete
path with a tabular fraud classifier, a FastAPI service, Docker, and GitHub
Actions.

The project is an engineering example, not a production-ready fraud decision
system. Its checked-in CSV is a small example dataset with 10,000 rows and a
3% positive class; it is not a substitute for representative, properly
labelled transaction data. Its checked-in ROC-AUC threshold of `0.70` is a
starter gate, not evidence that the model is safe or effective for a real
business.

## Project structure

```text
.
├── .github/
│   └── workflows/
│       └── model-ci-cd.yml
├── app/
│   ├── __init__.py
│   ├── index.html
│   └── main.py
├── data/
│   └── fraud_dataset.csv
├── mlops/
│   ├── drift_check.py
│   ├── export_onnx.py
│   ├── model_gate.py
│   └── score_batch.py
├── models/
│   ├── __init__.py
│   ├── baseline_auc.json
│   └── train.py
├── tests/
│   ├── test_api.py
│   ├── test_batch_scoring.py
│   └── test_drift.py
├── .dockerignore
├── .gitignore
├── BLOG_POST.md
├── Dockerfile
├── README.md
├── compose.yaml
├── requirements-ci.txt
└── requirements-serving.txt
```

The tree shows the checked-in source and configuration. Training outputs such
as the deployable model pipeline, metrics, and reference statistics are
generated under `models/` and are intentionally not checked in.

## What the system contains

The main journey through the repository is:

1. `data/fraud_dataset.csv` supplies labelled example transactions.
2. `models/train.py` sorts transactions in time order, trains an XGBoost
   pipeline, and reports ROC-AUC and PR-AUC on a later holdout.
3. `mlops/model_gate.py` runs candidate training and only writes the deployable
   model and feature-reference statistics if candidate ROC-AUC beats the
   configured baseline.
4. `app/main.py` validates prediction requests, loads the saved pipeline, and
   returns a probability and a threshold-based decision.
5. `mlops/score_batch.py` applies the same saved pipeline to CSV batches.
6. Docker Compose packages and runs the API, while the GitHub Actions workflow
   tests, validates, and (when configured) deploys approved artifacts.
7. `mlops/drift_check.py` compares recent logged input features with training
   reference distributions and can notify an operator and request retraining.

The key files are `README.md` for concise operating instructions,
`models/train.py` for model and feature construction, `app/main.py` for online
serving, `mlops/` for the promotion, batch, drift, and optional ONNX tools,
`tests/` for focused checks, and `.github/workflows/model-ci-cd.yml` for
automation.

## Why this project uses these tools

The stack favors common, interoperable tools that solve a concrete need in
this small fraud-scoring service; there is no single mandatory technology
stack for production machine learning.

| Tool | Why it is used here | Industry fit and trade-off |
|---|---|---|
| Python, pandas, and scikit-learn `Pipeline` | Python provides the model and service ecosystem; pandas handles tabular data, while the pipeline packages feature engineering and prediction into one reusable artifact. | Python and pandas are widely used for data science, and scikit-learn pipelines are a common way to keep preprocessing and inference together. The pipeline also reduces training-serving skew. Custom Python transforms can limit exporting the complete pipeline to another runtime. |
| XGBoost | A gradient-boosted tree classifier is a practical fit for tabular transaction features and class-probability scoring. | XGBoost is a mature, widely adopted open-source gradient-boosting library, but it is not automatically the best model for most tabular datasets.It has other competitors usch as lightgbm and Catboost. We compare it with a simple baseline and select the best model using appropriate, time-aware validation. |
| FastAPI, Pydantic, and Uvicorn | FastAPI exposes the prediction endpoint; Pydantic validates the input contract; Uvicorn runs the Asynchronous Server Gateway Interface(ASGI) application. | These are established Python web-service tools. JSON over HTTP is a familiar integration pattern, but FastAPI is one implementation choice rather than a universal model-serving standard. |
| Docker and Docker Compose | Docker packages the app and its runtime dependencies; Compose makes the local API and its log volume reproducible with one configuration. | Docker images follow the Open Container Initiative (OCI) container ecosystem, a broadly adopted packaging approach. Compose is convenient for local and single-host operation; larger platforms may use Kubernetes or a managed container service instead. |
| GitHub Actions | The workflow runs tests, validates a candidate, and deploys only after the required checks pass. | CI/CD automation is standard practice, while GitHub Actions is a GitHub-specific, commonly used implementation. A self-hosted runner is appropriate for this single-host example, but it has access to that host and must only run trusted deployment code. |
| Joblib and ONNX Runtime | Joblib saves and reloads the fitted Python pipeline. The optional ONNX path runs the exported XGBoost classifier and checks its scores against the original classifier. | Joblib is convenient within a compatible Python/scikit-learn environment, but serialized Python objects are version-sensitive and must only be loaded from trusted sources. ONNX is an open model-exchange format with a broad runtime ecosystem; this project's export covers only the classifier, not its custom feature engineering. |

These choices are proportionate to a demonstrator and a modest single-service
deployment. A feature store, model server, or orchestrator is not included
because the current example does not show a scale or feature-computation need
that would justify its operational cost. Consider Feast or another online
feature store when features need shared, low-latency, point-in-time-consistent
serving; consider Triton or another dedicated inference server when measured
throughput, accelerator use, or dynamic batching requirements justify it.
Choose based on workload evidence and operational requirements, not a tool's
popularity alone.

## 1. Prepare the environment

The project targets Python 3.12. To run it locally from PowerShell, create and
activate a virtual environment, then install the project and test
dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ci.txt
```

If PowerShell prevents activation, use the virtual environment's interpreter
directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ci.txt
```

`requirements-ci.txt` includes the serving dependencies plus test and ONNX
conversion packages. The container uses the smaller `requirements-serving.txt`
set. Docker Engine and the Docker Compose plugin are needed for the container
steps below. For local Python-only tests and training, Docker is not required.

## 2. Understand the data and the prediction contract

Each row contains eight input features and a binary `target`:

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

The same eight inputs are accepted by the API and batch scorer. Date values
should be ISO-8601 strings. Fields may be null or omitted; the training
pipeline uses sentinel substitutions for missing numeric, categorical, and
date data. The API rejects unknown fields and non-finite numeric values rather
than silently accepting a different input schema.

Before using a real dataset, check that its labels are complete and trustworthy,
the column names and data types match this contract, and its time span and fraud
prevalence reflect the intended use case. Understand where the labels came
from, whether the same customer or transaction family can cross the split, and
whether any input would only become available after the decision must be made.
The example workflow checks for a complete `target` column, but a real
deployment also needs explicit data validation and leakage review.

## 3. Train and evaluate the model

Run the end-to-end candidate gate from the repository root:

```powershell
python mlops/model_gate.py
```

The training function parses `transaction_datetime`, sorts rows chronologically,
and uses the earliest 70% for training and the latest 30% for holdout
evaluation. This is important: a random split could let future transaction
patterns influence evaluation of earlier transactions. The pipeline is fitted
after the split, so transformations are part of the estimator that is trained
on the training portion and reused at inference.

Feature engineering fills missing inputs, renames source columns with a
`_MISS` suffix, creates a few categorical membership indicators and amount /
account-age ratios, and calculates days since the last login. A selector then
keeps the engineered numeric features for an XGBoost binary classifier. Keep
this preprocessing inside the saved pipeline: training and serving must not
implement separate versions of the same feature logic.

The gate compares holdout ROC-AUC with `models/baseline_auc.json`. Promotion
requires a strict improvement (`candidate_roc_auc > baseline_roc_auc`); equal
or lower scores fail the command and do not produce the deployable model or
reference summary. A successful run writes the pipeline to
`models/fraud_model_pipeline.joblib`, metrics to `models/model_metrics.json`,
and feature distributions to `models/reference_stats.json`. These generated
files are ignored by Git. The baseline file is tracked and currently contains
`0.70`; it should be replaced only with a defensible champion result calculated
with the same data version, split, and metric procedure—not lowered to make a
candidate pass.

ROC-AUC measures ranking across thresholds; it does not select an operational
threshold or establish that a model is worth deploying. Fraud is the minority
class in the sample, so a real evaluation should also pay attention to
precision-recall performance, false-positive and false-negative costs,
calibration, and performance across relevant segments. The training function
reports PR-AUC as well, but the current promotion gate is ROC-AUC only. Any
threshold other than the serving default of 0.5 should be selected and
documented using validation data and business requirements.

Another optional diagnostic is the Kolmogorov-Smirnov (KS) statistic: the
largest gap between the cumulative fraud and non-fraud score distributions.
It summarizes how well the scores separate the two classes, but it does not
choose a decision threshold or replace PR-AUC and cost-based evaluation.
Calculate it on the same chronological holdout used for other metrics, and
interpret it alongside the number of positive examples and uncertainty in the
estimate.

## 4. Run checks and inspect the API

Run the focused automated test suite:

```powershell
pytest -q tests
```

The tests cover API validation and responses, prediction logging, batch-score
columns and input checks, and drift calculations and log-window filtering.
They use small fake models for service tests; they are not evidence of model
quality.

Once `model_gate.py` succeeds, start the API directly:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000/` for the interactive form or
`http://127.0.0.1:8000/docs` for the OpenAPI interface. `/health` (also
available as `/ping`) is a liveness check; `/ready` reports whether a model has
loaded. Predictions are available at `POST /predict` and its `/invocations`
alias. For example, from PowerShell:

```powershell
$body = @{
  items_purchased = 2
  transaction_amount = 24.5
  avg_7day_spend = 18
  account_age_months = 8
  mcc_code = "Retail_A"
  product_category = "Clothing"
  last_login_time = "2023-01-01T10:00:00"
  transaction_datetime = "2023-01-01T10:30:00"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
  -Method Post -ContentType "application/json" -Body $body
```

The response contains a `fraud_probability`, a `predicted_fraud` boolean,
`model_version`, `request_id`, and inference latency. The current decision
rule is probability greater than or equal to 0.5. A caller can supply an
`x-request-id`; otherwise the service creates one.

## 5. Score a batch

Prepare a CSV containing all eight feature columns, then run:

```powershell
python -m mlops.score_batch `
  --input data\transactions.csv `
  --output data\scored_transactions.csv
```

The script loads `models/fraud_model_pipeline.joblib` by default, preserves
other input columns such as transaction IDs, and appends `fraud_probability`
and `predicted_fraud`. The default threshold is 0.5; use `--threshold` to
provide a different value between 0 and 1, and `--model` to select another
pipeline. The output path cannot overwrite the input or model file. Keep
identifiers and any sensitive batch data protected according to your
organization's data-handling rules.

## 6. Package and run with Docker

After the gate has produced the model and reference statistics, build and run
the local container:

```powershell
docker compose --project-name fraud-model up --build -d
```

The multi-stage `Dockerfile` installs serving dependencies in a builder stage,
copies them to a smaller runtime image, and runs the service as a non-root
user. Compose maps the API to `127.0.0.1:8000` by default and stores rotating
prediction logs in a named Docker volume. Check
`http://127.0.0.1:8000/ready` and open the form or API docs to confirm it is
serving. To stop the service without removing its persistent log volume, run:

```powershell
docker compose --project-name fraud-model down
```

The API logs each prediction as JSON, including the input features, timestamp,
probability, decision, request ID, latency, and model version. Logs go to
stdout and, when configured in Compose, a rotating file. Treat these records
as sensitive data: restrict access to the host and Docker volume, choose a
retention period, and avoid exposing the unauthenticated API to an untrusted
network. The default localhost binding is intentional; changing
`API_BIND_ADDRESS` should only be done behind appropriate network controls.

## 7. Optional ONNX export

This project’s feature engineering could be included in an ONNX graph, but it currently isn’t:  `mlops/export_onnx.py`  exports only the XGBoost classifier. The API still runs the pandas/scikit-learn preprocessor before passing features to ONNX.

The project’s imputations, category-membership flags, ratios, and date differences could be reimplemented with ONNX operators or custom converters. The main complication is that the current steps use pandas  DataFrame  operations, and ONNX doesn’t natively parse the API’s ISO-8601 date strings. Date conversion to numeric timestamps before inference or implementation of  that parsing outside ONNX.

End-to-end ONNX inference is feasible, but it would require changing the export and serving paths and checking prediction parity. The current split is simpler; including preprocessing in ONNX would make deployment less dependent on Python/sklearn and help keep preprocessing and scoring together

`mlops/export_onnx.py` exports the XGBoost classifier and checks its predicted
probabilities against the classifier in the scikit-learn pipeline:

```powershell
python mlops/export_onnx.py `
  --model models/fraud_model_pipeline.joblib `
  --output models/fraud_classifier.onnx
```

Feature engineering remains in the Python pipeline; only the classifier is
converted. The parity check uses up to 100 example rows and fails if the
maximum probability difference exceeds `1e-5`. The API can use this classifier
when `ONNX_MODEL_PATH` points to the exported ONNX file. That environment
variable must be passed to the API process; the checked-in Compose file does
not enable ONNX by default. Keep the matching preprocessor and classifier
together when promoting an ONNX artifact.

## 8. CI, promotion, and deployment

The workflow in `.github/workflows/model-ci-cd.yml` runs tests on pushes and
pull requests to `main`. After tests pass, it trains and gates a candidate,
exports and validates ONNX, and uploads the successful model, reference
statistics, metrics, and ONNX file as a workflow artifact. A failed test or
model gate blocks the dependent work.

Automatic deployment is deliberately narrower: a push to `main` can deploy
only after both tests and model validation succeed, and only on a self-hosted
runner labelled `fraud-model-server`. To enable it, install Docker Engine and
the Compose plugin on a dedicated machine, register the runner with that label,
and keep it online as a service. Protect `main` and require the test and model
validation checks before merging. The runner can control Docker on its host,
so only reviewed, trusted changes should reach the deployment job.

The deployment job downloads the validated artifacts, builds the image on the
target machine, and recreates the Compose service. It sets `MODEL_VERSION` to
the deployed commit SHA so prediction records identify which source revision
served them. This is a local-machine deployment pattern; it does not require a
cloud host or container registry.

## 9. Drift monitoring and retraining

On its daily schedule, the workflow runs a PSI (Population Stability Index)
check on the most recent 24 hours of prediction logs. It requires at least
100 recent predictions; with fewer records or no log, the check skips rather
than making a low-sample drift claim. It compares numeric, date, and
categorical feature distributions with the reference summary generated from
the training data. A feature with PSI at or above 0.2 is reported as drifted.

For alerts, configure the repository secret `DRIFT_ALERT_WEBHOOK_URL` with an
HTTPS endpoint that accepts a JSON POST. To dispatch a candidate retraining
workflow, configure `DRIFT_DISPATCH_TOKEN` with a fine-scoped GitHub token
allowed to create repository dispatches. The scheduled job also needs the
self-hosted runner and deployed Compose log volume. To run the check manually:

```powershell
docker compose --project-name fraud-model run --rm --no-deps drift-check
```

Drift is a reason to investigate, not proof that the model has become
inaccurate. This monitor measures changes in inputs; it does not know whether
recent predictions were correct because it has no ground-truth labels. A
dispatch starts fresh tests and candidate training, but it does not bypass the
ROC-AUC gate or the merge-to-main deployment rule. Production retraining must
use refreshed, labelled training data; predictions must never be treated as
labels.

## What to improve before real-world use

The system provides a useful end-to-end skeleton, but real fraud decisions
need more than a successful build or an improved AUC. Replace the sample data
with governed, representative labels; validate schema, time availability,
leakage, and data quality; compare meaningful baselines; and choose an
operating threshold from the costs of missed fraud and unnecessary declines.
Add calibration, subgroup and temporal evaluation, access control, rate
limits, explicit operational alerting, and a process for reviewing human
feedback and confirmed outcomes. Define monitoring ownership and log retention
before collecting production features.

If serving latency or throughput becomes a concern, measure the full request
path first—including request validation, feature engineering, model
inference, and response handling—before choosing an optimization. In this
project, the optional ONNX export covers the classifier only; the Python
feature-engineering pipeline still runs separately. Batch scoring can improve
throughput when the caller can submit multiple records together. For systems
that require costly rolling or aggregate features, an online feature store
can avoid recalculating them for every request, but use the same feature
definitions for offline training and online serving to prevent training-serving
skew.

For an inference optimization, start with a representative benchmark of
latency (including tail latency), throughput, and resource use. If profiling
shows preprocessing is material, evaluate whether the fitted preprocessing
steps can be exported with a supported ONNX converter; custom transformations
may need to remain in Python, and parity tests should cover the complete
prediction path. Polars or PyArrow can be evaluated for a measured,
dataframe-heavy bottleneck, but replacing pandas adds complexity and is not
automatically faster for this service's small, single-record requests. Likewise,
micro-batching can increase throughput under load but adds queuing delay and
should be introduced only when its latency trade-off is acceptable. Treelite
or a dedicated server such as Triton are possible options for specific
tree-inference or serving workloads, not assumed upgrades to the current
ONNX Runtime/FastAPI design.

The most important end-to-end principle is consistency: preserve chronology
when evaluating, keep feature engineering with the model, promote only
validated artifacts, and monitor inputs without confusing drift for labelled
model performance. With those pieces in place, the repository offers a clear
path from a local example to a controlled, testable model-serving workflow.

## Deploy locally from PowerShell

From the repository root, first run the model gate, then start the Compose
service only if validation succeeds:

```powershell
python mlops/model_gate.py
if ($LASTEXITCODE -ne 0) { throw "Model validation failed; deployment stopped." }
docker compose --project-name fraud-model up --build -d
docker compose --project-name fraud-model ps
```

The API binds to `http://127.0.0.1:8000` by default; visit
`http://127.0.0.1:8000/docs` to inspect the API. Docker Desktop (or another
Docker Engine) must already be running. If Compose reports that it cannot open
the `dockerDesktopLinuxEngine` pipe on Windows, the Docker engine is not
available to the CLI; start Docker Desktop and verify `docker version` before
retrying the Compose command. The model gate can succeed independently, but
that does not mean the container has started.
