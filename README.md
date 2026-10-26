# Fraud model serving and delivery

## API

Build the image after placing a validated model and its ONNX classifier under
`models/`:

```powershell
python -m pip install -r requirements-ci.txt
python mlops/model_gate.py
python mlops/export_onnx.py
docker build -t fraud-api .
docker run --rm -p 8000:8000 fraud-api
```

`POST /predict` (also SageMaker's `/invocations`) accepts the following
features. Numeric and categorical inputs can be `null` or omitted; dates use
ISO-8601 strings. Unknown fields and non-finite numbers are rejected.

```json
{
  "items_purchased": 2,
  "transaction_amount": 24.5,
  "avg_7day_spend": 18,
  "account_age_months": 8,
  "mcc_code": "Retail_A",
  "product_category": "Clothing",
  "last_login_time": "2023-01-01T10:00:00",
  "transaction_datetime": "2023-01-01T10:30:00"
}
```

The response includes a fraud probability, binary decision, model version,
request ID, and measured prediction latency. Structured prediction logs include
the input features and are written to standard output for CloudWatch ingestion.
Treat these logs as sensitive data and restrict access/retention accordingly.
`/health` and `/ready` are health-check routes (`/ping` is SageMaker's health
route).

The optional ONNX export compiles the XGBoost estimator only. The existing
scikit-learn/pandas feature engineering remains in Python and is applied before
the ONNX Runtime prediction. This is not a fully compiled preprocessing graph;
the gated ECS and SageMaker deployments enable the ONNX classifier. Benchmark
it on representative traffic before enabling it in production.

## CI/CD and promotion

`.github/workflows/model-ci-cd.yml` runs API/drift tests on pushes, trains a
chronologically split candidate, compares holdout ROC-AUC against
`models/baseline_auc.json`, and blocks promotion unless the candidate strictly
beats the baseline. The model artifact is uploaded only after it passes. Merges
to `main` build/push the model image to ECR and update the ECS service; SageMaker
deployment is an optional parallel target.

Configure repository variables `AWS_REGION`, `ECR_REPOSITORY`,
`ECS_TASK_FAMILY`, `ECS_CONTAINER_NAME`, `ECS_CLUSTER`, and `ECS_SERVICE`; set
`AWS_DEPLOY_ROLE_ARN` as a secret. The GitHub OIDC role needs ECR push, ECS
register/update/describe, and `iam:PassRole` permissions for the task roles.
The ECS task should send container stdout to a CloudWatch log group and use
`/ready` as its target-group health check.

To enable SageMaker deployment, set repository variable `DEPLOY_SAGEMAKER` to
`true`, and configure `SAGEMAKER_MODEL_BUCKET` and `SAGEMAKER_ENDPOINT_NAME`;
set `SAGEMAKER_EXECUTION_ROLE_ARN` as a secret. The deployment creates or
updates a real-time endpoint from the gated model artifact and waits for the
endpoint to become healthy. The SageMaker role needs access to the model S3
object and permission to pull the ECR image.

The AUC value in `baseline_auc.json` is a starter threshold, not a measured
production baseline. Replace it with the current champion's holdout metric
using the same dataset version, chronological split, and metric calculation.
Do not lower the baseline simply to make a candidate pass.

## Drift monitoring and retraining signal

`deployment/drift-schedule.yaml` provisions an EventBridge Scheduler task that
runs `mlops/drift_check.py` daily. It compares recent CloudWatch prediction
features with aggregate reference distributions (numeric/date PSI and
categorical PSI), alerts an SNS topic when any feature reaches PSI 0.2, and
skips checks with fewer than 100 records. The task role requires
`logs:FilterLogEvents` and `sns:Publish`; the scheduler role needs `ecs:RunTask`
and `iam:PassRole`. Supply the cluster, task definition, subnet, security group,
log group, topic, and role parameters when deploying the template.

For an automated retraining signal, configure the ECS task definition with
`GITHUB_REPOSITORY` and a `DRIFT_DISPATCH_TOKEN` secret that can dispatch
repository events. The drift checker emits `model-drift-detected`; the workflow
re-trains and applies the same AUC gate. A drift alert does not bypass the
promotion gate or merge-to-main deployment rule. The sample workflow uses the
checked-in training CSV, so production must refresh it from the labeled
training-data source before retraining; unlabeled drift observations are never
treated as labels. Grant the ECS execution role access to Secrets Manager when
the dispatch token is injected as a task-definition secret.

Run tests locally with `pytest -q`. `mlops/export_onnx.py` can export a
standalone ONNX classifier for latency comparisons; the API uses it when
`ONNX_MODEL_PATH` points to that file.
