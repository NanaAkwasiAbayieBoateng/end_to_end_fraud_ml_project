import argparse
import json
import tarfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy a gated fraud model to a SageMaker endpoint.")
    parser.add_argument("--model", type=Path, default=Path("models/fraud_model_pipeline.joblib"))
    parser.add_argument("--image-uri", required=True, help="Gated custom inference image in ECR.")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN.")
    parser.add_argument("--bucket", required=True, help="S3 bucket for the model archive.")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--instance-type", default="ml.m5.large")
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Validated model artifact not found: {args.model}")

    region = boto3.Session().region_name
    if not region:
        raise RuntimeError("AWS region is not configured.")
    client = boto3.client("sagemaker", region_name=region)
    s3 = boto3.client("s3", region_name=region)
    endpoint = args.endpoint_name
    suffix = args.version[:12]
    model_name = f"{endpoint}-{suffix}"
    config_name = f"{endpoint}-{suffix}"
    archive = args.model.parent / "sagemaker-model.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(args.model, arcname="fraud_model_pipeline.joblib")
    object_key = f"fraud-models/{endpoint}/{suffix}/model.tar.gz"
    s3.upload_file(str(archive), args.bucket, object_key)
    model_data = f"s3://{args.bucket}/{object_key}"
    client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=args.role_arn,
        PrimaryContainer={
            "Image": args.image_uri,
            "ModelDataUrl": model_data,
            "Environment": {
                "MODEL_PATH": "/opt/ml/model/fraud_model_pipeline.joblib",
                "MODEL_VERSION": suffix,
                "PORT": "8080",
                "ONNX_MODEL_PATH": "/app/models/fraud_classifier.onnx",
            },
        },
    )
    client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": args.instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
    )
    try:
        client.describe_endpoint(EndpointName=endpoint)
    except ClientError as error:
        if error.response["Error"]["Code"] not in {"ValidationException", "ResourceNotFound"}:
            raise
        client.create_endpoint(EndpointName=endpoint, EndpointConfigName=config_name)
    else:
        client.update_endpoint(EndpointName=endpoint, EndpointConfigName=config_name)
    client.get_waiter("endpoint_in_service").wait(
        EndpointName=endpoint,
        WaiterConfig={"Delay": 30, "MaxAttempts": 40},
    )
    print(
        json.dumps(
            {
                "endpoint": endpoint,
                "model": model_name,
                "model_data": model_data,
                "status": "deployment_started",
            }
        )
    )


if __name__ == "__main__":
    main()
