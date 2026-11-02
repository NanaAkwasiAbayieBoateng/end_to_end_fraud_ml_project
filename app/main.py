import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Annotated, Optional
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field


FEATURE_NAMES = (
    "items_purchased",
    "transaction_amount",
    "avg_7day_spend",
    "account_age_months",
    "mcc_code",
    "product_category",
    "last_login_time",
    "transaction_datetime",
)
NumericFeature = Annotated[Optional[float], Field(allow_inf_nan=False)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items_purchased: NumericFeature = None
    transaction_amount: NumericFeature = None
    avg_7day_spend: NumericFeature = None
    account_age_months: NumericFeature = None
    mcc_code: Optional[str] = None
    product_category: Optional[str] = None
    last_login_time: Optional[datetime] = None
    transaction_datetime: Optional[datetime] = None


class PredictionResponse(BaseModel):
    request_id: str
    fraud_probability: float
    predicted_fraud: bool
    latency_ms: float
    model_version: str


def create_app(model=None, model_version: Optional[str] = None) -> FastAPI:
    state = {
        "model": model,
        "model_version": model_version,
        "onnx": None,
        "prediction_logger": None,
    }

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["model"] is None:
            model_path = Path(os.getenv("MODEL_PATH", "models/fraud_model_pipeline.joblib"))
            if not model_path.is_file():
                raise RuntimeError(f"Model artifact does not exist: {model_path}")
            state["model"] = joblib.load(model_path)
            state["model_version"] = os.getenv("MODEL_VERSION", model_path.stem)
            onnx_path = os.getenv("ONNX_MODEL_PATH")
            if onnx_path:
                if not Path(onnx_path).is_file():
                    raise RuntimeError(f"ONNX model artifact does not exist: {onnx_path}")
                import onnxruntime

                state["onnx"] = onnxruntime.InferenceSession(
                    onnx_path, providers=["CPUExecutionProvider"]
                )
        log_path = os.getenv("PREDICTION_LOG_PATH")
        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            prediction_logger = logging.Logger(f"fraud_predictions_{id(state)}")
            prediction_logger.setLevel(logging.INFO)
            prediction_logger.addHandler(
                RotatingFileHandler(path, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
            )
            prediction_logger.propagate = False
            state["prediction_logger"] = prediction_logger
        try:
            yield
        finally:
            prediction_logger = state["prediction_logger"]
            if prediction_logger is not None:
                for handler in prediction_logger.handlers:
                    handler.close()
                    prediction_logger.removeHandler(handler)

    api = FastAPI(title="Fraud Prediction API", version="1.0.0", lifespan=lifespan)

    @api.get("/", include_in_schema=False)
    def prediction_form():
        return FileResponse(Path(__file__).with_name("index.html"))

    @api.get("/health")
    @api.get("/ping")
    def health():
        return {"status": "ok"}

    @api.get("/ready")
    def readiness():
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return {"status": "ready", "model_version": state["model_version"]}

    @api.post("/predict", response_model=PredictionResponse)
    @api.post("/invocations", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request):
        model_instance = state["model"]
        if model_instance is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")

        started_at = time.perf_counter()
        request_id = request.headers.get("x-request-id", str(uuid4()))
        features = payload.model_dump()
        feature_frame = pd.DataFrame([features], columns=FEATURE_NAMES)
        if state["onnx"] is None:
            probabilities = model_instance.predict_proba(feature_frame)
            probability = float(np.asarray(probabilities)[0, 1])
        else:
            engineered = model_instance.named_steps["preprocessor"].transform(feature_frame)
            selected = model_instance.named_steps["selector"].transform(engineered)
            session = state["onnx"]
            raw_probabilities = session.run(
                None,
                {session.get_inputs()[0].name: np.asarray(selected, dtype=np.float32)},
            )[1][0]
            probability = float(raw_probabilities.get(1, raw_probabilities.get("1"))) if isinstance(
                raw_probabilities, dict
            ) else float(raw_probabilities[1])
        latency_ms = (time.perf_counter() - started_at) * 1000
        version = state["model_version"] or "unknown"

        record = {
            "event": "fraud_prediction",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "model_version": version,
            "fraud_probability": probability,
            "predicted_fraud": probability >= 0.5,
            "latency_ms": round(latency_ms, 3),
            "features": features,
        }
        serialized_record = json.dumps(record, default=str, separators=(",", ":"))
        print(serialized_record, flush=True)
        if state["prediction_logger"] is not None:
            state["prediction_logger"].info(serialized_record)
        return PredictionResponse(
            request_id=request_id,
            fraud_probability=probability,
            predicted_fraud=probability >= 0.5,
            latency_ms=round(latency_ms, 3),
            model_version=version,
        )

    return api


app = create_app()
