import argparse
import sys
from pathlib import Path

import joblib
import onnxmltools
import numpy as np
import onnxruntime
import pandas as pd
from onnxmltools.convert.common.data_types import FloatTensorType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the XGBoost estimator; pandas feature engineering stays in the sklearn pipeline."
    )
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "fraud_model_pipeline.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "fraud_classifier.onnx")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "fraud_dataset.csv")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Validation data not found: {args.data}")
    pipeline = joblib.load(args.model)
    selector = pipeline.named_steps["selector"]
    feature_count = len(selector.transformers_[0][2])
    classifier = pipeline.named_steps["classifier"]
    converted = onnxmltools.convert_xgboost(
        classifier,
        initial_types=[("features", FloatTensorType([None, feature_count]))],
        target_opset=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(converted.SerializeToString())
    sample = pd.read_csv(args.data).drop(columns=["target"], errors="ignore").head(100)
    engineered = pipeline.named_steps["preprocessor"].transform(sample)
    selected = np.asarray(selector.transform(engineered), dtype=np.float32)
    session = onnxruntime.InferenceSession(
        str(args.output), providers=["CPUExecutionProvider"]
    )
    outputs = session.run(None, {session.get_inputs()[0].name: selected})
    onnx_probabilities = outputs[1]
    if isinstance(onnx_probabilities[0], dict):
        actual = np.asarray(
            [row.get(1, row.get("1")) for row in onnx_probabilities],
            dtype=float,
        )
    else:
        actual = np.asarray(onnx_probabilities)[:, 1]
    expected = classifier.predict_proba(selected)[:, 1]
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 1e-5:
        raise RuntimeError(
            f"ONNX probability parity check failed; max absolute error={max_error:.8g}."
        )
    print(
        f"Exported ONNX classifier to {args.output}; "
        f"max probability error={max_error:.8g}. "
        "Feature engineering remains in the sklearn/pandas pipeline."
    )


if __name__ == "__main__":
    main()
