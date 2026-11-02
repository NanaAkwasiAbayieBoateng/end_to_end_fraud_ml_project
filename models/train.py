import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

# ==============================================================================
# 1. Configuration Constants & Transformer Callables
# (Must stay at module level for joblib/pickle serialization to work)
# ==============================================================================

MISSING_NUMERIC_SUBSTITUTION = -9999.01
MISSING_STRING_SUBSTITUTION = "MISS"
MISSING_DATETIME_SUBSTITUTION = "1900-01-01 00:00:00"
SKOPS_TRUSTED_TYPES = [
    f"{__name__}.{name}"
    for name in (
        "calculate_days_since",
        "cat_to_bool_transformer",
        "create_ratio_features",
        "impute_missing_values",
        "rename_columns_with_suffix",
    )
] + [
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]
SKOPS_TRUSTED_TYPES = [
    f"{__name__}.{name}"
    for name in (
        "calculate_days_since",
        "cat_to_bool_transformer",
        "create_ratio_features",
        "impute_missing_values",
        "rename_columns_with_suffix",
    )
] + [
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


def impute_missing_values(
    df: pd.DataFrame,
    numeric_cols: List[str],
    date_cols: List[str],
    categorical_cols: List[str],
    numeric_sub: float = MISSING_NUMERIC_SUBSTITUTION,
    date_sub: str = MISSING_DATETIME_SUBSTITUTION,
    cat_sub: str = MISSING_STRING_SUBSTITUTION,
) -> pd.DataFrame:
    """Imputes missing values with sentinel domain substitutions."""
    df_out = df.copy()

    for col in numeric_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce").fillna(numeric_sub)

    for col in date_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_datetime(df_out[col], errors="coerce").fillna(pd.to_datetime(date_sub))

    for col in categorical_cols:
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(cat_sub).astype(str)

    return df_out


def rename_columns_with_suffix(df: pd.DataFrame, suffix: str = "_MISS") -> pd.DataFrame:
    """Appends suffix to all dataframe column names."""
    df_out = df.copy()
    df_out.columns = [f"{col}{suffix}" for col in df_out.columns]
    return df_out


def cat_to_bool_transformer(
    df: pd.DataFrame, col_list: List[str], value_list: List[List[Any]]
) -> pd.DataFrame:
    """Encodes categorical membership rules into binary hotkey (hk) indicators."""
    df_out = df.copy()
    for col, values in zip(col_list, value_list):
        if col in df_out.columns:
            new_col_name = f"{col}_hk"
            df_out[new_col_name] = df_out[col].isin(values).astype(int)
    return df_out


def create_ratio_features(
    df: pd.DataFrame,
    numerators: List[str],
    denominators: List[str],
    new_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generates numeric ratio features with zero-division protection."""
    df_out = df.copy()
    if len(numerators) != len(denominators):
        raise ValueError("Numerator and denominator lists must have the same length.")
    if new_names and len(new_names) != len(numerators):
        raise ValueError("New names list must match numerator/denominator length.")

    for i, (num_col, den_col) in enumerate(zip(numerators, denominators)):
        if num_col in df_out.columns and den_col in df_out.columns:
            denom = df_out[den_col]
            num = df_out[num_col]
            ratio = np.where(denom == 0, num, num / denom)
            col_name = new_names[i] if new_names else f"Ratio_{num_col}_to_{den_col}"
            df_out[col_name] = ratio

    return df_out


def calculate_days_since(
    df: pd.DataFrame,
    dt1_col: str,
    dt2_cols: List[str],
    missing_subs_value: str = MISSING_DATETIME_SUBSTITUTION,
    missing_subs_value_2: float = MISSING_NUMERIC_SUBSTITUTION,
) -> pd.DataFrame:
    """Calculates elapsed days between a reference date and target dates."""
    df_out = df.copy()
    target_dt_missing = pd.to_datetime(missing_subs_value)
    ref_series = pd.to_datetime(df_out[dt1_col], errors="coerce")

    for col in dt2_cols:
        target_series = pd.to_datetime(df_out[col], errors="coerce")
        days = (ref_series - target_series).dt.total_seconds() / 86400.0

        new_var_name = "DaysSince_" + col.replace("_MISS", "")
        df_out[new_var_name] = days

        # Apply sentinel numeric substitution when either date was missing
        is_missing = (ref_series == target_dt_missing) | (target_series == target_dt_missing)
        df_out.loc[is_missing, new_var_name] = missing_subs_value_2

    return df_out


# ==============================================================================
# 2. Pipeline Builder
# ==============================================================================

def build_feature_engineering_pipeline(
    numeric_features: List[str],
    date_features: List[str],
    categorical_features: List[str],
    categorical_bool_rules: Dict[str, List[Any]],
    numerators: List[str],
    denominators: List[str],
    new_ratio_names: List[str],
    reference_dt_col: str,
    feature_dt_cols: List[str],
    suffix: str = "_MISS",
) -> Pipeline:
    """Constructs the scikit-learn preprocessing Pipeline."""
    return Pipeline([
        (
            "imputer",
            FunctionTransformer(
                impute_missing_values,
                kw_args={
                    "numeric_cols": numeric_features,
                    "date_cols": date_features,
                    "categorical_cols": categorical_features,
                },
            ),
        ),
        ("renamer", FunctionTransformer(rename_columns_with_suffix, kw_args={"suffix": suffix})),
        (
            "cat_to_bool",
            FunctionTransformer(
                cat_to_bool_transformer,
                kw_args={
                    "col_list": list(categorical_bool_rules.keys()),
                    "value_list": list(categorical_bool_rules.values()),
                },
            ),
        ),
        (
            "ratio_features",
            FunctionTransformer(
                create_ratio_features,
                kw_args={
                    "numerators": [f"{col}{suffix}" for col in numerators],
                    "denominators": [f"{col}{suffix}" for col in denominators],
                    "new_names": new_ratio_names,
                },
            ),
        ),
        (
            "date_diff",
            FunctionTransformer(
                calculate_days_since,
                kw_args={
                    "dt1_col": reference_dt_col,
                    "dt2_cols": feature_dt_cols,
                },
            ),
        ),
    ])


# ==============================================================================
# 3. Main Training & Serialization Function
# ==============================================================================

def train_fraud_model(
    data: Union[pd.DataFrame, str],
    target_col: str = "target",
    datetime_col: str = "transaction_datetime",
    train_ratio: float = 0.70,
    model_params: Optional[Dict[str, Any]] = None,
    save_artifacts: bool = True,
    output_dir: str = "models",
    mlflow_tracking_uri: Optional[str] = None,
    mlflow_experiment_name: str = "fraud-detection",
    serialization_format: str = "auto",
    log_mlflow_run: bool = True,
) -> Dict[str, Any]:
    """Preprocesses data, trains an XGBoost fraud detection pipeline, and exports artifacts.

    Parameters
    ----------
    data : pd.DataFrame or str
        Input dataset or file path to CSV.
    target_col : str, default='target'
        Name of the binary target label column.
    datetime_col : str, default='transaction_datetime'
        Datetime column used for chronological train/test splitting.
    train_ratio : float, default=0.70
        Proportion of chronological data assigned to training.
    model_params : dict, optional
        Custom hyperparameters for XGBClassifier.
    save_artifacts : bool, default=True
        Whether to serialize pipeline and model files to disk.
    output_dir : str, default='models'
        Target directory to save serialized models.
    mlflow_tracking_uri : str, optional
        MLflow tracking URI. Defaults to the MLFLOW_TRACKING_URI environment
        variable or a local SQLite database in the project root.
    mlflow_experiment_name : str, default='fraud-detection'
        MLflow experiment used to track this training run.
    serialization_format : {'auto', 'skops', 'cloudpickle'}, default='auto'
        MLflow model format. 'auto' prefers safer skops serialization and
        falls back to cloudpickle only when skops is unavailable.
    log_mlflow_run : bool, default=True
        Whether to log the training run and model to MLflow.

    Returns
    -------
    dict
        Dictionary containing trained pipeline, test predictions, metrics, and artifact paths.
    """
    supported_serialization_formats = {"auto", "skops", "cloudpickle"}
    if serialization_format not in supported_serialization_formats:
        raise ValueError(
            "serialization_format must be one of "
            f"{sorted(supported_serialization_formats)}."
        )

    # 1. Load and sort data chronologically
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(by=datetime_col).reset_index(drop=True)

    # 2. Chronological Split (prevents future-data leakage)
    split_index = int(len(df) * train_ratio)
    train_df = df.iloc[:split_index].reset_index(drop=True)
    test_df = df.iloc[split_index:].reset_index(drop=True)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print(f"Data split: Train={len(train_df):,} rows, Test={len(test_df):,} rows")

    # 3. Feature Configuration
    numeric_features = ["items_purchased", "transaction_amount", "avg_7day_spend", "account_age_months"]
    date_features = ["last_login_time", "transaction_datetime"]
    categorical_features = ["mcc_code", "product_category"]

    reference_dt_col = "transaction_datetime_MISS"
    feature_dt_cols = ["last_login_time_MISS"]

    numerators = ["transaction_amount", "account_age_months"]
    denominators = ["avg_7day_spend", "items_purchased"]
    new_ratio_names = ["Amount_Ratio", "Account_Ratio"]

    categorical_bool_rules = {
        "mcc_code_MISS": ["Retail_A", "Service_C"],
        "product_category_MISS": ["Luxury_Goods"],
    }

    final_numeric_cols = [
        "items_purchased_MISS",
        "transaction_amount_MISS",
        "avg_7day_spend_MISS",
        "account_age_months_MISS",
        "mcc_code_MISS_hk",
        "product_category_MISS_hk",
        "Amount_Ratio",
        "Account_Ratio",
        "DaysSince_last_login_time",
    ]

    # 4. Pipeline Assembly
    feature_pipeline = build_feature_engineering_pipeline(
        numeric_features=numeric_features,
        date_features=date_features,
        categorical_features=categorical_features,
        categorical_bool_rules=categorical_bool_rules,
        numerators=numerators,
        denominators=denominators,
        new_ratio_names=new_ratio_names,
        reference_dt_col=reference_dt_col,
        feature_dt_cols=feature_dt_cols,
    )

    selector = ColumnTransformer([("select", "passthrough", final_numeric_cols)])

    default_params = {
        "learning_rate": 0.1,
        "n_estimators": 120,
        "seed": 0,
        "objective": "binary:logistic",
        "max_depth": 8,
        "eval_metric": "auc",
        "subsample": 0.7,
        "colsample_bytree": 0.5,
        "max_delta_step": 1,
        "tree_method": "approx",
        "verbosity": 0,
        "gamma": 5,
        "n_jobs": -1,
        "min_child_weight": 2,
    }
    if model_params:
        default_params.update(model_params)

    xgb_clf = XGBClassifier(**default_params)

    final_pipeline = Pipeline([
        ("preprocessor", feature_pipeline),
        ("selector", selector),
        ("classifier", xgb_clf),
    ])

    # 5. Model Fitting
    print("Fitting full pipeline (Feature Engineering + Selector + XGBoost)...")
    final_pipeline.fit(X_train, y_train)
    print("Training complete.")

    # 6. Evaluation
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_proba)
    test_pr_auc = average_precision_score(y_test, y_proba)
    print(f"Test Set Evaluation -> ROC-AUC: {test_roc_auc:.4f} | PR-AUC: {test_pr_auc:.4f}")

    # 7. Model Persistence
    saved_paths = {}
    if save_artifacts:
        os.makedirs(output_dir, exist_ok=True)

        full_pipeline_path = os.path.join(output_dir, "fraud_model_pipeline.joblib")
        preprocessor_path = os.path.join(output_dir, "fraud_model_pipeline_preprocessor.joblib")
        xgb_json_path = os.path.join(output_dir, "fraud_model_pipeline.json")
        pmml_path = os.path.join(output_dir, "fraud_model_pipeline.pmml")

        # Save complete sklearn pipeline
        joblib.dump(final_pipeline, full_pipeline_path)
        # Save preprocessor independently
        joblib.dump(final_pipeline.named_steps["preprocessor"], preprocessor_path)
        # Save XGBoost model in native cross-version JSON
        final_pipeline.named_steps["classifier"].save_model(xgb_json_path)

        # Export the classifier as PMML using the engineered numeric features.
        from sklearn2pmml import PMMLPipeline, sklearn2pmml

        transformed_X_train = feature_pipeline.transform(X_train)
        pmml_pipeline = PMMLPipeline([
            ("classifier", XGBClassifier(**default_params)),
        ])
        pmml_training_X = transformed_X_train[final_numeric_cols]
        pmml_pipeline.fit(pmml_training_X, y_train)
        pmml_pipeline.verify(pmml_training_X.head(10))
        sklearn2pmml(pmml_pipeline, pmml_path, with_repr=True)

        saved_paths = {
            "pipeline_joblib": full_pipeline_path,
            "preprocessor_joblib": preprocessor_path,
            "xgb_json": xgb_json_path,
            "pipeline_pmml": pmml_path,
        }
        print(f"Artifacts successfully saved to: {os.path.abspath(output_dir)}")

    run_id = None
    if log_mlflow_run:
        import mlflow
        import mlflow.sklearn

        tracking_uri = (
            mlflow_tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI")
            or f"sqlite:///{(Path(__file__).resolve().parent.parent / 'mlflow.db').as_posix()}"
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)

        effective_format = serialization_format
        if effective_format in {"auto", "skops"}:
            try:
                import skops.io  # noqa: F401
            except ImportError:
                if effective_format == "skops":
                    raise
                effective_format = "cloudpickle"
                logging.getLogger(__name__).warning(
                    "skops is unavailable; falling back to cloudpickle, which can "
                    "execute arbitrary code when loading untrusted model files."
                )
            else:
                effective_format = "skops"

        with mlflow.start_run(run_name="fraud-model-training") as run:
            mlflow.log_params({f"xgb.{key}": value for key, value in default_params.items()})
            mlflow.log_params({
                "train_ratio": train_ratio,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "mlflow.serialization_format": effective_format,
            })
            mlflow.log_metrics({
                "test_roc_auc": test_roc_auc,
                "test_pr_auc": test_pr_auc,
            })
            mlflow.set_tags({
                "split_strategy": "chronological",
                "model_type": "XGBClassifier",
            })
            from mlflow.models import infer_signature

            input_example = X_test.iloc[:5].copy()
            integer_columns = input_example.select_dtypes(include=["integer"]).columns
            input_example[integer_columns] = input_example[integer_columns].astype(
                "float64"
            )
            signature = infer_signature(
                input_example, final_pipeline.predict(input_example)
            )
            log_model_kwargs = {
                "sk_model": final_pipeline,
                "name": "fraud-model",
                "serialization_format": effective_format,
                "signature": signature,
                "input_example": input_example,
            }
            if effective_format == "skops":
                log_model_kwargs["skops_trusted_types"] = SKOPS_TRUSTED_TYPES
            mlflow.sklearn.log_model(**log_model_kwargs)
            for artifact_path in saved_paths.values():
                mlflow.log_artifact(artifact_path, artifact_path="model_files")

        run_id = run.info.run_id
        print(f"MLflow run: {run_id} (tracking URI: {tracking_uri})")

    return {
        "pipeline": final_pipeline,
        "y_test": y_test,
        "y_proba": y_proba,
        "metrics": {"roc_auc": test_roc_auc, "pr_auc": test_pr_auc},
        "artifact_paths": saved_paths,
        "mlflow_run_id": run_id,
    }


# ==============================================================================
# 4. Example Execution
# ==============================================================================
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from models.train import train_fraud_model as run_training

    data_source = os.path.join(project_root, "data", "fraud_dataset.csv")
    
    if os.path.exists(data_source):
        results = run_training(
            data=data_source,
            output_dir=os.path.join(project_root, "models"),
            save_artifacts=True,
        )
        print("Model execution completed successfully.")
    else:
        
        print(f"Sample data file '{data_source}' not found. Pass an existing DataFrame or CSV path.")
