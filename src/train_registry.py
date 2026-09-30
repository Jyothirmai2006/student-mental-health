import sys
from pathlib import Path

# 🟢 FIX: Allow this file to find config, preprocess, and models in the parent/src directories
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_FILE,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME
)
from preprocess import prepare_data, build_preprocessor
from models import create_random_forest

def main():
    print("=" * 60)
    print("MODEL REGISTRY - STUDENT MENTAL HEALTH")
    print("=" * 60)

    print("Reading dataset...")
    df = prepare_data(RAW_DATA_FILE)
    print(f"Original shape: {df.shape}\n")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    model = create_random_forest(preprocessor)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="registry-run"):
        model.fit(X_train, y_train)

        # FIXED: Added cloudpickle and explicitly allowed the required types
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format="cloudpickle",
            skops_trusted_types=[
                "numpy.dtype", 
                "numpy.core.multiarray.scalar", 
                "sklearn.tree._tree.Tree"
            ]
        )
        print(f"\nModel registered successfully as: {REGISTERED_MODEL_NAME}")

if __name__ == "__main__":
    main()
