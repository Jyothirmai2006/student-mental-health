# 🟢 FIX 1: Turn off strict skops serialization to bypass numpy.dtype exceptions
import os
os.environ["MLFLOW_SKLEARN_SUPPORT_SKOPS"] = "False"

import sys
from pathlib import Path

# 🟢 FIX 2: Add src folder to Python path BEFORE importing from it
sys.path.append(
    str(Path(__file__).resolve().parent / "src")
)

import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from config import (
    RAW_DATA_FILE,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    MODEL_FILE,
    create_directories
)

from preprocess import prepare_data, build_preprocessor
from models import create_random_forest


def main():

    print("=" * 60)
    print("LAB 5 - STUDENT MENTAL HEALTH ML PIPELINE")
    print("=" * 60)

    create_directories()

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    df = prepare_data(
        RAW_DATA_FILE
    )

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN].map({
        "No": 0,
        "Yes": 1
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    preprocessor = build_preprocessor(
        X_train
    )

    model = create_random_forest(
        preprocessor
    )

    with mlflow.start_run(
        run_name="student-mental-health-random-forest"
    ):

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        mlflow.log_param(
            "model",
            "RandomForestClassifier"
        )

        mlflow.log_param(
            "n_estimators",
            200
        )

        mlflow.log_param(
            "max_depth",
            8
        )

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # 🟢 FIX 3: Explicitly set serialization parameters to ensure no security errors
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            serialization_format="cloudpickle",
            skops_trusted_types=["numpy.dtype", "numpy.core.multiarray.scalar"]
        )

        joblib.dump(
            model,
            MODEL_FILE
        )

        print("\nPipeline Results")
        print("------------------------")
        print("Accuracy :", accuracy)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1 Score :", f1)

        print("\nModel saved to:")
        print(MODEL_FILE)

    print("\nLAB 5 COMPLETED.")


if __name__ == "__main__":
    main()
