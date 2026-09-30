# 🟢 FIX: Turn off strict skops serialization to bypass numpy.dtype exceptions
import os
os.environ["MLFLOW_SKLEARN_SUPPORT_SKOPS"] = "False"

import sys
from pathlib import Path

# Add src folder to Python path
sys.path.append(
    str(Path(__file__).resolve().parent / "src")
)

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
    EXPERIMENT_NAME
)

from preprocess import (
    prepare_data,
    build_preprocessor
)

from models import create_logistic_regression


def main():

    print("=" * 60)
    print("LAB 4 - MLFLOW EXPERIMENT TRACKING")
    print("=" * 60)

    # -----------------------------------------
    # MLflow configuration
    # -----------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # -----------------------------------------
    # Load dataset
    # -----------------------------------------

    print("\nLoading dataset...")

    df = prepare_data(
        RAW_DATA_FILE
    )

    # -----------------------------------------
    # Separate X and y
    # -----------------------------------------

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN].map({
        "No": 0,
        "Yes": 1
    })

    # -----------------------------------------
    # Train/Test split
    # -----------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # -----------------------------------------
    # Preprocessing
    # -----------------------------------------

    preprocessor = build_preprocessor(
        X_train
    )

    # -----------------------------------------
    # Create model
    # -----------------------------------------

    model = create_logistic_regression(
        preprocessor
    )

    # -----------------------------------------
    # Start MLflow run
    # -----------------------------------------

    with mlflow.start_run(
        run_name="student-mental-health-baseline"
    ):

        print("\nTraining model...")

        model.fit(
            X_train,
            y_train
        )

        # -------------------------------------
        # Predictions
        # -------------------------------------

        predictions = model.predict(
            X_test
        )

        # -------------------------------------
        # Metrics
        # -------------------------------------

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

        # -------------------------------------
        # Log parameters
        # -------------------------------------

        mlflow.log_param(
            "model",
            "LogisticRegression"
        )

        mlflow.log_param(
            "test_size",
            TEST_SIZE
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE
        )

        # -------------------------------------
        # Log metrics
        # -------------------------------------

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

        # -------------------------------------
        # Log model (FIXED PARAMETERS)
        # -------------------------------------

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            serialization_format="cloudpickle",
            skops_trusted_types=["numpy.dtype", "numpy.core.multiarray.scalar"]
        )

        # -------------------------------------
        # Display results
        # -------------------------------------

        print("\n" + "=" * 50)
        print("MLFLOW RESULTS")
        print("=" * 50)

        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )

        print("\nRun ID:")

        print(
            mlflow.active_run().info.run_id
        )

    print("\n" + "=" * 60)
    print("LAB 4 COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
