import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from config import (
    RAW_DATA_FILE,
    MODEL_FILE,
    METRICS_FILE,
    PREDICTIONS_FILE,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE
)

from preprocess import prepare_data


def evaluate_model():

    print("\n========== MODEL EVALUATION ==========")

    df = prepare_data(RAW_DATA_FILE)

    X = df.drop(columns=[TARGET_COLUMN])

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

    model = joblib.load(MODEL_FILE)

    predictions = model.predict(X_test)

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

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    with open(METRICS_FILE, "w") as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    result = X_test.copy()
    result["actual"] = y_test.values
    result["predicted"] = predictions

    result.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    print("\nMetrics:")
    print(json.dumps(metrics, indent=4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_test,
        predictions
    ))

    print("\nEvaluation completed.")

    return metrics


if __name__ == "__main__":
    evaluate_model()