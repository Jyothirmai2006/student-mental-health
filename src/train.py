# 🟢 FIX: Turn off strict skops serialization to allow numpy.dtype types
import os
os.environ["MLFLOW_SKLEARN_SUPPORT_SKOPS"] = "False"

import joblib
from sklearn.model_selection import train_test_split

from config import (
    RAW_DATA_FILE,
    MODEL_FILE,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    create_directories
)

from preprocess import prepare_data, build_preprocessor
from models import create_logistic_regression


def train_model():

    create_directories()

    print("Loading dataset...")

    df = prepare_data(RAW_DATA_FILE)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Convert target to 0/1
    y = y.map({
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

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    preprocessor = build_preprocessor(X_train)

    model = create_logistic_regression(
        preprocessor
    )

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("\nModel trained successfully.")
    print("Model saved at:")
    print(MODEL_FILE)

    return model, X_test, y_test


if __name__ == "__main__":
    train_model()
