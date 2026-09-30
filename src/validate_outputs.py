from pathlib import Path

from config import (
    MODEL_FILE,
    METRICS_FILE,
    PREDICTIONS_FILE
)


def validate_outputs():

    print("\n========== OUTPUT VALIDATION ==========")

    files = [
        MODEL_FILE,
        METRICS_FILE,
        PREDICTIONS_FILE
    ]

    success = True

    for file in files:

        if Path(file).exists():
            print("PASS:", file)
        else:
            print("FAIL:", file)
            success = False

    if success:
        print("\nAll outputs are available.")
    else:
        print("\nSome outputs are missing.")

    return success


if __name__ == "__main__":
    validate_outputs()