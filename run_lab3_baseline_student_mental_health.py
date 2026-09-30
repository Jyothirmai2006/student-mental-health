import sys
from pathlib import Path

# Add src folder to Python path BEFORE importing from it
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from train import train_model
from evaluate import evaluate_model
from validate_outputs import validate_outputs

def main():
    print("=" * 60)
    print("LAB 3 - STUDENT MENTAL HEALTH BASELINE")
    print("=" * 60)
    train_model()
    evaluate_model()
    validate_outputs()
    print("\nLAB 3 COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
