import sys
from pathlib import Path

# Add src folder to Python path BEFORE importing anything else
sys.path.append(
    str(Path(__file__).resolve().parent / "src")
)

# Now Python can safely find the files inside the src directory
from train_registry import main as train_registry

def main():
    print("=" * 60)
    print("LAB 6 - STUDENT MENTAL HEALTH MODEL REGISTRY")
    print("=" * 60)
    
    # Run the model registration pipeline process
    train_registry()
    
    print("\nLAB 6 COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
