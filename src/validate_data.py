import pandas as pd


REQUIRED_COLUMNS = [
    "Gender",
    "Age",
    "Course",
    "Year_of_Study",
    "CGPA",
    "Marital_Status",
    "Depression",
    "Anxiety",
    "Panic_Attack",
    "Treatment_Sought"
]


def validate_data(df):

    print("\n========== DATA VALIDATION ==========")

    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print("Missing columns:", missing_columns)
        return False

    if df.empty:
        print("Dataset is empty.")
        return False

    print("Required columns: PASS")

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nTarget distribution:")
    print(df["Depression"].value_counts())

    print("\nData validation successful.")

    return True


if __name__ == "__main__":

    from config import RAW_DATA_FILE
    from preprocess import prepare_data

    df = prepare_data(RAW_DATA_FILE)

    validate_data(df)