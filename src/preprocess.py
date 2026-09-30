import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Original Kaggle column names -> clean names
COLUMN_MAPPING = {
    "Timestamp": "Timestamp",
    "Choose your gender": "Gender",
    "Age": "Age",
    "What is your course?": "Course",
    "Your current year of Study": "Year_of_Study",
    "What is your CGPA?": "CGPA",
    "Marital status": "Marital_Status",
    "Do you have Depression?": "Depression",
    "Do you have Anxiety?": "Anxiety",
    "Do you have Panic attack?": "Panic_Attack",
    "Did you seek any specialist for a treatment?": "Treatment_Sought"
}


def load_data(file_path):
    print("Reading dataset...")
    df = pd.read_csv(file_path)

    print("Original shape:", df.shape)

    return df


def clean_column_names(df):

    df = df.copy()

    rename_dict = {
        old: new
        for old, new in COLUMN_MAPPING.items()
        if old in df.columns
    }

    df.rename(
        columns=rename_dict,
        inplace=True
    )

    return df


def clean_data(df):

    df = df.copy()

    # -------------------------------------------------
    # Remove Timestamp
    # -------------------------------------------------

    if "Timestamp" in df.columns:
        df.drop(
            columns=["Timestamp"],
            inplace=True
        )

    # -------------------------------------------------
    # Clean text values
    # -------------------------------------------------

    for column in df.columns:

        if pd.api.types.is_string_dtype(df[column]):

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    # -------------------------------------------------
    # Gender
    # -------------------------------------------------

    if "Gender" in df.columns:

        df["Gender"] = (
            df["Gender"]
            .astype("string")
            .str.strip()
            .str.title()
        )

    # -------------------------------------------------
    # Age
    # -------------------------------------------------

    if "Age" in df.columns:

        df["Age"] = pd.to_numeric(
            df["Age"],
            errors="coerce"
        )

    # -------------------------------------------------
    # Year of Study
    # -------------------------------------------------

    if "Year_of_Study" in df.columns:

        df["Year_of_Study"] = (
            df["Year_of_Study"]
            .astype("string")
            .str.lower()
            .str.strip()
            .str.replace(
                "year",
                "",
                regex=False
            )
            .str.strip()
        )

        df["Year_of_Study"] = pd.to_numeric(
            df["Year_of_Study"],
            errors="coerce"
        )

    # -------------------------------------------------
    # CGPA
    # -------------------------------------------------

    if "CGPA" in df.columns:

        def convert_cgpa(value):

            if pd.isna(value):
                return np.nan

            value = str(value).strip()

            # Example:
            # 3.00 - 3.49
            if "-" in value:

                parts = value.split("-")

                try:

                    low = float(parts[0].strip())
                    high = float(parts[1].strip())

                    return (low + high) / 2

                except ValueError:

                    return np.nan

            # Direct numeric value
            try:

                return float(value)

            except ValueError:

                return np.nan

        df["CGPA"] = df["CGPA"].apply(
            convert_cgpa
        )

    # -------------------------------------------------
    # Yes / No columns
    # -------------------------------------------------

    binary_columns = [
        "Marital_Status",
        "Depression",
        "Anxiety",
        "Panic_Attack",
        "Treatment_Sought"
    ]

    for column in binary_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .str.title()
            )

    # -------------------------------------------------
    # Fill numeric missing values
    # -------------------------------------------------

    numeric_columns = [
        "Age",
        "Year_of_Study",
        "CGPA"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

            df[column] = df[column].fillna(
                df[column].median()
            )

    return df


def prepare_data(file_path):

    df = load_data(file_path)

    df = clean_column_names(df)

    df = clean_data(df)

    print("\nCleaned columns:")
    print(list(df.columns))

    print("\nData types:")
    print(df.dtypes)

    return df


def build_preprocessor(X):

    # IMPORTANT:
    # Use pandas' is_numeric_dtype instead of
    # dtype != "object"

    numerical_features = [
        column
        for column in X.columns
        if pd.api.types.is_numeric_dtype(X[column])
    ]

    categorical_features = [
        column
        for column in X.columns
        if not pd.api.types.is_numeric_dtype(X[column])
    ]

    print("\nNumerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    # -------------------------------------------------
    # Numerical Pipeline
    # -------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # -------------------------------------------------
    # Categorical Pipeline
    # -------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # -------------------------------------------------
    # Combine Pipelines
    # -------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor