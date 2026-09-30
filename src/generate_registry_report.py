import mlflow

from mlflow.tracking import MlflowClient

from config import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME
)


def main():

    print("=" * 60)
    print("MODEL REGISTRY REPORT")
    print("=" * 60)

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient()

    try:

        versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'"
        )

        if not versions:
            print("No registered model versions found.")
            return

        print("\nRegistered Model:")
        print(REGISTERED_MODEL_NAME)

        print("\nModel Versions:")
        print("-" * 50)

        for version in versions:

            print(
                f"Version      : {version.version}"
            )

            print(
                f"Run ID       : {version.run_id}"
            )

            print(
                f"Status       : {version.status}"
            )

            print(
                f"Created      : {version.creation_timestamp}"
            )

            print("-" * 50)

    except Exception as error:

        print("Registry error:")
        print(error)


if __name__ == "__main__":
    main()