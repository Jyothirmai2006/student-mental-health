import mlflow

from mlflow.tracking import MlflowClient

from config import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME
)


def main():

    print("=" * 60)
    print("MODEL LIFECYCLE AUTOMATION")
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

            print("No model versions available.")
            return

        latest = max(
            versions,
            key=lambda x: int(x.version)
        )

        print("\nLatest Model Version:")
        print(latest.version)

        print("\nRun ID:")
        print(latest.run_id)

        print("\nModel Name:")
        print(REGISTERED_MODEL_NAME)

        print("\nLifecycle check completed.")

    except Exception as error:

        print("Error:")
        print(error)


if __name__ == "__main__":
    main()