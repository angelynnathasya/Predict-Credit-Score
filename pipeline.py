import os
import tarfile
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.data_ingestion import DataIngestion
from src.preprocessing import Preprocessing
from src.training_model import ModelTrainer
from src.evaluation import ModelEvaluator

RANDOM_STATE = 42
ARTIFACT_DIR = "models_artifact"
MODEL_FILENAME = "model.joblib"
TARBALL_PATH = os.path.join(ARTIFACT_DIR, "model.tar.gz")

def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("Loading Dataset...")
    ingestion = DataIngestion("data/data_B.csv")
    df = ingestion.load_data()

    print("Preprocessing Data...") 
    preprocessor = Preprocessing()
    df = preprocessor.fit_transform(df)

    # Encode target (Credit_Score) 
    target_encoder = LabelEncoder()
    X = df.drop("Credit_Score", axis=1)
    y = target_encoder.fit_transform(df["Credit_Score"])
    print("\nTarget Encoding Mapping:")
    print(dict(
            zip(
                target_encoder.classes_,
                target_encoder.transform(
                    target_encoder.classes_
                )
            )
        )
    )

    print("===Feature Columns===")
    print(df.columns.tolist())

    # Split data
    print("Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Training Shape : {X_train.shape}")
    print(f"Test Shape : {X_test.shape}")

    trainer = ModelTrainer(random_state=RANDOM_STATE)
    evaluator = ModelEvaluator()

    print("\nTraining XGBoost Tuned...")

    best_model = trainer.train_xgboost(X_train, y_train)
    metrics = evaluator.evaluate_model(best_model, X_train, y_train, X_test, y_test)

    print("\n=== CONFUSION MATRIX ===")
    print(evaluator.confusion_matrix_result(best_model, X_test, y_test))

    print("\n=== CLASSIFICATION REPORT ===")
    print(evaluator.classification_report_result(best_model, X_test, y_test))

    print("\n=== METRICS ===")
    print(f"Accuracy            : {metrics['Test Accuracy']}")
    print(f"Precision           : {metrics['Precision']}")
    print(f"Recall              : {metrics['Recall']}")
    print(f"F1-Score            : {metrics['F1-Score']}")
    print(f"ROC-AUC             : {metrics['ROC-AUC']}")

    # Save artifacts
    model_path = os.path.join(ARTIFACT_DIR, MODEL_FILENAME)
    joblib.dump(best_model, model_path)

    joblib.dump(
        preprocessor.label_encoders,
        os.path.join(ARTIFACT_DIR, "label_encoders.joblib")
    )

    joblib.dump(
        target_encoder,
        os.path.join(ARTIFACT_DIR, "target_encoder.joblib")
    )

    joblib.dump(
        X.columns.tolist(),
        os.path.join(ARTIFACT_DIR, "features.joblib")
    )

    print("Best Model Saved")
    print("Label Encoders Saved")
    print("Target Encoder Saved")
    print("Feature Names Saved")

    # Package model for SageMaker
    with tarfile.open(TARBALL_PATH, "w:gz") as tar:

        tar.add(
            model_path,
            arcname=MODEL_FILENAME
        )

        tar.add(
            os.path.join(ARTIFACT_DIR, "label_encoders.joblib"),
            arcname="label_encoders.joblib"
        )

        tar.add(
            os.path.join(ARTIFACT_DIR, "target_encoder.joblib"),
            arcname="target_encoder.joblib"
        )

        tar.add(
            os.path.join(ARTIFACT_DIR, "features.joblib"),
            arcname="features.joblib"
        )

    print(f"\nPackaged: {TARBALL_PATH}")

    print("\nNext Steps:")
    print("\n1. Create an S3 Bucket")
    print("aws s3 mb s3://your-bucket-name --region us-east-1")

    print("\n2. Upload model.tar.gz")
    print(
        f"aws s3 cp {TARBALL_PATH} "
        "s3://your-bucket-name/credit-score/model.tar.gz"
    )

    print("\n3. Deploy SageMaker Endpoint")
    print("python deploy_endpoint.py")

    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()