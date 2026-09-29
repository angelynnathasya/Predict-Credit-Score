import pandas as pd
import numpy as np
import os
import pickle
import optuna
import mlflow
import mlflow.sklearn

from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, make_scorer
from mlflow import MlflowClient

from src.data_ingestion import DataIngestion
from src.preprocessing import Preprocessing
from src.training_model import ModelTrainer
from src.evaluation import ModelEvaluator

RANDOM_STATE = 42
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
f1_scorer = make_scorer(f1_score, average='weighted')

def main():
    # Load dataset
    print("Loading Dataset...")
    ingestion = DataIngestion("data/data_B.csv")
    df = ingestion.load_data()

    # Preprocessing
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

    # Scaling untuk model logreg
    print("Scaling Data for Logistic Regression Model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train baseline models
    trainer = ModelTrainer(random_state=RANDOM_STATE)

    print("\nTraining Baseline Models...")
    lr_baseline = trainer.train_logistic_regression(X_train_scaled, y_train)
    rf_baseline = trainer.train_random_forest(X_train, y_train)
    xgb_baseline = trainer.train_xgboost(X_train, y_train)

    # Evaluasi baseline models
    evaluator = ModelEvaluator()

    lr_result = evaluator.evaluate_model(lr_baseline, X_train_scaled, y_train, X_test_scaled, y_test, "Logistic Regression")
    rf_result = evaluator.evaluate_model(rf_baseline, X_train, y_train, X_test, y_test, "Random Forest")
    xgb_result = evaluator.evaluate_model(xgb_baseline, X_train, y_train, X_test, y_test, "XGBoost")

    baseline_df = evaluator.create_result_dataframe(
        [
            lr_result,
            rf_result,
            xgb_result
        ]
    )

    print("\n=== BASELINE RESULTS ===")
    print(baseline_df)

    # Optuna tuning
    print("Hyperparameter Tuning (Optuna)...")
    def objective_lr(trial):
        """Optuna objective for Logistic Regression"""

        params = {
            'C': trial.suggest_float('C', 0.0001, 100, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
            'max_iter': trial.suggest_int('max_iter', 500, 3000)
        }

        model = trainer.train_logistic_regression(X_train_scaled, y_train, params=params)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring=f1_scorer, n_jobs=1)

        return scores.mean()

    study_lr = optuna.create_study(direction="maximize", study_name="Logistic Regression", sampler=TPESampler(seed=RANDOM_STATE))
    study_lr.optimize(objective_lr, n_trials=30, show_progress_bar=True)


    def objective_rf(trial):
        """Optuna objective for Random Forest"""

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 250, 500),
            'max_depth': trial.suggest_int('max_depth', 10, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
        }

        model = trainer.train_random_forest(X_train, y_train, params=params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=f1_scorer, n_jobs=-1)

        return scores.mean()

    study_rf = optuna.create_study(direction="maximize", study_name="Random Forest", sampler=TPESampler(seed=RANDOM_STATE))
    study_rf.optimize(objective_rf, n_trials=30, show_progress_bar=True)

    def objective_xgb(trial):
        """Optuna objective for XGBoost"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10, log=True)
        }

        model = trainer.train_xgboost(X_train, y_train, params=params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=f1_scorer, n_jobs=-1)

        return scores.mean()

    study_xgb = optuna.create_study(direction="maximize", study_name="XGBoost", sampler=TPESampler(seed=RANDOM_STATE))
    study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=True)

    print("===BEST SCORES===") # best_score => f1_score
    print(f"Logistic Regression Tuned F1:{study_lr.best_value}")
    print(f"Random Forest Tuned F1:{study_rf.best_value}")
    print(f"XGBoost Tuned F1:{study_xgb.best_value}")

    # Train tuned models with best params
    print("Training tuned models with best params...")
    print(f"Best Logistic Regression parameters: {study_lr.best_params}")
    print(f"\nBest Random Forest parameters: {study_rf.best_params}")
    print(f"\nBest XGBoost parameters: {study_xgb.best_params}")

    tuned_lr = trainer.train_logistic_regression(X_train_scaled, y_train, study_lr.best_params)
    tuned_rf = trainer.train_random_forest(X_train, y_train, study_rf.best_params)
    tuned_xgb = trainer.train_xgboost(X_train, y_train, study_xgb.best_params)

    # Evaluation
    tuned_lr_result = evaluator.evaluate_model(tuned_lr, X_train_scaled, y_train, X_test_scaled, y_test, "Logistic Regression Tuned")
    tuned_rf_result = evaluator.evaluate_model(tuned_rf, X_train, y_train, X_test, y_test, "Random Forest Tuned")
    tuned_xgb_result  = evaluator.evaluate_model(tuned_xgb, X_train, y_train, X_test, y_test, "XGBoost Tuned")

    tuned_df = evaluator.create_result_dataframe(
        [
            tuned_lr_result,
            tuned_rf_result,
            tuned_xgb_result
        ]
    )

    print("\n=== TUNED RESULTS ===")
    print(tuned_df)

    # Compare baseline models vs tuned models
    comparison_df = pd.concat(
        [
            baseline_df,
            tuned_df
        ],
        axis=0
    ).reset_index(drop=True)

    comparison_df = comparison_df.sort_values(by="F1-Score", ascending=False)

    print("\n=== BASELINE VS TUNED ===")
    print(comparison_df)

    # Select best model dari tuned models
    all_models = {
        "Logistic Regression Tuned": tuned_lr,
        "Random Forest Tuned": tuned_rf,
        "XGBoost Tuned": tuned_xgb
    }

    best_model_name = tuned_df.loc[tuned_df["F1-Score"].idxmax(), "Model"]
    best_model = all_models[best_model_name]
    print(f"Best Model Selected: {best_model_name}")

    # Confusion matrix
    confusion_matrix = evaluator.confusion_matrix_result(best_model, X_test if "Logistic" not in best_model_name else X_test_scaled, y_test)
    print(confusion_matrix)

    # Classification report
    class_report = evaluator.classification_report_result(best_model, X_test if "Logistic" not in best_model_name else X_test_scaled, y_test)
    print(class_report)

    best_metrics = tuned_df[tuned_df["Model"] == best_model_name].iloc[0]

    print("\n=== BEST MODEL METRICS ===")
    print(f"Accuracy            : {best_metrics['Test Accuracy']}")
    print(f"Precision           : {best_metrics['Precision']}")
    print(f"Recall              : {best_metrics['Recall']}")
    print(f"F1-Score            : {best_metrics['F1-Score']}")
    print(f"ROC-AUC             : {best_metrics['ROC-AUC']}")

    # MLflow logging
    best_run_id = None
    best_model_artifact = None

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Credit Score Classification")

    # Baseline Models
    baseline_models = [
        (lr_result, lr_baseline),
        (rf_result, rf_baseline),
        (xgb_result, xgb_baseline)
    ]

    if mlflow.active_run():
        mlflow.end_run()

    for result, model in baseline_models:

        with mlflow.start_run(run_name=result["Model"]) as run:
            mlflow.log_metrics({
                "accuracy": result["Test Accuracy"],
                "precision": result["Precision"],
                "recall": result["Recall"],
                "f1_score": result["F1-Score"],
                "roc_auc": result["ROC-AUC"]
            })

            mlflow.sklearn.log_model(
                sk_model=model,
                name=result["Model"]
            )

            if result["Model"] == best_model_name:
                best_run_id = run.info.run_id
                best_model_artifact = result["Model"]

    # Tuned Models
    tuned_models = [
        (tuned_lr_result, tuned_lr, study_lr.best_params),
        (tuned_rf_result, tuned_rf, study_rf.best_params),
        (tuned_xgb_result, tuned_xgb, study_xgb.best_params)
    ]

    for result, model, params in tuned_models:

        with mlflow.start_run(run_name=result["Model"]) as run:
            mlflow.log_params(params)

            mlflow.log_metrics({
                "accuracy": result["Test Accuracy"],
                "precision": result["Precision"],
                "recall": result["Recall"],
                "f1_score": result["F1-Score"],
                "roc_auc": result["ROC-AUC"]
            })

            mlflow.sklearn.log_model(
                sk_model=model,
                name=result["Model"]
            )

            if result["Model"] == best_model_name:
                best_run_id = run.info.run_id
                best_model_artifact = result["Model"]

    if best_run_id is None:
        raise ValueError(
            "Best run id is not found"
        )

    # Register best model
    print("\nRegistering Best Model...")
    model_uri = f"runs:/{best_run_id}/{best_model_artifact}"
    print(best_run_id)
    print(best_model_artifact)
    print(model_uri)

    registered_model = mlflow.register_model(model_uri=model_uri, name="CreditScoreBestModel")
    print(f"Model registered successfully. Version: {registered_model.version}")

    # Champion for best model
    client = MlflowClient()
    client.set_registered_model_alias(
        name="CreditScoreBestModel",
        alias="champion",
        version=registered_model.version
    )
    print("Champion alias assigned.")


    # save artifacts
    os.makedirs("models", exist_ok=True)

    with open("models/features.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    preprocessor.save_encoders("models/label_encoders.pkl")

    with open("models/target_encoder.pkl", "wb") as f:
        pickle.dump(target_encoder, f)

    print("Best model Saved")
    print("Scaler Saved")
    print("Encoder Saved")
    print("Target encoder Saved")

if __name__ == "__main__":
    print("Starting Training Pipeline...")
    main()