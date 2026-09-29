"""Evaluate XGBoost Tuned model using the best hyperparameters."""

import pandas as pd

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix)

class ModelEvaluator:

    def evaluate_model(self, model, X_train, y_train, X_test, y_test) -> dict:
        
        # Prediction
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        test_prob = model.predict_proba(X_test)

        results = {

            "Train Accuracy":
                accuracy_score(
                    y_train,
                    train_pred
                ),

            "Test Accuracy":
                accuracy_score(
                    y_test,
                    test_pred
                ),

            "Precision":
                precision_score(
                    y_test,
                    test_pred,
                    average="weighted"
                ),

            "Recall":
                recall_score(
                    y_test,
                    test_pred,
                    average="weighted"
                ),

            "F1-Score":
                f1_score(
                    y_test,
                    test_pred,
                    average="weighted"
                ),

            "ROC-AUC":
                roc_auc_score(
                    y_test,
                    test_prob,
                    multi_class="ovr",
                    average="weighted"
                )

        }

        print("\nEvaluation completed.")
        print(results)

        return results

    def classification_report_result(self, model, X_test, y_test):
        pred = model.predict(X_test)
        return classification_report(y_test, pred)
     

    def confusion_matrix_result(self, model, X_test, y_test):
        pred = model.predict(X_test)
        return confusion_matrix(y_test, pred)