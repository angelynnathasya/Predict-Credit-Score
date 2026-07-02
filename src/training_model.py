"""Training models"""

import xgboost as xgb


class ModelTrainer:

    def __init__(self, random_state=42):
        self.random_state = random_state

    def train_xgboost(self, X_train, y_train):
        """Train XGBoost Tuned model using the best hyperparameters."""
        
        model = xgb.XGBClassifier(

            # Best parameters from Optuna (train.py)
            n_estimators=250,
            max_depth=10,
            learning_rate=0.1205712628744377,
            subsample=0.8795975452591109,
            colsample_bytree=0.7468055921327309,
            min_child_weight=2,
            gamma=0.2904180608409973,
            reg_alpha=3.9676050770529883,
            reg_lambda=0.6358358856676253,

            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=self.random_state
        )

        model.fit(X_train, y_train)

        return model