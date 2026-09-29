"""
Load model, scaler, label encoder, dan target encoder
Preprocess data baru
Return hasil prediksi
"""

import pickle
import pandas as pd

print("Start loading inference")

from src.preprocessing import Preprocessing

print("Import preprocessing success!")

class CreditScorePredictor:

    def __init__(self):

        with open("models/best_model.pkl", "rb") as f:
            self.model = pickle.load(f)

        with open("models/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open("models/target_encoder.pkl", "rb") as f:
            self.target_encoder = pickle.load(f)

        with open("models/features.pkl", "rb") as f:
            self.features = pickle.load(f)

        self.preprocessor = Preprocessing()

        self.preprocessor.load_encoders(
            "models/label_encoders.pkl"
        )

        print("Artifacts loaded successfully")
        print(f"Number of features: {len(self.features)}")

    def predict(self, input_df):
        
        # Preprocessing
        df = self.preprocessor.transform(input_df)
        
        df = df[self.features]

        # debug
        print("\n=== INFERENCE DATA ===")
        print(df.head())

        print("\n=== DATA TYPES ===")
        print(df.dtypes)

        object_cols = df.select_dtypes(include="object").columns

        if len(object_cols) > 0:
            raise ValueError(
                f"Masih terdapat kolom bertipe object: {list(object_cols)}")

        # scaling hanya untuk Logistic Regression
        model_name = type(self.model).__name__
        if model_name == "LogisticRegression":
            df = self.scaler.transform(df)

        # prediction
        prediction = self.model.predict(df)

        result = self.target_encoder.inverse_transform(prediction)

        return result[0]


if __name__ == "__main__":

    print("Testing inference pipeline...")

    predictor = CreditScorePredictor()

    print("Inference pipeline loaded successfully!")