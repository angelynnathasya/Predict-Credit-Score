"""
SageMaker inference script for Credit Score Classification
"""

import os
import json
import joblib
import pandas as pd

from preprocessing import Preprocessing

def model_fn(model_dir):
    """
    Load model artifacts from SageMaker model directory.
    """

    model = joblib.load(
        os.path.join(model_dir, "model.joblib")
    )

    target_encoder = joblib.load(
        os.path.join(model_dir, "target_encoder.joblib")
    )

    features = joblib.load(
        os.path.join(model_dir, "features.joblib")
    )

    label_encoders = joblib.load(
        os.path.join(model_dir, "label_encoders.joblib")
    )

    preprocessor = Preprocessing()
    preprocessor.label_encoders = label_encoders

    artifacts = {
        "model": model,
        "target_encoder": target_encoder,
        "features": features,
        "preprocessor": preprocessor
    }

    print("Model artifacts loaded successfully.")

    return artifacts


def input_fn(request_body, content_type):
    """
    Parse input request.
    """

    if content_type == "application/json":

        data = json.loads(request_body)

        if isinstance(data, dict):
            return pd.DataFrame([data])

        elif isinstance(data, list):
            return pd.DataFrame(data)

        else:
            raise ValueError("Invalid JSON format.")

    raise ValueError(
        f"Unsupported content type: {content_type}"
    )


def predict_fn(input_data, artifacts):
    """
    Preprocess input and generate prediction.
    """

    preprocessor = artifacts["preprocessor"]

    model = artifacts["model"]

    target_encoder = artifacts["target_encoder"]

    features = artifacts["features"]

    df = preprocessor.transform(input_data)

    df = df[features]

    prediction = model.predict(df)

    prediction = target_encoder.inverse_transform(prediction)

    return prediction.tolist()


def output_fn(prediction, accept):
    """
    Format prediction response.
    """

    if accept == "application/json":

        return json.dumps(
            {
                "prediction": prediction
            }
        ), accept

    raise ValueError(
        f"Unsupported accept type: {accept}"
    )