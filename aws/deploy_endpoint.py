"""
Deploy the Credit Score XGBoost model to a SageMaker real-time endpoint.
"""

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel


BUCKET = "your-bucket-name"
MODEL_S3_KEY = "credit-score/model.tar.gz"
ENDPOINT_NAME = "credit-score-endpoint"

REGION = "us-east-1"
INSTANCE_TYPE = "ml.m5.large"
FRAMEWORK_VERSION = "1.4-2"

def get_lab_role_arn():
    iam = boto3.client("iam")
    return iam.get_role(RoleName="LabRole")["Role"]["Arn"]

def main():
    boto3.setup_default_session(region_name=REGION)
    session = sagemaker.Session()
    role = get_lab_role_arn()
    model_uri = f"s3://{BUCKET}/{MODEL_S3_KEY}"

    print(f"Role          : {role}")
    print(f"Model URI     : {model_uri}")
    print(f"Endpoint Name : {ENDPOINT_NAME}")

    model = SKLearnModel(
        model_data=model_uri,
        role=role,
        entry_point="inference.py",
        source_dir="src",
        dependencies=["src"],
        framework_version=FRAMEWORK_VERSION,
        sagemaker_session=session
    )

    print("\nDeploying endpoint...")

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME
    )

    print("\nEndpoint deployed successfully.")

    sample = {
        "Month": "March",
        "Age": 35,
        "Occupation": "Engineer",
        "Annual_Income": 50000,
        "Monthly_Inhand_Salary": 3800,
        "Num_Bank_Accounts": 4,
        "Num_Credit_Card": 5,
        "Interest_Rate": 8,
        "Num_of_Loan": 2,
        "Type_of_Loan": "Personal Loan",
        "Delay_from_due_date": 3,
        "Num_of_Delayed_Payment": 2,
        "Changed_Credit_Limit": 10.5,
        "Num_Credit_Inquiries": 4,
        "Credit_Mix": "Good",
        "Outstanding_Debt": 450,
        "Credit_Utilization_Ratio": 28.4,
        "Payment_of_Min_Amount": "Yes",
        "Total_EMI_per_month": 120,
        "Amount_invested_monthly": 350,
        "Payment_Behaviour": "High_spent_Medium_value_payments",
        "Monthly_Balance": 900,
        "Credit_History_Age": "15 Years and 4 Months"
    }

    runtime = boto3.client(
        "sagemaker-runtime", region_name=REGION
    )

    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=str(sample).replace("'", '"')
    )

    print("\nPrediction")

    print(response["Body"].read().decode("utf-8"))

    print(f"\nEndpoint '{ENDPOINT_NAME}' is live in {REGION}.")

if __name__ == "__main__":
    main()