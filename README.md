# Credit Score Classification — End-to-End Machine Learning Pipeline & AWS Deployment

An end-to-end machine learning project for **Credit Score Classification**, covering data cleaning, exploratory data analysis (EDA), feature engineering, model development, hyperparameter optimization, experiment tracking with MLflow, model inference, local deployment, and cloud deployment using **AWS SageMaker, Amazon EC2, and Streamlit**.

The project was developed with a focus on building a reproducible and deployment-ready machine learning workflow rather than only training a classification model.

---

## Project Overview

The objective of this project is to classify customers into three credit score categories:

- **Good**
- **Standard**
- **Poor**

The project uses customer financial and credit-related information such as income, debt, loan information, payment behavior, credit inquiries, credit utilization, and payment history.

The complete workflow is:

```text
Raw Dataset
     ↓
Data Ingestion
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Categorical Encoding
     ↓
Train-Test Split
     ↓
Baseline Models
     ↓
Optuna Hyperparameter Tuning
     ↓
Model Evaluation
     ↓
MLflow Experiment Tracking
     ↓
Best Model Selection
     ↓
Inference Pipeline
     ↓
Streamlit Web Application
     ↓
AWS SageMaker + EC2 Deployment
```

---

## Key Objectives

This project focuses on:

- Building a complete machine learning classification pipeline
- Applying data cleaning and feature engineering to real-world financial data
- Comparing multiple machine learning algorithms
- Handling class imbalance during model evaluation and hyperparameter optimization
- Performing hyperparameter tuning with Optuna
- Tracking experiments and model artifacts using MLflow
- Creating a reusable inference pipeline
- Deploying the trained model as a web application
- Deploying the machine learning model using AWS SageMaker
- Hosting the Streamlit application on Amazon EC2
- Testing predictions across all target classes

---

## Dataset

The project uses a credit score dataset containing **25,000 records and 29 original columns**.

The target variable is:

```text
Credit_Score
```

with three classes:

```text
Good
Standard
Poor
```

### Target Distribution

| Credit Score | Records | Approx. Proportion |
|---|---:|---:|
| Standard | 13,257 | 53.0% |
| Poor | 7,197 | 28.8% |
| Good | 4,546 | 18.2% |
| **Total** | **25,000** | **100%** |

The target distribution is imbalanced, with **Standard** being the dominant class. This was considered during model evaluation and hyperparameter optimization.

---

## Data Cleaning

The original dataset contains inconsistent data types, missing values, invalid values, and unnecessary identifier columns.

### Data Type Conversion

Several numerical columns were originally stored as strings or objects.

The following integer-related features were converted to numeric integer types:

- `Age`
- `Num_of_Loan`
- `Num_of_Delayed_Payment`

The following features were converted to floating-point values:

- `Annual_Income`
- `Changed_Credit_Limit`
- `Outstanding_Debt`
- `Amount_invested_monthly`
- `Monthly_Balance`

Invalid characters such as `_` were removed before numerical conversion.

### Missing Value Handling

Missing values were identified across the dataset.

Examples of columns containing missing values included:

- `Monthly_Inhand_Salary`
- `Type_of_Loan`
- `Num_of_Delayed_Payment`
- `Changed_Credit_Limit`
- `Num_Credit_Inquiries`
- `Credit_History_Age`
- `Amount_invested_monthly`
- `Monthly_Balance`

Numerical missing values were handled using median imputation, while categorical missing values were handled during preprocessing.

### Invalid Values

Several columns contained invalid or unrealistic values. These were cleaned and converted before model development.

Examples include:

- Invalid numeric strings
- Underscore characters in numerical values
- Invalid occupation values
- Invalid credit mix values
- Invalid payment behavior values

---

## Exploratory Data Analysis

EDA was performed to understand the distribution of the target variable and identify features associated with credit score.

### Categorical Feature Analysis

The categorical variables analyzed included:

- `Month`
- `Occupation`
- `Credit_Mix`
- `Payment_of_Min_Amount`
- `Payment_Behaviour`

The analysis showed that:

- **Credit_Mix** has a strong relationship with `Credit_Score`.
- **Payment_of_Min_Amount** shows noticeable differences across credit score classes.
- **Payment_Behaviour** shows clear differences between Good, Standard, and Poor customers.
- `Month` and `Occupation` show relatively similar credit score distributions across categories.

Overall, credit-related behavior and payment-related features showed stronger relationships with the target than `Month` and `Occupation`.

### Numerical Feature Analysis

The numerical features analyzed included:

- `Age`
- `Num_Bank_Accounts`
- `Num_Credit_Card`
- `Interest_Rate`
- `Delay_from_due_date`
- `Num_Credit_Inquiries`
- `Credit_Utilization_Ratio`
- `Total_EMI_per_month`

Key observations:

- **`Delay_from_due_date`** showed the strongest relationship with `Credit_Score`.
- Customers with a **Poor** credit score tended to have higher payment delays.
- **`Interest_Rate`** showed a noticeable relationship with credit score, with Poor customers tending to have higher interest rates.
- **`Num_Credit_Inquiries`** also showed differences across credit score classes.
- `Age`, `Credit_Utilization_Ratio`, and `Total_EMI_per_month` showed less distinct separation between the classes.

These findings were used to guide feature engineering and model development.

---

## Feature Engineering

Several derived features were created to represent financial relationships that may not be directly captured by the original variables.

### Credit History Transformation

The original `Credit_History_Age` feature was stored as text, for example:

```text
22 Years and 1 Months
```

It was transformed into:

```text
265 months
```

and stored as:

```text
Credit_History_Months
```

### Engineered Features

The following features were created:

| Feature | Description |
|---|---|
| `Debt_vs_Income` | Relationship between outstanding debt and annual income |
| `EMI_burden` | Monthly EMI relative to monthly income |
| `Investment_Habit` | Monthly investment relative to monthly income |
| `Credit_History_Quality` | Credit history duration relative to age |
| `Delayed_Payment_Ratio` | Delayed payments relative to loan-related activity |
| `Inquiry_per_Loan` | Credit inquiries relative to number of loans |
| `Loan_per_Account` | Number of loans relative to bank accounts |
| `Total_Credit_Exposure` | Combined representation of debt and EMI exposure |
| `Financial_Stability_Score` | Derived financial stability indicator |
| `High_Risk_Customer` | Derived indicator representing higher-risk financial behavior |

These engineered variables were included together with the original relevant features during model training.

---

## Feature Encoding

Categorical features were transformed using `LabelEncoder`.

The categorical features used in the final modeling pipeline were:

```text
Month
Occupation
Type_of_Loan
Credit_Mix
Payment_of_Min_Amount
Payment_Behaviour
```

The target variable `Credit_Score` was also label encoded for model training.

The encoders were saved as model artifacts so that the same transformations could be applied during inference.

---

# Machine Learning

## Train-Test Split

The dataset was divided into:

- **80% training data**
- **20% testing data**

A stratified split was used to preserve the class distribution across the training and testing sets.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## Baseline Models

Three machine learning algorithms were developed and compared:

1. **Logistic Regression**
2. **Random Forest**
3. **XGBoost**

StandardScaler was applied to the input features for Logistic Regression because it is a linear model sensitive to feature scale.

Random Forest and XGBoost were trained using the encoded feature values without standardization.

---

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

Because this is a multiclass classification problem with imbalanced classes, weighted averages were used for:

- Precision
- Recall
- F1-Score

ROC-AUC was calculated using the **One-vs-Rest (OVR)** approach.

---

## Baseline Performance

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 1.0000 | 0.7396 | 0.7414 | 0.7396 | 0.7401 | 0.8650 |
| XGBoost | 0.9239 | 0.7250 | 0.7251 | 0.7250 | 0.7248 | 0.8565 |
| Logistic Regression | 0.6447 | 0.6506 | 0.6507 | 0.6506 | 0.6460 | 0.7836 |

### Baseline Analysis

**Logistic Regression**

The training and testing accuracy were very close, indicating limited overfitting. However, its overall performance was lower than the tree-based models. This suggests that the linear model was not able to capture the complex and non-linear relationships present in the credit score data.

**Random Forest**

Random Forest achieved strong baseline performance but had a large gap between training and testing accuracy:

```text
Train Accuracy = 1.0000
Test Accuracy  = 0.7396
```

This indicates a tendency toward overfitting.

**XGBoost**

XGBoost achieved:

```text
Train Accuracy = 0.9239
Test Accuracy  = 0.7250
```

The smaller train-test gap compared with Random Forest indicated better generalization behavior, although its baseline predictive performance was slightly lower.

---

# Hyperparameter Optimization

## Optuna

Hyperparameter tuning was performed using **Optuna** to search for better parameter combinations for all three models.

A **TPE (Tree-structured Parzen Estimator)** sampler was used with:

```python
TPESampler(seed=42)
```

Each model was optimized for **30 trials**.

---

## Cross-Validation Strategy

Because the target classes were imbalanced, **StratifiedKFold** was used:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

This ensures that each fold maintains a similar proportion of the three credit score classes.

The optimization objective was:

```text
Weighted F1-Score
```

Weighted F1-Score was selected because the problem is multiclass and imbalanced, while F1-Score provides a balance between precision and recall.

---

## Hyperparameters Tuned

### Logistic Regression

The following parameters were optimized:

- `C`
- `penalty`
- `solver`
- `max_iter`

### Random Forest

The following parameters were optimized:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `criterion`
- `class_weight`

### XGBoost

The following parameters were optimized:

- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `gamma`
- `reg_alpha`
- `reg_lambda`

---

# Tuned Model Performance

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest Tuned | 0.9892 | 0.7356 | 0.7432 | 0.7356 | 0.7375 | 0.8661 |
| XGBoost Tuned | 0.9198 | 0.7350 | 0.7365 | 0.7350 | 0.7354 | 0.8643 |
| Logistic Regression Tuned | 0.6446 | 0.6504 | 0.6505 | 0.6504 | 0.6458 | 0.7836 |

### Tuning Analysis

**Logistic Regression Tuned**

The tuned Logistic Regression model showed almost no improvement compared with the baseline model. Its performance remained around 0.65 test accuracy and 0.646 F1-Score.

This suggests that hyperparameter optimization could not overcome the limitation of using a linear decision boundary for this dataset.

**Random Forest Tuned**

Random Forest Tuned reduced training accuracy from 1.0000 to approximately 0.9892, indicating reduced model complexity and less overfitting.

However, its test accuracy and F1-Score decreased slightly compared with the baseline model.

**XGBoost Tuned**

XGBoost Tuned improved test accuracy from:

```text
Baseline: 0.7250
Tuned:    0.7350
```

Its F1-Score also improved:

```text
Baseline: 0.7248
Tuned:    0.7354
```

The train-test gap remained relatively small:

```text
Train Accuracy = 0.9198
Test Accuracy  = 0.7350
```

This indicates better generalization behavior compared with the Random Forest models.

---

# Final Model Selection

The final model selected for deployment was:

## XGBoost Tuned

Final evaluation:

| Metric | Score |
|---|---:|
| Train Accuracy | **0.9198** |
| Test Accuracy | **0.7350** |
| Precision | **0.7365** |
| Recall | **0.7350** |
| F1-Score | **0.7354** |
| ROC-AUC | **0.8643** |

Random Forest Tuned achieved slightly higher values on several test metrics. However, it maintained a larger gap between training and testing performance.

XGBoost Tuned showed a smaller train-test gap and better generalization behavior. The performance difference between the two tree-based models was relatively small, while XGBoost showed lower overfitting.

Therefore, **XGBoost Tuned was selected for deployment**.

---

# MLflow Experiment Tracking

MLflow was used to track machine learning experiments and model artifacts.

The experiment was configured as:

```text
Credit Score Classification
```

The MLflow tracking URI used during local development was:

```text
sqlite:///mlflow.db
```

The following experiments were tracked:

### Baseline Models

- Logistic Regression
- Random Forest
- XGBoost

### Tuned Models

- Logistic Regression Tuned
- Random Forest Tuned
- XGBoost Tuned

For each experiment, the following information was recorded:

- Model parameters
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Trained model artifact

The selected model was registered in the MLflow Model Registry as:

```text
CreditScoreBestModel
```

The selected model version was assigned the alias:

```text
champion
```

This allows the best model to be identified and managed separately from individual training runs.

---

# OOP-Based Machine Learning Architecture

The project was structured using Object-Oriented Programming principles.

Core components include:

### `DataIngestion`

Responsible for:

- Loading the dataset
- Checking whether the input file exists
- Returning the loaded DataFrame

### `Preprocessing`

Responsible for:

- Data type conversion
- Invalid value handling
- Missing value handling
- Credit history transformation
- Feature engineering
- Categorical encoding
- Saving and loading encoders

### `ModelTrainer`

Responsible for training:

- Logistic Regression
- Random Forest
- XGBoost

### `ModelEvaluator`

Responsible for:

- Accuracy calculation
- Precision calculation
- Recall calculation
- F1-Score calculation
- ROC-AUC calculation
- Confusion matrix
- Classification report
- Result comparison

This structure separates the responsibilities of each component and makes the training pipeline easier to maintain and retrain.

---

# Model Artifacts

After training, the required artifacts are saved for inference and deployment.

Examples include:

```text
best_model.pkl
scaler.pkl
label_encoders.pkl
target_encoder.pkl
features.pkl
```

For the AWS deployment package, the model and inference artifacts are packaged into:

```text
model.tar.gz
```

The deployment package contains:

```text
model.joblib
label_encoders.joblib
target_encoder.joblib
features.joblib
```

---

# Inference Pipeline

The inference pipeline applies the same transformations required during training before generating a prediction.

```text
Input Customer Data
        ↓
Input Validation
        ↓
Preprocessing
        ↓
Categorical Encoding
        ↓
Feature Selection
        ↓
XGBoost Tuned Model
        ↓
Target Label Decoding
        ↓
Credit Score
```

The prediction output is returned in JSON format:

```json
{
  "prediction": ["Good"]
}
```

The inference code supports JSON input containing customer information and converts it into a pandas DataFrame before applying preprocessing and prediction.

---

# Streamlit Application

A Streamlit web application was developed as the user interface for the prediction system.

The application allows users to enter customer financial information and request a credit score prediction.

The interface contains fields related to:

- Customer age
- Income
- Bank accounts
- Credit cards
- Interest rate
- Loans
- Payment delays
- Credit inquiries
- Credit mix
- Outstanding debt
- Credit utilization
- Credit history
- EMI
- Monthly investment
- Payment behavior
- Monthly balance

The application then sends the input data to the deployed inference endpoint and displays one of:

```text
Good
Standard
Poor
```

---

# AWS Cloud Deployment

The machine learning application was designed using the following AWS architecture:

```text
User
  ↓
Streamlit Web Application
  ↓
Amazon EC2
  ↓
SageMaker Runtime
  ↓
Amazon SageMaker Endpoint
  ↓
XGBoost Tuned Model
  ↓
Prediction
```

### AWS Services Used

- **Amazon S3** — stores the model deployment package
- **Amazon SageMaker** — hosts the machine learning inference endpoint
- **Amazon EC2** — hosts the Streamlit application
- **AWS IAM** — provides permissions for AWS services
- **Amazon CloudWatch** — used for SageMaker endpoint logs and troubleshooting

---

## Amazon S3

The trained model package was compressed into:

```text
model.tar.gz
```

and uploaded to an Amazon S3 bucket.

Example structure:

```text
s3://<bucket-name>/credit-score/model.tar.gz
```

The S3 object is used as the model artifact for SageMaker deployment.

---

## Amazon SageMaker

The trained XGBoost model was deployed as a SageMaker real-time endpoint.

The endpoint is responsible for:

1. Receiving customer data
2. Loading the trained model
3. Applying preprocessing
4. Running inference
5. Returning the predicted credit score

Example endpoint name:

```text
credit-score-endpoint
```

---

## Amazon EC2

The Streamlit application is hosted on an Amazon EC2 instance.

The EC2 instance:

1. Clones the project repository
2. Creates a Python virtual environment
3. Installs Streamlit and boto3
4. Starts the Streamlit application
5. Sends prediction requests to SageMaker

The application runs on:

```text
0.0.0.0:8501
```

A systemd service is used so that the Streamlit application can automatically restart if the process stops.

---

# Deployment Testing

The deployed application was tested using customer inputs representing each target class.

The required test scenarios include:

| Test Case | Expected Class |
|---|---|
| Test Case 1 | Good |
| Test Case 2 | Standard |
| Test Case 3 | Poor |

The testing process verifies that:

- The Streamlit interface accepts user input
- The request reaches the SageMaker endpoint
- The inference pipeline processes the input correctly
- The model returns a valid prediction
- The prediction is displayed correctly in the Streamlit application

Screenshots of the deployed application and prediction results can be added to the repository under a dedicated `screenshots/` directory.

---

# Local vs Cloud Deployment

| Aspect | Local Deployment | AWS Cloud Deployment |
|---|---|---|
| Development | Simple and fast | More infrastructure required |
| Cost | Low / no cloud cost | Cloud resources may incur cost |
| Accessibility | Usually local machine only | Can be accessed remotely |
| Scalability | Limited by local hardware | More scalable |
| Model Serving | Local inference | SageMaker endpoint |
| Monitoring | Manual / local logs | CloudWatch integration |
| Infrastructure | Minimal | EC2 + S3 + SageMaker + IAM |
| Production Readiness | Suitable for development/testing | More suitable for cloud-based deployment |

### Local Deployment

Local deployment is useful for:

- Development
- Debugging
- Experimentation
- Rapid model iteration
- Testing inference logic

### Cloud Deployment

Cloud deployment provides:

- Remote accessibility
- Managed model hosting
- Scalable infrastructure
- Cloud-based monitoring
- Separation between application and model serving

The trade-off is increased infrastructure complexity and cloud resource costs.

---

# Project Structure

```text
Predict-Credit-Score/
│
├── data/
│   └── data_B.csv
│
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── training_model.py
│   ├── evaluation.py
│   └── inference.py
│
├── models./
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── target_encoder.pkl
│   └── features.pkl
│
├── models_artifact/
│   └── model.tar.gz
│
├── notebooks/
│   └── credit_score_classification.ipynb
│
├── train.py
├── pipeline.py
├── deploy_endpoint.py
├── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Technologies Used

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy
- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn
- XGBoost

### Hyperparameter Optimization

- Optuna

### Experiment Tracking

- MLflow

### Model Deployment

- Streamlit
- AWS SageMaker
- Amazon EC2
- Amazon S3
- AWS IAM
- Amazon CloudWatch

### Development Environment

- Jupyter Notebook
- VS Code
- GitHub

---

# Key Takeaways

This project demonstrates an end-to-end machine learning workflow rather than only model training.

The main technical outcomes include:

- Data cleaning and preprocessing
- Exploratory data analysis
- Feature engineering
- Multiclass classification
- Handling imbalanced classes
- Model comparison
- Hyperparameter optimization with Optuna
- Cross-validation with StratifiedKFold
- Model evaluation using multiple metrics
- MLflow experiment tracking
- Model artifact management
- OOP-based ML architecture
- Model inference development
- Streamlit application development
- AWS SageMaker deployment
- Amazon EC2 deployment
- S3 model artifact storage
- Cloud-based model serving

The final deployed model is **XGBoost Tuned**, selected based on its predictive performance and generalization behavior.

---

ess assessment, security controls, and domain-specific requirements.
