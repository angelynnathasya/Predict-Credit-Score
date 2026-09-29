# Credit Score Classification: End-to-End Machine Learning Pipeline & AWS Deployment

An end-to-end machine learning project for predicting customer credit scores using financial and behavioral data.

This project covers the complete machine learning lifecycle, starting from data ingestion and preprocessing, model training and evaluation, experiment tracking with MLflow, model inference, local deployment using Streamlit, and cloud deployment using AWS SageMaker and EC2.

The project was developed using an Object-Oriented Programming (OOP) approach to make the machine learning pipeline modular, reusable, and easier to maintain.

---

## Project Overview

Credit scoring is an important application of machine learning in the financial industry. The objective of this project is to classify customers into three credit score categories:

- **Good**
- **Standard**
- **Bad**

The model uses customer financial information, credit behavior, payment history, and other related features to predict the customer's credit score category.

The project implements both **local machine learning deployment** and **cloud-based deployment on AWS**.

---

## Project Objectives

The main objectives of this project are:

1. Build a reusable machine learning training pipeline.
2. Perform data preprocessing and feature engineering.
3. Train and compare multiple machine learning models.
4. Perform hyperparameter tuning to improve model performance.
5. Track experiments and evaluation results using MLflow.
6. Select and save the best-performing model.
7. Build a reusable inference pipeline.
8. Deploy the model locally using Streamlit.
9. Deploy the model to AWS SageMaker.
10. Build a public web interface for model inference using Streamlit and AWS.

---

## Machine Learning Workflow

```text
Raw Dataset
     │
     ▼
Data Ingestion
     │
     ▼
Data Preprocessing
     │
     ├── Data Type Conversion
     ├── Invalid Value Handling
     ├── Missing Value Handling
     ├── Categorical Encoding
     └── Feature Engineering
     │
     ▼
Train-Test Split
     │
     ▼
Model Training
     │
     ├── Logistic Regression
     ├── Random Forest
     └── XGBoost
     │
     ▼
Hyperparameter Tuning
     │
     ▼
Model Evaluation
     │
     ├── Accuracy
     ├── Precision
     ├── Recall
     ├── F1-Score
     └── ROC-AUC
     │
     ▼
MLflow Experiment Tracking
     │
     ▼
Best Model
     │
     ▼
Model Artifacts
     │
     ▼
Inference Pipeline
     │
     ├───────────────┐
     ▼               ▼
Local Deployment   AWS Deployment
Streamlit          S3 + SageMaker
                       │
                       ▼
                    Streamlit
                       │
                       ▼
                  Public Web App
