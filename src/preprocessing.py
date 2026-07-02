"""Preprocessing and feature engineering"""

import pandas as pd
import numpy as np
import joblib
import re

from sklearn.preprocessing import LabelEncoder

class Preprocessing:

    def __init__(self):
        self.label_encoders = {}

    def convert_datatypes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        int_cols = [
            'Age', 
            'Num_of_Loan', 
            'Num_of_Delayed_Payment'
        ]

        for col in int_cols:
            df[col] = (df[col].astype(str).str.replace("_", "", regex=False))
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].astype("Int64")
        
        float_cols = [
            "Annual_Income", 
            "Changed_Credit_Limit", 
            "Outstanding_Debt", 
            "Amount_invested_monthly", 
            "Monthly_Balance"
        ]

        for col in float_cols:
            df[col] = (df[col].astype(str).str.replace("_", "", regex=False).str.strip())
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df
    
    def handle_invalid_string(self, df):
        df = df.copy()

        invalid_string = {
            'Payment_Behaviour': '!@9#%8',
            'Credit_Mix': '_',
            'Occupation': '_______'
        }

        for col, invalid_value in invalid_string.items():
            if col in df.columns:
                df[col] = df[col].replace(invalid_value, np.nan)
            
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])

        return df
    
    def handle_invalid_values(self, df):
        df = df.copy()

        df.loc[(df['Age'] < 18) | (df['Age'] > 100), 'Age'] = np.nan

        df.loc[(df['Num_Bank_Accounts'] < 0) | (df['Num_Bank_Accounts'] > 20), 'Num_Bank_Accounts'] = np.nan

        df.loc[df['Num_Credit_Card'] > 20, 'Num_Credit_Card'] = np.nan

        df.loc[df['Interest_Rate'] > 100, 'Interest_Rate'] = np.nan

        df.loc[(df['Num_of_Loan'] < 0) | (df['Num_of_Loan'] > 50), 'Num_of_Loan'] = np.nan

        df.loc[df['Delay_from_due_date'] < 0, 'Delay_from_due_date'] = np.nan

        df.loc[(df['Num_of_Delayed_Payment'] < 0) | (df['Num_of_Delayed_Payment'] > 100), 'Num_of_Delayed_Payment'] = np.nan

        df.loc[df['Num_Credit_Inquiries'] > 100, 'Num_Credit_Inquiries'] = np.nan

        df.loc[df['Monthly_Balance'].abs() > 1e6, 'Monthly_Balance'] = np.nan

        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        return df
    
    def handle_missing_values(self, df):
        df = df.copy()

        numeric_missing_cols = [
            'Monthly_Inhand_Salary',
            'Num_of_Delayed_Payment',
            'Changed_Credit_Limit',
            'Num_Credit_Inquiries',
            'Amount_invested_monthly',
            'Monthly_Balance'
        ]

        for col in numeric_missing_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        categorical_missing_cols = [
            'Type_of_Loan'
        ]

        for col in categorical_missing_cols:
            if col in df.columns:
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode().iloc[0])

        return df
    
    def convert_credit_history(self, text):
        if pd.isna(text):
            return np.nan

        match = re.search(
            r'(\d+)\s+Years?\s+and\s+(\d+)\s+Months?',
            str(text)
        )

        if match:
            years = int(match.group(1))
            months = int(match.group(2))

            return years * 12 + months

        return np.nan

    def create_credit_history_months(self, df):
        df = df.copy()

        # Credit History
        if "Credit_History_Age" in df.columns:
            df["Credit_History_Months"] = (df["Credit_History_Age"].apply(self.convert_credit_history))
            df["Credit_History_Months"] = (df["Credit_History_Months"].fillna(df["Credit_History_Months"].median()))
        
        return df

    def drop_unnecessary_columns(self, df):
        df = df.copy()

        drop_cols = [
            'Unnamed: 0',
            'ID',
            'Customer_ID',
            'Name',
            'SSN',
            'Credit_History_Age'
        ]

        df.drop(
            columns=drop_cols,
            inplace=True,
            errors='ignore'
        )

        return df 

    def feature_engineering(self, df):

        df = df.copy()

        df['Debt_vs_Income'] = (df['Outstanding_Debt'] / (df['Annual_Income'] + 1))
        df['EMI_burden'] = (df['Total_EMI_per_month'] / (df['Monthly_Inhand_Salary'] + 1))
        df['Investment_Habit'] = (df['Amount_invested_monthly'] / (df['Monthly_Inhand_Salary'] + 1))
        df['Credit_History_Quality'] = (df['Credit_History_Months'] / (df['Age'] * 12 + 1))
        df['Delayed_Payment_Ratio'] = (df['Num_of_Delayed_Payment'] / (df['Num_of_Loan'] + 1))
        df['Inquiry_per_Loan'] = (df['Num_Credit_Inquiries'] / (df['Num_of_Loan'] + 1))
        df['Loan_per_Account'] = (df['Num_of_Loan'] / (df['Num_Bank_Accounts'] + 1))
        df['Total_Credit_Exposure'] = (df['Outstanding_Debt'] + df['Total_EMI_per_month'])
        df['Financial_Stability_Score'] = (df['Monthly_Balance'] + df['Amount_invested_monthly'] - df['Outstanding_Debt'])
        df['High_Risk_Customer'] = (df['Delay_from_due_date'] > 30).astype(int)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())

        return df

    def encode_features(self, df, fit=True):
        df = df.copy()

        categorical_cols = ['Month', 'Occupation', 'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']

        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col not in self.label_encoders:
                     raise ValueError(f"Encoder untuk {col} tidak ditemukan")
                
                le = self.label_encoders[col]
                df[col] = df[col].astype(str)

                df[col] = df[col].apply(
                    lambda x:
                    le.transform([x])[0]
                    if x in le.classes_
                    else -1
                    )

        return df
    
    # save dan load encoder
    def save_encoders(self, path):
        joblib.dump(self.label_encoders, path)

    def load_encoders(self, path):
        self.label_encoders = joblib.load(path)

    def fit_transform(self, df):

        df = self.convert_datatypes(df)
        df = self.handle_invalid_string(df)
        df = self.handle_invalid_values(df)
        df = self.handle_missing_values(df)
        df = self.create_credit_history_months(df)
        df = self.drop_unnecessary_columns(df)
        df = self.feature_engineering(df)
        df = self.encode_features(df, fit=True)
        
        print("Preprocessing completed!")

        return df
    
    # inference
    def transform(self, df):

        df = self.convert_datatypes(df)
        df = self.handle_invalid_string(df)
        df = self.handle_invalid_values(df)
        df = self.handle_missing_values(df)
        df = self.create_credit_history_months(df)
        df = self.drop_unnecessary_columns(df)
        df = self.feature_engineering(df)
        df = self.encode_features(df, fit=False)

        return df
    
if __name__ == "__main__":
    print("ModelTrainer module loaded successfully")