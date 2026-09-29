import streamlit as st
import pandas as pd
import re

from inference import CreditScorePredictor

predictor = CreditScorePredictor()

st.title("Credit Score Classification")
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:

    month = st.selectbox(
        "Month",
        [
            "January","February","March","April",
            "May","June","July","August"
        ]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    occupation = st.selectbox(
        "Occupation",
        [
            "Lawyer", "Scientist", "Accountant", "Engineer", "Architect",
            "Mechanic", "Writer", "Entrepreneur", "Media_Manager", "Developer", 
            "Doctor", "Journalist", "Teacher", "Manager", "Musician"
        ]
    )

    annual_income = st.number_input(
        "Annual Income",
        min_value=7006035.0,
        max_value=24177150000.0,
        value=37824960.0
    )

    monthly_salary = st.number_input(
        "Monthly Inhand Salary",
        min_value=303.645417,
        max_value=15204.633333,
        value=3118.399167
    )

    num_bank_accounts = st.number_input(
        "Number of Bank Accounts",
        min_value=0,
        max_value=11,
        value=5
    )

    num_credit_card = st.number_input(
        "Number of Credit Cards",
        min_value=0,
        max_value=20,
        value=5
    )

    interest_rate = st.number_input(
        "Interest Rate",
        min_value=1,
        max_value=100,
        value=13
    )

    num_loan = st.number_input(
        "Number of Loans",
        min_value=0,
        max_value=49,
        value=5
    )

    loan_options = [
        "Auto Loan",
        "Credit-Builder Loan",
        "Debt Consolidation Loan",
        "Home Equity Loan",
        "Mortgage Loan",
        "Payday Loan",
        "Personal Loan",
        "Student Loan",
        "Not Specified"
    ]

    type_of_loan = st.selectbox(
        "Type of Loan",
        loan_options
    )

with col2:

    delay_due = st.number_input(
        "Delay From Due Date",
        min_value=0,
        max_value=67,
        value=18
    )

    delayed_payment = st.number_input(
        "Number of Delayed Payments",
        min_value=0,
        max_value=84,
        value=14
    )

    changed_credit_limit = st.number_input(
        "Changed Credit Limit",
        min_value=-6.49,
        max_value=36.49,
        value=9.37
    )

    credit_inquiries = st.number_input(
        "Number Credit Inquiries",
        min_value=0,
        max_value=93,
        value=50
    )

    credit_mix = st.selectbox(
        "Credit Mix",
        ["Good","Standard","Bad"]
    )

    outstanding_debt = st.number_input(
        "Outstanding Debt",
        min_value=0.23,
        max_value=4998.07,
        value=1170.25
    )

    credit_utilization = st.number_input(
        "Credit Utilization Ratio",
        min_value=5.13,
        max_value=49.2549,
        value=32.27
    )

    payment_min = st.selectbox(
        "Payment of Minimum Amount",
        ["Yes","No", "NM"]
    )

    total_emi = st.number_input(
        "Total EMI per Month",
        min_value=0.0,
        value=150.0
    )

    amount_invested = st.number_input(
        "Amount Invested Monthly",
        min_value=0.0,
        value=300.0
    )

    payment_behaviour = st.selectbox(
        "Payment Behaviour",
        [
            "High_spent_Small_value_payments",
            "Low_spent_Small_value_payments",
            "High_spent_Medium_value_payments",
            "Low_spent_Medium_value_payments",
            "High_spent_Large_value_payments",
            "Low_spent_Large_value_payments"
        ]
    )

    monthly_balance = st.number_input(
        "Monthly Balance",
        value=500.0
    )

    credit_history_age = st.text_input(
        "Credit History Age",
        value="10 Years and 5 Months"
    )

if st.button("Predict Credit Score", type="primary"):

    try:
        
        # Validation
        if not re.match(
            r"^\d+\s+Years?\s+and\s+\d+\s+Months?$",
            credit_history_age
        ):
            st.error(
                "Format Credit History Age harus seperti: 10 Years and 5 Months"
            )
            st.stop()


        input_df = pd.DataFrame({

            "Month":[month],
            "Age":[age],
            "Occupation":[occupation],
            "Annual_Income":[annual_income],
            "Monthly_Inhand_Salary":[monthly_salary],
            "Num_Bank_Accounts":[num_bank_accounts],
            "Num_Credit_Card":[num_credit_card],
            "Interest_Rate":[interest_rate],
            "Num_of_Loan":[num_loan],
            "Type_of_Loan":[type_of_loan],
            "Delay_from_due_date":[delay_due],
            "Num_of_Delayed_Payment":[delayed_payment],
            "Changed_Credit_Limit":[changed_credit_limit],
            "Num_Credit_Inquiries":[credit_inquiries],
            "Credit_Mix":[credit_mix],
            "Outstanding_Debt":[outstanding_debt],
            "Credit_Utilization_Ratio":[credit_utilization],
            "Payment_of_Min_Amount":[payment_min],
            "Total_EMI_per_month":[total_emi],
            "Amount_invested_monthly":[amount_invested],
            "Payment_Behaviour":[payment_behaviour],
            "Monthly_Balance":[monthly_balance],
            "Credit_History_Age":[credit_history_age]
        })

        with st.expander("View Input Data"):
            st.dataframe(input_df)

        with st.spinner("Predicting Credit Score..."):
            prediction = predictor.predict(input_df)

        if prediction == "Good":
            st.success(f"Credit Score Prediction: {prediction}")

        elif prediction == "Standard":
            st.warning(f"Credit Score Prediction: {prediction}")

        else:
            st.error(f"Credit Score Prediction: {prediction}")

    except Exception as e:
        
        st.error("Prediction Failed!")

        st.exception(e)
