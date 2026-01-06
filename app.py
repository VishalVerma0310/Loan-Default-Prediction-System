import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# LOAD SAVED ARTIFACTS
# =========================
model = joblib.load("model/loan_model.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")
selected_features = joblib.load("model/feature_names.pkl")

# =========================
# RAW INPUT FEATURES
# (Before feature selection)
# =========================
raw_numerical_features = [
    'person_age',
    'person_income',
    'person_emp_exp',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'credit_score',
    'previous_loan_defaults_on_file'
]


# =========================
# PREPROCESSING FUNCTION
# (Same logic as notebook)
# =========================
def preprocess_input(df, label_encoders):
    # Outlier capping (IQR)
    numerical_cols_with_outliers = [
        'person_age',
        'person_income',
        'person_emp_exp',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length',
        'credit_score'
    ]

    for col in numerical_cols_with_outliers:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])

    # Label encoding (same encoders as training)
    for col in label_encoders:
        df[col] = label_encoders[col].transform(df[col])

    # Feature engineering
    df['debt_to_income_ratio'] = df['loan_amnt'] / df['person_income']
    df['age_to_experience_ratio'] = df['person_age'] / df['person_emp_exp'].replace(0, 1)

    return df

# =========================
# STREAMLIT UI
# =========================
st.title("Loan Default Prediction System")

st.subheader("Enter Applicant Details")

input_data = {}

input_data = {}

st.subheader("Numerical Inputs")
for feature in raw_numerical_features:
    input_data[feature] = st.number_input(
        f"Enter {feature}",
        value=0.0
    )

st.subheader("Categorical Inputs")
for col in label_encoders:
    input_data[col] = st.selectbox(
        f"Select {col}",
        options=label_encoders[col].classes_
    )



# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    # Apply preprocessing
    input_df = preprocess_input(input_df, label_encoders)

    # Select final features (8 features)
    input_df = input_df[selected_features]

    # Predict probability
    probability = model.predict_proba(input_df)[0][1]

    st.write(f"Predicted Default Probability: {probability:.2f}")

    # Business threshold
    if probability >= 0.30:
        st.error("High Risk of Loan Default")
    else:
        st.success("Low Risk of Loan Default")
