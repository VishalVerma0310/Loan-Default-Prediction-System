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
# =========================
def preprocess_input(df, label_encoders):
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

    # Label encoding
    for col in label_encoders:
        df[col] = label_encoders[col].transform(df[col])

    # Feature engineering
    df['debt_to_income_ratio'] = df['loan_amnt'] / df['person_income'].replace(0, 1)
    df['age_to_experience_ratio'] = df['person_age'] / df['person_emp_exp'].replace(0, 1)

    return df

# =========================
# STREAMLIT UI SETUP
# =========================
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide"
)

# =========================
# APP HEADER
# =========================
st.markdown(
    "<h1 style='text-align: center; color: #4B0082;'>💰 Loan Default Prediction System</h1>",
    unsafe_allow_html=True
)
st.markdown("<h4 style='text-align: center; color: gray;'>Interactive ML Dashboard to Predict Loan Default Risk</h4>", unsafe_allow_html=True)
st.markdown("---")

# =========================
# INPUT SECTION
# =========================
st.header("Applicant Details")

col1, col2 = st.columns(2)

with col1:
    input_data = {}
    st.subheader("Numerical Inputs")
    for feature in raw_numerical_features[:5]:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)

with col2:
    st.subheader("Numerical & Categorical Inputs")
    for feature in raw_numerical_features[5:]:
        input_data[feature] = st.number_input(f"{feature}", value=0.0)
    for col in label_encoders:
        input_data[col] = st.selectbox(f"{col}", options=label_encoders[col].classes_)

st.markdown("---")

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_df = preprocess_input(input_df, label_encoders)
    input_df = input_df[selected_features]

    probability = model.predict_proba(input_df)[0][1]

    # Risk evaluation
    if probability >= 0.30:
        risk_status = "High Risk of Default"
        risk_color = "#FF4B4B"  # Red
        risk_emoji = "🔴"
    else:
        risk_status = "Low Risk of Default"
        risk_color = "#4BB543"  # Green
        risk_emoji = "🟢"

    # Display metrics in cards
    col1, col2 = st.columns(2)
    col1.metric("Default Probability", f"{probability:.2f}")
    col2.markdown(f"<h3 style='color:{risk_color};'>{risk_emoji} {risk_status}</h3>", unsafe_allow_html=True)

    # Optional: Add insight text
    st.info(
        "💡 This prediction is based on historical loan data and risk features. "
        "A probability above 0.30 indicates a higher risk of loan default."
    )

st.markdown("---")

# =========================
# FOOTER
# =========================
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>Developed by Vishal Verma | Portfolio Project</div>",
    unsafe_allow_html=True
)
