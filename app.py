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
# USER FRIENDLY LABELS
# =========================
feature_labels = {
    "person_age": "Age (Years)",
    "person_income": "Annual Income",
    "person_emp_exp": "Work Experience (Years)",
    "loan_amnt": "Loan Amount",
    "loan_int_rate": "Interest Rate (%)",
    "loan_percent_income": "Loan to Income Ratio",
    "cb_person_cred_hist_length": "Credit History Length (Years)",
    "credit_score": "Credit Score",
    "previous_loan_defaults_on_file": "Previous Loan Default?"
}

# =========================
# NUMERICAL FEATURES
# =========================
numerical_features = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score"
]

# =========================
# PREPROCESSING FUNCTION
# =========================
def preprocess_input(df, label_encoders):
    for col in numerical_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 1.5 * IQR
        lower = Q1 - 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)

    for col in label_encoders:
        df[col] = label_encoders[col].transform(df[col])

    df["debt_to_income_ratio"] = df["loan_amnt"] / df["person_income"].replace(0, 1)
    df["age_to_experience_ratio"] = df["person_age"] / df["person_emp_exp"].replace(0, 1)

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
    """
    <h1 style="text-align:center; color:#ff6b08;">
        🏦 Loan Default Prediction System
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align:center; color:gray;'>Interactive ML Dashboard to Predict Loan Default Risk</h4>",
    unsafe_allow_html=True
)
st.markdown("---")

# =========================
# INPUT SECTION
# =========================
st.header("Applicant Details")

input_data = {}
col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial & Personal Information")
    for feature in numerical_features[:4]:
        input_data[feature] = st.number_input(
            feature_labels[feature],
            value=None,
            placeholder="Enter value"
        )

with col2:
    st.subheader("Credit & Loan Details")
    for feature in numerical_features[4:]:
        input_data[feature] = st.number_input(
            feature_labels[feature],
            value=None,
            placeholder="Enter value"
        )

    input_data["previous_loan_defaults_on_file"] = (
        1 if st.selectbox(
            feature_labels["previous_loan_defaults_on_file"],
            ["No", "Yes"]
        ) == "Yes" else 0
    )

    for col in label_encoders:
        if col == "previous_loan_defaults_on_file":
            continue

        input_data[col] = st.selectbox(
            col.replace("_", " ").title(),
            options=label_encoders[col].classes_
        )

st.markdown("---")

# =========================
# PREDICTION
# =========================
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_df = preprocess_input(input_df, label_encoders)
    input_df = input_df[selected_features]

    probability = model.predict_proba(input_df)[0][1]

    if probability >= 0.30:
        status = "High Risk of Default"
        icon = "🔴"
        color = "red"
    else:
        status = "Low Risk of Default"
        icon = "🟢"
        color = "green"

    col1, col2 = st.columns(2)
    col1.metric("Default Probability", f"{probability:.2f}")
    col2.markdown(
        f"<h3 style='color:{color}'>{icon} {status}</h3>",
        unsafe_allow_html=True
    )

st.markdown("---")

# =========================
# FOOTER
# =========================
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>Developed by Vishal Verma | Portfolio Project</div>",
    unsafe_allow_html=True
)
