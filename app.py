import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Loan Default Prediction",
    layout="wide"
)

st.title("🏦 Loan Default Prediction System")
st.caption("Machine Learning powered credit risk assessment")

# --------------------------------------------------
# Load Model & Encoders
# --------------------------------------------------
model = joblib.load("model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
selected_features = joblib.load("selected_features.pkl")

# --------------------------------------------------
# Feature Lists (use your original ones if named differently)
# --------------------------------------------------
raw_numerical_features = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score"
]

default_values = {feature: 0.0 for feature in raw_numerical_features}

# --------------------------------------------------
# Preprocessing Function (UNCHANGED LOGIC)
# --------------------------------------------------
def preprocess_input(df, encoders):
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])
    return df

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("Applicant Information")

input_data = {}

st.sidebar.subheader("Numerical Inputs")
for feature in raw_numerical_features:
    input_data[feature] = st.sidebar.number_input(
        feature,
        value=default_values.get(feature, 0.0)
    )

st.sidebar.subheader("Categorical Inputs")
for col in label_encoders:
    input_data[col] = st.sidebar.selectbox(
        col,
        options=label_encoders[col].classes_
    )

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
if st.sidebar.button("Predict Loan Default Risk"):
    input_df = pd.DataFrame([input_data])

    input_df = preprocess_input(input_df, label_encoders)
    input_df = input_df[selected_features]

    probability = model.predict_proba(input_df)[0][1]

    if probability >= 0.30:
        risk_color = "#d9534f"
        risk_status = "High Risk of Default"
        emoji = "⚠️"
    else:
        risk_color = "#5cb85c"
        risk_status = "Low Risk of Default"
        emoji = "✅"

    st.markdown(
        f"""
        <div style="
            padding:25px;
            border-radius:15px;
            background-color:{risk_color};
            color:white;
            text-align:center;
        ">
            <h2>{emoji} {risk_status}</h2>
            <h3>Default Probability: {probability:.2f}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Risk Level Indicator")
    st.progress(min(probability, 1.0))
