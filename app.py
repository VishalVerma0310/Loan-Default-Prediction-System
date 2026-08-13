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
# SESSION STATE INITIALIZATION
# =========================
if 'show_probability' not in st.session_state:
    st.session_state.show_probability = True
if 'show_risk_details' not in st.session_state:
    st.session_state.show_risk_details = True

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

    # Clip numerical outliers
    for col in numerical_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 1.5 * IQR
        lower = Q1 - 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)

    # SAFE label encoding
    for col in label_encoders:
        le = label_encoders[col]

        # ensure values exist in encoder classes
        df[col] = df[col].apply(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )

        df[col] = le.transform(df[col])

    # feature engineering
    df["debt_to_income_ratio"] = df["loan_amnt"] / df["person_income"].replace(0, 1)
    df["age_to_experience_ratio"] = df["person_age"] / df["person_emp_exp"].replace(0, 1)

    return df

# =========================
# STREAMLIT UI SETUP
# =========================
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# SIDEBAR - SETTINGS
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.subheader("Display Options")
    st.session_state.show_probability = st.checkbox(
        "Show Default Probability",
        value=st.session_state.show_probability
    )
    
    st.session_state.show_risk_details = st.checkbox(
        "Show Risk Assessment Details",
        value=st.session_state.show_risk_details
    )
    
    st.markdown("---")
    
    st.subheader("Model Information")
    st.markdown("""
    **Prediction Type:**  
    Binary Classification
    
    **Output:**  
    Default Probability
    
    **Risk Threshold:**  
    30%
    """)



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

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# INPUT SECTION
# =========================
st.markdown("### 📝 Applicant Details")
st.markdown("<br>", unsafe_allow_html=True)

input_data = {}
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 💼 Financial & Personal Information")
    for feature in numerical_features[:4]:
        input_data[feature] = st.number_input(
            feature_labels[feature],
            value=None,
            placeholder="Enter value"
        )

with col2:
    st.markdown("#### 💳 Credit & Loan Details")
    for feature in numerical_features[4:]:
        input_data[feature] = st.number_input(
            feature_labels[feature],
            value=None,
            placeholder="Enter value"
        )

    # FIX: use encoder classes instead of manual 0/1
    input_data["previous_loan_defaults_on_file"] = st.selectbox(
        feature_labels["previous_loan_defaults_on_file"],
        options=label_encoders["previous_loan_defaults_on_file"].classes_
    )

    # other categorical features
    for col in label_encoders:
        if col == "previous_loan_defaults_on_file":
            continue

        input_data[col] = st.selectbox(
            col.replace("_", " ").title(),
            options=label_encoders[col].classes_
        )

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# ACTION BUTTONS
# =========================
button_col1, button_col2, button_col3 = st.columns([1, 1, 3])

with button_col1:
    predict_button = st.button("🔍 Predict", type="primary", use_container_width=True)

with button_col2:
    reset_button = st.button("🔄 Reset", use_container_width=True)

if reset_button:
    st.rerun()

# =========================
# PREDICTION
# =========================
if predict_button:
    input_df = pd.DataFrame([input_data])

    input_df = preprocess_input(input_df, label_encoders)
    input_df = input_df[selected_features]

    probability = model.predict_proba(input_df)[0][1]
    probability_percent = probability * 100
    
    # Fixed 30% threshold
    if probability >= 0.30:
        status = "High Risk of Default"
        icon = "🔴"
        color = "red"
        risk_level = "HIGH RISK"
    else:
        status = "Low Risk of Default"
        icon = "🟢"
        color = "green"
        risk_level = "LOW RISK"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📊 Risk Assessment")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Risk Assessment Card
    st.markdown(
        f"""
        <div style='background-color: {'#ffebee' if color == 'red' else '#e8f5e9'}; 
                    padding: 30px; 
                    border-radius: 10px; 
                    border-left: 5px solid {color};
                    text-align: center;'>
            <h1 style='color: {color}; margin: 0;'>{icon} {risk_level}</h1>
            <p style='font-size: 18px; color: #555; margin-top: 10px;'>{status}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Probability Display
    if st.session_state.show_probability:
        st.markdown("#### Default Probability")
        st.progress(probability, text=f"**{probability_percent:.1f}%**")
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Risk Details
    if st.session_state.show_risk_details:
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        
        with detail_col1:
            st.metric(
                label="Default Probability",
                value=f"{probability_percent:.1f}%"
            )
        
        with detail_col2:
            st.metric(
                label="Risk Threshold",
                value="30%"
            )
        
        with detail_col3:
            st.metric(
                label="Classification",
                value=risk_level,
                delta="Above Threshold" if probability >= 0.30 else "Below Threshold",
                delta_color="inverse"
            )

st.markdown("---")

# =========================
# HOW IT WORKS SECTION
# =========================
st.markdown("### 🔄 How It Works")
st.markdown("<br>", unsafe_allow_html=True)

workflow_cols = st.columns(5)

with workflow_cols[0]:
    st.markdown(
        """
        <div style='text-align: center;'>
            <div style='font-size: 40px;'>📝</div>
            <strong>Applicant Details</strong>
            <p style='font-size: 12px; color: gray;'>Input financial and credit information</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with workflow_cols[1]:
    st.markdown(
        """
        <div style='text-align: center;'>
            <div style='font-size: 40px;'>⚙️</div>
            <strong>Preprocessing</strong>
            <p style='font-size: 12px; color: gray;'>Data cleaning and feature engineering</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with workflow_cols[2]:
    st.markdown(
        """
        <div style='text-align: center;'>
            <div style='font-size: 40px;'>🤖</div>
            <strong>ML Model</strong>
            <p style='font-size: 12px; color: gray;'>Ensemble classification model</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with workflow_cols[3]:
    st.markdown(
        """
        <div style='text-align: center;'>
            <div style='font-size: 40px;'>📊</div>
            <strong>Probability</strong>
            <p style='font-size: 12px; color: gray;'>Calculate default likelihood</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with workflow_cols[4]:
    st.markdown(
        """
        <div style='text-align: center;'>
            <div style='font-size: 40px;'>🎯</div>
            <strong>Classification</strong>
            <p style='font-size: 12px; color: gray;'>Determine risk level</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# =========================
# FOOTER
# =========================
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>Developed by Vishal Verma | Portfolio Project</div>",
    unsafe_allow_html=True
)
