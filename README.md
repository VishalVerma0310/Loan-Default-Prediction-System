# 💳 Loan Default Prediction System

An end-to-end Machine Learning project that predicts the likelihood of loan default using ensemble learning models and a full data science pipeline — from exploratory data analysis to a deployed Streamlit application.

This project showcases practical ML skills in preprocessing, feature engineering, handling class imbalance, model tuning, and evaluation for real-world credit risk assessment.

---

## 📌 Project Overview

Loan default prediction is critical for financial institutions to reduce credit risk and improve lending decisions.

This project builds classification models to predict whether a borrower is likely to default on a loan based on demographic, financial, and credit-related attributes.

The workflow covers the complete ML lifecycle:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Class imbalance handling
- Feature selection
- Hyperparameter tuning
- Model comparison & evaluation
- Deployment via Streamlit

---

## 🚀 Live Demo

The project is deployed using Streamlit:

👉 **[Live Streamlit App](loan-default-prediction-system-vv.streamlit.app)**

Users can input borrower details and receive instant loan default risk predictions.

---

## 📂 Dataset

**Source:** Kaggle  
**Dataset:** Loan Approval Classification Dataset  
**Link:** https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data  

**Target Variable:** Loan Status (Default / Non-Default)

### Example Features

- Applicant Income  
- Loan Amount  
- Credit Score  
- Employment Length  
- Home Ownership  
- Loan Intent  
- Loan Grade  
- Interest Rate  

---

## ⚙️ Project Pipeline

### 🔍 Exploratory Data Analysis (EDA)

- Distribution analysis  
- Correlation heatmaps  
- Category-level comparisons  
- Feature relationship visualization  

---

### 🧹 Data Preprocessing

- Handling missing values  
- Encoding categorical variables  
- Outlier detection and treatment  
- Feature scaling where necessary  

---

### ⚖️ Handling Class Imbalance

- RandomOverSampler used to balance classes  
- Improved minority class prediction  

---

### 🎯 Feature Selection

- Identification of influential features  
- Removal of low-impact variables  

---

## 🤖 Models Implemented

### ✅ CatBoost Classifier
- Strong handling of categorical data  
- Hyperparameter tuning applied  

---

### ✅ ExtraTrees Classifier
- Ensemble-based tree model  
- Robust to noise and variance  

---

### ✅ LightGBM Classifier
- Efficient gradient boosting framework  
- Fast training and high performance  

---

## 📊 Evaluation Metrics

Models were evaluated using:

- Accuracy  
- Confusion Matrix  
- ROC-AUC Score  
- Precision & Recall  
- Precision-Recall Curve  
- Feature Importance Analysis  

---

## 📈 Model Comparison

The project includes:

- ROC curve comparisons  
- Precision-Recall comparisons  
- Performance visualizations across models  

These comparisons help determine the most reliable model for deployment.

---

## 🛠️ Tech Stack

- Python  
- Pandas & NumPy  
- Scikit-learn  
- CatBoost  
- LightGBM  
- Matplotlib & Seaborn  
- Streamlit  

---

## 🎯 Key Highlights

✔ End-to-end ML pipeline  
✔ Real-world financial dataset  
✔ Class imbalance handling  
✔ Multiple ensemble models  
✔ Hyperparameter tuning  
✔ Deployed ML application  
✔ Business-focused use case  

---

## 📌 Future Improvements

- Explainable AI integration (SHAP/LIME)  
- Cloud deployment scaling  
- Automated ML pipeline  
- API-based real-time predictions  

---

## 👤 Author

**Vishal Verma**

If you found this project useful, consider giving it a ⭐ on GitHub!

---
