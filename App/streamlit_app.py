"""
Loan Approval Checker – Streamlit Application

This module implements an end-to-end Streamlit web application for predicting
loan approval eligibility using a pre-trained machine learning model.

The application:
- Loads a trained loan approval model (Pipeline) from disk
- Provides a user-friendly, screenshot-friendly UI for loan application inputs
- Validates user input against the model’s expected feature schema
- Runs predictions and displays approval status and probability
- Includes optional debug logging visible both in terminal and Streamlit sidebar

Key Components
--------------
1. Path & Environment Setup
   - Resolves project base directory dynamically
   - Locates the trained model file using pathlib
   - Ensures portability across environments

2. Logging System
   - Standard logging to terminal (INFO / ERROR)
   - Optional UI logging via Streamlit sidebar checkbox
   - Centralized helper functions for logging consistency

3. Model Loading
   - Cached model loading using st.cache_resource
   - Safe loading with existence checks and exception handling
   - Stops execution gracefully if model is missing or invalid

4. User Interface (Streamlit)
   - Configured page metadata (title, icon, layout)
   - Structured input form using st.form
   - Numeric and categorical inputs aligned with training features
   - Clean UX suitable for demos and screenshots

5. Data Validation
   - Ensures column names exactly match training-time features
   - Detects missing or extra columns
   - Reorders columns to match model expectations
   - Prevents prediction on incomplete or invalid data

6. Prediction & Output
   - Generates binary loan approval prediction
   - Displays approval decision clearly (Approved / Not Approved)
   - Shows model confidence as approval probability
   - Handles prediction errors safely

7. Debugging & Transparency
   - Optional sidebar debug logs for development and troubleshooting
   - Logs key execution steps, inputs, and model expectations

Dependencies
------------
- pandas
- streamlit
- joblib
- logging
- pathlib
- os

Expected Model Requirements
---------------------------
- The model must be a scikit-learn Pipeline
- Must include a preprocessing step named "preprocess"
- Must expose `feature_names_in_` for input validation
- Must support both `predict` and `predict_proba`

Usage
-----
Run the application using Streamlit:

    streamlit run app.py

Ensure that the trained model file exists at:
    Project_01_LoanApp/Models/loan_approval_model.pkl

Author
------
Designed for educational and production-style ML deployment workflows
using Streamlit and scikit-learn Pipelines.
"""

import pandas as pd
import streamlit as st
import logging
import joblib
import os
from pathlib import Path

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent.parent   # Project_01_LoanApp
MODEL_PATH = BASE_DIR / "Models" / "loan_approval_model.pkl"
MODEL_PATH = str(MODEL_PATH)

# ---------------- LOGGING CONFIG ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="Loan Approval Checker",
    page_icon="🏦",
    layout="centered"
)

# Title
st.title("🏦 Loan Approval Checker")





# ---------------- Optional: show logs inside Streamlit ----------------
DEBUG_UI = st.sidebar.checkbox("Show debug logs", value=False)
ui_logs = []

def log_info(msg: str):
    logger.info(msg)                 # terminal
    if DEBUG_UI:
        ui_logs.append(f"{msg}")  # UI

def log_error(msg: str):
    logger.error(msg)
    if DEBUG_UI:
        ui_logs.append(f" {msg}")


# ---------------- Load model ----------------
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


log_info(f"Current working directory: {os.getcwd()}")
log_info(f"Looking for model at: {MODEL_PATH}")


if not os.path.exists(MODEL_PATH):
    st.warning("The system is initializing, please wait...")
    st.error("Model file not found. Please run train.py first.")
    log_error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

try:
    log_info(f"Loading model from: {MODEL_PATH}")
    clf = load_model(MODEL_PATH)
    log_info("Model loaded successfully")
except Exception as e:
    st.error("Failed to load model.")
    log_error(f"Failed to load model: {e}")
    st.stop()



# ---- Input form (screenshot-friendly UI) ----
with st.form("loan_form"):
    st.subheader("Loan Application Details")

    applicant_income = st.number_input(
        "Applicant Income (monthly)",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    loan_amount = st.number_input(
        "Requested Loan Amount",
        min_value=0.0,
        value=150.0,
        step=10.0
    )

    loan_term = st.number_input(
        "Loan Term (months)",
        min_value=1.0,
        value=360.0,
        step=12.0
    )

    credit_history = st.selectbox(
        "Credit History",
        options=[1.0, 0.0],
        index=0,
        format_func=lambda x: "Exists (1)" if x == 1.0 else "Does not exist (0)"
    )
    dict_opt = {"Yes": "Married", "No": "Single"}

    married = st.selectbox(
        "Marital Status",
        options=["Yes", "No"],
        index=0,
        format_func=lambda x: dict_opt[x]
    )
    education = st.selectbox(
        "Education",
        options=["Graduate", "Not Graduate"],
        index=0)
    property_area = st.selectbox(
        "Property Area",
        options=["Urban", "Semiurban", "Rural"]
    )

    submitted = st.form_submit_button("Check Loan Eligibility")

# Runs only after clicking the button
if submitted:

    # IMPORTANT:
    # Column names must exactly match those used during model training
    X = pd.DataFrame([{
        "ApplicantIncome": float(applicant_income),
        "LoanAmount": float(loan_amount),
        "Loan_Amount_Term": float(loan_term),
        "Credit_History": float(credit_history),
        "Married": married,
        "Education": education,
        "Property_Area": property_area
    }])


    log_info(f"Input DataFrame created with columns: {list(X.columns)}")
    log_info(f"Input values: {X.iloc[0].to_dict()}")

    # Validate columns match expected
    expected = list(clf.named_steps["preprocess"].feature_names_in_)
    log_info(f"Expected columns from model: {expected}")

    missing_cols = set(expected) - set(X.columns)
    extra_cols = set(X.columns) - set(expected)

    if missing_cols:
        st.error(f"Missing required fields: {sorted(list(missing_cols))}")
        log_error(f"Missing required fields: {missing_cols}")
        st.stop()

    if extra_cols:
        # Not fatal, but good to log
        log_info(f"Extra fields will be ignored: {extra_cols}")

    # Align order + drop extras
    X = X.reindex(columns=expected)
    log_info("Columns aligned to expected order")

    # Check for NaN values (missing inputs)
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        st.error(f"Please fill these fields: {nan_cols}")
        log_error(f"NaN detected in columns: {nan_cols}")
        st.stop()
    log_info("No missing values detected")

    # Predict
    try:
        log_info("Running prediction...")
        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0, 1]  # probability of class 1 (approved)
        # =========================
        # Display result to user
        # =========================

        if pred == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Not Approved")
        log_info(f"Raw prediction: {pred}")

        # Display probability
        log_info(f"Approval probability: {proba:.2%}")
        st.markdown("### 📊 Model Confidence")
        st.info(f"Estimated approval probability: {proba:.2%}")

    except Exception as e:
        st.error("Prediction failed.")
        log_error(f"Prediction failed: {e}")
        st.stop()

# ---------------- Show debug logs in UI ----------------
if DEBUG_UI and ui_logs:
    st.sidebar.subheader("Debug logs")
    for line in ui_logs:
        st.sidebar.write(line)