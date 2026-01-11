"""
Loan Approval Prediction - Inference Script

This script loads a trained ML pipeline (preprocessing + model),
validates a single customer input, and returns:
- predicted class (Y/N)
- approval probability

Author: Amit Zalman
Project: Project_01_LoanApp
Date: 2026-01-10
"""

import logging
import joblib
import pandas as pd

# ---------------- LOGGING CONFIG ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- PATHS ----------------
MODEL_PATH = "../Models/loan_approval_model.pkl"


def main() -> None:
    """
    Run inference for a single loan application.

    This function performs the following steps:
    1. Loads a trained machine learning pipeline (preprocessing + model).
    2. Creates a single customer input sample.
    3. Validates that the input contains all required feature names.
    4. Aligns the input columns to match the exact order expected by the model.
    5. Checks for missing (NaN) values in required fields.
    6. Runs prediction using the trained model.
    7. Outputs:
       - The predicted loan decision (Y / N).
       - The probability of loan approval.

    The function is designed to simulate the same behavior that will later
    be triggered by the Streamlit user interface.

    Returns
    -------
    None
        The function prints the prediction results and logs all steps.
    """

    # =========================
    # Load trained pipeline
    # =========================
    logger.info(f"Loading model from: {MODEL_PATH}")
    clf = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")

    # =========================
    # Create new input (one customer)
    # =========================
    new_customer = pd.DataFrame([{
        "Married": "Yes",
        "Education": "Graduate",
        "ApplicantIncome": 5000,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": 1.0,
        "Property_Area": "Urban"
    }])

    logger.info("New customer input created")
    logger.info(f"Input columns: {list(new_customer.columns)}")
    logger.info(f"Input row:\n{new_customer.iloc[0]}")

    # =========================
    # Get expected columns from the preprocessor
    #    (these are the original feature names BEFORE one-hot encoding)
    # =========================
    expected_cols = list(clf.named_steps["preprocess"].feature_names_in_)
    logger.info(f"Expected columns: {expected_cols}")

    # =========================
    # Validate column names (missing + extra)
    # =========================
    missing_cols = set(expected_cols) - set(new_customer.columns)
    extra_cols = set(new_customer.columns) - set(expected_cols)

    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        raise ValueError(f"Missing required columns: {missing_cols}")

    if extra_cols:
        # Not always an error, but we want to know about it
        logger.warning(f"Extra columns will be dropped: {extra_cols}")

    logger.info("Column names validation passed")

    # =========================
    # Align columns (order + drop extras)
    # =========================
    new_customer = new_customer.reindex(columns=expected_cols)
    logger.info("Columns aligned to expected order")

    # =========================
    # Validate missing values (NaN)
    # =========================
    nan_cols = new_customer.columns[new_customer.isna().any()].tolist()
    if nan_cols:
        logger.error(f"Missing values found in columns: {nan_cols}")
        raise ValueError(f"Missing values found in columns: {nan_cols}")

    logger.info("No missing values found in required fields")

    # =========================
    #  Predict (class + probability)
    # =========================
    logger.info("Running prediction...")
    pred_num = clf.predict(new_customer)[0]          # 0 or 1
    proba_yes = clf.predict_proba(new_customer)[0, 1]  # probability of class 1

    result = "Y" if pred_num == 1 else "N"

    logger.info(f"Raw prediction (0/1): {pred_num}")
    logger.info(f"Final prediction (Y/N): {result}")
    logger.info(f"Approval probability: {proba_yes:.2%}")

    print("Prediction:", result)
    print("Probability (Approve=Y):", f"{proba_yes:.2%}")


if __name__ == "__main__":
    main()