"""
Loan Approval Prediction - Training Script

This script loads the loan dataset, performs basic exploratory data analysis (EDA),
and prepares the data for further preprocessing and modeling.

Author: Amit Zalman
Project: Project_01_LoanApp
Date: 2026-01-10
"""
import logging
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report







# ---------------- LOGGING CONFIG ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- PATHS ----------------
TRAIN_PATH = "../Data/raw/train.csv"

def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV dataset from the given path.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    logger.info("Dataset loaded successfully")
    return df




def main():
    def main():
        """
        Main training pipeline for the Loan Approval model.

        This function performs the full training workflow:
        1. Loads the loan dataset from disk
        2. Performs basic exploratory logging on the target variable
        3. Selects a limited set of numeric and categorical features
        4. Builds preprocessing pipelines for numeric and categorical data
           - Numeric: median imputation + standard scaling
           - Categorical: most-frequent imputation + one-hot encoding
        5. Combines preprocessing using a ColumnTransformer
        6. Builds a full ML pipeline (preprocessing + SVM classifier)
        7. Evaluates model stability using stratified cross-validation
        8. Splits the data into train and test sets with preserved class distribution
        9. Trains the model on the training set
        10. Evaluates performance on the test set
            - Accuracy
            - Confusion matrix
            - Classification report
        11. Saves the trained pipeline (preprocessor + model) to disk

        The saved model is later used for inference and deployment (e.g., Streamlit UI).

        Notes
        -----
        - The pipeline prevents data leakage by fitting preprocessing
          only on training data.
        - Class imbalance is handled using class_weight="balanced".
        - Feature selection is intentionally limited to 7 features
          for academic requirements.

        Returns
        -------
        None
        """
    df = load_data(TRAIN_PATH)
    #loading the data

    logger.info("Target distribution (counts):") #im about to print the target distribution
    logger.info(f"\n{df['Loan_Status'].value_counts()}") #Counts how many times there are Y/N

    logger.info("Target distribution (percentages):") #im about to print the percentages target distribution
    logger.info(f"\n{(df['Loan_Status'].value_counts(normalize=True) * 100).round(2)}") #normalize=True gives me the percentages

    df['Loan_Status_Num'] = df['Loan_Status'].map({'Y': 1, 'N': 0}) # creating a new column (.map = take each value (Y/N) and if its in the dic replace it with (0/1)
    logger.info("(0 / 1) target distribution:") #Test to see if it worked
    logger.info(df['Loan_Status_Num'].value_counts())
    logger.info(df[['Loan_Status', 'Loan_Status_Num']].head()) #give me only the first 5 lines (just for  if the coding is right checking)

    # ----- numeric  values ----
    numeric_cols = [ #a list of numeric columns
        'ApplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'Credit_History'
    ]

    # ----- categorical values ----
    categorical_cols = [  # difinenig the categorical columns
        'Married',
        'Education',
        'Property_Area'
    ]

    # ----- For numeric missing values ----
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")), #fills the NaN with the median
        ("scaler", StandardScaler()) # does regular scaling
    ])

    # ----- For categorical missing values ----
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # we use most frequent because its categorical its less fair but more correct mathmaticly
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
        # we use OneHotEncoder in order to turn the values into 1 and 0 and fot the numbers not have value's that the model will notice
    ])  # handle_unknown="ignore" prevents the model from collapsing incase the use enters a categorical not in the categorical train values


    # ----- Only working on the 7 features ----
    FEATURES = numeric_cols + categorical_cols

    x = df[FEATURES]
    y = df["Loan_Status_Num"]

    logger.info("Missing values in NUMERIC features (used by model):")
    logger.info(x[numeric_cols].isna().sum())

    logger.info("Missing values in CATEGORICAL features (used by model):")
    logger.info(x[categorical_cols].isna().sum())


    # ----- For ColumnTransformer missing values ----
    preprocessor = ColumnTransformer( #splits the columns and makes one matrix that the model can workon
        transformers=[
            ("num", numeric_transformer, numeric_cols),#all the numeric he sends to the numeric pipline
            ("cat", categorical_transformer, categorical_cols) #all the categorical he sends to the categorical pipline
        ],
        remainder="drop"
    )

    # =========================
    #  Build pipeline
    # =========================
    clf = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
    ])

    # =========================
    # Cross Validation
    # =========================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #doing cross validation to make sure the model is stable

    scores = cross_val_score(
        clf,
        x,
        y,
        cv=cv,
        scoring="accuracy"
    )

    logger.info(f"CV accuracies: {scores}")
    logger.info(f"Mean CV accuracy: {scores.mean():.3f}")
    logger.info(f"Std CV accuracy: {scores.std():.3f}")

    # =========================
    # ----- Splitting into Train / Test  ----
    # =========================

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    # we log to make sure the split is done correctly
    logger.info(f"Train shape: X={x_train.shape}, y={y_train.shape}")
    logger.info(f"Test  shape: X={x_test.shape},  y={y_test.shape}")

    logger.info("Target distribution in TRAIN (%):")
    logger.info(f"\n{(y_train.value_counts(normalize=True) * 100).round(2)}") #shows me the  train values

    logger.info("Target distribution in TEST (%):")
    logger.info(f"\n{(y_test.value_counts(normalize=True) * 100).round(2)}") #shows me that the test values a re very close to the train values

    # =========================
    #  Building the Model
    # ========================
    logger.info("Training model on train set...")
    clf.fit(x_train, y_train)

    logger.info("Evaluating on test set...")
    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    logger.info(f"Test Accuracy: {acc * 100:.2f}%")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    # =========================
    # ----- For logging the preprocessor values ----
    # =========================

    pre = clf.named_steps["preprocess"]
    X_processed = pre.transform(x_train)

    logger.info("First raw row (x_train.iloc[0]):")
    logger.info(f"\n{x_train.iloc[0]}")
    logger.info("First processed row (first 20 values):")
    logger.info(f"{X_processed[0][:20]}")


    # =========================
    #  Saving the model the Model
    # ========================

    MODEL_PATH = "../Models/loan_approval_model.pkl" #Saving the model in the folder models!

    logger.info("Saving trained model...")
    joblib.dump(clf, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")



if __name__ == "__main__":
    main()
