# Loan Approval Checker 🏦

An end-to-end machine learning project that predicts loan approval eligibility using a trained scikit-learn pipeline and a Streamlit web application.

The project demonstrates the full ML workflow — from data preprocessing and model training to deployment with a clean, user-friendly UI.

---

## 🚀 Project Overview

This application allows users to input loan application details and receive:
- A loan approval / rejection decision
- The model’s confidence (approval probability)

The model is loaded dynamically and validated against the exact feature schema used during training to ensure safe and reliable predictions.

---

## 🧠 Key Features

- Trained machine learning pipeline (scikit-learn)
- Robust input validation (column names, order, missing values)
- Screenshot-friendly Streamlit UI
- Approval probability display
- Optional debug logging inside the Streamlit sidebar
- Clean and modular project structure

---

## 🛠 Tech Stack

- **Python**
- **pandas**
- **scikit-learn**
- **Streamlit**
- **joblib**
- **pathlib & logging**

---

## 📁 Project Structure

```
Project_01_LoanApp/
├── App/                # Streamlit application (UI)
├── SRC/                # Model training & preprocessing scripts
├── Models/             # Trained ML model (.pkl)
├── Data/               # Datasets
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/USERNAME/Project_01_LoanApp.git
cd Project_01_LoanApp
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run App/app.py
```

---

## 📊 Model Requirements

The trained model must:
- Be a **scikit-learn Pipeline**
- Include a preprocessing step named **`preprocess`**
- Expose `feature_names_in_`
- Support both `predict()` and `predict_proba()`

---

## 📌 Notes

- The virtual environment (`.venv`) is excluded from version control
- Ensure the trained model file exists in the `Models/` directory before running the app

---

## 👤 Author
Amit zalman
A.I 
Built as part of a hands-on machine learning and deployment learning process.
