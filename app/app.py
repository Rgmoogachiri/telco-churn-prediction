"""Streamlit app: churn risk for a single customer."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"

st.set_page_config(page_title="Telco churn risk", page_icon="📉")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["threshold"]


pipeline, threshold = load_model()

st.title("Customer churn risk")
st.write("Enter a customer's details in the sidebar to see their churn risk score.")

st.sidebar.header("Customer details")

# Account
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox(
    "Payment method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)
paperless = st.sidebar.selectbox("Paperless billing", ["Yes", "No"])
monthly = st.sidebar.number_input("Monthly charges", 18.0, 120.0, 70.0, step=0.5)

# Demographics
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = 1 if st.sidebar.selectbox("Senior citizen", ["No", "Yes"]) == "Yes" else 0
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

# Services (some options depend on others, as in the training data)
phone = st.sidebar.selectbox("Phone service", ["Yes", "No"])
if phone == "No":
    multiple_lines = "No phone service"
else:
    multiple_lines = st.sidebar.selectbox("Multiple lines", ["No", "Yes"])

internet = st.sidebar.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
if internet == "No":
    online_security = online_backup = device_protection = "No internet service"
    tech_support = streaming_tv = streaming_movies = "No internet service"
else:
    online_security = st.sidebar.selectbox("Online security", ["No", "Yes"])
    online_backup = st.sidebar.selectbox("Online backup", ["No", "Yes"])
    device_protection = st.sidebar.selectbox("Device protection", ["No", "Yes"])
    tech_support = st.sidebar.selectbox("Tech support", ["No", "Yes"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.sidebar.selectbox("Streaming movies", ["No", "Yes"])

# Total charges are estimated from tenure and monthly charges
total = round(tenure * monthly, 2)

customer = pd.DataFrame(
    [
        {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total),
        }
    ]
)

score = float(pipeline.predict_proba(customer)[0, 1])
high_risk = score >= threshold

col1, col2 = st.columns(2)
col1.metric("Risk score", f"{score:.2f}")
col2.metric("High-risk threshold", f"{threshold:.2f}")

if high_risk:
    st.error("High churn risk: consider a retention offer.")
else:
    st.success("Lower churn risk.")

st.caption(
    "The score ranks customers by risk. Because the model is weighted towards "
    "catching churners, it is not a literal probability of leaving."
)

with st.expander("Customer record sent to the model"):
    st.dataframe(customer.T.rename(columns={0: "value"}))
    st.caption("TotalCharges is estimated as tenure x monthly charges.")