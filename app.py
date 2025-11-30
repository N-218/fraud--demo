import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# App Title
# ---------------------------
st.title("Fraud Detection: CASH-OUT Transactions")
st.markdown(
    """
Predict whether a CASH-OUT transaction is fraudulent based on transaction features.
"""
)

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("fraud_sample.csv")
    df['balance_delta'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    return df

df = load_data()
st.sidebar.subheader("Dataset Preview")
if st.sidebar.checkbox("Show dataset"):
    st.write(df.head())
    st.write("Fraud distribution:")
    st.write(df['isFraud'].value_counts())

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

# ---------------------------
# User Input
# ---------------------------
st.sidebar.subheader("Enter Transaction Details")
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)
oldbalance = st.sidebar.number_input("Old Balance", min_value=0.0, value=1000.0)
newbalance = st.sidebar.number_input("New Balance", min_value=0.0, value=900.0)
balance_delta = newbalance - oldbalance

# ---------------------------
# Prediction
# ---------------------------
if st.sidebar.button("Predict Fraud"):
    features = np.array([[amount, oldbalance, newbalance, balance_delta]])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if pred == 1:
        st.error(f"⚠ Fraud detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✓ Transaction appears legitimate (Probability: {prob:.2f})")

# ---------------------------
# Feature Importance
# ---------------------------
st.subheader("Feature Importance")
try:
    importances = model.feature_importances_
    feature_names = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'balance_delta']
    fig, ax = plt.subplots()
    ax.bar(feature_names, importances)
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)
except AttributeError:
    st.write("Model does not have feature_importances_ attribute.")
