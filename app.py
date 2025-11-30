import streamlit as st
import joblib

# Load model
model = joblib.load("fraud_model.pkl")

st.title("Fraud Detection – Mobile Money CASH-OUT")

# User Inputs
amount = st.number_input("Transaction Amount", 1, 50000, 1000)
old_balance = st.number_input("Old Balance", 0, 100000, 5000)
new_balance = st.number_input("New Balance", 0, 100000, 4000)

# Feature engineering
balance_delta = old_balance - new_balance
features = [[amount, old_balance, new_balance, balance_delta]]

# Predict
prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0][1]

# Display result
if prediction == 1:
    st.error(f"⚠️ FRAUD ALERT! Probability: {probability:.2%}")
else:
    st.success(f"✅ Legitimate Transaction. Probability: {probability:.2%}")

# Show probability bar
st.progress(probability)
