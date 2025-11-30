import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.title("Fraud Detection: CASH-OUT Transactions")

# Load dataset from GitHub or local CSV
DATA_URL = "https://github.com/N-218/fraud--demo/blob/72c49bdc366577d61d36431dea4c9076ee01d259/fraud_sample.csv"
df = pd.read_csv(DATA_URL)
df['balance_delta'] = df['newbalanceOrig'] - df['oldbalanceOrg']

features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'balance_delta']
X = df[features]
y = df['isFraud']

# Train model inside Streamlit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.sidebar.header("Transaction Input")
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0)
oldbalance = st.sidebar.number_input("Old Balance", min_value=0.0)
newbalance = st.sidebar.number_input("New Balance", min_value=0.0)

balance_delta = newbalance - oldbalance
input_df = pd.DataFrame({
    'amount': [amount],
    'oldbalanceOrg': [oldbalance],
    'newbalanceOrig': [newbalance],
    'balance_delta': [balance_delta]
})

# Predict
if st.button("Check Transaction"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Transaction flagged as FRAUD (probability {probability:.2f})")
    else:
        st.success(f"✅ Transaction appears LEGIT (probability {probability:.2f})")

# Feature importance
st.subheader("Feature Importance")
importances = model.feature_importances_
plt.bar(features, importances)
st.pyplot(plt)
