import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("💼 Customer Churn Predictor")

st.markdown("""
# Customer Churn Prediction App  

**Class Task:** Build a Streamlit app for churn prediction  
**Submitted by:** Zuraiz Khan  
**Instructor:** Eng. Kamran  
**Due Date:** 20 Oct 2025  
""")


# Load artifacts
model = joblib.load('best_churn_model.pkl')
scaler = joblib.load('scaler.pkl')
le_gender = joblib.load('le_gender.pkl')
le_sub = joblib.load('le_subscription.pkl')
le_contract = joblib.load('le_contract.pkl')

# --- Input fields (match your training columns)
# Get numeric range suggestions from your data if you want; here are generic inputs:
age = st.number_input("Age", min_value=10, max_value=100, value=30)
gender = st.selectbox("Gender", options=list(le_gender.classes_))
tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
usage = st.number_input("Usage Frequency (per month)", min_value=0, max_value=1000, value=50)
support_calls = st.number_input("Support Calls (last period)", min_value=0, max_value=100, value=1)
payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=365, value=0)
subscription_type = st.selectbox("Subscription Type", options=list(le_sub.classes_))
contract_length = st.selectbox("Contract Length", options=list(le_contract.classes_))
total_spend = st.number_input("Total Spend", min_value=0, max_value=1000000, value=1000)
last_interaction = st.number_input("Days since Last Interaction", min_value=0, max_value=365, value=7)

# Prepare the input dataframe (ensure column ordering same as training)
input_df = pd.DataFrame([{
    'Age': age,
    'Gender': gender,
    'Tenure': tenure,
    'Usage Frequency': usage,
    'Support Calls': support_calls,
    'Payment Delay': payment_delay,
    'Subscription Type': subscription_type,
    'Contract Length': contract_length,
    'Total Spend': total_spend,
    'Last Interaction': last_interaction
}])

# Encode with loaded label encoders
input_df['Gender'] = le_gender.transform(input_df['Gender'])
input_df['Subscription Type'] = le_sub.transform(input_df['Subscription Type'])
input_df['Contract Length'] = le_contract.transform(input_df['Contract Length'])

# Scale numeric features
X_input = scaler.transform(input_df)

if st.button("Predict Churn"):
    try:
        # Make prediction
        pred = model.predict(X_input)[0]

        # Safely get churn probability
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]  # [prob_stay, prob_churn]
            prob_churn = float(proba[1])
        else:
            prob_churn = None

        # Display results
        st.markdown("---")
        if pred == 1:  # churn class
            st.error(f"🚨 **Customer is likely to churn**")

        else:
            st.success(f"✅ **Customer is likely to stay**")


    except Exception as e:
        st.error(f"⚠️ Error while making prediction: {e}")
