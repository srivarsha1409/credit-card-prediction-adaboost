import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("default_model.sav", "rb") as f:
    model = pickle.load(f)

st.title("💳 Credit Card Default Prediction App")
st.write("Enter customer details to predict whether they will **default next month** (1 = Default, 0 = No Default).")

# Collect inputs
limit_bal = st.number_input("Credit Limit (LIMIT_BAL)", min_value=1000, max_value=1000000, step=1000)
age = st.number_input("Age", min_value=18, max_value=100, step=1)
pay_0 = st.selectbox("Repayment Status Last Month (PAY_0)", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
bill_amt1 = st.number_input("Bill Amount (Last Month)", min_value=-10000, max_value=1000000, step=1000)
pay_amt1 = st.number_input("Payment Amount (Last Month)", min_value=0, max_value=1000000, step=1000)

# Build input row (default values for other features = 0)
sample_dict = {col: 0 for col in model.feature_names_in_}

# Fill in the values user gave
sample_dict["LIMIT_BAL"] = limit_bal
sample_dict["AGE"] = age
sample_dict["PAY_0"] = pay_0
sample_dict["BILL_AMT1"] = bill_amt1
sample_dict["PAY_AMT1"] = pay_amt1

# Convert into DataFrame with correct columns
sample = pd.DataFrame([sample_dict])

# Predict
if st.button("Predict"):
    prediction = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0]

    if prediction == 1:
        st.error(f"⚠️ Prediction: **Default** (Probability {prob[1]:.2f})")
    else:
        st.success(f"✅ Prediction: **No Default** (Probability {prob[0]:.2f})")
