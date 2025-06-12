import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Customer Category Predictor", layout="centered")
st.title("🏦 Customer Category Prediction App")

# Load model and scaler
model = joblib.load("rf_final_model1.pkl")
scaler = joblib.load("scaler1.pkl")

# Final feature list used during training
top_features = [
    'Credit_History_Age_Months',
    'Num_Credit_Card',
    'Num_Bank_Accounts',
    'Monthly_Balance',
    'Monthly_Inhand_Salary',
    'Annual_Income',
    'Age',
    'Num_of_Loan',
    'Changed_Credit_Limit',
    'Occupation',
    'Amount_invested_monthly',
    'Payment_of_Min_Amount',
    'Credit_Mix',
    'Payment_Behaviour'
]

# Example defaults for manual mode
example_defaults = {
    'Credit_History_Age_Months': 120,
    'Num_Credit_Card': 2,
    'Num_Bank_Accounts': 3,
    'Monthly_Balance': 5000,
    'Monthly_Inhand_Salary': 25000,
    'Annual_Income': 600000,
    'Age': 35,
    'Num_of_Loan': 1,
    'Changed_Credit_Limit': 1000,
    'Occupation': 'Engineer',
    'Amount_invested_monthly': 1500,
    'Payment_of_Min_Amount': 'Yes',
    'Credit_Mix': 'Good',
    'Payment_Behaviour': 'High_spent_Large_value_payments'
}

# Categorical columns (must match training)
categorical_cols = ['Occupation', 'Payment_of_Min_Amount', 'Credit_Mix', 'Payment_Behaviour']

# Load encoders for categorical features
encoders = joblib.load("encoders_selected.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Sidebar input method
st.sidebar.header("📥 Select Input Method")
input_mode = st.sidebar.radio("Choose how you want to enter data:", ["Manual Entry", "Upload CSV"])

# --- Manual Mode ---
if input_mode == "Manual Entry":
    st.subheader("📝 Enter Customer Details")
    with st.form("manual_form"):
        user_input = {}
        for col in top_features:
            if col in categorical_cols:
                user_input[col] = st.selectbox(f"{col}", encoders[col].classes_.tolist())
            else:
                user_input[col] = st.number_input(f"{col}", value=example_defaults.get(col, 0.0))
        submitted = st.form_submit_button("Predict")

    if submitted:
        df_input = pd.DataFrame([user_input])
        for col in categorical_cols:
            df_input[col] = encoders[col].transform(df_input[col])
        df_scaled = scaler.transform(df_input)
        pred = model.predict(df_scaled)[0]
        label = target_encoder.inverse_transform([pred])[0]
        st.success(f"🎯 Predicted Category: **{label}**")

# --- CSV Mode ---
else:
    st.subheader("📁 Upload CSV File for Bulk Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            missing_cols = [col for col in top_features if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns in CSV: {', '.join(missing_cols)}")
            else:
                df = df[top_features].copy()
                for col in categorical_cols:
                    df[col] = encoders[col].transform(df[col].fillna(method="ffill"))
                df = df.fillna(0)
                df_scaled = scaler.transform(df)
                predictions = model.predict(df_scaled)
                df["Predicted Category"] = target_encoder.inverse_transform(predictions)
                st.success("✅ Prediction Complete!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
