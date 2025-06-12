import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page configuration
st.set_page_config(page_title="Customer Category Predictor", layout="centered")
st.title("🏦 Customer Category Prediction App")

# Load model and scaler
model = joblib.load("rf_final_model1.pkl")
scaler = joblib.load("scaler1.pkl")

# Corrected feature names (must match those used in training)
top_features = [
    'Credit_History_Age_Months', 'Outstanding_Debt', 'Num_Credit_Inquiries',
    'Interest_Rate', 'Delay_from_due_date', 'Num_Bank_Accounts',
    'Num_Credit_Card', 'Monthly_Balance', 'Annual_Income', 'Age',
    'Num_of_Delayed_Payment', 'Monthly_Inhand_Salary', 'Personal Loan',
    'Credit_Utilization_Ratio', 'Mortgage Loan'
]

# Sidebar: select input method
st.sidebar.header("📥 Select Input Method")
input_mode = st.sidebar.radio("Choose how you want to enter data:", ["Manual Entry", "Upload CSV"])

# --- Manual Input Mode ---
if input_mode == "Manual Entry":
    st.subheader("📝 Enter Customer Details")
    with st.form("manual_form"):
        user_input = {}
        for feature in top_features:
            user_input[feature] = st.number_input(f"{feature}", value=0.0)
        submitted = st.form_submit_button("Predict")

    if submitted:
        df_input = pd.DataFrame([user_input])
        df_scaled = scaler.transform(df_input)
        pred = model.predict(df_scaled)[0]
        label_map = {
        0: "Established Customer",
        1: "Growing Customer",
        2: "Legacy Customer",
        3: "Loyal Customer",
        4: "New Customer"}
        label = label_map.get(pred, "Unknown")
        st.success(f"🎯 Predicted Category: **{label}**")

# --- CSV Upload Mode ---
else:
    st.subheader("📁 Upload CSV File for Bulk Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if not all(feature in df.columns for feature in top_features):
                st.error(f"❌ CSV must contain these columns: {', '.join(top_features)}")
            else:
                X = df[top_features].fillna(0)
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)
                label_map = {
                    0: "Established Customer",
                    1: "Growing Customer",
                    2: "Legacy Customer",
                    3: "Loyal Customer",
                    4: "New Customer"}
                df["Predicted Category"] = [label_map.get(p, "Unknown") for p in predictions]

                st.success("✅ Prediction Complete!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
