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

# Correct top features used in training
top_features = [
    "Credit_History_Age_Months", "Outstanding_Debt", "Num_of_Loan",
    "Interest_Rate", "Payment_of_Min_Amount", "Num_Credit_Inquiries",
    "Delay_from_due_date", "Annual_Income", "Total_EMI_per_month", "Age"
]

# Optional: example defaults for manual input
example_defaults = {
    "Credit_History_Age_Months": 120,
    "Outstanding_Debt": 1500.5,
    "Num_of_Loan": 3,
    "Interest_Rate": 12.5,
    "Payment_of_Min_Amount": 1,  # 1 for Yes, 0 for No
    "Num_Credit_Inquiries": 2,
    "Delay_from_due_date": 4,
    "Annual_Income": 75000,
    "Total_EMI_per_month": 1200,
    "Age": 35
}

# Sidebar input method
st.sidebar.header("📥 Select Input Method")
input_mode = st.sidebar.radio("Choose how you want to enter data:", ["Manual Entry", "Upload CSV"])

# --- Manual Mode ---
if input_mode == "Manual Entry":
    st.subheader("📝 Enter Customer Details")
    with st.form("manual_form"):
        user_input = {}
        for feature in top_features:
            default = example_defaults.get(feature, 0.0)
            user_input[feature] = st.number_input(f"{feature}", value=default)
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
            4: "New Customer"
        }
        label = label_map.get(pred, "Unknown")
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
                X = df[top_features].fillna(0)
                X_scaled = scaler.transform(X)
                predictions = model.predict(X_scaled)
                label_map = {
                    0: "Established Customer",
                    1: "Growing Customer",
                    2: "Legacy Customer",
                    3: "Loyal Customer",
                    4: "New Customer"
                }
                df["Predicted Category"] = [label_map.get(p, "Unknown") for p in predictions]
                st.success("✅ Prediction Complete!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
