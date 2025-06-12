import streamlit as st
import pandas as pd
import numpy as np
import joblib

# App configuration
st.set_page_config(page_title="Customer Category Predictor", layout="centered")
st.title("🏦 Customer Category Prediction App")

# Load trained model and scaler
model = joblib.load("rf_final_model1.pkl") 
scaler = joblib.load("scaler1.pkl")         

# Final top 10 features used in model training
top_features = [
    "Credit_History_Age_Months",
    "Outstanding_Debt",
    "Num_of_Loan",
    "Interest_Rate",
    "Payment_of_Min_Amount",
    "Num_Credit_Inquiries",
    "Delay_from_due_date",
    "Annual_Income",
    "Total_EMI_per_month",
    "Age"
]

# Customer category label map
label_map = {
    0: "Established Customer",
    1: "Growing Customer",
    2: "Legacy Customer",
    3: "Loyal Customer",
    4: "New Customer"
}

# Sidebar: input method
st.sidebar.header("📥 Select Input Method")
input_mode = st.sidebar.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

# --- Manual Input ---
if input_mode == "Manual Entry":
    st.subheader("📝 Enter Customer Details")
    with st.form("input_form"):
        user_input = {}
        for feature in top_features:
            user_input[feature] = st.number_input(f"{feature}", value=0.0, step=0.1)
        submitted = st.form_submit_button("Predict")

    if submitted:
        df_input = pd.DataFrame([user_input])
        df_scaled = scaler.transform(df_input)
        pred = model.predict(df_scaled)[0]
        label = label_map.get(pred, "Unknown")
        st.success(f"🎯 Predicted Category: **{label}**")

# --- CSV Upload ---
else:
    st.subheader("📁 Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if not all(col in df.columns for col in top_features):
                missing = set(top_features) - set(df.columns)
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            else:
                X = df[top_features].fillna(0)
                X_scaled = scaler.transform(X)
                preds = model.predict(X_scaled)
                df["Predicted Category"] = [label_map.get(p, "Unknown") for p in preds]

                st.success("✅ Prediction complete!")
                st.dataframe(df)

                # Download button
                csv = df.to_csv(index=False).encode()
                st.download_button("📥 Download Predictions", csv, "customer_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
