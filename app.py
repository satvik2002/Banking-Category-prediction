import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page setup
st.set_page_config(page_title="Customer Category Predictor", layout="centered")
st.title("🏦 Customer Category Prediction App")

# Load model, scaler, encoder
model = joblib.load("rf_final_model1.pkl")
scaler = joblib.load("scaler1.pkl")
encoder = joblib.load("encoder.pkl")  # LabelEncoder

# Feature list used during training
top_features = [
    'Credit_History_Age_Months', 'Outstanding_Debt', 'Num_Credit_Inquiries',
    'Interest_Rate', 'Delay_from_due_date', 'Num_Bank_Accounts',
    'Num_Credit_Card', 'Monthly_Balance', 'Annual_Income', 'Age',
    'Num_of_Delayed_Payment', 'Monthly_Inhand_Salary', 'Personal Loan',
    'Credit_Utilization_Ratio', 'Mortgage Loan'
]

# Sidebar input method selection
st.sidebar.header("📥 Select Input Method")
input_mode = st.sidebar.radio("Choose how to enter data:", ["Manual Entry", "Upload CSV"])

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
        pred_class = model.predict(df_scaled)[0]
        pred_label = encoder.inverse_transform([pred_class])[0]
        st.success(f"🎯 Predicted Category: **{pred_label}**")

# --- CSV Upload Mode ---
else:
    st.subheader("📁 Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            # Ensure required columns are present
            if not all(f in df.columns for f in top_features):
                st.error(f"❌ CSV must include: {', '.join(top_features)}")
            else:
                X = df[top_features].fillna(0)
                X_scaled = scaler.transform(X)
                preds = model.predict(X_scaled)
                df["Predicted Category"] = encoder.inverse_transform(preds)

                st.success("✅ Predictions complete")
                st.dataframe(df)

                csv_data = df.to_csv(index=False).encode()
                st.download_button("📥 Download Predictions", csv_data, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
