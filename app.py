import streamlit as st
import pandas as pd
from src.Pipeline.predict_pipeline import PredictPipeline, CustomData

st.title("📊 Customer Lifetime Value (CLV) Prediction")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
location = st.text_input("Location", "New York")
income_level = st.selectbox("Income Level", ["Low", "Medium", "High"])
purchase_freq = st.number_input("Purchase Frequency", min_value=1, value=5)
avg_order_value = st.number_input("Average Order Value", min_value=1, value=100)
recency_days = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
tenure_days = st.number_input("Tenure (days as customer)", min_value=1, value=365)
churned = st.selectbox("Churned?", [0, 1])

if st.button("Predict CLV"):
    # Create custom data
    input_data = CustomData(
        Age=age,
        Gender=gender,
        Location=location,
        IncomeLevel=income_level,
        PurchaseFrequency=purchase_freq,
        AvgOrderValue=avg_order_value,
        RecencyDays=recency_days,
        TenureDays=tenure_days,
        Churned=churned,
    )

    df = input_data.get_data_as_data_frame()
    st.write("✅ Input data:", df)

    # Run prediction
    pipeline = PredictPipeline()
    prediction = pipeline.predict(df)

    st.success(f"Predicted CLV: {prediction[0]:.2f}")
