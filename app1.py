import streamlit as st
import os
import joblib
import pandas as pd

st.write("Current folder:", os.getcwd())

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Customer Segmentation App")
st.write("Enter customer details to predict the segment.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    income = st.number_input("Income", min_value=0, max_value=200000, value=50000)
    total_spent = st.number_input("Total Spent (sum of all purchases)", min_value=0, max_value=3000, value=600)
    total_purchases = st.number_input("Total Purchases (web + catalog + store)", min_value=0, max_value=50, value=12)
    recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=100, value=50)

with col2:
    children = st.number_input("Children (kids + teens at home)", min_value=0, max_value=5, value=1)
    web_visits = st.number_input("Web Visits per Month", min_value=0, max_value=30, value=5)
    deals = st.number_input("Deals Purchases", min_value=0, max_value=20, value=2)
    campaigns = st.number_input("Campaigns Accepted (0-5)", min_value=0, max_value=5, value=0)
    customer_days = st.number_input("Customer Days (days since joining)", min_value=0, max_value=1000, value=350)

# Same column names and order as the 'features' list used for training
input_data = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Total_Spent": [total_spent],
    "Total_Purchases": [total_purchases],
    "Recency": [recency],
    "Children": [children],
    "NumWebVisitsMonth": [web_visits],
    "NumDealsPurchases": [deals],
    "Total_Campaigns": [campaigns],
    "Customer_Days": [customer_days],
})

input_scaled = scaler.transform(input_data)

if st.button("Predict Segment"):
    cluster = kmeans.predict(input_scaled)[0]
    st.success(f"Predicted Segment: Cluster {cluster}")
