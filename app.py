import pandas as pd
import streamlit as st
import pickle
from datetime import date
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load model and scaler
# -----------------------------
with open("models.pkl", "rb") as f:
    models = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

model = models["RandomForestRegressor"]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏠 Real Estate House Price Prediction App")

col1, col2 = st.columns(2)

with col1:
    X1 = st.date_input(
    "📅 Select a transaction date:",
    value=date(2012,9, 17),           # Default date (start at 2015)
    min_value=date(2010, 1, 1),       # Earliest possible date
    max_value=date(2025, 12, 31)      # Latest possible date
)
    X2 = st.number_input("House age (years):", min_value=0.0, value=19.5)
    X3 = st.number_input("Distance to nearest MRT station (m):", value=306.5947)

with col2:
    X4 = st.number_input("Number of convenience stores:", min_value=0.0, value=9.0)
    X5 = st.number_input("Latitude:", min_value=0.0, value=24.98034)
    X6 = st.number_input("Longitude:", min_value=0.0, value=121.53951)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Price"):
    # Convert the date to numeric
    X1_num = X1.year + X1.month / 12

    input_data = pd.DataFrame([[
        X1_num, X2, X3, X4, X5, X6
    ]], columns=[
        'X1 transaction date',
        'X2 house age',
        'X3 distance to the nearest MRT station',
        'X4 number of convenience stores',
        'X5 latitude',
        'X6 longitude'
    ])

    # Scale the data
    scaled_input = scaler.transform(input_data)

    # Predict
    price = model.predict(scaled_input)[0]

    st.success(f"### 💰 Predicted house price per unit area: **{price:.2f}**")

st.write("---")
st.caption("Built with ❤️ using Streamlit and Scikit-learn")
