import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Apple Stock Price Prediction", layout="wide")

st.title("📈 Apple Stock Price Prediction")
st.write("SARIMA & XGBoost based Time Series Forecasting")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("AAPL.csv")   # use same CSV as notebook
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

df = load_data()

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_models():
    with open("models/sarima_model.pkl", "rb") as f:
        sarima_model = pickle.load(f)

    with open("models/xgb_model.pkl", "rb") as f:
        tuned_model = pickle.load(f)

    return sarima_model, tuned_model

sarima_model, tuned_model = load_models()

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Model Options")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ("SARIMA", "XGBoost")
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=1,
    max_value=60,
    value=30
)

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("📊 Data Preview")
st.dataframe(df.tail())

# -------------------------------
# HISTORICAL PLOT
# -------------------------------
st.subheader("📈 Historical Close Price")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df['Close'], label="Close Price")
ax.legend()
st.pyplot(fig)

# -------------------------------
# FORECASTING
# -------------------------------
st.subheader("🔮 Forecast")

if model_choice == "SARIMA":
    forecast = sarima_model.forecast(steps=forecast_days)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Close'], label="Historical")
    ax.plot(
        pd.date_range(df.index[-1], periods=forecast_days + 1, freq='B')[1:],
        forecast,
        label="SARIMA Forecast"
    )
    ax.legend()
    st.pyplot(fig)

    st.write("📌 SARIMA Forecast Values")
    st.dataframe(forecast)

# -------------------------------
# XGBOOST PREDICTION
# -------------------------------
else:
    # Same feature engineering logic as notebook
    df_feat = df.copy()
    df_feat['Returns'] = df_feat['Close'].pct_change()
    df_feat.dropna(inplace=True)

    X = df_feat[['Returns']]
    y = df_feat['Returns']

    future_returns = tuned_model.predict(X.tail(forecast_days))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y.tail(100).values, label="Actual Returns")
    ax.plot(future_returns, label="XGBoost Predicted Returns")
    ax.legend()
    st.pyplot(fig)

    st.write("📌 XGBoost Forecasted Returns")
    st.dataframe(future_returns)

# -------------------------------
# MODEL METRICS (STATIC FROM NOTEBOOK)
# -------------------------------
st.subheader("📐 Model Performance (From Training)")

col1, col2 = st.columns(2)

with col1:
    st.metric("XGBoost Test MAE", "0.0124")
    st.metric("XGBoost Test RMSE", "0.0175")

with col2:
    st.metric("SARIMA MAE", "20.76")
    st.metric("SARIMA RMSE", "26.36")
