import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
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
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)
    return df

df = load_data()

# -------------------------------
# LOAD MODELS
# -------------------------------
@st.cache_resource
def load_xgb_model():
    # XGBoost is usually small enough to load from disk
    return joblib.load("models/xgboost_model.pkl")

@st.cache_resource
def get_sarima_model(_df):
    # Fit SARIMA on the fly using your best parameters from the notebook
    # Replace these orders with your actual tuned parameters
    p, d, q = 1, 1, 1 
    P, D, Q, s = 1, 1, 1, 12 
    
    model = SARIMAX(_df['Close'], 
                    order=(p, d, q), 
                    seasonal_order=(P, D, Q, s))
    results = model.fit(disp=False)
    return results

# Load data first
df = load_data()

# Initialize models
tuned_model = load_xgb_model()
sarima_model = get_sarima_model(df)

st.write(f"Model type: {type(sarima_model)}")
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
    # 1. Recreate the features the model was trained on
    df_feat = df.copy()
    
    # Simple Moving Averages
    df_feat['MA_5'] = df_feat['Close'].rolling(window=5).mean()
    df_feat['MA_10'] = df_feat['Close'].rolling(window=10).mean()
    
    # Volatility (Standard Deviation of returns)
    df_feat['Returns'] = df_feat['Close'].pct_change()
    df_feat['Volatility_5'] = df_feat['Returns'].rolling(window=5).std()
    
    # Momentum (Price difference)
    df_feat['Momentum_5'] = df_feat['Close'] - df_feat['Close'].shift(5)
    
    # 2. Clean up NAs caused by rolling windows
    df_feat.dropna(inplace=True)

    # 3. SELECT THE EXACT FEATURES IN THE EXACT ORDER
    # The model expects these 9 specific columns
    features = ['Open', 'High', 'Low', 'Adj Close', 'Volume', 'MA_5', 'MA_10', 'Volatility_5', 'Momentum_5']
    X = df_feat[features]

    # 4. Predict
    # We take the most recent data to predict the next steps
    future_returns = tuned_model.predict(X.tail(forecast_days))

    # --- Plotting logic remains the same ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_feat['Returns'].tail(100).values, label="Actual Returns")
    ax.plot(future_returns, label="XGBoost Predicted Returns", linestyle='--')
    ax.legend()
    st.pyplot(fig)

    st.write("📌 XGBoost Forecasted Returns")
    st.dataframe(pd.DataFrame(future_returns, columns=['Predicted Returns']))

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

