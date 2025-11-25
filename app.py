import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import streamlit as st

# Page config
st.set_page_config(
    page_title="Stock Forecasting LSTM",
    page_icon="📈",
    layout="wide"
)

# Sidebar controls
with st.sidebar:
    st.title("⚙ Controls")

    user_input = st.text_input("Enter Stock Ticker:", "TATASTEEL.NS")
    start = st.date_input("Start Date", datetime(2020, 1, 1))
    end = st.date_input("End Date", datetime.today())

    st.write("### Show/Hide Sections")
    show_ma = st.checkbox("Moving Averages", True)
    show_pred_plot = st.checkbox("Prediction vs Original", True)
    show_forecast = st.checkbox("10-Day Forecast", True)

# Fetch data
with st.spinner("📥 Fetching Stock Data..."):
    df = yf.download(user_input, start=start, end=end, auto_adjust=True)

# Page title
st.title("📈 Stock Price Forecasting with LSTM")

# Stock info
st.subheader("Stock Information")
st.write(df.describe())

# Splitting data into Train & Test sets
train_df = df[['Close', 'Open', 'High', 'Low', 'Volume']][0: int(len(df) * 0.80)]
test_df = df[['Close', 'Open', 'High', 'Low', 'Volume']][int(len(df) * 0.80):]

# Feature scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_scaled = sc.fit_transform(train_df)

# Loading the trained model
best_model = load_model("optimized_lstm_model.keras")

# Preparing test data
past_100 = train_df.tail(100)
final_df = pd.concat([past_100, test_df], ignore_index=True)
input_data = sc.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Model predictions
with st.spinner("🔮 Predicting test data…"):
    y_pred = best_model.predict(x_test)

# Inverse transform predictions
temp_pred = np.zeros((len(y_pred), 5))
temp_pred[:, 0] = y_pred.flatten()
y_pred_unscaled = sc.inverse_transform(temp_pred)[:, 0]

temp_test = np.zeros((len(y_test), 5))
temp_test[:, 0] = y_test.flatten()
y_test_unscaled = sc.inverse_transform(temp_test)[:, 0]

# Model evaluation metrics
mape = mean_absolute_percentage_error(y_test_unscaled, y_pred_unscaled) * 100
r2 = r2_score(y_test_unscaled, y_pred_unscaled)
accuracy = 100 - mape

st.subheader("📊 Model Evaluation Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("MAPE (%)", f"{mape:.2f}")
col2.metric("Accuracy (%)", f"{accuracy:.2f}")
col3.metric("R² Score", f"{r2:.4f}")

# Moving average charts
if show_ma:
    st.subheader("Closing Price with Moving Averages")

    mavg100 = df.Close.rolling(100).mean()
    mavg200 = df.Close.rolling(200).mean()

    fig2 = plt.figure(figsize=(12, 6))
    plt.plot(df.Close, label="Close")
    plt.plot(mavg100, label="100-day MA", color='red')
    plt.plot(mavg200, label="200-day MA", color='orange')
    plt.legend()
    st.pyplot(fig2)

# Prediction vs Original Graph
if show_pred_plot:
    st.subheader("Prediction vs Original (Test Data)")

    fig3 = plt.figure(figsize=(12, 6))
    plt.plot(y_test_unscaled, label="Original")
    plt.plot(y_pred_unscaled, 'green', label="Predicted")
    plt.legend()
    st.pyplot(fig3)

# 10-day Future Prediction
if show_forecast:
    st.subheader("📅 10-Day Future Forecast")

    last_100_scaled = input_data[-100:]
    window = last_100_scaled.copy()

    future_predictions_scaled = []

    for _ in range(10):  # changed from 30 → 10
        pred_close_scaled = best_model.predict(window.reshape(1, 100, 5))[0][0]

        last_row = window[-1].copy()
        next_row = last_row.copy()
        next_row[0] = pred_close_scaled  # update close only

        window = np.vstack([window[1:], next_row])
        future_predictions_scaled.append(pred_close_scaled)

    temp_future = np.zeros((len(future_predictions_scaled), 5))
    temp_future[:, 0] = future_predictions_scaled
    future_unscaled = sc.inverse_transform(temp_future)[:, 0]

    future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=10)

    fig4 = plt.figure(figsize=(12, 6))
    plt.plot(df.Close[-100:], label="Last 100 Days")
    plt.plot(future_dates, future_unscaled, label="10-Day Forecast", color='green')
    plt.legend()
    st.pyplot(fig4)

st.markdown("---")
st.markdown("#### © By **Shivang Verma – 11|2025**")
