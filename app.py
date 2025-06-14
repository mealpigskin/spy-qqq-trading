import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="SPY/QQQ Options Timing", layout="wide")

def fetch_data(symbol, period="1y", interval="1d"):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    return df

def calculate_indicators(df):
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['ATR'] = pd.concat([df['High'] - df['Low'], (df['High'] - df['Close'].shift()).abs(), (df['Low'] - df['Close'].shift()).abs()], axis=1).max(axis=1).rolling(window=14).mean()
    df['ATR_Percentile'] = df['ATR'].rank(pct=True) * 100
    df['Vol_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    return df

def calculate_timing_score(df):
    df['EMA_Signal'] = np.where((df['EMA9'] > df['EMA21']) & (df['EMA21'] > df['SMA50']), 1, 0)
    df['RSI_Signal'] = np.where((df['RSI'] >= 35) & (df['RSI'] <= 50), 1, 0)
    df['Stoch_Signal'] = np.where((df['Stoch_K'] > df['Stoch_D']) & (df['Stoch_K'] < 20), 1, 0)
    df['VWAP_Signal'] = np.where(df['Close'] > df['VWAP'], 1, 0)
    df['IVP_Signal'] = np.where((df['ATR_Percentile'] >= 20) & (df['ATR_Percentile'] <= 40), 1, 0)
    df['PC_Signal'] = np.where(df['Vol_Ratio'] > 1.2, 1, 0)
    df['Timing_Score'] = (df['EMA_Signal'] * 0.3 + df['RSI_Signal'] * 0.2 + 
                          df['Stoch_Signal'] * 0.2 + df['VWAP_Signal'] * 0.1 + 
                          df['IVP_Signal'] * 0.1 + df['PC_Signal'] * 0.1) * 100
    df['Percentile'] = df['Timing_Score'].rank(pct=True) * 100
    return df

st.title("SPY/QQQ Options Trading Strategy Dashboard")
st.write("Real-time timing score and percentile for SPY/QQQ call options")

symbols = ["SPY", "QQQ"]
data = {}
for symbol in symbols:
    df = fetch_data(symbol)
    df = calculate_indicators(df)
    df = calculate_timing_score(df)
    data[symbol] = df

col1, col2 = st.columns(2)
for symbol, df in data.items():
    latest = df.iloc[-1]
    percentile = latest['Percentile']
    score = latest['Timing_Score']
    signal = "Buy Call" if percentile >= 80 else "Hold" if percentile >= 50 else "Avoid"
    with col1 if symbol == "SPY" else col2:
        st.subheader(symbol)
        st.metric("Timing Percentile", f"{percentile:.1f}%")
        st.metric("Timing Score", f"{score:.1f}")
        st.metric("Trade Signal", signal, delta_color="normal")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=percentile,
            title={'text': f"{symbol} Percentile"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "green" if percentile >= 80 else "yellow" if percentile >= 50 else "red"},
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}}))
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Historical Timing Score")
for symbol in symbols:
    df = data[symbol]
    fig = px.line(df, x=df.index, y="Timing_Score", title=f"{symbol} Timing Score (1 Year)")
    fig.add_scatter(x=df.index, y=df['Percentile'], mode='lines', name='Percentile', yaxis="y2")
    fig.update_layout(yaxis2={'title': 'Percentile', 'overlaying': 'y', 'side': 'right'})
    st.plotly_chart(fig, use_container_width=True)

if st.button("Refresh Data"):
    st.rerun()

st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")