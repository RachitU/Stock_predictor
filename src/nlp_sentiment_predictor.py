import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import streamlit as st

# ------------------------------------------------------------
# Technical Indicator Helpers
# ------------------------------------------------------------

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, min_periods=1).mean()
    ema_slow = series.ewm(span=slow, min_periods=1).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, min_periods=1).mean()
    return macd, signal_line, macd - signal_line

def compute_bollinger_bands(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = upper - lower
    return upper, lower, width

def compute_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return tr.rolling(window=period).mean()

def compute_obv(df):
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0)
    return obv.cumsum()

# ------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------

def prepare_features(stock_df: pd.DataFrame, sentiment_value: float = 0.0) -> pd.DataFrame:
    """Generate technical indicators + sentiment feature."""
    df = stock_df.copy()

    # Moving averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    # Indicators
    df['RSI'] = compute_rsi(df['Close'], 14)
    df['MACD'], df['MACD_signal'], df['MACD_diff'] = compute_macd(df['Close'])
    df['volatility'] = df['Close'].rolling(window=10).std()
    df['BB_upper'], df['BB_lower'], df['BB_width'] = compute_bollinger_bands(df['Close'])
    df['ATR'] = compute_atr(df)
    df['OBV'] = compute_obv(df)
    df['sentiment_score'] = sentiment_value

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_lag_{lag}'] = df['Close'].shift(lag)

    # Targets + date features
    df['Target_T_plus_1'] = df['Close'].shift(-1)
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    return df.dropna()

# ------------------------------------------------------------
# Model Training
# ------------------------------------------------------------

@st.cache_resource
def train_prediction_model(stock_df: pd.DataFrame, sentiment_value: float = 0.0):
    """Train lightweight XGBoost model suitable for Streamlit Cloud."""
    df = prepare_features(stock_df, sentiment_value)

    feature_cols = [
        'MA_5', 'MA_10', 'MA_20', 'RSI', 'MACD', 'MACD_signal', 'MACD_diff',
        'volatility', 'BB_width', 'ATR', 'OBV',
        'Close_lag_1', 'Close_lag_2', 'Close_lag_3', 'Close_lag_5', 'Close_lag_10',
        'day_of_week', 'month', 'sentiment_score'
    ]
    target_col = 'Target_T_plus_1'

    model_data_cols = feature_cols + [target_col]
    valid_cols = [col for col in model_data_cols if col in df.columns]
    model_data = df[valid_cols].dropna()

    if len(model_data) < 50:
        st.warning("⚠️ Insufficient data for training — try a longer time period.")
        return None, None, df, None

    X = model_data[feature_cols].values
    y = model_data[target_col].values
    split_idx = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Simple, efficient model
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        objective='reg:squarederror',
        n_jobs=1
    )

    xgb_model.fit(X_train_scaled, y_train)
    score = xgb_model.score(X_test_scaled, y_test)
    print(f"✅ Model trained successfully. R² Score: {score:.4f}")

    return xgb_model, feature_cols, df, scaler

# ------------------------------------------------------------
# Forecast Function
# ------------------------------------------------------------

def forecast_future_prices(model, scaler, df, feature_cols, forecast_days=7):
    """Predict next few days of stock prices."""
    if model is None or scaler is None:
        return None

    future_prices = []
    df_copy = df.copy()
    for _ in range(forecast_days):
        last_row = df_copy.iloc[-1:].copy()
        next_features = last_row[feature_cols].values
        next_scaled = scaler.transform(next_features)
        next_pred = model.predict(next_scaled)[0]
        future_prices.append(next_pred)

        # Append prediction as next day for next iteration
        new_row = last_row.copy()
        new_row['Close'] = next_pred
        new_row['Target_T_plus_1'] = next_pred
        df_copy = pd.concat([df_copy, new_row]).iloc[1:]

    return future_prices
