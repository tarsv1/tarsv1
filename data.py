# data.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta # Technical Analysis library - KEPT FOR INDICATORS NOT IN SCANNER
from datetime import datetime
import traceback

# Import display_log from utils for consistent logging
from utils import display_log

# --- INDICATOR FUNCTIONS COPIED FROM 1_Scanner.py ---
# This centralizes the logic so the scanner and backtester use the exact same calculations.

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=1).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd_lines(series: pd.Series, fast=12, slow=26, signal=9):
    macd = ema(series, fast) - ema(series, slow)
    sig = ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    # Wilder-style smoothing approximated with EWM alpha=1/period
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _di_lines(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return +DI, -DI, ADX (Wilder-style smoothing via EWM alpha=1/period)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    tr_smoothed = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_sm = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_dm_sm = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm_sm / tr_smoothed.replace(0, np.nan))
    minus_di = 100 * (minus_dm_sm / tr_smoothed.replace(0, np.nan))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_series = dx.ewm(alpha=1/period, adjust=False).mean()
    return plus_di, minus_di, adx_series

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Kept for compatibility where only ADX is needed."""
    return _di_lines(df, period)[2]

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume.fillna(0)).cumsum()

# --- END OF COPIED FUNCTIONS ---
# Add these two functions to data.py

def _find_swings(series: pd.Series, window: int = 5) -> tuple[list[int], list[int]]:
    """Return indices of swing highs and swing lows using a rolling window."""
    arr = series.to_numpy()
    highs, lows = [], []
    for i in range(window, len(arr) - window):
        seg = arr[i - window:i + window + 1]
        if np.argmax(seg) == window:
            highs.append(i)
        if np.argmin(seg) == window:
            lows.append(i)
    return highs, lows

def detect_divergences(df: pd.DataFrame, rsi_vals: pd.Series, macd_line: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Detects historical divergences and returns them as Series.
    Returns two boolean series: one for bullish divergences, one for bearish.
    """
    close = df["Close"]
    highs, lows = _find_swings(close, window=5) # Use a fixed window for performance

    bullish_div = pd.Series(False, index=df.index)
    bearish_div = pd.Series(False, index=df.index)

    # Simplified loop for historical detection
    for i in range(1, len(lows)):
        idx1, idx2 = lows[i-1], lows[i]
        # Price lower low, RSI higher low -> Bullish
        if close.iloc[idx2] < close.iloc[idx1] and rsi_vals.iloc[idx2] > rsi_vals.iloc[idx1]:
            bullish_div.iloc[idx2] = True
        # Price lower low, MACD higher low -> Bullish
        if close.iloc[idx2] < close.iloc[idx1] and macd_line.iloc[idx2] > macd_line.iloc[idx1]:
            bullish_div.iloc[idx2] = True

    for i in range(1, len(highs)):
        idx1, idx2 = highs[i-1], highs[i]
        # Price higher high, RSI lower high -> Bearish
        if close.iloc[idx2] > close.iloc[idx1] and rsi_vals.iloc[idx2] < rsi_vals.iloc[idx1]:
            bearish_div.iloc[idx2] = True
        # Price higher high, MACD lower high -> Bearish
        if close.iloc[idx2] > close.iloc[idx1] and macd_line.iloc[idx2] < macd_line.iloc[idx1]:
            bearish_div.iloc[idx2] = True

    return bullish_div, bearish_div


def fetch_stock_data(ticker: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance, with robust column handling.
    """
    display_log(f"🔄 Data Fetching Started for {ticker}...", "info")
    try:
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        df = yf.download(ticker, start=start_str, end=end_str, progress=False, auto_adjust=False)

        if df.empty:
            display_log(f"❗ No data found for {ticker}.", "warning")
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
            if 'Adj Close' != 'Close':
                df.drop(columns=['Adj Close'], inplace=True)

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            display_log(f"❌ Critical: Missing one of {required_cols}. Cannot proceed.", "error")
            return pd.DataFrame()

        df = df[required_cols]
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(inplace=True)
        display_log(f"📈 Data fetched and cleaned for {ticker} → final shape: {df.shape}", "info")
        return df
    except Exception as e:
        display_log(f"❌ Error in fetch_stock_data for {ticker}: {e}", "error")
        return pd.DataFrame()


# --- REPLACED FUNCTION ---
def add_technical_indicators(df: pd.DataFrame, selected_indicators: list,
                             rsi_window: int = 14,
                             macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                             atr_window: int = 14, adx_window: int = 14) -> pd.DataFrame:
    """
    Adds selected technical indicators to the DataFrame.
    MODIFIED: This version uses the exact indicator calculation functions from 1_Scanner.py
    to ensure consistency between the scanner and the backtester. Indicators not
    present in the scanner (e.g., Bollinger Bands) will still use the 'ta' library.
    """
    display_log("⚡ Adding Technical Indicators (SYNCED WITH SCANNER)...", "info")
    df_copy = df.copy()

    if df_copy.empty:
        return df

    try:
        # Standardize required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df_copy.columns:
                display_log(f"❗ Missing required column '{col}' for TA calculation.", "warning")
                return df
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').ffill().bfill()

        close_series = df_copy['Close']
        volume_series = df_copy['Volume']

        # --- Synced Indicators (using scanner functions) ---
        # The `selected_indicators` list contains keys from the config.py file.
        
        if 'RSI_WINDOW' in selected_indicators:
            df_copy['RSI'] = rsi(close_series, period=rsi_window)

        if 'MACD_SHORT_WINDOW' in selected_indicators:
            macd_line, macd_sig, macd_hist = macd_lines(close_series, fast=macd_fast, slow=macd_slow, signal=macd_signal)
            df_copy['MACD'] = macd_line
            df_copy['MACD_Signal'] = macd_sig
            df_copy['MACD_Diff'] = macd_hist

        if 'ATR_WINDOW' in selected_indicators:
            df_copy['ATR'] = atr(df_copy, period=atr_window)

        if 'ADX_WINDOW' in selected_indicators:
            plus_di, minus_di, adx_series = _di_lines(df_copy, period=adx_window)
            df_copy['ADX'] = adx_series
            df_copy['PLUS_DI'] = plus_di
            df_copy['MINUS_DI'] = minus_di

        if 'OBV' in selected_indicators:
            df_copy['OBV'] = obv(close_series, volume_series)

        # --- Unsynced Indicators (not in scanner, using 'ta' library) ---

        if 'MA_WINDOWS' in selected_indicators:
            # NOTE: These are general MAs. The critical MA Cross signals are generated in
            # `signal_features.py`, which correctly uses the synced scanner parameters.
            for window in [10, 20, 50, 100]:
                 df_copy[f'SMA_{window}'] = sma(close_series, period=window)

        if 'BB_WINDOW' in selected_indicators:
            bb = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
            df_copy['BBL'] = bb.bollinger_lband()
            df_copy['BBM'] = bb.bollinger_mavg()
            df_copy['BBU'] = bb.bollinger_hband()

        # Final cleanup
        df_copy.fillna(method='ffill', inplace=True)
        df_copy.fillna(method='bfill', inplace=True)

        display_log(f"✅ Synced technical indicators added. Current shape: {df_copy.shape}", "info")
        return df_copy
    except Exception as e:
        display_log(f"❌ Error adding synced technical indicators: {e}. Traceback: {traceback.format_exc()}", "error")
        st.exception(e)
        return df


def add_macro_indicators(df: pd.DataFrame, global_factors: list) -> pd.DataFrame:
    """
    Fetches and adds real macroeconomic indicators to the DataFrame using yfinance.
    (This function is unchanged)
    """
    display_log("🌍 Adding Real Macroeconomic Indicators...", "info")
    df_copy = df.copy()
    
    macro_tickers_map = {name: ticker for name, ticker in st.session_state.get('global_market_tickers', {}).items()}
    selected_macro_tickers = {k: v for k, v in macro_tickers_map.items() if k in global_factors}

    start_date = df_copy.index.min().strftime('%Y-%m-%d')
    end_date = df_copy.index.max().strftime('%Y-%m-%d')

    for factor_name, yf_ticker in selected_macro_tickers.items():
        col_name = f"{factor_name.replace(' ', '_').replace('-', '_')}_Close"
        try:
            macro_df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not macro_df.empty:
                macro_series = macro_df['Close'].rename(col_name)
                df_copy = df_copy.join(macro_series)
        except Exception as e:
            display_log(f"❌ Error fetching/adding {yf_ticker} ({factor_name}): {e}", "error")

    df_copy.fillna(method='ffill', inplace=True)
    df_copy.fillna(method='bfill', inplace=True)
    
    display_log(f"✅ Real macroeconomic indicators added → new shape: {df_copy.shape}", "info")
    return df_copy


def add_fundamental_indicators(df: pd.DataFrame, ticker: str, selected_fund_indicators: list) -> pd.DataFrame:
    """
    Fetches and adds selected fundamental indicators to the DataFrame.
    (This function is unchanged)
    """
    display_log("📈 Adding Fundamental Indicators...", "info")
    df_copy = df.copy()
    
    if not selected_fund_indicators:
        return df_copy

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        fund_data = {name: info.get(key) for name, key in st.session_state.get('fundamental_metrics', {}).items() if name in selected_fund_indicators}
        
        for name, value in fund_data.items():
            if value is not None:
                df_copy[name.replace(' ', '_')] = value
        
        display_log(f"✅ Fundamental indicators added. Current shape: {df_copy.shape}", "info")
        return df_copy
        
    except Exception as e:
        display_log(f"❌ Error adding fundamental indicators for {ticker}: {e}", "error")
        return df