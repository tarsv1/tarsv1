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

# --- HELPER FUNCTION ---
def to_scalar(x) -> float:
    """Return last scalar value from pandas object / ndarray / scalar safely."""
    try:
        if isinstance(x, pd.DataFrame):
            if x.shape[1] > 0: return float(x.iloc[-1, 0])
            return float("nan")
        if isinstance(x, pd.Series): return float(x.iloc[-1])
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x).ravel()
            return float(arr[-1]) if arr.size else float("nan")
        return float(x)
    except (ValueError, IndexError):
        return float("nan")

# --- INDICATOR FUNCTIONS ---
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
    adx_filled = adx_series.bfill().ffill().fillna(0)
    plus_di_filled = plus_di.bfill().ffill().fillna(0)
    minus_di_filled = minus_di.bfill().ffill().fillna(0)
    return plus_di_filled, minus_di_filled, adx_filled

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _di_lines(df, period)[2]

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume.fillna(0)).cumsum()

def _find_swings(series: pd.Series, window: int = 5) -> tuple[list[int], list[int]]:
    """Return indices of swing highs and swing lows using a rolling window."""
    arr = series.to_numpy()
    highs, lows = [], []
    for i in range(window, len(arr) - window):
        seg = arr[i - window:i + window + 1]
        if np.argmax(seg) == window: highs.append(i)
        if np.argmin(seg) == window: lows.append(i)
    return highs, lows

def detect_divergences(df: pd.DataFrame, rsi_vals: pd.Series, macd_line: pd.Series,
                         confirmation_data: dict = None) -> tuple[pd.Series, pd.Series]:
    """
    Detects historical divergences and returns them as boolean Series.
    If confirmation_data is provided, it filters these signals for higher quality.
    """
    close = df["Close"]
    highs, lows = _find_swings(close, window=5)

    bullish_div = pd.Series(False, index=df.index)
    bearish_div = pd.Series(False, index=df.index)
    min_price_change_pct = 0.005

    for i in range(1, len(lows)):
        idx1, idx2 = lows[i-1], lows[i]
        # FIX: Ensure values are scalars before comparison to prevent ValueError
        price1, price2 = to_scalar(close.iloc[idx1]), to_scalar(close.iloc[idx2])
        
        if np.isfinite(price1) and np.isfinite(price2) and price2 < price1 * (1 - min_price_change_pct):
            if to_scalar(rsi_vals.iloc[idx2]) > to_scalar(rsi_vals.iloc[idx1]) or to_scalar(macd_line.iloc[idx2]) > to_scalar(macd_line.iloc[idx1]):
                bullish_div.iloc[idx2] = True

    for i in range(1, len(highs)):
        idx1, idx2 = highs[i-1], highs[i]
        # FIX: Ensure values are scalars before comparison
        price1, price2 = to_scalar(close.iloc[idx1]), to_scalar(close.iloc[idx2])

        if np.isfinite(price1) and np.isfinite(price2) and price2 > price1 * (1 + min_price_change_pct):
            if to_scalar(rsi_vals.iloc[idx2]) < to_scalar(rsi_vals.iloc[idx1]) or to_scalar(macd_line.iloc[idx2]) < to_scalar(macd_line.iloc[idx1]):
                bearish_div.iloc[idx2] = True

    if confirmation_data:
        sma200 = confirmation_data.get("sma200")
        vol_avg = confirmation_data.get("vol_avg")
        rsi_long = confirmation_data.get("rsi_long")
        vol_mult = confirmation_data.get("vol_mult", 1.5)
        vol = df["Volume"]
        base_index = df.index

        # FIX: Reindex all filter masks to ensure they align perfectly before boolean operations.
        # Fill missing values (from rolling/diff operations) with False.
        is_in_uptrend = (close > sma200).reindex(base_index, fill_value=False)
        is_in_downtrend = (close < sma200).reindex(base_index, fill_value=False)
        has_volume_spike = (vol > (vol_avg * vol_mult)).reindex(base_index, fill_value=False)
        htf_momentum_agrees_bullish = (rsi_long.diff() > 0).reindex(base_index, fill_value=False)
        htf_momentum_agrees_bearish = (rsi_long.diff() < 0).reindex(base_index, fill_value=False)

        # Create combined filter masks
        bullish_filter = is_in_downtrend & has_volume_spike & htf_momentum_agrees_bullish
        bearish_filter = is_in_uptrend & has_volume_spike & htf_momentum_agrees_bearish

        # Apply the filters. Use standard '&' instead of in-place '&=' for safety.
        bullish_div = bullish_div & bullish_filter
        bearish_div = bearish_div & bearish_filter

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


def add_technical_indicators(df: pd.DataFrame, selected_indicators: list,
                             rsi_window: int = 14,
                             macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                             atr_window: int = 14, adx_window: int = 14) -> pd.DataFrame:
    """
    Adds selected technical indicators to the DataFrame.
    """
    display_log("⚡ Adding Technical Indicators (SYNCED WITH SCANNER)...", "info")
    df_copy = df.copy()

    if df_copy.empty: return df

    try:
        close_series = df_copy['Close'].squeeze()
        volume_series = df_copy['Volume'].squeeze()
        
        if 'RSI_WINDOW' in selected_indicators:
            df_copy['RSI'] = rsi(close_series, period=rsi_window)

        if 'MACD_SHORT_WINDOW' in selected_indicators:
            macd_line, macd_sig, macd_hist = macd_lines(close_series, fast=macd_fast, slow=macd_slow, signal=macd_signal)
            df_copy['MACD'], df_copy['MACD_Signal'], df_copy['MACD_Diff'] = macd_line, macd_sig, macd_hist

        if 'ATR_WINDOW' in selected_indicators:
            df_copy['ATR'] = atr(df_copy, period=atr_window)

        if 'ADX_WINDOW' in selected_indicators:
            plus_di, minus_di, adx_series = _di_lines(df_copy, period=adx_window)
            df_copy['ADX'], df_copy['PLUS_DI'], df_copy['MINUS_DI'] = adx_series, plus_di, minus_di

        if 'OBV' in selected_indicators:
            df_copy['OBV'] = obv(close_series, volume_series)

        if 'MA_WINDOWS' in selected_indicators:
            for window in [10, 20, 50, 100]:
                 df_copy[f'SMA_{window}'] = sma(close_series, period=window)

        if 'BB_WINDOW' in selected_indicators:
            bb = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
            df_copy['BBL'], df_copy['BBM'], df_copy['BBU'] = bb.bollinger_lband(), bb.bollinger_mavg(), bb.bollinger_hband()

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
    """
    display_log("🌍 Adding Real Macroeconomic Indicators...", "info")
    df_copy = df.copy()
    macro_tickers_map = {name: ticker for name, ticker in st.session_state.get('global_market_tickers', {}).items()}
    selected_macro_tickers = {k: v for k, v in macro_tickers_map.items() if k in global_factors}
    start_date, end_date = df_copy.index.min().strftime('%Y-%m-%d'), df_copy.index.max().strftime('%Y-%m-%d')
    for factor_name, yf_ticker in selected_macro_tickers.items():
        col_name = f"{factor_name.replace(' ', '_').replace('-', '_')}_Close"
        try:
            macro_df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not macro_df.empty:
                macro_series = macro_df['Close'].rename(col_name)
                df_copy = df_copy.join(macro_series)
        except Exception as e:
            display_log(f"❌ Error fetching/adding {yf_ticker} ({factor_name}): {e}", "error")
    df_copy.ffill(inplace=True)
    df_copy.bfill(inplace=True)
    display_log(f"✅ Real macroeconomic indicators added → new shape: {df_copy.shape}", "info")
    return df_copy


def add_fundamental_indicators(df: pd.DataFrame, ticker: str, selected_fund_indicators: list) -> pd.DataFrame:
    """
    Fetches and adds selected fundamental indicators to the DataFrame.
    """
    display_log("📈 Adding Fundamental Indicators...", "info")
    df_copy = df.copy()
    if not selected_fund_indicators: return df_copy
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

