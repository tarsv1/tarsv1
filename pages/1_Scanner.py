import io
import base64
import re
from typing import Any, Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# FIX: Import indicator logic from data.py to act as a single source of truth
from data import sma, ema, rsi, macd_lines, atr, _di_lines, adx, obv

# Page config
st.set_page_config(page_title="Patterns & Strategy Scanner (NSE)", layout="wide")
st.title("📊 Patterns & Strategy Scanner (NSE)")

# ------------------------
# Safe helpers
# ------------------------
def to_scalar(x) -> float:
    """Return last scalar value from pandas object / ndarray / scalar safely."""
    try:
        if isinstance(x, pd.DataFrame):
            if x.shape[1] > 0:
                return float(x.iloc[-1, 0])
            return float("nan")
        if isinstance(x, pd.Series):
            return float(x.iloc[-1])
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x).ravel()
            return float(arr[-1]) if arr.size else float("nan")
        return float(x)
    except Exception:
        return float("nan")

def last_val(series: pd.Series) -> float:
    try:
        return float(series.to_numpy().ravel()[-1])
    except Exception:
        return float("nan")

def prev_val(series: pd.Series) -> float:
    try:
        arr = series.to_numpy().ravel()
        return float(arr[-2]) if arr.size >= 2 else float("nan")
    except Exception:
        return float("nan")

# ------------------------
# Profiles
# ------------------------
PROFILES: Dict[str, Dict[str, Any]] = {
    "Short Term": {
        "ma_fast": 5, "ma_slow": 20,
        "rsi_period": 7, "rsi_low": 35, "rsi_high": 65,
        "macd_fast": 8, "macd_slow": 21, "macd_signal": 5,
        "adx_period": 7, "adx_threshold": 20,
        "atr_period": 7, "atr_mult": 1.5,
        "days": 120,
        "vol_spike_mult": 1.8,
        "darvas_lookback": 20, "darvas_min_bars": 5
    },
    "Medium Term": {
        "ma_fast": 20, "ma_slow": 50,
        "rsi_period": 14, "rsi_low": 30, "rsi_high": 70,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "adx_period": 14, "adx_threshold": 25,
        "atr_period": 14, "atr_mult": 2.0,
        "days": 250,
        "vol_spike_mult": 2.0,
        "darvas_lookback": 30, "darvas_min_bars": 7
    },
    "Long Term": {
        "ma_fast": 50, "ma_slow": 200,
        "rsi_period": 21, "rsi_low": 25, "rsi_high": 75,
        "macd_fast": 19, "macd_slow": 39, "macd_signal": 9,
        "adx_period": 21, "adx_threshold": 20,
        "atr_period": 21, "atr_mult": 2.5,
        "days": 760,
        "vol_spike_mult": 2.2,
        "darvas_lookback": 40, "darvas_min_bars": 9
    }
}

# ------------------------
# Session-state profile defaults / manual override
# ------------------------
if "active_profile" not in st.session_state:
    st.session_state.active_profile = "Medium Term"
    for k, v in PROFILES["Medium Term"].items():
        st.session_state[k] = v

profile_choice = st.sidebar.radio("Profile", list(PROFILES.keys()))
if profile_choice != st.session_state["active_profile"]:
    for k, v in PROFILES[profile_choice].items():
        st.session_state[k] = v
    st.session_state["active_profile"] = profile_choice
    st.rerun()

st.sidebar.markdown("**Manual overrides (override profile defaults)**")
st.sidebar.number_input("Fast MA", min_value=1, max_value=500, key="ma_fast")
st.sidebar.number_input("Slow MA", min_value=1, max_value=2000, key="ma_slow")
st.sidebar.number_input("RSI Period", min_value=2, max_value=100, key="rsi_period")
st.sidebar.number_input("RSI Low (base)", min_value=1, max_value=60, key="rsi_low")
st.sidebar.number_input("RSI High (base)", min_value=40, max_value=99, key="rsi_high")
st.sidebar.number_input("MACD Fast", min_value=1, max_value=200, key="macd_fast")
st.sidebar.number_input("MACD Slow", min_value=1, max_value=500, key="macd_slow")
st.sidebar.number_input("MACD Signal", min_value=1, max_value=200, key="macd_signal")
st.sidebar.number_input("ADX Period", min_value=5, max_value=100, key="adx_period")
st.sidebar.number_input("ADX Threshold", min_value=1, max_value=100, key="adx_threshold")
st.sidebar.number_input("ATR Period", min_value=1, max_value=200, key="atr_period")
st.sidebar.number_input("ATR Multiplier", min_value=1.0, max_value=5.0, step=0.1, key="atr_mult", help="Affects volatility bands for RSI and other calculations.")
st.sidebar.number_input("Lookback days", min_value=10, max_value=2000, key="days")
st.sidebar.number_input("Volume spike multiplier", min_value=1.0, max_value=5.0, step=0.1, key="vol_spike_mult")
st.sidebar.number_input("Darvas lookback", min_value=5, max_value=200, key="darvas_lookback")
st.sidebar.number_input("Darvas min bars", min_value=2, max_value=60, key="darvas_min_bars")

if st.sidebar.button("🔄 Reset to Profile Defaults"):
    for k, v in PROFILES[profile_choice].items():
        st.session_state[k] = v
    st.session_state["active_profile"] = profile_choice
    st.rerun()

PARAMS = {
    "ma_fast": int(st.session_state["ma_fast"]),
    "ma_slow": int(st.session_state["ma_slow"]),
    "rsi_period": int(st.session_state["rsi_period"]),
    "rsi_low": float(st.session_state["rsi_low"]),
    "rsi_high": float(st.session_state["rsi_high"]),
    "macd_fast": int(st.session_state["macd_fast"]),
    "macd_slow": int(st.session_state["macd_slow"]),
    "macd_signal": int(st.session_state["macd_signal"]),
    "adx_period": int(st.session_state["adx_period"]),
    "adx_threshold": float(st.session_state["adx_threshold"]),
    "atr_period": int(st.session_state["atr_period"]),
    "atr_mult": float(st.session_state["atr_mult"]),
    "days": int(st.session_state["days"]),
    "vol_spike_mult": float(st.session_state["vol_spike_mult"]),
    "darvas_lookback": int(st.session_state["darvas_lookback"]),
    "darvas_min_bars": int(st.session_state["darvas_min_bars"]),
}

# ------------------------
# Robust ticker fetch / fallback
# ------------------------
@st.cache_data(ttl=24 * 3600)
def get_nse_tickers() -> List[str]:
    urls = [
        "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
        "https://www1.nseindia.com/content/equities/EQUITY_L.csv",
        "https://www.nseindia.com/content/equities/EQUITY_L.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if "SYMBOL" in df.columns:
                syms = df["SYMBOL"].astype(str).str.strip().str.upper().tolist()
                syms = [s for s in syms if re.fullmatch(r"[A-Z0-9\-]+", s)]
                if syms:
                    return sorted(set([s + ".NS" for s in syms]))
        except Exception:
            continue
    st.warning("Could not fetch NSE ticker list — using fallback subset.")
    return ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","ITC.NS","LT.NS","BHARTIARTL.NS"]

# ------------------------
# Robust fetch history (yfinance) with fallback attempts
# ------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_history(ticker: str, days: int) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    try:
        df = yf.download(ticker, period=f"{int(days)}d", progress=False, auto_adjust=False, threads=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    try:
        tk = yf.Ticker(ticker)
        df2 = tk.history(period=f"{int(days)}d", auto_adjust=False, actions=False)
        if isinstance(df2, pd.DataFrame) and not df2.empty:
            for c in ["Open","High","Low","Close","Volume"]:
                if c not in df2.columns:
                    lower = c.lower()
                    if lower in df2.columns:
                        df2[c] = df2[lower]
                    else:
                        df2[c] = np.nan
            return df2
    except Exception:
        pass
    if not ticker.endswith(".NS"):
        try:
            df3 = yf.download(ticker + ".NS", period=f"{int(days)}d", progress=False, auto_adjust=False, threads=False)
            if isinstance(df3, pd.DataFrame) and not df3.empty:
                return df3
        except Exception:
            pass
    return pd.DataFrame()

# ------------------------
# Technical Indicators (safe)
#
# FIX: All local indicator functions have been REMOVED.
# They are now imported from data.py to prevent code duplication.
# ------------------------

# ------------------------
# Candlestick patterns (stricter)
# ------------------------
def _is_bull(o, c): return np.isfinite(o) and np.isfinite(c) and c > o
def _is_bear(o, c): return np.isfinite(o) and np.isfinite(c) and c < o

def detect_patterns(df: pd.DataFrame) -> List[str]:
    # FIX 1: Guard short dataframes
    if df.shape[0] < 20:
        return []
    patterns: List[str] = []
    n = len(df)
    if n < 2:
        return patterns
    last = df.iloc[-1]
    prev = df.iloc[-2]
    o1, c1, h1, l1 = map(to_scalar, [prev["Open"], prev["Close"], prev["High"], prev["Low"]])
    o2, c2, h2, l2 = map(to_scalar, [last["Open"], last["Close"], last["High"], last["Low"]])
    if np.isfinite(o1) and np.isfinite(c1) and np.isfinite(o2) and np.isfinite(c2):
        if _is_bear(o1, c1) and _is_bull(o2, c2) and (c2 > o1) and (o2 < c1):
            patterns.append("Bullish Engulfing")
        if _is_bull(o1, c1) and _is_bear(o2, c2) and (c2 < o1) and (o2 > c1):
            patterns.append("Bearish Engulfing")
    rng2 = h2 - l2 if np.isfinite(h2) and np.isfinite(l2) else 0.0
    body2 = abs(c2 - o2) if np.isfinite(c2) and np.isfinite(o2) else 0.0
    if rng2 > 0 and body2 <= 0.1 * rng2:
        patterns.append("Doji")
    upper = h2 - max(o2, c2) if np.isfinite(h2) and np.isfinite(o2) and np.isfinite(c2) else 0.0
    lower = min(o2, c2) - l2 if np.isfinite(l2) and np.isfinite(o2) and np.isfinite(c2) else 0.0
    if rng2 > 0 and body2 > 0:
        if (lower >= 2 * body2) and (upper <= body2):
            patterns.append("Hammer")
        if (upper >= 2 * body2) and (lower <= body2):
            patterns.append("Shooting Star")
    if n >= 3:
        c0 = df.iloc[-3]
        o0, c0c = map(to_scalar, [c0["Open"], c0["Close"]])
        if _is_bear(o0, c0c) and body2 > body2 * 0:
            mid_prev = (o1 + c1) / 2 if np.isfinite(o1) and np.isfinite(c1) else np.nan
            small_mid = abs(c1 - o1) <= 0.6 * abs(c0c - o0) if np.isfinite(o0) and np.isfinite(c0c) and np.isfinite(o1) and np.isfinite(c1) else False
            if _is_bear(o0, c0c) and small_mid and _is_bull(o2, c2) and np.isfinite(mid_prev) and c2 > mid_prev:
                patterns.append("Morning Star")
        if _is_bull(o0, c0c):
            mid_prev = (o1 + c1) / 2 if np.isfinite(o1) and np.isfinite(c1) else np.nan
            small_mid = abs(c1 - o1) <= 0.6 * abs(c0c - o0) if np.isfinite(o0) and np.isfinite(c0c) and np.isfinite(o1) and np.isfinite(c1) else False
            if _is_bull(o0, c0c) and small_mid and _is_bear(o2, c2) and np.isfinite(mid_prev) and c2 < mid_prev:
                patterns.append("Evening Star")
    if _is_bear(o1, c1) and _is_bull(o2, c2):
        mid_prev = (o1 + c1) / 2
        if np.isfinite(mid_prev) and c2 > mid_prev and c2 < o1:
            patterns.append("Piercing Line")
    return patterns

# ------------------------
# Divergences (swing-based)
# ------------------------
def _find_swings(series: pd.Series, window: int = 5) -> Tuple[List[int], List[int]]:
    arr = series.to_numpy()
    highs, lows = [], []
    for i in range(len(arr)):
        start = max(0, i - window)
        end = min(len(arr), i + window + 1)
        segment = arr[start:end]
        if len(segment) > 0 and arr[i] == np.max(segment):
            highs.append(i)
        if len(segment) > 0 and arr[i] == np.min(segment):
            lows.append(i)
    unique_highs = [h for i, h in enumerate(highs) if i == 0 or highs[i-1] != h]
    unique_lows = [l for i, l in enumerate(lows) if i == 0 or lows[i-1] != l]
    return unique_highs, unique_lows

def detect_divergences(df: pd.DataFrame, rsi_vals: pd.Series, macd_line: pd.Series) -> List[str]:
    # FIX 1: Guard short dataframes
    if df.shape[0] < 20:
        return []
    out: List[str] = []
    if len(df) < 15:
        return out
    close = df["Close"]
    highs, lows = _find_swings(close, window=3)
    def last_two(idx_list: List[int]) -> Optional[Tuple[int, int]]:
        if len(idx_list) >= 2:
            return idx_list[-2], idx_list[-1]
        return None
    hpair = last_two(highs)
    lpair = last_two(lows)
    if hpair:
        i1, i2 = hpair
        if to_scalar(close.iloc[i2]) > to_scalar(close.iloc[i1]) and to_scalar(rsi_vals.iloc[i2]) < to_scalar(rsi_vals.iloc[i1]):
            out.append("Bearish RSI Divergence")
        if to_scalar(close.iloc[i2]) < to_scalar(close.iloc[i1]) and to_scalar(rsi_vals.iloc[i2]) > to_scalar(rsi_vals.iloc[i1]):
            out.append("Bullish Hidden RSI Divergence")
    if lpair:
        i1, i2 = lpair
        if to_scalar(close.iloc[i2]) < to_scalar(close.iloc[i1]) and to_scalar(rsi_vals.iloc[i2]) > to_scalar(rsi_vals.iloc[i1]):
            out.append("Bullish RSI Divergence")
        if to_scalar(close.iloc[i2]) > to_scalar(close.iloc[i1]) and to_scalar(rsi_vals.iloc[i2]) < to_scalar(rsi_vals.iloc[i1]):
            out.append("Bearish Hidden RSI Divergence")
    if hpair:
        i1, i2 = hpair
        if to_scalar(close.iloc[i2]) > to_scalar(close.iloc[i1]) and to_scalar(macd_line.iloc[i2]) < to_scalar(macd_line.iloc[i1]):
            out.append("Bearish MACD Divergence")
        if to_scalar(close.iloc[i2]) < to_scalar(close.iloc[i1]) and to_scalar(macd_line.iloc[i2]) > to_scalar(macd_line.iloc[i1]):
            out.append("Bullish Hidden MACD Divergence")
    if lpair:
        i1, i2 = lpair
        if to_scalar(close.iloc[i2]) < to_scalar(close.iloc[i1]) and to_scalar(macd_line.iloc[i2]) > to_scalar(macd_line.iloc[i1]):
            out.append("Bullish MACD Divergence")
        if to_scalar(close.iloc[i2]) > to_scalar(close.iloc[i1]) and to_scalar(macd_line.iloc[i2]) < to_scalar(macd_line.iloc[i1]):
            out.append("Bearish Hidden MACD Divergence")
    return out

# ------------------------
# Volume spike (window-safe, no look-ahead)
# ------------------------
def volume_spike(vol: pd.Series, mult: float = 2.0, window: int = 20) -> bool:
    arr = vol.to_numpy().ravel()
    n = arr.size
    if n < 6:
        return False
    look = min(window, max(1, n - 1))
    avg_prev = pd.Series(arr[:-1]).tail(look).mean()
    if not np.isfinite(avg_prev) or avg_prev == 0:
        return False
    return bool(arr[-1] > mult * avg_prev)

# ------------------------
# Darvas Box (tightened)
# ------------------------
def darvas_box_breakout(df: pd.DataFrame, lookback: int = 20, min_bars: int = 5, min_width: float = 0.30, buffer: float = 0.001) -> Tuple[bool, float, float]:
    # FIX 1: Guard short dataframes
    if df.shape[0] < 20:
        return False, float("nan"), float("nan")

    n = len(df)
    if n < min_bars + 3:
        return False, float("nan"), float("nan")
    _df = df.copy()
    if isinstance(_df.columns, pd.MultiIndex):
        _df.columns = [c[0] for c in _df.columns]
    try:
        high = _df["High"].squeeze().astype(float)
        low  = _df["Low"].squeeze().astype(float)
        close = _df["Close"].squeeze().astype(float)
    except Exception:
        return False, float("nan"), float("nan")
    window = lookback + min_bars
    start = max(0, n - (window + 1))
    stop  = n - 1
    high_hist = high.iloc[start:stop]
    low_hist  = low.iloc[start:stop]
    if high_hist.empty or low_hist.empty:
        return False, float("nan"), float("nan")
    tol = 1e-9
    box_high = float(high_hist.max())
    peak_idx = high_hist.index[high_hist >= (box_high - tol)]
    if peak_idx.empty:
        return False, float("nan"), float("nan")
    last_peak_time = peak_idx[-1]
    cons_high = high_hist.loc[high_hist.index > last_peak_time]
    cons_low  = low_hist.loc[low_hist.index > last_peak_time]
    if cons_high.shape[0] < min_bars:
        return False, float("nan"), float("nan")
    if (cons_high > box_high + tol).any():
        return False, float("nan"), float("nan")
    box_low = float(cons_low.min()) if cons_low.size else float("nan")
    if not np.isfinite(box_low):
        return False, float("nan"), float("nan")
    if np.isfinite(box_low) and box_high > 0:
        tight_pct = (box_high - box_low) / box_high
        # FIX 4: Adaptive Darvas tightness
        atr_vals = atr(df, 14) if "Close" in df else pd.Series()
        last_atr = to_scalar(atr_vals.iloc[-1]) if not atr_vals.empty else np.nan
        last_close = to_scalar(df["Close"].iloc[-1])
        atr_pct = (last_atr / last_close) if np.isfinite(last_atr) and np.isfinite(last_close) else 0.02
        max_width = max(min_width, atr_pct * 5)
        if tight_pct > max_width:
            return False, float("nan"), float("nan")
    last_close = float(close.iloc[-1])
    last_high  = float(high.iloc[-1])
    is_breakout = (
        (np.isfinite(last_close) and last_close > box_high * (1.0 + buffer)) or
        (np.isfinite(last_high)  and last_high  > box_high * (1.0 + tol))
    )
    return bool(is_breakout), box_high, box_low

# ------------------------
# Compute signals (with improved confirmations)
# ------------------------
def _linear_slope(y: pd.Series) -> float:
    arr = y.to_numpy().ravel()
    n = arr.size
    if n < 5:
        return float("nan")
    x = np.arange(n, dtype=float)
    denom = (x - x.mean()).var() * n
    if denom == 0:
        return float("nan")
    slope = np.cov(x, arr, bias=True)[0,1] / (x.var() + 1e-12)
    return float(slope)

def compute_signals(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    # FIX 1: Guard short dataframes
    if df.shape[0] < 20:
        return {"Error": "Insufficient data"}
    out: Dict[str, Any] = {}
    if df.empty or df.shape[0] < 10:
        return out
    close = df["Close"]
    vol = df["Volume"].fillna(0)
    ma_fast = sma(close, params["ma_fast"])
    ma_slow = sma(close, params["ma_slow"])
    rsi_vals = rsi(close, params["rsi_period"])
    macd_line, macd_sig, macd_hist = macd_lines(close, params["macd_fast"], params["macd_slow"], params["macd_signal"])
    atr_vals = atr(df, params["atr_period"])
    plus_di, minus_di, adx_vals = _di_lines(df, params["adx_period"])
    obv_vals = obv(close, vol)
    last_close = to_scalar(close.iloc[-1])
    last_atr = to_scalar(atr_vals.iloc[-1])
    last_rsi = to_scalar(rsi_vals.iloc[-1])
    last_macd = to_scalar(macd_line.iloc[-1])
    last_macd_sig = to_scalar(macd_sig.iloc[-1])
    last_adx = to_scalar(adx_vals.iloc[-1])
    last_pdi = to_scalar(plus_di.iloc[-1])
    last_mdi = to_scalar(minus_di.iloc[-1])
    atr_pct = (last_atr / last_close) if (np.isfinite(last_atr) and np.isfinite(last_close) and last_close != 0) else 0.0
    widen = np.clip(params.get("atr_mult", 1.0) * (atr_pct * 100) * 0.6, 0.0, 8.0)
    low_thr = max(1.0, params["rsi_low"] - widen)
    high_thr = min(99.0, params["rsi_high"] + widen)
    if np.isfinite(last_rsi) and last_rsi < low_thr:
        out["RSI_Event"] = "Oversold"
    elif np.isfinite(last_rsi) and last_rsi > high_thr:
        out["RSI_Event"] = "Overbought"
    else:
        out["RSI_Event"] = ""
    macd_diff = macd_line - macd_sig
    macd_now = to_scalar(macd_diff.iloc[-1])
    macd_prev = to_scalar(macd_diff.iloc[-2]) if df.shape[0] >= 2 else float("nan")
    if np.isfinite(macd_now) and np.isfinite(macd_prev):
        if macd_now > 0 and macd_prev <= 0:
            out["MACD_Event"] = "Bullish Cross"
        elif macd_now < 0 and macd_prev >= 0:
            out["MACD_Event"] = "Bearish Cross"
        else:
            out["MACD_Event"] = ""
    else:
        out["MACD_Event"] = ""
    out["ADX_Strength"] = "Strong Trend" if np.isfinite(last_adx) and last_adx > params["adx_threshold"] else "Weak Trend"
    if np.isfinite(last_pdi) and np.isfinite(last_mdi) and np.isfinite(last_adx) and last_adx > params["adx_threshold"]:
        out["ADX_Direction"] = "Up" if last_pdi > last_mdi else "Down" if last_pdi < last_mdi else ""
    else:
        out["ADX_Direction"] = ""
    ma_diff = ma_fast - ma_slow
    ma_now = to_scalar(ma_diff.iloc[-1])
    ma_prev = to_scalar(ma_diff.iloc[-2]) if df.shape[0] >= 2 else float("nan")
    if np.isfinite(ma_now) and np.isfinite(ma_prev):
        if ma_now > 0 and ma_prev <= 0:
            out["MA_Cross"] = "Golden Cross"
        elif ma_now < 0 and ma_prev >= 0:
            out["MA_Cross"] = "Death Cross"
        else:
            out["MA_Cross"] = ""
    else:
        out["MA_Cross"] = ""
    sma200 = sma(close, 200)
    out["Above_SMA200"] = bool(np.isfinite(last_close) and np.isfinite(to_scalar(sma200.iloc[-1])) and last_close > to_scalar(sma200.iloc[-1]))
    out["Patterns"] = detect_patterns(df)
    out["Divergences"] = detect_divergences(df, rsi_vals, macd_line)
    out["Vol_Spike"] = bool(volume_spike(vol, params.get("vol_spike_mult", 2.0), window=20))
    if obv_vals.shape[0] >= 12:
        look = obv_vals.tail(20) if obv_vals.shape[0] >= 20 else obv_vals
        slope = _linear_slope(look)
        out["OBV_Trend"] = "Up" if np.isfinite(slope) and slope > 0 else "Down" if np.isfinite(slope) and slope < 0 else ""
    else:
        out["OBV_Trend"] = ""
    is_bo, box_hi, box_lo = darvas_box_breakout(df, params.get("darvas_lookback", 20), params.get("darvas_min_bars", 5))
    out["Darvas_Breakout"] = bool(is_bo)
    out["Darvas_Box"] = (box_hi, box_lo)
    darvas_confirm = False
    if out["Darvas_Breakout"]:
        try:
            vol_arr = vol.to_numpy().ravel()
            n = vol_arr.size
            look = min(20, max(1, n - 1))
            avg_prev = pd.Series(vol_arr[:-1]).tail(look).mean()
            if np.isfinite(avg_prev) and vol_arr[-1] > params.get("vol_spike_mult", 2.0) * avg_prev:
                darvas_confirm = True
        except Exception:
            pass
        try:
            if obv_vals.shape[0] >= 12:
                look_obv = obv_vals.tail(20) if obv_vals.shape[0] >= 20 else obv_vals
                slope = _linear_slope(look_obv)
                if np.isfinite(slope) and slope > 0:
                    darvas_confirm = True
        except Exception:
            pass
    out["Darvas_Confirmed"] = bool(darvas_confirm)
    
    # --- NEW: Composite Score (WITH WEIGHTS, GRADUATIONS, AND COMBOS) ---
    score = 0
    
    # Major Trend Signals (higher weight)
    score += 3 if out.get("MA_Cross") == "Golden Cross" else 0
    score -= 3 if out.get("MA_Cross") == "Death Cross" else 0
    score += 2 if out["Above_SMA200"] else -1 # Above 200 is good, below is a warning
    
    # Momentum & Breakout Signals (medium weight)
    score += 2 if out["MACD_Event"] == "Bullish Cross" else 0
    score -= 2 if out["MACD_Event"] == "Bearish Cross" else 0
    score += 2 if out["Darvas_Confirmed"] else 0 # Only confirmed breakouts get points
    
    # FIX 3: Re-weight divergences
    bull_div = sum(2 for d in out["Divergences"] if "Bullish" in d)
    bear_div = sum(2 for d in out["Divergences"] if "Bearish" in d)
    score += np.clip(bull_div, 0, 2)
    score -= np.clip(bear_div, 0, 2)
    
    # Minor Confirmation Signals & Graduated Scores
    # FIX 2: Tame RSI double-counting
    if np.isfinite(last_rsi):
        if last_rsi < 30:
            score += 1
        elif last_rsi > 70:
            score -= 1
        
    if np.isfinite(last_adx):
        if last_adx > params["adx_threshold"]:
            score += 1 # Trend is strengthening
            if out.get("ADX_Direction") == "Up": score += 1
            if out.get("ADX_Direction") == "Down": score -=1
        if last_adx > 40:
            score += 1 # Extra point for an extremely strong trend
            
    score += 1 if out["OBV_Trend"] == "Up" else 0
    score += 1 if out["Vol_Spike"] else 0
    
    # FIX 3: Re-weight patterns
    bull_patterns = sum(1 for p in out["Patterns"] if "Bullish" in p)
    bear_patterns = sum(1 for p in out["Patterns"] if "Bearish" in p)
    score += np.clip(bull_patterns, 0, 2)
    score -= np.clip(bear_patterns, 0, 2)

    # --- Combo Score Bonuses (applied on top of other scores) ---
    if out["MACD_Event"] == "Bullish Cross" and out["RSI_Event"] == "Oversold":
        score += 3  # Add 3 bonus points for this powerful reversal signal
    if out["MACD_Event"] == "Bearish Cross" and out["RSI_Event"] == "Overbought":
        score -= 3  # Subtract 3 bonus points
    if out["Darvas_Confirmed"] and out["ADX_Strength"] == "Strong Trend":
        score += 2 # Add 2 bonus points for confirmed breakout in a strong trend
        
    out["Score"] = int(score)
    # --- END OF NEW SCORING LOGIC ---

    out["Close"] = float(last_close) if np.isfinite(last_close) else float("nan")
    out["RSI"] = float(last_rsi) if np.isfinite(last_rsi) else float("nan")
    out["ADX"] = float(last_adx) if np.isfinite(last_adx) else float("nan")
    out["ATR%"] = float((last_atr / last_close) * 100) if (np.isfinite(last_atr) and np.isfinite(last_close) and last_close != 0) else float("nan")
    
    long_bias = (out["Above_SMA200"] and out.get("ADX_Direction") == "Up") or (out.get("MA_Cross") == "Golden Cross")
    short_bias = (not out["Above_SMA200"] and out.get("ADX_Direction") == "Down") or (out.get("MA_Cross") == "Death Cross")
    if out["Darvas_Confirmed"] and long_bias:
        out["Strategy"] = "Breakout Long (Darvas + Confirmed)"
    elif out["Darvas_Breakout"] and long_bias:
        out["Strategy"] = "Watchlist Long (Darvas)"
    elif out["MACD_Event"] == "Bullish Cross" and long_bias and out["RSI_Event"] != "Overbought":
        out["Strategy"] = "Momentum Long (MACD)"
    elif out["MACD_Event"] == "Bearish Cross" and short_bias and out["RSI_Event"] != "Oversold":
        out["Strategy"] = "Momentum Short (MACD)"
    else:
        out["Strategy"] = ""
    out["Trend_Dir"] = out.get("ADX_Direction", "")
    return out

# ------------------------
# Plotting + sparklines
# ------------------------
def plot_single_ticker(df: pd.DataFrame, params: Dict[str, Any], title: str):
    # FIX: Add .squeeze() to ensure 'close' is a Series, preventing plot errors.
    close = df["Close"].squeeze()
    dates = df.index
    fma = sma(close, params["ma_fast"])
    sma_slow = sma(close, params["ma_slow"])
    r = rsi(close, params["rsi_period"])
    macd_line, macd_sig, macd_hist = macd_lines(close, params["macd_fast"], params["macd_slow"], params["macd_signal"])
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dates, close, label="Close", color="black")
    ax1.plot(dates, fma, label=f"SMA{params['ma_fast']}", color="blue")
    ax1.plot(dates, sma_slow, label=f"SMA{params['ma_slow']}", color="red")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(dates, r, label="RSI", color="purple")
    ax2.axhline(params["rsi_low"], color="green", linestyle="--")
    ax2.axhline(params["rsi_high"], color="red", linestyle="--")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(dates, macd_line, label="MACD", color="blue")
    ax3.plot(dates, macd_sig, label="Signal", color="orange")
    ax3.bar(dates, macd_hist, label="Hist", color="grey", alpha=0.5)
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax3.get_xticklabels():
        label.set_rotation(45)
    st.pyplot(fig)

@st.cache_data(show_spinner=False)
def make_sparkline_base64(series: pd.Series, fast: int, slow: int) -> str:
    data = series.tail(90)
    fig, ax = plt.subplots(figsize=(2.0, 0.6), dpi=72)
    ax.plot(data.index, data.values, linewidth=1)
    if len(data) > slow:
        ax.plot(data.index, sma(series, fast).tail(90).values, linewidth=0.8)
        ax.plot(data.index, sma(series, slow).tail(90).values, linewidth=0.8)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{encoded}" width="120" />'

# ------------------------
# NSE tickers helper
# ------------------------
def build_universe_choice(choice: str) -> List[str]:
    all_tickers = get_nse_tickers()
    if choice == "NIFTY50":
        return all_tickers[:50]
    if choice == "NIFTY100":
        return all_tickers[:100]
    if choice == "NIFTY500":
        return all_tickers[:500]
    return all_tickers

# ------------------------
# App modes
# ------------------------
mode = st.sidebar.radio("Mode", ["Single", "Scanner"])
st.sidebar.markdown("---")
st.sidebar.markdown("Profile: **%s**" % profile_choice)

# Single mode
if mode == "Single":
    st.header("Single Ticker Analysis")
    ticker = st.text_input("Ticker (NSE)", "RELIANCE.NS").strip().upper()
    if st.button("▶ Run Scan"):
        df = fetch_history(ticker, PARAMS["days"])
        if df.empty:
            st.error(
                f"No data found for ticker {ticker}. Possible causes:\n"
                "- network / yfinance issue\n"
                "- wrong ticker (try add/remove '.NS')\n\n"
                "Try increasing lookback days or check connection."
            )
        else:
            signals = compute_signals(df, PARAMS)
            st.session_state["last_single"] = {"ticker": ticker, "signals": signals}
            st.success("✅ Scan completed")
    
    if "last_single" in st.session_state:
        res = st.session_state["last_single"]
        st.subheader(f"{res['ticker']} — Signals")

        # Display signals as a clean table instead of raw dictionary
        signals_dict = res.get("signals", {})
        if signals_dict:
            signals_df = pd.DataFrame(signals_dict.items(), columns=['Signal', 'Value'])
            st.dataframe(signals_df, use_container_width=True)

        if "signals" in res and res["signals"]:
            df = fetch_history(res["ticker"], PARAMS["days"])
            if not df.empty:
                plot_single_ticker(df, PARAMS, f"{res['ticker']} Chart")

# Scanner mode
else:
    st.header("Scanner")
    universe_choice = st.selectbox("Universe", ["NIFTY50", "NIFTY100", "NIFTY500", "All NSE Stocks", "Custom"])
    if universe_choice == "Custom":
        tickers_text = st.text_area("Tickers (comma separated)", "RELIANCE.NS,TCS.NS,INFY.NS")
        tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    else:
        tickers = build_universe_choice(universe_choice)
    count = st.number_input("Scan first N tickers (for speed)", min_value=1, max_value=len(tickers), value=30)
    tickers = tickers[:count]
    if st.button("Run Scan"):
        results: List[Dict[str, Any]] = []
        progress = st.progress(0.0)
        for i, t in enumerate(tickers):
            df = fetch_history(t, PARAMS["days"])
            if df.empty:
                progress.progress((i + 1) / len(tickers))
                continue
            sigs = compute_signals(df, PARAMS)
            # FIX 1: Check for error in signals dictionary
            if "Error" in sigs:
                progress.progress((i + 1) / len(tickers))
                continue

            img_html = make_sparkline_base64(df["Close"], PARAMS["ma_fast"], PARAMS["ma_slow"])
            row = {
                "Ticker": t,
                "Close": sigs.get("Close"),
                "Score": sigs.get("Score"),
                "RSI": sigs.get("RSI"),
                "ADX": sigs.get("ADX"),
                "ATR%": sigs.get("ATR%"),
                "MACD": sigs.get("MACD_Event"),
                "MA Cross": sigs.get("MA_Cross"),
                "SMA200+": sigs.get("Above_SMA200"),
                "VolSpike": sigs.get("Vol_Spike"),
                "OBV": sigs.get("OBV_Trend"),
                "Darvas": sigs.get("Darvas_Breakout"),
                "Darvas_Confirmed": sigs.get("Darvas_Confirmed"),
                "Patterns": ", ".join(sigs.get("Patterns", [])),
                "Divergences": ", ".join(sigs.get("Divergences", [])),
                "Chart": img_html
            }
            results.append(row)
            progress.progress((i + 1) / len(tickers))
        st.session_state["last_scan"] = results
        st.success("✅ Scan completed")

    if "last_scan" in st.session_state and st.session_state["last_scan"]:
        df_out = pd.DataFrame(st.session_state["last_scan"])
        
        # Sort the results by score in descending order
        df_out.sort_values(by="Score", ascending=False, inplace=True)
        
        def color_score(val):
            try:
                v = float(val)
                if v > 0:
                    return f'<span style="color:green;font-weight:bold">{v}</span>'
                elif v < 0:
                    return f'<span style="color:red;font-weight:bold">{v}</span>'
                else:
                    return f'<span>{v}</span>'
            except Exception:
                return val
        df_disp = df_out.copy()
        df_disp["Score"] = df_disp["Score"].apply(color_score)
        df_disp["Close"] = df_disp["Close"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        df_disp["RSI"] = df_disp["RSI"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "")
        df_disp["ADX"] = df_disp["ADX"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "")
        df_disp["ATR%"] = df_disp["ATR%"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        html = df_disp.to_html(escape=False, index=False)
        st.write(html, unsafe_allow_html=True)
        csv = df_out.drop(columns=["Chart"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", csv, "scan_results.csv", "text/csv")

if "last_scan" in st.session_state and st.session_state["last_scan"]:
    st.markdown("---")
    st.header("Send Results to Backtester")
    df_results = pd.DataFrame(st.session_state["last_scan"])
    df_results.sort_values(by="Score", ascending=False, inplace=True)
    st.write("You can filter the scanned results before sending them for backtesting.")
    min_score = st.slider("Minimum Score", int(df_results['Score'].min()), int(df_results['Score'].max()), 0)
    available_events = df_results['MACD'].unique().tolist()
    selected_events = st.multiselect("Filter by MACD Event", options=[e for e in available_events if e], default=[])
    filtered_df = df_results[df_results['Score'] >= min_score]
    if selected_events:
        filtered_df = filtered_df[filtered_df['MACD'].isin(selected_events)]
    st.write(f"**Found {len(filtered_df)} tickers matching filters.**")
    st.dataframe(filtered_df[['Ticker', 'Score', 'Close', 'MACD', 'Patterns']].head())
    page_options = {
        "Neural Network Backtester": ("pages/2_Neural_Net_Backtest.py", None),
        "Classical ML Backtester": ("pages/3_Classic_ML_Backtest.py", None),
        "Both (Run Neural Net First)": ("pages/2_Neural_Net_Backtest.py", "pages/3_Classic_ML_Backtest.py"),
        "Both (Run Classical First)": ("pages/3_Classic_ML_Backtest.py", "pages/2_Neural_Net_Backtest.py")
    }
    destination_page_name = st.selectbox(
        "Choose a backtester to open:",
        options=list(page_options.keys())
    )
    if st.button("Send Tickers and Open Backtester", type="primary"):
        tickers_to_send = filtered_df['Ticker'].tolist()
        if not tickers_to_send:
            st.warning("No tickers selected to send.")
        else:
            st.session_state['tickers_for_backtest'] = tickers_to_send
            if 'PARAMS' in locals() or 'PARAMS' in globals():
                st.session_state['scanner_params'] = PARAMS
            first_page, next_page = page_options[destination_page_name]
            if next_page:
                st.session_state['next_page_to_run'] = next_page
            elif 'next_page_to_run' in st.session_state:
                del st.session_state['next_page_to_run']
            st.switch_page(first_page)