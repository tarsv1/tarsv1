# pages/4_Intraday_Scanner.py

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

# Import indicator logic from data.py to act as a single source of truth
from data import sma, ema, rsi, macd_lines, atr, _di_lines, obv

# Page config
st.set_page_config(page_title="Intraday Scanner (NSE)", layout="wide")
st.title("🚀 Intraday Scanner (NSE)")
st.info("This scanner is optimized for intraday timeframes. Use the controls below to find opportunities and send them to the new Unified Intraday Backtester.")

# ------------------------
# Hardcoded Futures Stock list
# ------------------------
FUTURES_STOCKS = [
    "AARTIIND.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABFRL.NS", "ACC.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "ADANIPOWER.NS", "ADANITRANS.NS", "ADANITOTAL.NS", "AEGISCHEM.NS", "AIAENG.NS",
    "APLLTD.NS", "ALKEM.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS",
    "ASIANPAINT.NS", "ASTRAL.NS", "ATUL.NS", "AUBANK.NS", "AURIONPRO.NS", "AUROPHARMA.NS",
    "AVANTIFEED.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BALAMINES.NS",
    "BALRAMCHIN.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BATAINDIA.NS", "BEL.NS", "BHARATFORG.NS",
    "BHEL.NS", "BIKAJI.NS", "BIOCON.NS", "BIRLACORPN.NS", "BSOFT.NS", "BOSCHLTD.NS", "BPCL.NS",
    "BRITANNIA.NS", "BSLLTD.NS", "CANBK.NS", "CANFINHOME.NS", "CESC.NS", "CHAMBLFERT.NS", "CHOLAFIN.NS",
    "CIPLA.NS", "COALINDIA.NS", "COFORGE.NS", "COLPAL.NS", "CONCOR.NS", "COROMANDEL.NS", "CROMPTON.NS",
    "CUB.NS", "CUMMINSIND.NS", "DABUR.NS", "DALBHARAT.NS", "DEEPAKNI.NS", "DELTACORP.NS", "DIVISLAB.NS",
    "DIXON.NS", "DLF.NS", "DRREDDY.NS", "EICHERMOT.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS",
    "FSL.NS", "GAIL.NS", "GLENMARK.NS", "GMRINFRA.NS", "GNFC.NS", "GODREJCP.NS", "GODREJPROP.NS",
    "GRANULES.NS", "GRASIM.NS", "GSPL.NS", "GUJGASLTD.NS", "HAL.NS", "HAVELLS.NS", "HCLTECH.NS",
    "HDFC.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "HINDCOPPER.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "HONAUT.NS", "IBULHSGFIN.NS", "ICICIBANK.NS",
    "ICICIGI.NS", "ICICIPRULI.NS", "IDEA.NS", "IEX.NS", "IGL.NS", "INDHOTEL.NS", "INDIACEM.NS",
    "INDIAMART.NS", "INDIGO.NS", "INDUSINDBK.NS", "INDUSTOWER.NS", "INFY.NS", "IRCTC.NS", "IRFC.NS",
    "IRB.NS", "ITC.NS", "JBCHEPHARM.NS", "JINDALSTEL.NS", "JSWENERGY.NS", "JSWSTEEL.NS", "JUBLFOOD.NS",
    "KALYANKJIL.NS", "KAYNES.NS", "KEI.NS", "KFINTECH.NS", "KOTAKBANK.NS", "KPITTECH.NS", "L&TF.NS",
    "LAURUSLABS.NS", "LICI.NS", "LICHSGFIN.NS", "LODHA.NS", "LT.NS", "LTIM.NS", "LUPIN.NS", "M&M.NS",
    "MANAPPURAM.NS", "MANKIND.NS", "MARICO.NS", "MARUTI.NS", "MAXHEALTH.NS", "MAZDOCK.NS", "MCX.NS",
    "MFSL.NS", "MGL.NS", "MOTHERSON.NS", "MPHASIS.NS", "MRF.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS",
    "NAUKRI.NS", "NAVINFLUOR.NS", "NESTLEIND.NS", "NHPC.NS", "NMDC.NS", "NTPC.NS", "NUVAMA.NS",
    "NYKAA.NS", "OBEROIRLTY.NS", "OFSS.NS", "OIL.NS", "ONGC.NS", "PAGEIND.NS", "PATANJALI.NS",
    "PAYTM.NS", "PEL.NS", "PERSISTENT.NS", "PETRONET.NS", "PFC.NS", "PIDILITIND.NS", "PIIND.NS",
    "PNB.NS", "POLYCAB.NS", "POWERGRID.NS", "POOONAWALLA.NS", "PRINCEPIPE.NS", "PVRINOX.NS", "RAIN.NS",
    "RBLBANK.NS", "RECLTD.NS", "RELIANCE.NS", "RVNL.NS", "SAIL.NS", "SBICARD.NS", "SBILIFE.NS",
    "SBIN.NS", "SHREECEM.NS", "SIEMENS.NS", "SJVN.NS", "SRF.NS", "SUNPHARMA.NS", "SUNTV.NS",
    "SUZLON.NS", "SYRMA.NS", "TATACHEM.NS", "TATACOMM.NS", "TATACONSUM.NS", "TATAMOTORS.NS",
    "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS", "TORNTPOWER.NS",
    "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "ULTRACEMCO.NS", "UPL.NS", "VEDL.NS", "VGUARD.NS",
    "VOLTAS.NS", "WHIRLPOOL.NS", "WIPRO.NS", "ZEEL.NS", "ZYDUSLIFE.NS"
]

# ------------------------
# Safe helpers
# ------------------------
def to_scalar(x) -> float:
    try:
        if isinstance(x, pd.DataFrame):
            if x.shape[1] > 0: return float(x.iloc[-1, 0])
            return float("nan")
        if isinstance(x, pd.Series): return float(x.iloc[-1])
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x).ravel()
            return float(arr[-1]) if arr.size else float("nan")
        return float(x)
    except Exception: return float("nan")

# ------------------------
# Intraday Profiles
# ------------------------
PROFILES: Dict[str, Dict[str, Any]] = {
    "Scalping (5-min)": {
        "ma_fast": 9, "ma_slow": 21, "rsi_period": 9, "rsi_low": 30, "rsi_high": 70,
        "macd_fast": 8, "macd_slow": 21, "macd_signal": 5, "adx_period": 10, "adx_threshold": 25,
        "atr_period": 10, "atr_mult": 2.0, "days": 5, "vol_spike_mult": 1.8,
        "darvas_lookback": 50, "darvas_min_bars": 7
    },
    "Momentum (15-min)": {
        "ma_fast": 10, "ma_slow": 30, "rsi_period": 14, "rsi_low": 40, "rsi_high": 60,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "adx_period": 14, "adx_threshold": 25,
        "atr_period": 14, "atr_mult": 2.0, "days": 15, "vol_spike_mult": 2.0,
        "darvas_lookback": 60, "darvas_min_bars": 10
    },
    "Reversal (1-hour)": {
        "ma_fast": 8, "ma_slow": 21, "rsi_period": 20, "rsi_low": 25, "rsi_high": 75,
        "macd_fast": 15, "macd_slow": 30, "macd_signal": 9, "adx_period": 20, "adx_threshold": 20,
        "atr_period": 20, "atr_mult": 2.5, "days": 45, "vol_spike_mult": 2.2,
        "darvas_lookback": 75, "darvas_min_bars": 12
    }
}

# ------------------------
# Sidebar UI
# ------------------------
st.sidebar.header("Intraday Configuration")
timeframe_options = ["5 Minute", "15 Minute", "30 Minute", "1 Hour"]
timeframe_choice = st.sidebar.selectbox("Select Timeframe", timeframe_options)
interval_map = {"5 Minute": "5m", "15 Minute": "15m", "30 Minute": "30m", "1 Hour": "1h"}
selected_interval = interval_map[timeframe_choice]

default_profile = "Scalping (5-min)"
if "15 Minute" in timeframe_choice: default_profile = "Momentum (15-min)"
elif "Hour" in timeframe_choice: default_profile = "Reversal (1-hour)"

if "active_profile_intra" not in st.session_state:
    st.session_state.active_profile_intra = default_profile
    for k, v in PROFILES[default_profile].items(): st.session_state[k + "_intra"] = v

profile_choice = st.sidebar.radio("Profile", list(PROFILES.keys()), index=list(PROFILES.keys()).index(st.session_state.get("active_profile_intra", default_profile)))
if profile_choice != st.session_state.get("active_profile_intra"):
    for k, v in PROFILES[profile_choice].items(): st.session_state[k + "_intra"] = v
    st.session_state.active_profile_intra = profile_choice
    st.rerun()

st.sidebar.markdown("**Manual Overrides**")
st.sidebar.number_input("Fast MA", 1, 500, key="ma_fast_intra"); st.sidebar.number_input("Slow MA", 1, 2000, key="ma_slow_intra")
st.sidebar.number_input("RSI Period", 2, 100, key="rsi_period_intra"); st.sidebar.number_input("RSI Low", 1, 60, key="rsi_low_intra")
st.sidebar.number_input("RSI High", 40, 99, key="rsi_high_intra"); st.sidebar.number_input("MACD Fast", 1, 200, key="macd_fast_intra")
st.sidebar.number_input("MACD Slow", 1, 500, key="macd_slow_intra"); st.sidebar.number_input("MACD Signal", 1, 200, key="macd_signal_intra")
st.sidebar.number_input("ADX Period", 5, 100, key="adx_period_intra"); st.sidebar.number_input("ADX Threshold", 1, 100, key="adx_threshold_intra")
st.sidebar.number_input("ATR Period", 1, 200, key="atr_period_intra"); st.sidebar.number_input("ATR Multiplier", 1.0, 5.0, step=0.1, key="atr_mult_intra")
st.sidebar.number_input("Lookback Days", 1, 60, key="days_intra", help="Max 60 days for intraday data from yfinance"); st.sidebar.number_input("Volume Spike Multiplier", 1.0, 5.0, step=0.1, key="vol_spike_mult_intra")
st.sidebar.number_input("Darvas Lookback (candles)", 5, 200, key="darvas_lookback_intra"); st.sidebar.number_input("Darvas Min Bars", 2, 60, key="darvas_min_bars_intra")

if st.sidebar.button("🔄 Reset to Profile", key="reset_intra"):
    for k, v in PROFILES[profile_choice].items(): st.session_state[k + "_intra"] = v
    st.session_state.active_profile_intra = profile_choice
    st.rerun()

# --- FIX: Explicitly define parameter keys to avoid passing unwanted session_state items ---
param_keys = [
    "ma_fast", "ma_slow", "rsi_period", "rsi_low", "rsi_high",
    "macd_fast", "macd_slow", "macd_signal", "adx_period", "adx_threshold",
    "atr_period", "atr_mult", "days", "vol_spike_mult",
    "darvas_lookback", "darvas_min_bars"
]
PARAMS = {key: st.session_state[key + "_intra"] for key in param_keys}
PARAMS['interval'] = selected_interval # Add interval to params

# ------------------------
# Data Fetching
# ------------------------
@st.cache_data(ttl=24 * 3600)
def get_nse_tickers() -> List[str]:
    urls = ["https://archives.nseindia.com/content/equities/EQUITY_L.csv", "https://www1.nseindia.com/content/equities/EQUITY_L.csv"]
    for url in urls:
        try:
            df = pd.read_csv(url)
            if "SYMBOL" in df.columns:
                syms = df["SYMBOL"].astype(str).str.strip().str.upper().tolist()
                syms = [s for s in syms if re.fullmatch(r"[A-Z0-9\-]+", s)]
                if syms: return sorted(set([s + ".NS" for s in syms]))
        except Exception: continue
    st.warning("Could not fetch NSE ticker list — using fallback subset.")
    return FUTURES_STOCKS

@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker: str, days: int, interval: str) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    period = f"{min(days, 60)}d"
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
        if isinstance(df, pd.DataFrame) and not df.empty: return df
    except Exception: pass
    return pd.DataFrame()

# ------------------------
# Signal Calculation Functions (Assumed to be correct and unchanged)
# ------------------------
def _is_bull(o, c): return np.isfinite(o) and np.isfinite(c) and c > o
def _is_bear(o, c): return np.isfinite(o) and np.isfinite(c) and c < o

def detect_patterns(df: pd.DataFrame) -> List[str]:
    if df.shape[0] < 3: return []
    patterns: List[str] = []
    last, prev = df.iloc[-1], df.iloc[-2]
    o1, c1 = to_scalar(prev["Open"]), to_scalar(prev["Close"])
    o2, c2 = to_scalar(last["Open"]), to_scalar(last["Close"])
    if np.isfinite(o1) and np.isfinite(c1) and np.isfinite(o2) and np.isfinite(c2):
        if _is_bear(o1, c1) and _is_bull(o2, c2) and (c2 > o1) and (o2 < c1): patterns.append("Bullish Engulfing")
        if _is_bull(o1, c1) and _is_bear(o2, c2) and (c2 < o1) and (o2 > c1): patterns.append("Bearish Engulfing")
    return patterns

def _find_swings(series: pd.Series, window: int = 5) -> Tuple[List[int], List[int]]:
    arr = series.to_numpy(); highs, lows = [], []
    for i in range(len(arr)):
        start, end = max(0, i - window), min(len(arr), i + window + 1)
        segment = arr[start:end]
        if len(segment) > 0 and arr[i] == np.max(segment): highs.append(i)
        if len(segment) > 0 and arr[i] == np.min(segment): lows.append(i)
    unique_highs = [h for i, h in enumerate(highs) if i == 0 or highs[i-1] != h]
    unique_lows = [l for i, l in enumerate(lows) if i == 0 or lows[i-1] != l]
    return unique_highs, unique_lows

def detect_divergences(df: pd.DataFrame, rsi_vals: pd.Series, macd_line: pd.Series) -> List[str]:
    if df.shape[0] < 20: return []
    out: List[str] = []; close = df["Close"]
    highs, lows = _find_swings(close, window=3)
    min_price_change_pct = 0.005 # 0.5%
    def last_two(idx_list: List[int]) -> Optional[Tuple[int, int]]:
        return (idx_list[-2], idx_list[-1]) if len(idx_list) >= 2 else None
    hpair, lpair = last_two(highs), last_two(lows)
    if hpair:
        i1, i2 = hpair; p1, p2 = to_scalar(close.iloc[i1]), to_scalar(close.iloc[i2])
        r1, r2 = to_scalar(rsi_vals.iloc[i1]), to_scalar(rsi_vals.iloc[i2])
        m1, m2 = to_scalar(macd_line.iloc[i1]), to_scalar(macd_line.iloc[i2])
        if p2 > p1 * (1 + min_price_change_pct):
            if r2 < r1: out.append("Bearish RSI Divergence")
            if m2 < m1: out.append("Bearish MACD Divergence")
    if lpair:
        i1, i2 = lpair; p1, p2 = to_scalar(close.iloc[i1]), to_scalar(close.iloc[i2])
        r1, r2 = to_scalar(rsi_vals.iloc[i1]), to_scalar(rsi_vals.iloc[i2])
        m1, m2 = to_scalar(macd_line.iloc[i1]), to_scalar(macd_line.iloc[i2])
        if p2 < p1 * (1 - min_price_change_pct):
            if r2 > r1: out.append("Bullish RSI Divergence")
            if m2 > m1: out.append("Bullish MACD Divergence")
    return out

def volume_spike(vol: pd.Series, mult: float = 2.0, window: int = 20) -> bool:
    arr = vol.to_numpy().ravel()
    if arr.size < 6: return False
    look = min(window, max(1, arr.size - 1))
    avg_prev = pd.Series(arr[:-1]).tail(look).mean()
    if not np.isfinite(avg_prev) or avg_prev == 0: return False
    return bool(arr[-1] > mult * avg_prev)

def darvas_box_breakout(df: pd.DataFrame, lookback: int, min_bars: int, min_width: float = 0.30, buffer: float = 0.001) -> Tuple[bool, float, float]:
    if df.shape[0] < 20: return False, float("nan"), float("nan")
    n = len(df)
    if n < min_bars + 3: return False, float("nan"), float("nan")
    _df = df.copy()
    if isinstance(_df.columns, pd.MultiIndex): _df.columns = [c[0] for c in _df.columns]
    try:
        high = _df["High"].squeeze().astype(float)
        low  = _df["Low"].squeeze().astype(float)
        close = _df["Close"].squeeze().astype(float)
    except Exception: return False, float("nan"), float("nan")
    window = lookback + min_bars
    start = max(0, n - (window + 1)); stop  = n - 1
    high_hist = high.iloc[start:stop]
    if high_hist.empty: return False, float("nan"), float("nan")
    tol = 1e-9
    box_high = float(high_hist.max())
    peak_idx = high_hist.index[high_hist >= (box_high - tol)]
    if peak_idx.empty: return False, float("nan"), float("nan")
    last_peak_time = peak_idx[-1]
    cons_high = high.loc[high.index > last_peak_time]
    cons_low  = low.loc[low.index > last_peak_time]
    if cons_high.shape[0] < min_bars: return False, float("nan"), float("nan")
    if (cons_high > box_high + tol).any(): return False, float("nan"), float("nan")
    box_low = float(cons_low.min()) if cons_low.size else float("nan")
    if not np.isfinite(box_low): return False, float("nan"), float("nan")
    last_close = float(close.iloc[-1])
    is_breakout = (np.isfinite(last_close) and last_close > box_high * (1.0 + buffer))
    return bool(is_breakout), box_high, box_low

def _linear_slope(y: pd.Series) -> float:
    arr = y.to_numpy().ravel()
    if arr.size < 5: return np.nan
    x = np.arange(arr.size, dtype=float)
    slope = np.polyfit(x, arr, 1)[0]
    return float(slope)

def compute_signals(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    if df.shape[0] < 20: return {"Error": "Insufficient data"}
    out: Dict[str, Any] = {}
    close = df["Close"]; vol = df["Volume"].fillna(0)
    
    ma_fast = sma(close, int(params["ma_fast"]))
    ma_slow = sma(close, int(params["ma_slow"]))
    rsi_vals = rsi(close, int(params["rsi_period"]))
    macd_line, macd_sig, _ = macd_lines(close, int(params["macd_fast"]), int(params["macd_slow"]), int(params["macd_signal"]))
    atr_vals = atr(df, int(params["atr_period"]))
    plus_di, minus_di, adx_vals = _di_lines(df, int(params["adx_period"]))
    obv_vals = obv(close, vol)
    
    last_close = to_scalar(close.iloc[-1])
    last_atr = to_scalar(atr_vals.iloc[-1])
    last_rsi = to_scalar(rsi_vals.iloc[-1])
    last_adx = to_scalar(adx_vals.iloc[-1])
    last_pdi = to_scalar(plus_di.iloc[-1])
    last_mdi = to_scalar(minus_di.iloc[-1])
    
    atr_pct = (last_atr / last_close) * 100 if (np.isfinite(last_atr) and np.isfinite(last_close) and last_close != 0) else 0.0
    widen = np.clip(float(params.get("atr_mult", 1.0)) * atr_pct * 0.6, 0.0, 8.0)
    low_thr = max(1.0, float(params["rsi_low"]) - widen)
    high_thr = min(99.0, float(params["rsi_high"]) + widen)
    
    out["RSI_Event"] = "Oversold" if np.isfinite(last_rsi) and last_rsi < low_thr else "Overbought" if np.isfinite(last_rsi) and last_rsi > high_thr else ""
    
    macd_diff = macd_line - macd_sig
    macd_now = to_scalar(macd_diff.iloc[-1])
    macd_prev = to_scalar(macd_diff.iloc[-2]) if len(macd_diff) > 1 else np.nan
    out["MACD_Event"] = "Bullish Cross" if macd_now > 0 and macd_prev <= 0 else "Bearish Cross" if macd_now < 0 and macd_prev >= 0 else ""

    ma_diff = ma_fast - ma_slow
    ma_now = to_scalar(ma_diff.iloc[-1])
    ma_prev = to_scalar(ma_diff.iloc[-2]) if len(ma_diff) > 1 else np.nan
    out["MA_Cross"] = "Golden Cross" if ma_now > 0 and ma_prev <= 0 else "Death Cross" if ma_now < 0 and ma_prev >= 0 else ""
    
    out["ADX_Strength"] = "Strong Trend" if np.isfinite(last_adx) and last_adx > float(params["adx_threshold"]) else "Weak Trend"
    out["ADX_Direction"] = "Up" if last_pdi > last_mdi else "Down" if last_pdi < last_mdi else ""
    
    out["Patterns"] = detect_patterns(df)
    out["Divergences"] = detect_divergences(df, rsi_vals, macd_line)
    out["Vol_Spike"] = volume_spike(vol, float(params.get("vol_spike_mult", 2.0)))
    
    is_bo, _, _ = darvas_box_breakout(df, int(params["darvas_lookback"]), int(params["darvas_min_bars"]))
    out["Darvas_Breakout"] = is_bo
    
    if obv_vals.shape[0] >= 12:
        look = obv_vals.tail(20) if obv_vals.shape[0] >= 20 else obv_vals
        slope = _linear_slope(look)
        out["OBV_Trend"] = "Up" if np.isfinite(slope) and slope > 0 else "Down" if np.isfinite(slope) and slope < 0 else ""
    else:
        out["OBV_Trend"] = ""
    
    score = 0
    if out["MA_Cross"] == "Golden Cross": score += 5
    elif out["MA_Cross"] == "Death Cross": score -= 5
    if out["MACD_Event"] == "Bullish Cross": score += 3
    elif out["MACD_Event"] == "Bearish Cross": score -= 3
    if out["ADX_Strength"] == "Strong Trend":
        if out["ADX_Direction"] == "Up": score += 3
        elif out["ADX_Direction"] == "Down": score -= 3
    if out["Darvas_Breakout"]: score += 5
    if out["Vol_Spike"]: score += 1
    if out["OBV_Trend"] == "Up": score += 1
    elif out["OBV_Trend"] == "Down": score -= 1
    if any("Bullish" in d for d in out["Divergences"]): score += 5
    if any("Bearish" in d for d in out["Divergences"]): score -= 5
    if any("Bullish" in p for p in out["Patterns"]): score += 2
    if any("Bearish" in p for p in out["Patterns"]): score -= 2
    if out["RSI_Event"] == "Oversold": score += 2
    elif out["RSI_Event"] == "Overbought": score -= 2
    
    out["Score"] = int(score)
    out["Close"] = float(last_close)
    out["RSI"] = float(last_rsi)
    out["ADX"] = float(last_adx)
    out["ATR%"] = float(atr_pct)
    out["Strategy"] = "Breakout Long" if out["Darvas_Breakout"] and score > 5 else "Momentum Long" if out["MACD_Event"] == "Bullish Cross" and score > 3 else ""

    return out

def plot_single_ticker(df: pd.DataFrame, params: Dict[str, Any], title: str, interval: str):
    close = df["Close"].squeeze(); dates = df.index
    fma = sma(close, int(params["ma_fast"])); sma_slow = sma(close, int(params["ma_slow"]))
    r = rsi(close, int(params["rsi_period"]))
    macd_line, macd_sig, macd_hist = macd_lines(close, int(params["macd_fast"]), int(params["macd_slow"]), int(params["macd_signal"]))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    plt.subplots_adjust(hspace=0.08)
    ax1.plot(dates, close, label="Close", color="black"); ax1.plot(dates, fma, label=f"SMA{params['ma_fast']}", color="blue")
    ax1.plot(dates, sma_slow, label=f"SMA{params['ma_slow']}", color="red"); ax1.set_title(title); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(dates, r, label="RSI", color="purple"); ax2.axhline(params["rsi_low"], color="green", ls="--"); ax2.axhline(params["rsi_high"], color="red", ls="--")
    ax2.set_ylim(0, 100); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax3.plot(dates, macd_line, label="MACD", color="blue"); ax3.plot(dates, macd_sig, label="Signal", color="orange")
    ax3.bar(dates, macd_hist, label="Hist", color="grey", alpha=0.5); ax3.legend(); ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%d-%b"))
    plt.setp(ax3.get_xticklabels(), rotation=45)
    st.pyplot(fig)

@st.cache_data(show_spinner=False)
def make_sparkline_base64(series: pd.Series, fast: int, slow: int) -> str:
    data = series.tail(300); fig, ax = plt.subplots(figsize=(2.0, 0.6), dpi=72)
    ax.plot(data.index, data.values, linewidth=1)
    if len(data) > slow:
        ax.plot(data.index, sma(series, fast).tail(300).values, linewidth=0.8)
        ax.plot(data.index, sma(series, slow).tail(300).values, linewidth=0.8)
    ax.axis("off"); buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0); plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode("utf-8")}" width="120" />'

# ------------------------
# App Body
# ------------------------
mode = st.sidebar.radio("Mode", ["Scanner", "Single"], key="mode_intra")
st.sidebar.markdown("---")
st.sidebar.info(f"Timeframe: **{timeframe_choice}**\n\nProfile: **{profile_choice}**")

if mode == "Single":
    st.header("Single Ticker Analysis")
    ticker = st.text_input("Ticker (NSE)", "RELIANCE.NS", key="ticker_intra").strip().upper()
    if st.button("▶ Run Analysis", key="run_single_intra"):
        with st.spinner(f"Fetching {timeframe_choice} data for {ticker}..."):
            df = fetch_history(ticker, int(PARAMS["days"]), selected_interval)
        if df.empty: st.error(f"No data for {ticker} on {timeframe_choice} interval.")
        else:
            signals = compute_signals(df, PARAMS)
            st.session_state["last_single_intra"] = {"ticker": ticker, "signals": signals, "interval": selected_interval}
            st.success("✅ Analysis completed")
    if "last_single_intra" in st.session_state:
        res = st.session_state["last_single_intra"]
        st.subheader(f"{res['ticker']} — Signals ({res['interval']})")
        signals_df = pd.DataFrame(res.get("signals", {}).items(), columns=['Signal', 'Value'])
        st.dataframe(signals_df, use_container_width=True)
        df = fetch_history(res["ticker"], int(PARAMS["days"]), res["interval"])
        if not df.empty: plot_single_ticker(df, PARAMS, f"{res['ticker']} Chart", res['interval'])

else: # Scanner Mode
    st.header(f"Scanner ({timeframe_choice})")
    universe_options = ["Futures Stock", "NIFTY100", "NIFTY500", "All NSE Stocks", "Custom"]
    universe_choice = st.selectbox("Universe", universe_options, key="universe_intra")
    
    tickers = []
    if universe_choice == "Custom":
        tickers_text = st.text_area("Tickers (comma separated)", "RELIANCE.NS,TCS.NS,INFY.NS", key="custom_tickers_intra")
        tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    else:
        if universe_choice == "Futures Stock": tickers = FUTURES_STOCKS
        else:
            all_tickers = get_nse_tickers()
            if universe_choice == "NIFTY100": tickers = all_tickers[:100]
            elif universe_choice == "NIFTY500": tickers = all_tickers[:500]
            elif universe_choice == "All NSE Stocks": tickers = all_tickers
    
    min_volume = st.number_input("Min Avg Volume (20-period)", 0, value=100000, step=10000, key="min_vol_intra")
    count = st.number_input("Scan first N tickers", 1, len(tickers), 50, key="count_intra")
    tickers = tickers[:count]

    if st.button("▶ Run Scan", key="run_scan_intra"):
        results: List[Dict[str, Any]] = []
        progress = st.progress(0.0, text="Initializing scan...")
        for i, t in enumerate(tickers):
            progress.progress(i / len(tickers), text=f"Scanning {t}...")
            df = fetch_history(t, int(PARAMS["days"]), selected_interval)
            if df.empty or 'Volume' not in df.columns or df.shape[0] < 20: continue
            avg_volume = df['Volume'].squeeze().tail(20).mean()
            if avg_volume < min_volume: continue
            sigs = compute_signals(df, PARAMS)
            if "Error" in sigs: continue
            img_html = make_sparkline_base64(df["Close"], int(PARAMS["ma_fast"]), int(PARAMS["ma_slow"]))
            
            row = {
                "Ticker": t, "Close": sigs.get("Close"), "Avg Volume": avg_volume, "Chart": img_html,
                "Score": sigs.get("Score"), "RSI": sigs.get("RSI"), "ADX": sigs.get("ADX"),
                "ATR%": sigs.get("ATR%"), "RSI Event": sigs.get("RSI_Event"), "MACD Event": sigs.get("MACD_Event"),
                "MA Cross": sigs.get("MA_Cross"), "ADX Strength": sigs.get("ADX_Strength"),
                "Volume Spike": sigs.get("Vol_Spike"), "Patterns": ", ".join(sigs.get("Patterns", [])),
                "Divergences": ", ".join(sigs.get("Divergences", [])),
                "Darvas Breakout": sigs.get("Darvas_Breakout"),
                "OBV Trend": sigs.get("OBV_Trend"),
                "Strategy": sigs.get("Strategy")
            }
            results.append(row)
        progress.progress(1.0, text="Scan complete!")
        st.session_state["last_scan_intra"] = results
        st.success("✅ Scan completed")

    if "last_scan_intra" in st.session_state and st.session_state["last_scan_intra"]:
        df_out = pd.DataFrame(st.session_state["last_scan_intra"])
        df_out.sort_values(by="Score", ascending=False, inplace=True)
        
        def color_score(val):
            try:
                v = float(val)
                color = "green" if v > 0 else "red" if v < 0 else "black"
                return f'<span style="color:{color};font-weight:bold">{v}</span>'
            except: return val
        df_disp = df_out.copy()
        df_disp["Score"] = df_disp["Score"].apply(color_score)
        
        for col in ["Close", "RSI", "ADX", "ATR%"]:
            if col in df_disp.columns: df_disp[col] = df_disp[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        if "Avg Volume" in df_disp.columns: df_disp["Avg Volume"] = df_disp["Avg Volume"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
        
        html = df_disp.to_html(escape=False, index=False)
        st.write(html, unsafe_allow_html=True)
        csv = df_out.drop(columns=["Chart"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "intraday_scan.csv", "text/csv", key="download_intra")

# ----------------------------------------------------
# THIS IS THE UPDATED SECTION
# ----------------------------------------------------
if "last_scan_intra" in st.session_state and st.session_state["last_scan_intra"]:
    st.markdown("---")
    st.header("🔬 Send Results to Unified Backtester")
    df_results = pd.DataFrame(st.session_state["last_scan_intra"])
    df_results.sort_values(by="Score", ascending=False, inplace=True)
    st.write("You can filter the scanned results before sending them for backtesting.")
    
    min_score = st.slider("Minimum Score", int(df_results['Score'].min()), int(df_results['Score'].max()), 5, key="intra_min_score")
    
    filtered_df = df_results[df_results['Score'] >= min_score]
    st.write(f"**Found {len(filtered_df)} tickers matching filters.**")
    st.dataframe(filtered_df[['Ticker', 'Score', 'Close', 'MACD Event', 'Strategy']].head())
    
    if st.button("Send Tickers and Open Unified Backtester", type="primary"):
        tickers_to_send = filtered_df['Ticker'].tolist()
        if not tickers_to_send:
            st.warning("No tickers selected to send.")
        else:
            # Use unique session state keys for intraday workflow
            st.session_state['tickers_for_backtest_intra'] = tickers_to_send
            st.session_state['scanner_params_intra'] = PARAMS

            # Clear any leftover daily data to ensure the backtester starts in 'Intraday' mode
            if 'tickers_for_backtest' in st.session_state:
                del st.session_state['tickers_for_backtest']
            if 'scanner_params' in st.session_state:
                del st.session_state['scanner_params']
            
            # **FIXED PATH:** Use the page's filename without the 'pages/' prefix
            st.switch_page("2_Unified_Backtester")