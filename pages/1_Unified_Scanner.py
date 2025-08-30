# 1_Unified_Scanner.py

import io
import base64
import re
from typing import Any, Dict, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Import centralized logic from data.py
from data import sma, ema, rsi, macd_lines, atr, _di_lines, obv, detect_divergences, to_scalar

# --- Page Configuration ---
st.set_page_config(page_title="Unified Scanner (NSE)", layout="wide")
st.title("📊 Unified Patterns & Strategy Scanner (NSE)")
st.info("Select a scan type from the sidebar to switch between Daily and Intraday analysis.")

# ------------------------
# Hardcoded Stock Lists & Helpers
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
# Profile Dictionaries for Both Modes
# ------------------------
DAILY_PROFILES: Dict[str, Dict[str, Any]] = {
    "Short Term": {"ma_fast": 5, "ma_slow": 20, "rsi_period": 7, "rsi_low": 35, "rsi_high": 65, "macd_fast": 8, "macd_slow": 21, "macd_signal": 5, "adx_period": 7, "adx_threshold": 20, "atr_period": 7, "atr_mult": 1.5, "days": 120, "vol_spike_mult": 1.8, "darvas_lookback": 20, "darvas_min_bars": 5},
    "Medium Term": {"ma_fast": 20, "ma_slow": 50, "rsi_period": 14, "rsi_low": 30, "rsi_high": 70, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "adx_period": 14, "adx_threshold": 25, "atr_period": 14, "atr_mult": 2.0, "days": 250, "vol_spike_mult": 2.0, "darvas_lookback": 30, "darvas_min_bars": 7},
    "Long Term": {"ma_fast": 50, "ma_slow": 200, "rsi_period": 21, "rsi_low": 25, "rsi_high": 75, "macd_fast": 19, "macd_slow": 39, "macd_signal": 9, "adx_period": 21, "adx_threshold": 20, "atr_period": 21, "atr_mult": 2.5, "days": 760, "vol_spike_mult": 2.2, "darvas_lookback": 40, "darvas_min_bars": 9}
}

INTRADAY_PROFILES: Dict[str, Dict[str, Any]] = {
    "Scalping": {"ma_fast": 9, "ma_slow": 21, "rsi_period": 9, "rsi_low": 30, "rsi_high": 70, "macd_fast": 8, "macd_slow": 21, "macd_signal": 5, "adx_period": 10, "adx_threshold": 25, "atr_period": 10, "atr_mult": 2.0, "days": 5, "vol_spike_mult": 1.8, "darvas_lookback": 50, "darvas_min_bars": 7},
    "Momentum": {"ma_fast": 10, "ma_slow": 30, "rsi_period": 14, "rsi_low": 40, "rsi_high": 60, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "adx_period": 14, "adx_threshold": 25, "atr_period": 14, "atr_mult": 2.0, "days": 15, "vol_spike_mult": 2.0, "darvas_lookback": 60, "darvas_min_bars": 10},
    "Reversal": {"ma_fast": 8, "ma_slow": 21, "rsi_period": 20, "rsi_low": 25, "rsi_high": 75, "macd_fast": 15, "macd_slow": 30, "macd_signal": 9, "adx_period": 20, "adx_threshold": 20, "atr_period": 20, "atr_mult": 2.5, "days": 45, "vol_spike_mult": 2.2, "darvas_lookback": 75, "darvas_min_bars": 12}
}

# ------------------------
# Sidebar UI & Parameter Management
# ------------------------
st.sidebar.header("Scanner Configuration")
scan_type = st.sidebar.radio("Scan Type", ["Daily", "Intraday"], key="scan_type_selector")

if 'last_scan_type' not in st.session_state:
    st.session_state.last_scan_type = scan_type

if st.session_state.last_scan_type != scan_type:
    st.session_state.last_scan_type = scan_type
    if scan_type == "Daily":
        profile_to_load, PROFILES, param_suffix = "Medium Term", DAILY_PROFILES, ""
    else:
        profile_to_load, PROFILES, param_suffix = "Momentum", INTRADAY_PROFILES, "_intra"
    for k, v in PROFILES[profile_to_load].items():
        st.session_state[k + param_suffix] = v
    st.session_state['active_profile' + param_suffix] = profile_to_load
    st.rerun()

if scan_type == "Intraday":
    timeframe_options = ["5 Minute", "15 Minute", "30 Minute", "1 Hour"]
    timeframe_choice = st.sidebar.selectbox("Select Timeframe", timeframe_options)
    interval_map = {"5 Minute": "5m", "15 Minute": "15m", "30 Minute": "30m", "1 Hour": "1h"}
    selected_interval = interval_map[timeframe_choice]
    PROFILES, param_suffix, default_profile = INTRADAY_PROFILES, "_intra", "Momentum"
    if "active_profile" + param_suffix not in st.session_state:
        st.session_state["active_profile" + param_suffix] = default_profile
        for k, v in PROFILES[default_profile].items():
            st.session_state[k + param_suffix] = v
    profile_choice = st.sidebar.radio("Profile", list(PROFILES.keys()), key="profile_choice_intra", index=list(PROFILES.keys()).index(st.session_state.get('active_profile_intra', default_profile)))
else:
    PROFILES, param_suffix, default_profile = DAILY_PROFILES, "", "Medium Term"
    if "active_profile" not in st.session_state:
        st.session_state["active_profile"] = default_profile
        for k, v in PROFILES[default_profile].items():
            st.session_state[k] = v
    profile_choice = st.sidebar.radio("Profile", list(PROFILES.keys()), key="profile_choice_daily", index=list(PROFILES.keys()).index(st.session_state.get('active_profile', default_profile)))

active_profile_key = "active_profile" + param_suffix
if profile_choice != st.session_state.get(active_profile_key):
    for k, v in PROFILES[profile_choice].items():
        st.session_state[k + param_suffix] = v
    st.session_state[active_profile_key] = profile_choice
    st.rerun()

st.sidebar.markdown("**Manual Overrides**")
st.sidebar.number_input("Fast MA", 1, 500, key="ma_fast" + param_suffix)
st.sidebar.number_input("Slow MA", 1, 2000, key="ma_slow" + param_suffix)
st.sidebar.number_input("RSI Period", 2, 100, key="rsi_period" + param_suffix)
st.sidebar.number_input("RSI Low", 1, 60, key="rsi_low" + param_suffix)
st.sidebar.number_input("RSI High", 40, 99, key="rsi_high" + param_suffix)
st.sidebar.number_input("MACD Fast", 1, 200, key="macd_fast" + param_suffix)
st.sidebar.number_input("MACD Slow", 1, 500, key="macd_slow" + param_suffix)
st.sidebar.number_input("MACD Signal", 1, 200, key="macd_signal" + param_suffix)
st.sidebar.number_input("ADX Period", 5, 100, key="adx_period" + param_suffix)
st.sidebar.number_input("ADX Threshold", 1, 100, key="adx_threshold" + param_suffix)
st.sidebar.number_input("ATR Period", 1, 200, key="atr_period" + param_suffix)
st.sidebar.number_input("ATR Multiplier", 1.0, 5.0, step=0.1, key="atr_mult" + param_suffix)
days_max = 60 if scan_type == "Intraday" else 2000
days_help = "Max 60 for Intraday" if scan_type == "Intraday" else None
min_lookback_days = 30 if scan_type == "Daily" else 3
st.sidebar.number_input("Lookback Days", min_value=min_lookback_days, max_value=days_max, key="days" + param_suffix, help=days_help)
st.sidebar.number_input("Volume Spike Multiplier", 1.0, 5.0, step=0.1, key="vol_spike_mult" + param_suffix)
st.sidebar.number_input("Darvas Lookback (candles)", 5, 200, key="darvas_lookback" + param_suffix)
st.sidebar.number_input("Darvas Min Bars", 2, 60, key="darvas_min_bars" + param_suffix)

if st.sidebar.button("🔄 Reset to Profile Defaults", key="reset_button" + param_suffix):
    for k, v in PROFILES[profile_choice].items():
        st.session_state[k + param_suffix] = v
    st.session_state[active_profile_key] = profile_choice
    st.rerun()

PARAM_KEYS = ["ma_fast", "ma_slow", "rsi_period", "rsi_low", "rsi_high", "macd_fast", "macd_slow", "macd_signal", "adx_period", "adx_threshold", "atr_period", "atr_mult", "days", "vol_spike_mult", "darvas_lookback", "darvas_min_bars"]
PARAMS = {key: st.session_state[key + param_suffix] for key in PARAM_KEYS}
if scan_type == "Intraday":
    PARAMS['interval'] = selected_interval

# ------------------------
# Data Fetching (Unified)
# ------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker: str, days: int, interval: str) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    period = f"{min(days, 60 if 'm' in interval or 'h' in interval else 5000)}d"
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=24 * 3600)
def get_nse_tickers() -> List[str]:
    try:
        df = pd.read_csv("https://archives.nseindia.com/content/equities/EQUITY_L.csv")
        if "SYMBOL" in df.columns:
            syms = df["SYMBOL"].astype(str).str.strip().str.upper().tolist()
            return sorted(set([s + ".NS" for s in syms if re.fullmatch(r"[A-Z0-9\-]+", s)]))
    except Exception:
        st.warning("Could not fetch full NSE ticker list — using fallback F&O list.")
    return FUTURES_STOCKS

# ------------------------
# Signal Computation (Unified - logic applicable to any timeframe)
# ------------------------
def _is_bull(o, c): return np.isfinite(o) and np.isfinite(c) and c > o
def _is_bear(o, c): return np.isfinite(o) and np.isfinite(c) and c < o

def detect_patterns(df: pd.DataFrame) -> List[str]:
    if df.shape[0] < 3: return []
    patterns = []
    last, prev = df.iloc[-1], df.iloc[-2]
    o1, c1, o2, c2 = to_scalar(prev["Open"]), to_scalar(prev["Close"]), to_scalar(last["Open"]), to_scalar(last["Close"])
    if np.isfinite(o1) and np.isfinite(c1) and np.isfinite(o2) and np.isfinite(c2):
        if _is_bear(o1, c1) and _is_bull(o2, c2) and (c2 > o1) and (o2 < c1): patterns.append("Bullish Engulfing")
        if _is_bull(o1, c1) and _is_bear(o2, c2) and (c2 < o1) and (o2 > c1): patterns.append("Bearish Engulfing")
    return patterns

def volume_spike(vol: pd.Series, mult: float = 2.0, window: int = 20) -> bool:
    arr = vol.to_numpy().ravel()
    if arr.size < 6: return False
    avg_prev = pd.Series(arr[:-1]).tail(min(window, arr.size - 1)).mean()
    return bool(np.isfinite(avg_prev) and avg_prev > 0 and arr[-1] > mult * avg_prev)

def darvas_box_breakout(df: pd.DataFrame, lookback: int, min_bars: int) -> Tuple[bool, float, float]:
    if df.shape[0] < lookback + min_bars + 2: return False, np.nan, np.nan
    window = df.iloc[-(lookback + 1):-1]
    if window.empty: return False, np.nan, np.nan
    high_series = window['High'].squeeze()
    box_high = high_series.max()
    peak_indices = high_series.index[high_series >= box_high]
    if peak_indices.empty: return False, np.nan, np.nan
    last_peak_time = peak_indices[-1]
    consolidation_period = window.loc[window.index > last_peak_time]
    if len(consolidation_period) < min_bars or (consolidation_period['High'].squeeze() > box_high).any(): return False, np.nan, np.nan
    box_low = consolidation_period['Low'].min()
    is_breakout = to_scalar(df['Close']) > box_high
    return bool(is_breakout), box_high, box_low

def _linear_slope(y: pd.Series) -> float:
    arr = y.dropna().to_numpy().ravel()
    if arr.size < 5: return np.nan
    return float(np.polyfit(np.arange(arr.size), arr, 1)[0])

def compute_signals(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    if df.shape[0] < 20: return {"Error": "Insufficient data"}
    out = {}
    close, vol = df["Close"], df["Volume"].fillna(0)
    
    # --- Indicator Calculations ---
    rsi_vals = rsi(close, int(params["rsi_period"]))
    macd_line, macd_sig, _ = macd_lines(close, int(params["macd_fast"]), int(params["macd_slow"]), int(params["macd_signal"]))
    plus_di, minus_di, adx_vals = _di_lines(df, int(params["adx_period"]))
    obv_vals = obv(close, vol)
    atr_vals = atr(df, int(params["atr_period"]))

    # --- Data for Confirmation Filters ---
    sma200 = sma(close, 200)
    vol_avg = vol.rolling(window=20).mean()
    rsi_long = rsi(close, int(params["rsi_period"] * 4)) 
    confirmation_data = {"sma200": sma200, "vol_avg": vol_avg, "rsi_long": rsi_long, "vol_mult": 1.5}

    last_close = to_scalar(close)
    last_rsi = to_scalar(rsi_vals)
    last_adx = to_scalar(adx_vals)
    last_atr = to_scalar(atr_vals)

    # --- Events ---
    macd_diff = macd_line - macd_sig
    out["MACD_Event"] = "Bullish Cross" if to_scalar(macd_diff) > 0 and to_scalar(macd_diff.shift(1)) <= 0 else "Bearish Cross" if to_scalar(macd_diff) < 0 and to_scalar(macd_diff.shift(1)) >= 0 else ""
    ma_diff = sma(close, int(params["ma_fast"])) - sma(close, int(params["ma_slow"]))
    out["MA_Cross"] = "Golden Cross" if to_scalar(ma_diff) > 0 and to_scalar(ma_diff.shift(1)) <= 0 else "Death Cross" if to_scalar(ma_diff) < 0 and to_scalar(ma_diff.shift(1)) >= 0 else ""
    out["RSI_Event"] = "Oversold" if last_rsi < params["rsi_low"] else "Overbought" if last_rsi > params["rsi_high"] else ""

    # --- Trends & Strength ---
    out["ADX_Strength"] = "Strong Trend" if last_adx > params["adx_threshold"] else "Weak Trend"
    out["ADX_Direction"] = "Up" if to_scalar(plus_di) > to_scalar(minus_di) else "Down"
    out["OBV_Trend"] = "Up" if _linear_slope(obv_vals.tail(20)) > 0 else "Down" if _linear_slope(obv_vals.tail(20)) < 0 else ""

    # --- Patterns & Other Signals ---
    out["Patterns"] = detect_patterns(df)
    
    bullish_divs, bearish_divs = detect_divergences(df, rsi_vals, macd_line, confirmation_data)
    divergence_signals = []
    if to_scalar(bullish_divs.iloc[-1]): divergence_signals.append("Bullish Divergence (Confirmed)")
    if to_scalar(bearish_divs.iloc[-1]): divergence_signals.append("Bearish Divergence (Confirmed)")
    out["Divergences"] = divergence_signals
    
    out["Vol_Spike"] = volume_spike(vol, params["vol_spike_mult"])
    is_breakout, box_high, box_low = darvas_box_breakout(df, int(params["darvas_lookback"]), int(params["darvas_min_bars"]))
    out["Darvas_Breakout"] = is_breakout
    out["Darvas_High"] = box_high
    out["Darvas_Low"] = box_low
    out["ATR_Stop_Loss"] = last_close - (last_atr * params["atr_mult"]) if np.isfinite(last_close) and np.isfinite(last_atr) else np.nan
    
    out["Above_SMA200"] = last_close > to_scalar(sma200) if not sma200.empty else False

    # --- Scoring ---
    score = 0
    if out["Above_SMA200"]: score += 5
    if out.get("MA_Cross") == "Golden Cross": score += 5
    elif out.get("MA_Cross") == "Death Cross": score -= 5
    if out["MACD_Event"] == "Bullish Cross": score += 3
    elif out["MACD_Event"] == "Bearish Cross": score -= 3
    if out.get("ADX_Strength") == "Strong Trend": score += 3 if out.get("ADX_Direction") == "Up" else -3
    if out["Darvas_Breakout"]: score += 5
    if out["Vol_Spike"]: score += 1
    if out["OBV_Trend"] == "Up": score += 1
    elif out["OBV_Trend"] == "Down": score -= 1
    
    # Divergence scoring
    # FIX: Use a more robust check for boolean Series
    if to_scalar(bullish_divs.iloc[-1]):
        score += 2
    if to_scalar(bearish_divs.iloc[-1]):
        score -= 2

    out["Score"] = int(score)
    out["Close"] = float(last_close) if np.isfinite(last_close) else 0.0
    out["RSI"] = float(last_rsi) if np.isfinite(last_rsi) else 0.0
    out["ADX"] = float(last_adx) if np.isfinite(last_adx) else 0.0
    return out

# ------------------------
# Plotting
# ------------------------
@st.cache_data(show_spinner=False)
def make_sparkline_base64(series: pd.Series, fast: int, slow: int) -> str:
    data = series.tail(120)
    fig, ax = plt.subplots(figsize=(2.0, 0.6), dpi=72)
    ax.plot(data.index, data.values, linewidth=1)
    if len(data) > slow:
        ax.plot(data.index, sma(series, fast).tail(120).values, linewidth=0.8, color='orange')
        ax.plot(data.index, sma(series, slow).tail(120).values, linewidth=0.8, color='purple')
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode("utf-8")}" width="120" />'

# ------------------------
# Main App Body
# ------------------------
st.sidebar.markdown("---")
st.sidebar.info(f"Scan Type: **{scan_type}**\n\nProfile: **{profile_choice}**")

st.header(f"Scanner ({scan_type})")
universe_choice = st.selectbox("Universe", ["Futures Stock", "NIFTY100", "NIFTY500", "All NSE Stocks", "Custom"])

if universe_choice == "Custom":
    tickers_text = st.text_area("Tickers (comma separated)", "RELIANCE.NS,TCS.NS,INFY.NS")
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
else:
    all_nse = get_nse_tickers()
    tickers_map = {"Futures Stock": FUTURES_STOCKS, "NIFTY100": all_nse[:100], "NIFTY500": all_nse[:500], "All NSE Stocks": all_nse}
    tickers = tickers_map[universe_choice]

min_volume = st.number_input("Minimum Average Volume (20-period)", 0, 1000000, 50000, step=10000)
count = st.number_input("Scan first N tickers", 1, len(tickers), 50)
tickers = tickers[:count]

scan_key_suffix = "_intra" if scan_type == "Intraday" else "_daily"

if st.button("▶ Run Scan", key="run_scan_button"):
    results: List[Dict[str, Any]] = []
    progress = st.progress(0.0, text="Initializing scan...")
    interval_to_fetch = PARAMS.get('interval', '1d')

    for i, t in enumerate(tickers):
        progress.progress((i + 1) / len(tickers), text=f"Scanning {t} ({interval_to_fetch})...")
        df = fetch_history(t, int(PARAMS["days"]), interval_to_fetch)
        if df.empty or 'Volume' not in df.columns or df['Volume'].shape[0] < 20: continue
        if df['Volume'].squeeze().tail(20).mean() < min_volume: continue

        sigs = compute_signals(df, PARAMS)
        if "Error" in sigs: continue

        img_html = make_sparkline_base64(df["Close"], int(PARAMS["ma_fast"]), int(PARAMS["ma_slow"]))
        row = {"Ticker": t, "Chart": img_html, **sigs}
        results.append(row)

    progress.progress(1.0, text="Scan complete!")
    st.session_state["last_scan" + scan_key_suffix] = results
    st.success(f"✅ {scan_type} scan completed for {len(results)} tickers.")

if "last_scan" + scan_key_suffix in st.session_state:
    results = st.session_state["last_scan" + scan_key_suffix]
    if results:
        df_out = pd.DataFrame(results).drop(columns=['Patterns', 'Divergences'], errors='ignore')
        df_out['Divergences_str'] = [', '.join(d) for d in [r.get('Divergences', []) for r in results]]
        df_out.sort_values(by="Score", ascending=False, inplace=True)

        def color_score(val):
            color = "#28a745" if val > 0 else "#dc3545" if val < 0 else "grey"
            return f'<span style="color:{color};font-weight:bold">{val}</span>'
        
        def format_value_robust(val):
            """Safely formats a value if it's a number, otherwise returns an empty string."""
            if isinstance(val, (int, float, np.number)) and pd.notna(val):
                return f"{val:.2f}"
            return ""

        df_disp = df_out.copy()
        df_disp["Score"] = df_disp["Score"].apply(color_score)
        
        format_cols = ["Close", "RSI", "ADX", "Darvas_High", "Darvas_Low", "ATR_Stop_Loss"]
        for col in format_cols:
            if col in df_disp.columns:
                df_disp[col] = df_disp[col].apply(format_value_robust)

        st.write(df_disp.to_html(escape=False, index=False), unsafe_allow_html=True)

        st.markdown("---")
        st.header("🔬 Send Results to Backtester")

        min_score_val = int(df_out['Score'].min()) if not df_out.empty else 0
        max_score_val = int(df_out['Score'].max()) if not df_out.empty else 1
        default_score = int(df_out['Score'].quantile(0.8)) if not df_out.empty else 0

        min_score = st.slider("Minimum Score to Send", min_score_val, max_score_val, default_score)
        filtered_df = df_out[df_out['Score'] >= min_score]
        st.write(f"**Found {len(filtered_df)} tickers matching filters.**")
        
        cols_to_show = ['Ticker', 'Score', 'Close', 'Darvas_High', 'ATR_Stop_Loss', 'MACD_Event', 'MA_Cross']
        display_cols = [c for c in cols_to_show if c in filtered_df.columns]
        st.dataframe(filtered_df[display_cols].head())

        if st.button("Send Tickers and Open Unified Backtester", type="primary"):
            tickers_to_send = filtered_df['Ticker'].tolist()
            if not tickers_to_send:
                st.warning("No tickers selected to send.")
            else:
                if scan_type == "Intraday":
                    st.session_state['tickers_for_backtest_intra'] = tickers_to_send
                    st.session_state['scanner_params_intra'] = PARAMS
                    if 'tickers_for_backtest' in st.session_state: del st.session_state['tickers_for_backtest']
                    if 'scanner_params' in st.session_state: del st.session_state['scanner_params']
                else: # Daily
                    st.session_state['tickers_for_backtest'] = tickers_to_send
                    st.session_state['scanner_params'] = PARAMS
                    if 'tickers_for_backtest_intra' in st.session_state: del st.session_state['tickers_for_backtest_intra']
                    if 'scanner_params_intra' in st.session_state: del st.session_state['scanner_params_intra']

                st.switch_page("pages/2_Unified_Backtester.py")

