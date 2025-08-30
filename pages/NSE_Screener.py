# pages/NSE_Screener.py - A Streamlit page to automatically identify potential multibagger and intraday momentum stocks from the NSE.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
import concurrent.futures
from functools import partial
import threading

# Interactive charts
import plotly.graph_objects as go

# --- NaN/None → manual input fallback helper ---
def manual_value(label, value, default=0.0, fmt="%.4f", key=None, help_text=None):
    def _is_nan(x):
        try:
            return isinstance(x, float) and np.isnan(x)
        except Exception:
            return False

    if value is None or _is_nan(value):
        return st.number_input(
            f"{label} (missing – enter manually)",
            value=float(default),
            format=fmt,
            key=key or f"manual_{label}",
            help=help_text
        )
    return value


# Import shared valuation utilities
from valuation_utils import (
    normalize_nse_ticker,
    first_existing_row,
    cagr_from_series,
    safe_ratio,
    calculate_cagr,
    calculate_dcf_fair_value,
    calculate_graham_number,
    calculate_ddm_fair_value,
)

# --- Page Configuration ---
st.set_page_config(page_title="Automated NSE Screener", page_icon="🤖", layout="wide")

# --- Hide Streamlit Menu & Footer ---
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar for User-defined Criteria ---
st.sidebar.header("⚙️ Screener Parameters")

st.sidebar.subheader("Potential Multibagger Criteria")
min_rev_cagr = st.sidebar.slider("Min Revenue CAGR (%)", 0, 100, 15, 1) / 100.0
min_ni_cagr = st.sidebar.slider("Min Net Income CAGR (%)", 0, 100, 18, 1) / 100.0
min_roe = st.sidebar.slider("Min Average ROE (%)", 0, 100, 15, 1) / 100.0
max_d_e = st.sidebar.slider("Max Debt to Equity", 0.0, 5.0, 0.6, 0.1)
max_pe = st.sidebar.slider("Max P/E Ratio", 0, 200, 50, 1)

user_multibagger_criteria = {
    'min_rev_cagr': min_rev_cagr,
    'min_ni_cagr': min_ni_cagr,
    'min_roe': min_roe,
    'max_d_e': max_d_e,
    'max_pe': max_pe
}

st.sidebar.subheader("Intraday Momentum Criteria")
st.sidebar.markdown("**High Momentum**")
high_min_price_change = st.sidebar.slider("Min Price Change (%)", 0.0, 10.0, 2.0, 0.1)
high_min_volume_ratio = st.sidebar.slider("Min Volume Ratio (vs 20-day avg)", 1.0, 10.0, 1.5, 0.1)

st.sidebar.markdown("**Moderate Momentum**")
mod_min_price_change = st.sidebar.slider("Min Price Change (%) ", 0.0, 10.0, 1.0, 0.1)
mod_min_volume_ratio = st.sidebar.slider("Min Volume Ratio (vs 20-day avg) ", 1.0, 10.0, 1.2, 0.1)

user_intraday_criteria = {
    'high_price_change': high_min_price_change,
    'high_volume_ratio': high_min_volume_ratio,
    'mod_price_change': mod_min_price_change,
    'mod_volume_ratio': mod_min_volume_ratio,
}

# Dynamic Fair Value Parameters in Sidebar
st.sidebar.subheader("💸 Fair Value Parameters")
fv_growth = st.sidebar.slider("Expected Growth Rate (DCF) %", 0, 50, 10) / 100.0
fv_discount = st.sidebar.slider("Discount Rate (DCF) %", 0, 20, 8) / 100.0
fv_terminal = st.sidebar.slider("Terminal Growth (DCF) %", 0, 10, 2) / 100.0
fv_required_rate = st.sidebar.slider("Required Return (DDM) %", 0, 20, 8) / 100.0
fv_div_growth = st.sidebar.slider("Dividend Growth (DDM) %", 0, 15, 2) / 100.0

# --- Session State Initialization ---
if 'scan_running' not in st.session_state:
    st.session_state.scan_running = False
if 'stop_scan' not in st.session_state:
    st.session_state.stop_scan = False
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = None
if 'sector_momentum_scores' not in st.session_state:
    st.session_state.sector_momentum_scores = {}

# --- Helper Functions ---
@st.cache_data
def get_nse_tickers():
    """Fetch the list of all equity tickers from the NSE archives."""
    try:
        url = 'https://archives.nseindia.com/content/equities/EQUITY_L.csv'
        df = pd.read_csv(url)
        return (df['SYMBOL'] + '.NS').tolist()
    except Exception as e:
        st.error(f"Failed to fetch NSE ticker list: {e}")
        return []

@st.cache_data
def get_screening_data(ticker):
    """Fetch fundamental and recent price data for analysis."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or 'currentPrice' not in info:
            return None, f"Incomplete market data for '{ticker}'."
        
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        if financials.empty or balance_sheet.empty:
            return None, f"Missing financial statements for '{ticker}'."

        history = stock.history(period="60d")
        if history.empty:
            return None, f"Could not fetch price history for {ticker}."

        return {
            "info": info,
            "financials": financials,
            "balance_sheet": balance_sheet,
            "history": history
        }, None
    except Exception:
        return None, f"An error occurred while fetching data for {ticker}."

def analyze_multibagger_potential(data, criteria):
    """Robust multibagger analysis with fallbacks for NSE stocks."""
    info, financials, balance_sheet = data['info'], data['financials'], data['balance_sheet']
    results, score = {}, 0
    try:
        # Flexible labels
        revenue = first_existing_row(financials, ["Total Revenue","TotalRevenue","Revenue","Operating Revenue"])
        net_income = first_existing_row(financials, ["Net Income","NetIncome","Profit After Tax"])
        total_equity = first_existing_row(balance_sheet, ["Total Stockholder Equity","Total Equity","Shareholders Equity"])

        results['Revenue CAGR'] = cagr_from_series(revenue)
        results['Net Income CAGR'] = cagr_from_series(net_income)

        roe_series = safe_ratio(net_income, total_equity)
        results['Average ROE'] = float(roe_series.mean()) if roe_series is not None else 0.0

        dte = info.get('debtToEquity', None)
        if isinstance(dte, (int, float)) and 0 < dte < 10:
            results['Debt to Equity'] = dte
        else:
            total_debt = first_existing_row(balance_sheet, ["Total Debt","Borrowings","Long Term Debt"])
            dte_series = safe_ratio(total_debt, total_equity) if total_debt is not None and total_equity is not None else None
            results['Debt to Equity'] = float(dte_series.mean()) if dte_series is not None else np.nan

        pe = info.get('trailingPE', None)
        if not isinstance(pe, (int, float)) or pe <= 0 or (isinstance(pe, float) and np.isnan(pe)):
            price, eps = info.get('currentPrice', None), info.get('trailingEps', None)
            pe = (price/eps) if price and eps and eps > 0 else np.nan
        results['P/E Ratio'] = pe

        # Scoring
        if results['Revenue CAGR'] > criteria['min_rev_cagr']: score += 1
        if results['Net Income CAGR'] > criteria['min_ni_cagr']: score += 1
        if results['Average ROE'] > criteria['min_roe']: score += 1
        if isinstance(results['Debt to Equity'], (int,float)) and results['Debt to Equity'] < criteria['max_d_e']: score += 1
        if isinstance(results['P/E Ratio'], (int,float)) and results['P/E Ratio'] < criteria['max_pe']: score += 1

    except Exception:
        results = {'Revenue CAGR': 0, 'Net Income CAGR': 0, 'Average ROE': 0, 'Debt to Equity': np.nan, 'P/E Ratio': np.nan}
        score = 0
    return results, score

def analyze_intraday_momentum(history, criteria):
    """
    Analyze recent price history for signs of intraday momentum,
    differentiating between buy (positive) and sell (negative) momentum.
    """
    if len(history) < 2:
        return "N/A"
    today, yesterday = history.iloc[-1], history.iloc[-2]
    avg_volume = history['Volume'].iloc[-21:-1].mean()
    volume_ratio = today['Volume'] / avg_volume if avg_volume > 0 else 0
    price_change = ((today['Close'] - yesterday['Close']) / yesterday['Close']) * 100
    abs_price_change = abs(price_change)
    direction = "Buy" if price_change > 0 else "Sell"
    if abs_price_change > criteria['high_price_change'] and volume_ratio > criteria['high_volume_ratio']:
        return f"High {direction} Momentum"
    if abs_price_change > criteria['mod_price_change'] and volume_ratio > criteria['mod_volume_ratio']:
        return f"Moderate {direction} Momentum"
    return "Low"

def get_recommendation(score):
    if score == 5: return "Strong Buy"
    if score == 4: return "Buy"
    if score == 3: return "Consider"
    return "Avoid"

# Momentum scoring for sectors
def momentum_to_score(signal: str) -> int:
    """
    Map text signals to numeric scores for aggregation:
    High Buy +2, Moderate Buy +1, Low 0, Moderate Sell -1, High Sell -2
    """
    if not isinstance(signal, str):
        return 0
    s = signal.lower()
    if "high" in s and "buy" in s: return 2
    if "moderate" in s and "buy" in s: return 1
    if "moderate" in s and "sell" in s: return -1
    if "high" in s and "sell" in s: return -2
    return 0

def process_ticker(ticker, multibagger_criteria, intraday_criteria, stop_event):
    """Fetch + analyze a single ticker (used in parallel scan)."""
    if stop_event.is_set():
        return None
    stock_data, error = get_screening_data(ticker)
    if error: return None

    multi_analysis, multi_score = analyze_multibagger_potential(stock_data, multibagger_criteria)
    intraday_signal = analyze_intraday_momentum(stock_data['history'], intraday_criteria)

    # --- NEW: compute daily Price Change % from last 2 closes ---
    price_change_pct = np.nan
    try:
        hist = stock_data['history']
        if hist is not None and len(hist) >= 2:
            today, yest = hist.iloc[-1], hist.iloc[-2]
            if yest['Close'] and yest['Close'] != 0:
                price_change_pct = ((today['Close'] - yest['Close']) / yest['Close']) * 100.0
    except Exception:
        price_change_pct = np.nan

    if multi_score >= 3 or intraday_signal != "Low":
        return {
            "Ticker": ticker,
            "Sector": stock_data['info'].get('sector', 'N/A'),
            "Price": stock_data['info'].get('currentPrice', 'N/A'),
            "Price Change %": price_change_pct,  # <--- ADDED COLUMN
            "Multibagger Score": multi_score,
            "Recommendation": get_recommendation(multi_score),
            "Intraday Signal": intraday_signal,
            "Revenue CAGR": multi_analysis.get('Revenue CAGR'),
            "Net Income CAGR": multi_analysis.get('Net Income CAGR'),
            "Avg ROE": multi_analysis.get('Average ROE'),
        }
    return None

# --- Main UI ---
st.title("🇮🇳 Automated NSE Screener")
st.markdown("This tool scans all stocks on the **National Stock Exchange (NSE)** for:")
st.markdown("- **Potential Multibaggers** (long-term investment)")
st.markdown("- **Intraday Momentum** (short-term trading)")

# --- Control Buttons ---
col1, col2, col3 = st.columns([1,1,5])
if col1.button("🚀 Start Scan", use_container_width=True, type="primary", disabled=st.session_state.scan_running):
    st.session_state.scan_running = True
    st.session_state.stop_scan = False
    st.session_state.scan_results = []
    st.session_state.sector_momentum_scores = {}
    st.session_state.stop_event = threading.Event()
    st.session_state.run_multibagger_criteria = user_multibagger_criteria
    st.session_state.run_intraday_criteria = user_intraday_criteria
    st.rerun()

if col2.button("🛑 Stop Scan", use_container_width=True, disabled=not st.session_state.scan_running):
    if st.session_state.stop_event:
        st.session_state.stop_event.set()
    st.session_state.stop_scan = True
    st.session_state.scan_running = False
    st.rerun()

st.markdown("---")

# --- Automated Execution with LIVE streaming results + Sector Momentum ---
if st.session_state.scan_running:
    tickers = get_nse_tickers()
    if not tickers:
        st.error("Could not retrieve NSE tickers. Halting scan.")
        st.session_state.scan_running = False
        st.stop()

    all_results = []
    results_placeholder = st.empty()   # live results table (flat)
    momentum_placeholder = st.empty()  # live momentum leaderboard
    progress_bar = st.progress(0, text=f"Initializing scan of {len(tickers)} NSE stocks...")
    processed_count = 0

    task_function = partial(
        process_ticker,
        multibagger_criteria=st.session_state.run_multibagger_criteria,
        intraday_criteria=st.session_state.run_intraday_criteria,
        stop_event=st.session_state.stop_event
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_ticker = {executor.submit(task_function, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            if st.session_state.stop_event.is_set():
                for f in future_to_ticker:
                    try:
                        f.cancel()
                    except:
                        pass
                break

            processed_count += 1
            ticker = future_to_ticker[future]
            progress_bar.progress(processed_count / len(tickers), text=f"Analyzing {ticker} ({processed_count}/{len(tickers)})...")

            try:
                result = future.result()
                if result:
                    all_results.append(result)

                    # --- LIVE update: Results table (flat preview, now includes Price Change %) ---
                    live_df = pd.DataFrame(all_results).sort_values(by=["Sector", "Multibagger Score"], ascending=[True, False])
                    results_placeholder.dataframe(
                        live_df[["Ticker", "Sector", "Price", "Price Change %", "Multibagger Score", "Recommendation", "Intraday Signal"]],
                        use_container_width=True,
                        hide_index=True
                    )

                    # --- LIVE update: Sector Momentum leaderboard ---
                    if not live_df.empty:
                        live_df["MomentumScore"] = live_df["Intraday Signal"].apply(momentum_to_score)
                        sector_scores = live_df.groupby("Sector")["MomentumScore"].sum().sort_values(ascending=False)
                        st.session_state.sector_momentum_scores = sector_scores.to_dict()

                        fig = go.Figure(data=[go.Bar(x=list(sector_scores.index), y=list(sector_scores.values))])
                        fig.update_layout(
                            title="📈 Live Sector Momentum (High Buy=+2 • Mod Buy=+1 • Mod Sell=-1 • High Sell=-2)",
                            xaxis_title="Sector",
                            yaxis_title="Net Momentum Score",
                            margin=dict(l=10, r=10, t=40, b=10),
                            height=350
                        )
                        momentum_placeholder.plotly_chart(fig, use_container_width=True)
            except:
                # Ignore individual ticker errors during scan
                pass

    st.session_state.scan_results = all_results
    st.session_state.scan_running = False
    st.rerun()

# --- Display Results ---
if st.session_state.stop_scan:
    st.warning("Scan stopped by user.")
    st.session_state.stop_scan = False

# Show final Sector Momentum leaderboard (from state) even after scan
if st.session_state.sector_momentum_scores:
    final_scores = pd.Series(st.session_state.sector_momentum_scores).sort_values(ascending=False)
    st.subheader("🏭 Sector Momentum Leaderboard (Final)")
    fig_final = go.Figure(data=[go.Bar(x=list(final_scores.index), y=list(final_scores.values))])
    fig_final.update_layout(xaxis_title="Sector", yaxis_title="Net Momentum Score", margin=dict(l=10, r=10, t=30, b=10), height=350)
    st.plotly_chart(fig_final, use_container_width=True)

if st.session_state.scan_results:
    results_df = pd.DataFrame(st.session_state.scan_results).sort_values(by=["Sector", "Multibagger Score"], ascending=[True, False])

    # CSV Export (now includes Price Change % automatically)
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Results as CSV",
        data=csv_bytes,
        file_name="nse_screener_results.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.header("🏆 Screening Results")
    for sector, group in results_df.groupby('Sector'):
        st.subheader(sector)

        # Color styling for recommendation and intraday momentum (as before) + Price Change % (new)
        def style_recommendation(val):
            if val == "Strong Buy": return 'background-color: #1E5631; color: white;'
            if val == "Buy": return 'background-color: #2E8B57; color: white;'
            if val == "Consider": return 'background-color: #DAA520;'
            return ''
        def style_intraday(val):
            if isinstance(val, str) and "Buy" in val:
                if "High" in val: return 'background-color: #1E5631; color: white;'
                if "Moderate" in val: return 'background-color: #2E8B57; color: white;'
            elif isinstance(val, str) and "Sell" in val:
                if "High" in val: return 'background-color: #9B111E; color: white;'
                if "Moderate" in val: return 'background-color: #FF4B4B; color: white;'
            return ''
        def style_price_change(val):
            try:
                if val > 0: return 'color: green; font-weight: 700;'
                if val < 0: return 'color: red; font-weight: 700;'
            except Exception:
                pass
            return ''

        styled_group = group.style.applymap(style_recommendation, subset=['Recommendation'])\
                                  .applymap(style_intraday, subset=['Intraday Signal'])\
                                  .applymap(style_price_change, subset=['Price Change %'])\
                                  .format({
                                      "Price": "₹{:,.2f}",
                                      "Price Change %": "{:+.2f}%",
                                      "Multibagger Score": "{}/5",
                                      "Revenue CAGR": "{:.2%}",
                                      "Net Income CAGR": "{:.2%}",
                                      "Avg ROE": "{:.2%}"
                                  }, na_rep="N/A")
        st.dataframe(styled_group, use_container_width=True, hide_index=True)

    # --- On-demand Interactive Stock Detail Viewer (unchanged) ---
    st.subheader("📊 Interactive Stock Viewer")
    all_tickers = results_df["Ticker"].unique().tolist()
    selected_ticker = st.selectbox("Select a stock to view chart & purchase-relevant details:", all_tickers)

    if selected_ticker:
        try:
            stock = yf.Ticker(selected_ticker)
            info = stock.info
            hist = stock.history(period="6mo", interval="1d")

            # Screener row for this ticker
            row = results_df[results_df["Ticker"] == selected_ticker].iloc[0]

            # --- Key Purchase-Relevant Metrics ---
            st.markdown(f"### 🏦 {selected_ticker} — Stock Details")
            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Current Price", f"₹{info.get('currentPrice','N/A')}")
                st.metric("Sector", info.get("sector","N/A"))
            with colB:
                st.metric("P/E Ratio", f"{info.get('trailingPE','N/A')}")
                st.metric("Debt/Equity", f"{info.get('debtToEquity','N/A')}")
            with colC:
                st.metric("Multibagger Score", f"{row['Multibagger Score']}/5")
                st.metric("Intraday Signal", row["Intraday Signal"])

            # --- Fair Value Estimation (DCF, Graham, DDM) ---
            cashflow = getattr(stock, "cashflow", pd.DataFrame())
            fcf = None
            if isinstance(cashflow, pd.DataFrame) and not cashflow.empty and 'Free Cash Flow' in cashflow.index:
                try:
                    fcf = cashflow.loc['Free Cash Flow'].iloc[0]
                except Exception:
                    fcf = None

            shares = info.get('sharesOutstanding')
            dcf_value = calculate_dcf_fair_value(fcf, fv_growth, fv_terminal, fv_discount, shares)
            graham_value = calculate_graham_number(info.get('trailingEps'), info.get('bookValue'))
            ddm_value = calculate_ddm_fair_value(info.get('dividendRate'), fv_required_rate, fv_div_growth)

            fair_values = [v for v in [dcf_value, graham_value, ddm_value] if isinstance(v,(int,float)) and v>0]
            st.markdown("#### 📐 Fair Value Models")
            fv_cols = st.columns(3)
            with fv_cols[0]:
                st.write(f"**DCF:** {'₹{:,.2f}'.format(dcf_value) if isinstance(dcf_value,(int,float)) and dcf_value>0 else 'N/A'}")
            with fv_cols[1]:
                st.write(f"**Graham:** {'₹{:,.2f}'.format(graham_value) if isinstance(graham_value,(int,float)) and graham_value>0 else 'N/A'}")
            with fv_cols[2]:
                st.write(f"**DDM:** {'₹{:,.2f}'.format(ddm_value) if isinstance(ddm_value,(int,float)) and ddm_value>0 else 'N/A'}")

            if fair_values and info.get("currentPrice"):
                avg_fv = float(np.mean(fair_values))
                current_price = info["currentPrice"]
                upside = ((avg_fv - current_price) / current_price) * 100
                st.write(f"**Estimated Fair Value (Avg of models):** ₹{avg_fv:,.2f}")
                if upside > 20:
                    st.success(f"✅ Strong Buy — Potential Upside {upside:.2f}%")
                elif 10 < upside <= 20:
                    st.info(f"🟢 Buy — Potential Upside {upside:.2f}%")
                elif -10 <= upside <= 10:
                    st.warning(f"➖ Hold — Fairly Priced (Upside {upside:.2f}%)")
                else:
                    st.error(f"❌ Sell — Overvalued (Downside {upside:.2f}%)")
            else:
                st.info("⚠️ Could not compute aggregated fair value (insufficient inputs).")

            st.caption(
                f"DCF params → Growth: {fv_growth:.0%}, Discount: {fv_discount:.0%}, Terminal: {fv_terminal:.0%} | "
                f"DDM params → Required: {fv_required_rate:.0%}, Dividend growth: {fv_div_growth:.0%}"
            )

            # --- Chart (on-demand only) ---
            st.markdown("#### 📈 6-Month Price Chart")
            if hist is None or hist.empty:
                st.info("No price history available.")
            else:
                fig = go.Figure(data=[go.Candlestick(
                    x=hist.index,
                    open=hist['Open'],
                    high=hist['High'],
                    low=hist['Low'],
                    close=hist['Close']
                )])
                fig.update_layout(
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=10, r=10, t=20, b=10),
                    height=420,
                    title=f"{selected_ticker} — 6M Price History"
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not render details for {selected_ticker}: {e}")

elif not st.session_state.scan_running:
    st.info("Click 'Start Scan' to begin screening NSE stocks.")

# --- Manual Stock Search ---
st.markdown("---")
st.header("🔎 Search Individual Stock")
search_ticker = st.text_input("Enter NSE Ticker (e.g., RELIANCE.NS, TCS.NS, HDFCBANK.NS):", "")
if st.button("Search"):
    ticker = normalize_nse_ticker(search_ticker)
    if not ticker:
        st.warning("Please enter a valid ticker (with .NS).")
    else:
        data, error = get_screening_data(ticker)
        if error:
            st.error(error)
        else:
            multi_analysis, multi_score = analyze_multibagger_potential(data, user_multibagger_criteria)
            intraday_signal = analyze_intraday_momentum(data['history'], user_intraday_criteria)
            recommendation = get_recommendation(multi_score)
            info = data['info']
            current_price = info.get('currentPrice', None)
            st.subheader(f"📊 Analysis for {ticker}")
            st.write("**Multibagger Score:**", f"{multi_score}/5")
            st.write("**Recommendation:**", recommendation)
            st.write("**Intraday Signal:**", intraday_signal)
            st.write("**Current Price:**", f"₹{current_price}" if current_price else "N/A")

            # --- Fair Value Estimation (dynamic via sidebar inputs) ---
            stock = yf.Ticker(ticker)
            cashflow = getattr(stock, "cashflow", pd.DataFrame())
            fcf = None
            if isinstance(cashflow, pd.DataFrame) and not cashflow.empty and 'Free Cash Flow' in cashflow.index:
                try:
                    fcf = cashflow.loc['Free Cash Flow'].iloc[0]
                except Exception:
                    fcf = None
            shares = info.get('sharesOutstanding')

            dcf_value = calculate_dcf_fair_value(fcf, fv_growth, fv_terminal, fv_discount, shares)
            graham_value = calculate_graham_number(info.get('trailingEps'), info.get('bookValue'))
            ddm_value = calculate_ddm_fair_value(info.get('dividendRate'), fv_required_rate, fv_div_growth)

            fair_values = [v for v in [dcf_value, graham_value, ddm_value] if isinstance(v, (int, float)) and v > 0]
            if fair_values:
                avg_fair_value = float(np.mean(fair_values))
                st.write(f"**Estimated Fair Value (Avg of models):** ₹{avg_fair_value:,.2f}")
                if current_price:
                    upside = ((avg_fair_value - current_price) / current_price) * 100
                    if upside > 20: st.success(f"✅ Strong Buy — Potential Upside {upside:.2f}%")
                    elif 10 < upside <= 20: st.info(f"🟢 Buy — Potential Upside {upside:.2f}%")
                    elif -10 <= upside <= 10: st.warning(f"➖ Hold — Fairly Priced (Upside {upside:.2f}%)")
                    else: st.error(f"❌ Sell — Overvalued (Downside {upside:.2f}%)")
            else:
                st.info("⚠️ Could not estimate fair value (missing EPS or growth data).")

            st.caption(
                f"DCF params → Growth: {fv_growth:.0%}, Discount: {fv_discount:.0%}, Terminal: {fv_terminal:.0%} | "
                f"DDM params → Required: {fv_required_rate:.0%}, Dividend growth: {fv_div_growth:.0%}"
            )

            st.write("### Key Metrics")
            st.json(multi_analysis)
