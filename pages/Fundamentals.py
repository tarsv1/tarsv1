# pages/fundamentals.py - Fundamental analysis & valuation page (DCF, Graham, DDM)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
# --- NaN/None → manual input fallback helper ---
def manual_value(label, value, default=0.0, fmt="%.4f", key=None, help_text=None):
    """
    If 'value' is None or NaN, shows a Streamlit number_input and returns the user entry.
    Otherwise returns 'value' unchanged.
    """
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


from valuation_utils import (
    normalize_nse_ticker,
    calculate_cagr,
    calculate_dcf_fair_value,
    calculate_graham_number,
    calculate_ddm_fair_value,
)

st.set_page_config(page_title="Fundamental Analysis", page_icon="🧾", layout="wide")

# -------- Data fetch --------
@st.cache_data
def get_fundamental_data(ticker):
    """Fetches fundamental data for a given stock ticker from yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or 'currentPrice' not in info:
            return None, f"Could not retrieve valid data for '{ticker}'."
        return {
            "info": info,
            "financials": stock.financials,
            "balance_sheet": stock.balance_sheet,
            "cashflow": stock.cashflow
        }, None
    except Exception as e:
        return None, f"Error while fetching {ticker}: {e}"

# -------- UI --------
st.title("🧾 Fundamental Analysis & Stock Valuation")
st.markdown("Enter an NSE ticker to get financials plus fair value via DCF, Graham, and DDM.")

raw = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, TCS.NS)", value="RELIANCE.NS")
ticker = normalize_nse_ticker(raw)

if st.button("Analyze Stock", use_container_width=True):
    if not ticker:
        st.warning("Please enter a stock ticker.")
    else:
        with st.spinner(f"Fetching and analyzing data for {ticker}..."):
            stock_data, error = get_fundamental_data(ticker)
            if error:
                st.error(error)
            else:
                info = stock_data['info']
                financials = stock_data['financials']
                cashflow = stock_data['cashflow']
                current_price = info.get('currentPrice', 'N/A')

                st.header(f"Key Metrics for {info.get('longName', ticker)}")
                cols = st.columns(4)
                cols[0].metric("Current Price (₹)", f"{current_price:,.2f}" if isinstance(current_price, (int, float)) else "N/A")
                cols[1].metric("P/E", f"{info.get('trailingPE'):.2f}" if info.get('trailingPE') else "N/A")
                cols[2].metric("P/B", f"{info.get('priceToBook'):.2f}" if info.get('priceToBook') else "N/A")
                cols[3].metric("Debt/Equity", f"{info.get('debtToEquity'):.2f}" if info.get('debtToEquity') else "N/A")

                st.markdown("---")
                st.header("Fair Value Estimation Models")

                # ----- DCF -----
                with st.expander("Discounted Cash Flow (DCF) Model", expanded=True):
                    fcf = cashflow.loc['Free Cash Flow'].iloc[0] if (isinstance(cashflow, pd.DataFrame) and 'Free Cash Flow' in cashflow.index and not cashflow.empty) else None
                    shares = info.get('sharesOutstanding')

                    historical_cagr = calculate_cagr(financials)
                    default_growth_rate = historical_cagr if historical_cagr is not None else 0.05

                    c1, c2, c3 = st.columns(3)
                    g_rate = c1.slider("5-Year Growth Rate (g)", -0.10, 0.50, float(default_growth_rate or 0.05), 0.01, key='dcf_g')
                    t_rate = c2.slider("Perpetual Growth Rate", 0.00, 0.10, 0.02, 0.005, key='dcf_t')
                    d_rate = c3.slider("Discount Rate (WACC)", 0.05, 0.20, 0.08, 0.005, key='dcf_d')

                    dcf_value = calculate_dcf_fair_value(fcf, g_rate, t_rate, d_rate, shares)
                    st.metric("DCF Fair Value (₹ / share)", f"{dcf_value:,.2f}" if isinstance(dcf_value, (int, float)) else dcf_value)

                # ----- Graham -----
                with st.expander("Graham Number"):
                    eps = info.get('trailingEps')
                    bvps = info.get('bookValue')
                    graham_value = calculate_graham_number(eps, bvps)
                    st.metric("Graham Number (₹ / share)", f"{graham_value:,.2f}" if isinstance(graham_value, (int, float)) else graham_value)

                # ----- DDM -----
                with st.expander("Dividend Discount Model (DDM)"):
                    dividend = info.get('dividendRate')
                    c4, c5 = st.columns(2)
                    req_rate = c4.slider("Required Rate of Return", 0.05, 0.20, 0.08, 0.005, key='ddm_req')
                    div_growth = c5.slider("Dividend Growth Rate", 0.00, 0.10, 0.02, 0.005, key='ddm_growth')
                    ddm_value = calculate_ddm_fair_value(dividend, req_rate, div_growth)
                    st.metric("DDM Fair Value (₹ / share)", f"{ddm_value:,.2f}" if isinstance(ddm_value, (int, float)) else ddm_value)

                # ----- Final Signal -----
                st.markdown("---")
                st.header("Investment Signal")

                valid_vals = [v for v in [dcf_value, graham_value, ddm_value] if isinstance(v, (int, float)) and v > 0]
                if valid_vals and isinstance(current_price, (int, float)):
                    avg_fair_value = float(np.mean(valid_vals))
                    upside = ((avg_fair_value - current_price) / current_price) * 100

                    if upside > 20:
                        signal, color = "Strong Buy", "success"
                    elif 10 < upside <= 20:
                        signal, color = "Buy", "success"
                    elif -10 <= upside <= 10:
                        signal, color = "Hold", "warning"
                    else:
                        signal, color = "Sell", "error"

                    getattr(st, color)(f"**Signal: {signal}**")
                    st.write(f"Average fair value: **₹{avg_fair_value:,.2f}** | Upside: **{upside:.2f}%**")
                    st.write({"DCF": dcf_value, "Graham": graham_value, "DDM": ddm_value})
                else:
                    st.info("Not enough valid valuation data to generate a signal.")

                # ----- Financial statements -----
                st.markdown("---")
                st.header("Financial Statements")
                with st.expander("Income Statement (Annual)"):
                    if isinstance(financials, pd.DataFrame) and not financials.empty:
                        st.dataframe(financials.style.format('₹{:,.0f}'))
                    else:
                        st.write("N/A")
                with st.expander("Balance Sheet (Annual)"):
                    bs = stock_data['balance_sheet']
                    if isinstance(bs, pd.DataFrame) and not bs.empty:
                        st.dataframe(bs.style.format('₹{:,.0f}'))
                    else:
                        st.write("N/A")
                with st.expander("Cash Flow Statement (Annual)"):
                    cf = stock_data['cashflow']
                    if isinstance(cf, pd.DataFrame) and not cf.empty:
                        st.dataframe(cf.style.format('₹{:,.0f}'))
                    else:
                        st.write("N/A")
