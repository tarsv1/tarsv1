# valuation_utils.py
import numpy as np
import pandas as pd

# --------- Generic helpers ---------
def normalize_nse_ticker(t: str) -> str:
    """Normalize user input to UPPER + ensure '.NS' for NSE tickers."""
    t = (t or "").strip().upper()
    if not t:
        return t
    if not t.endswith(".NS"):
        if "." not in t:
            t = t + ".NS"
        else:
            base, dot, suf = t.partition(".")
            if suf.lower() == "ns":
                t = base + ".NS"
    return t

def first_existing_row(df: pd.DataFrame, candidates):
    """Return a Series for the first matching row name in candidates; else None."""
    if df is None or df.empty:
        return None
    for k in candidates:
        if k in df.index:
            s = df.loc[k]
            s = s.squeeze() if isinstance(s, (pd.Series, pd.DataFrame)) else s
            return pd.to_numeric(s, errors="coerce")
    return None

def cagr_from_series(series: pd.Series) -> float:
    """
    CAGR using yfinance rows that are in reverse-chronological order (most recent first across columns).
    CAGR = (latest / oldest)^(1/n) - 1
    """
    if series is None:
        return 0.0
    s = pd.to_numeric(series.dropna(), errors="coerce")
    if len(s) < 2:
        return 0.0
    latest = s.iloc[0]   # most recent
    oldest = s.iloc[-1]  # oldest
    if oldest <= 0 or latest <= 0:
        return 0.0
    n = len(s) - 1
    return (latest / oldest) ** (1 / n) - 1

def safe_ratio(num: pd.Series, den: pd.Series):
    """Elementwise ratio with alignment and protection against zeros/NaNs."""
    if num is None or den is None:
        return None
    num, den = num.align(den, join="inner")
    den = den.replace(0, np.nan)
    r = (num / den).replace([np.inf, -np.inf], np.nan).dropna()
    return r if not r.empty else None

# --------- Valuation models ---------
def calculate_cagr(financials) -> float:
    """Revenue CAGR (uses multiple label variants)."""
    try:
        revenue = first_existing_row(
            financials,
            ["Total Revenue", "TotalRevenue", "Revenue", "Operating Revenue", "Total Operating Revenue"]
        )
        return cagr_from_series(revenue)
    except Exception:
        return 0.0

def calculate_dcf_fair_value(fcf, growth_rate, terminal_growth_rate, discount_rate, shares_outstanding):
    """
    Simple 5-year DCF model. Returns 'N/A' if inputs invalid.
    - Ignores if FCF <= 0 or shares missing.
    """
    if not isinstance(fcf, (int, float)) or not isinstance(shares_outstanding, (int, float)):
        return "N/A"
    if fcf <= 0 or shares_outstanding <= 0:
        return "N/A"
    if discount_rate <= terminal_growth_rate:
        return "N/A"
    try:
        future_fcf = [fcf * (1 + growth_rate) ** i for i in range(1, 6)]
        terminal_value = (future_fcf[-1] * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
        discounted_fcf = [fcf_val / (1 + discount_rate) ** (i + 1) for i, fcf_val in enumerate(future_fcf)]
        discounted_terminal_value = terminal_value / (1 + discount_rate) ** 5
        total_intrinsic_value = sum(discounted_fcf) + discounted_terminal_value
        return total_intrinsic_value / shares_outstanding
    except Exception:
        return "N/A"

def calculate_graham_number(eps, book_value_per_share):
    """Graham number; uses EPS and Book Value per Share (both must be >0)."""
    if isinstance(eps, (int, float)) and isinstance(book_value_per_share, (int, float)) and eps > 0 and book_value_per_share > 0:
        return float(np.sqrt(22.5 * eps * book_value_per_share))
    return "N/A"

def calculate_ddm_fair_value(last_dividend, required_rate, dividend_growth_rate):
    """Gordon Growth DDM; returns 'N/A' for missing dividend."""
    if not isinstance(last_dividend, (int, float)) or last_dividend <= 0:
        return "N/A"
    if required_rate <= dividend_growth_rate:
        return "Growth > Rate"
    return last_dividend / (required_rate - dividend_growth_rate)
