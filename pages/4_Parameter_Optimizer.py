# pages/4_Parameter_Optimizer.py

import streamlit as st
import pandas as pd
import numpy as np
import optuna
from datetime import date, timedelta

# Import all necessary functions from your data.py
from data import fetch_stock_data, sma, rsi, macd_lines, _di_lines, obv, detect_divergences

st.set_page_config(layout="wide")
st.title("⚙️ Ultimate Strategy Optimizer")
st.warning("⚠️ This optimizer is computationally intensive. Start with fewer trials (100-200) and a shorter date range (1-2 years).")

# --- UI for selecting stock and optimization settings ---
st.sidebar.header("Optimizer Configuration")
ticker = st.sidebar.text_input("Enter Ticker Symbol", "A2ZINFRA.NS")
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=2*365))
end_date = st.sidebar.date_input("End Date", value=date.today())
n_trials = st.sidebar.slider("Number of Optimization Trials", 50, 2000, 200, 50)

# --- The Ultimate Strategy Backtest Function ---
def run_ultimate_strategy_backtest(df, params):
    """
    Runs a backtest by calculating a daily score based on the scanner's full logic.
    """
    df_strat = df.copy()

    # --- 1. Calculate all indicators with trial parameters ---
    df_strat['fast_ma'] = sma(df_strat['Close'], period=params['ma_fast'])
    df_strat['slow_ma'] = sma(df_strat['Close'], period=params['ma_slow'])
    df_strat['rsi'] = rsi(df_strat['Close'], period=params['rsi_period'])
    macd_line, macd_sig, _ = macd_lines(df_strat['Close'], params['macd_fast'], params['macd_slow'], params['macd_signal'])
    df_strat['macd_line'] = macd_line
    df_strat['macd_sig'] = macd_sig
    plus_di, minus_di, adx = _di_lines(df_strat, period=params['adx_period'])
    df_strat['plus_di'], df_strat['minus_di'], df_strat['adx'] = plus_di, minus_di, adx
    
    # Divergence detection (computationally expensive)
    bullish_div, bearish_div = detect_divergences(df_strat, df_strat['rsi'], df_strat['macd_line'])
    
    df_strat.dropna(inplace=True)
    if df_strat.empty: return -100

    # --- 2. Calculate a daily score based on scanner logic ---
    score = pd.Series(0, index=df_strat.index)
    
    # Trend Signals
    score += np.where(df_strat['fast_ma'] > df_strat['slow_ma'], 1, -1)
    score += np.where((df_strat['adx'] > params['adx_threshold']) & (df_strat['plus_di'] > df_strat['minus_di']), 1, 0)
    score += np.where((df_strat['adx'] > params['adx_threshold']) & (df_strat['minus_di'] > df_strat['plus_di']), -1, 0)
    
    # Momentum Signals
    score += np.where(df_strat['rsi'] < params['rsi_low'], 1, 0)
    score += np.where(df_strat['rsi'] > params['rsi_high'], -1, 0)
    score += np.where(df_strat['macd_line'] > df_strat['macd_sig'], 1, -1)
    
    # Divergence Signals
    score += np.where(bullish_div, 2, 0)  # Divergences are strong signals, give them more weight
    score += np.where(bearish_div, -2, 0)

    df_strat['score'] = score
    
    # --- 3. Generate trades based on score thresholds ---
    df_strat['signal'] = 0
    buy_cond = (df_strat['score'] > params['buy_score_threshold']) & (df_strat['score'].shift(1) <= params['buy_score_threshold'])
    sell_cond = (df_strat['score'] < params['sell_score_threshold']) & (df_strat['score'].shift(1) >= params['sell_score_threshold'])
    df_strat.loc[buy_cond, 'signal'] = 1
    df_strat.loc[sell_cond, 'signal'] = -1

    df_strat['position'] = df_strat['signal'].replace(to_replace=-1, value=0).ffill()
    df_strat['daily_return'] = df_strat['Close'].pct_change()
    df_strat['strategy_return'] = df_strat['daily_return'] * df_strat['position'].shift(1)

    cumulative_return = (1 + df_strat['strategy_return']).cumprod()
    total_return_pct = (cumulative_return.iloc[-1] - 1) * 100

    return total_return_pct if not np.isnan(total_return_pct) else -100

# --- The Optuna Objective Function ---
def objective(trial, df):
    params = {
        'ma_fast': trial.suggest_int('ma_fast', 5, 30, step=1),
        'ma_slow': trial.suggest_int('ma_slow', 35, 100, step=5),
        'rsi_period': trial.suggest_int('rsi_period', 5, 25, step=1),
        'rsi_low': trial.suggest_int('rsi_low', 20, 40, step=2),
        'rsi_high': trial.suggest_int('rsi_high', 60, 80, step=2),
        'macd_fast': trial.suggest_int('macd_fast', 5, 20, step=1),
        'macd_slow': trial.suggest_int('macd_slow', 21, 50, step=1),
        'macd_signal': trial.suggest_int('macd_signal', 5, 15, step=1),
        'adx_period': trial.suggest_int('adx_period', 5, 25, step=1),
        'adx_threshold': trial.suggest_int('adx_threshold', 18, 35, step=1),
        # Strategy thresholds
        'buy_score_threshold': trial.suggest_int('buy_score_threshold', 1, 5, step=1),
        'sell_score_threshold': trial.suggest_int('sell_score_threshold', -5, -1, step=1)
    }
    # Ensure MACD fast < slow
    if params['macd_fast'] >= params['macd_slow']:
        return -100 # Penalize invalid combination

    return run_ultimate_strategy_backtest(df, params)

# --- Main App Logic ---
if st.button(f"🚀 Optimize ULTIMATE Strategy for {ticker}", type="primary"):
    with st.spinner(f"Running {n_trials} trials... This will take a long time. Please be patient."):
        data = fetch_stock_data(ticker, start_date, end_date)
        if data.empty or len(data) < 250:
            st.error("Not enough data. Please select a longer date range or a different stock.")
        else:
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: objective(trial, data), n_trials=n_trials, n_jobs=-1)
            st.session_state['optuna_study'] = study
            st.session_state['last_optimized_ticker'] = ticker
            st.rerun()

# --- Display Results ---
if 'optuna_study' in st.session_state:
    st.markdown("---")
    st.header("Optimization Results")
    study = st.session_state['optuna_study']
    ticker_name = st.session_state['last_optimized_ticker']

    st.subheader(f"🏆 Best Parameter Combination for {ticker_name}")
    best_params_df = pd.DataFrame.from_dict(study.best_params, orient='index', columns=['Optimized Value'])
    st.dataframe(best_params_df, use_container_width=True)

    st.subheader("📈 Best Strategy Return")
    st.metric("Total Return Over Period", f"{study.best_value:.2f}%")

    st.subheader("All Trials Data")
    st.info("You can sort this table to see how different parameter values affected the return.")
    trials_df = study.trials_dataframe()
    st.dataframe(trials_df, use_container_width=True)