# pages/2_Unified_Backtester.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import os
from datetime import date, timedelta
import yfinance as yf
import re
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import json

# --- Import Project Modules ---
from config import MODEL_CHOICES, MODEL_TUNING_GRIDS, TECHNICAL_INDICATORS_DEFAULTS, FUNDAMENTAL_METRICS, GLOBAL_MARKET_TICKERS
from data import fetch_stock_data, add_technical_indicators, add_macro_indicators, add_fundamental_indicators
from utils import plot_predictions, calculate_metrics, create_sequences, display_log
from model import build_lstm_model, build_transformer_model, build_hybrid_model, train_model, run_hyperparameter_tuning, KERAS_TUNER_AVAILABLE

# --- Page Configuration ---
st.set_page_config(layout="wide")
st.title("🎯 Unified Backtester & Predictor")
st.info("This backtester automatically adapts to EOD (daily) or Intraday data sent from the scanner pages.")

# --- Paths for saving tuner models ---
BEST_MODELS_DIR = "best_models"
os.makedirs(BEST_MODELS_DIR, exist_ok=True)

# --- Mode Detection ---
if 'tickers_for_backtest_intra' in st.session_state and st.session_state['tickers_for_backtest_intra']:
    BACKTEST_MODE = "Intraday"
    tickers_to_run = st.session_state.get('tickers_for_backtest_intra', [])
    synced_params = st.session_state.get('scanner_params_intra', {})
else:
    BACKTEST_MODE = "Daily"
    tickers_to_run = st.session_state.get('tickers_for_backtest', [])
    synced_params = st.session_state.get('scanner_params', {})

st.sidebar.header(f"Mode: {BACKTEST_MODE}")

# --- Data Fetching ---
def fetch_data_for_backtest(ticker, mode, params, start_dt=None, end_dt=None):
    if mode == "Intraday":
        try:
            days_lookback = params.get('days', 15) + 10
            period = f"{min(days_lookback, 60)}d"
            interval = params.get('interval', '15m')
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
            if df.empty: return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1) if 'Ticker' in df.columns.names else [col[0] for col in df.columns]
            return df
        except Exception:
            return pd.DataFrame()
    else: # Daily mode
        return fetch_stock_data(ticker, start_dt, end_dt)

# --- Unified Backtesting Functions ---

def run_classical_ml_backtest(ticker, model_name, full_data, backtest_period, mode, params, feature_cfg):
    st.markdown(f"##### Running Model: **{model_name}**")
    data = full_data.copy()
    
    data = add_technical_indicators(
        df=data, selected_indicators=feature_cfg['tech'],
        rsi_window=params.get('rsi_period', 14), macd_fast=params.get('macd_fast', 12),
        macd_slow=params.get('macd_slow', 26), macd_signal=params.get('macd_signal', 9),
        atr_window=params.get('atr_period', 14), adx_window=params.get('adx_period', 14)
    )
    
    if mode == "Daily":
        if feature_cfg['macro']: data = add_macro_indicators(data, feature_cfg['macro'])
        if feature_cfg['fund']: data = add_fundamental_indicators(data, ticker, feature_cfg['fund'])
    
    data['target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    data.columns = [re.sub(r'[^A-Za-z0-9_]+', '', str(col)) for col in data.columns]

    if len(data) < backtest_period + 30:
        st.warning(f"Not enough data for {ticker} after feature engineering ({len(data)} rows). Skipping.")
        return None

    features = [col for col in data.columns if col not in ['target']]
    X, y = data[features], data['target']
    X_train, X_test = X.iloc[:-backtest_period], X.iloc[-backtest_period:]
    y_train, y_test = y.iloc[:-backtest_period], y.iloc[-backtest_period:]

    models = {
        'Random Forest': RandomForestRegressor(random_state=42), 'XGBoost': xgb.XGBRegressor(random_state=42),
        'LightGBM': lgb.LGBMRegressor(random_state=42), 'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'CatBoost': cb.CatBoostRegressor(random_state=42, verbose=0),
    }
    model = models.get(model_name, RandomForestRegressor())
    
    best_params_str = "Defaults"
    with st.spinner(f"Training {model_name} for {ticker}..."):
        if feature_cfg.get('enable_tuning', False) and model_name in MODEL_TUNING_GRIDS and mode == "Daily":
            tscv = TimeSeriesSplit(n_splits=3)
            random_search = RandomizedSearchCV(estimator=model, param_distributions=MODEL_TUNING_GRIDS[model_name], n_iter=feature_cfg['n_iter'], cv=tscv, scoring='neg_mean_absolute_error', n_jobs=1, random_state=42)
            random_search.fit(X_train, y_train)
            model = random_search.best_estimator_
            best_params_str = str(random_search.best_params_)
        else:
            model.fit(X_train, y_train)
            
        predictions = model.predict(X_test)
        # --- FIX: Always calculate next prediction ---
        next_prediction = model.predict(X.iloc[[-1]])[0]

    rmse, mae, r2 = calculate_metrics(y_test, predictions)
    plot_title = f"{ticker} ({model_name} - {params.get('interval', 'Daily')})"
    fig = plot_predictions(actual_prices=full_data['Close'], predicted_prices=pd.Series(predictions, index=y_test.index), ticker=plot_title)
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Figure': fig, 'Next Prediction': next_prediction, 'Best Params': best_params_str}


def run_nn_backtest(ticker, model_type, full_data, backtest_period, mode, params, feature_cfg, nn_params):
    st.markdown(f"##### Running Model: **{model_type}**")
    from sklearn.preprocessing import MinMaxScaler
    data = full_data.copy()
    
    data = add_technical_indicators(
        df=data, selected_indicators=feature_cfg['tech'],
        rsi_window=params.get('rsi_period', 14), macd_fast=params.get('macd_fast', 12),
        macd_slow=params.get('macd_slow', 26), macd_signal=params.get('macd_signal', 9),
        atr_window=params.get('atr_period', 14), adx_window=params.get('adx_period', 14)
    )
    
    if mode == "Daily" and feature_cfg['macro']:
        data = add_macro_indicators(data, feature_cfg['macro'])

    data.dropna(inplace=True)
    data.columns = [re.sub(r'[^A-Za-z0-9_]+', '', str(col)) for col in data.columns]
    
    if len(data) < nn_params['time_steps'] + backtest_period:
        st.warning(f"Not enough data for {ticker} ({len(data)} rows) for NN backtest. Skipping.")
        return None

    train_data = data.iloc[:-backtest_period]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    
    scaled_data = scaler.transform(data)
    close_col_index = data.columns.get_loc('Close')
    
    X, y = create_sequences(scaled_data, close_col_index, nn_params['time_steps'], 1)
    
    if X.size == 0:
        st.warning(f"Not enough data to form sequences for {ticker}. Skipping NN model."); return None
        
    train_end_idx = len(X) - backtest_period
    X_train, X_test = X[:train_end_idx], X[train_end_idx:]
    y_train, y_test = y[:train_end_idx], y[train_end_idx:]

    K.clear_session()
    input_shape = (X_train.shape[1], X_train.shape[2])
    model_builder = {'LSTM': build_lstm_model, 'Transformer': build_transformer_model, 'Hybrid (LSTM + Transformer)': build_hybrid_model}
    
    best_params_str = "Manual"
    build_params = nn_params.get('manual_arch', {})
    model = None
    
    if nn_params.get('hp_mode') == 'Automatic (Tuner)' and KERAS_TUNER_AVAILABLE:
        display_log(f"🚀 Starting NN Hyperparameter Tuning for {ticker}...", "info")
        tuned_model, best_hps, _, _ = run_hyperparameter_tuning(
            model_type=model_type, input_shape=input_shape, output_dim=1,
            X_train=X_train, y_train=y_train,
            num_trials=nn_params.get('num_trials', 10),
            executions_per_trial=nn_params.get('executions_per_trial', 1),
            best_models_dir=BEST_MODELS_DIR,
            force_retune=nn_params.get('force_retune', False),
            epochs=nn_params.get('tuner_epochs', 20)
        )
        if tuned_model and best_hps:
            model = tuned_model
            best_params_str = json.dumps(best_hps.values)
            display_log(f"✅ Tuner finished. Using best HPs.", "info")
        else:
            display_log("⚠️ Tuner failed to find a model. Reverting to manual.", "warning")
            
    if model is None:
        model = model_builder[model_type](input_shape, 1, manual_params=build_params, learning_rate=nn_params['lr'])
    
    with st.spinner(f"Training {model_type} for {ticker}..."):
        train_model(model, X_train, y_train, epochs=nn_params['epochs'], batch_size=nn_params['batch_size'], learning_rate=nn_params['lr'], X_val=X_test, y_val=y_test)
        predicted_scaled = model.predict(X_test).flatten()
        
        # --- FIX: Always calculate next prediction ---
        last_sequence = np.expand_dims(scaled_data[-nn_params['time_steps']:], axis=0)
        next_pred_scaled = model.predict(last_sequence).flatten()[0]

    def inverse_transform_prices(scaled_prices):
        # Ensure input is an array
        prices = np.asarray(scaled_prices)
        if prices.ndim == 0: # Handle single scalar value
            prices = prices.flatten()
        dummy_array = np.zeros((len(prices), scaler.n_features_in_))
        dummy_array[:, close_col_index] = prices
        return scaler.inverse_transform(dummy_array)[:, close_col_index]

    predicted_prices = inverse_transform_prices(predicted_scaled)
    next_prediction = inverse_transform_prices(np.array([next_pred_scaled]))[0]
    actual_prices = inverse_transform_prices(y_test.flatten())
    
    rmse, mae, r2 = calculate_metrics(actual_prices, predicted_prices)
    plot_title = f"{ticker} ({model_type} - {params.get('interval', 'Daily')})"
    fig = plot_predictions(actual_prices=full_data['Close'], predicted_prices=pd.Series(predicted_prices, index=data.index[-backtest_period:]), ticker=plot_title)
    
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Figure': fig, 'Next Prediction': next_prediction, 'Best Params': best_params_str}


# --- Sidebar UI Configuration (no changes needed here) ---
st.sidebar.subheader("Backtest Parameters")
if BACKTEST_MODE == "Daily":
    start_date = st.sidebar.date_input("Start Date for History", value=date.today() - timedelta(days=5*365))
    end_date = st.sidebar.date_input("End Date for History", value=date.today())
    backtest_period = st.sidebar.slider("Days to Backtest", 30, 500, 90, 10, key="unified_backtest_days")
else:
    start_date, end_date = None, None
    backtest_period = st.sidebar.slider("Candles to Backtest", 10, 500, 60, 10, key="unified_backtest_candles")

st.sidebar.subheader("Model Selection")
classical_models = [m for m in MODEL_CHOICES if m in ['LightGBM', 'XGBoost', 'Random Forest', 'Gradient Boosting', 'CatBoost']]
nn_models = ['LSTM', 'Transformer', 'Hybrid (LSTM + Transformer)']
selected_models = st.sidebar.multiselect("Choose Model(s) to Run", options=classical_models + nn_models, default=['LightGBM', 'LSTM'])

feature_config = {}
with st.sidebar.expander("Feature Selection"):
    if BACKTEST_MODE == "Intraday":
        st.info("Intraday mode uses a core set of scanner-synced technical indicators.")
        feature_config['tech'] = ['RSI_WINDOW', 'MACD_SHORT_WINDOW', 'ATR_WINDOW', 'ADX_WINDOW', 'OBV', 'MA_WINDOWS']
        feature_config['fund'], feature_config['macro'] = [], []
    else: # Daily
        st.write("**Technical Indicators**"); feature_config['tech'] = [ind for ind, (_, en) in TECHNICAL_INDICATORS_DEFAULTS.items() if st.checkbox(ind, en, key=f"unified_tech_{ind}")]
        st.write("**Fundamental Metrics**"); feature_config['fund'] = [name for name, _ in FUNDAMENTAL_METRICS.items() if st.checkbox(name, False, key=f"unified_fund_{name}")]
        st.write("**Macroeconomic Indicators**"); feature_config['macro'] = [name for name, _ in GLOBAL_MARKET_TICKERS.items() if st.checkbox(name, False, key=f"unified_macro_{name}")]

if BACKTEST_MODE == "Daily":
    st.sidebar.subheader("Hyperparameter Tuning (Classic ML)")
    feature_config['enable_tuning'] = st.sidebar.checkbox("Enable RandomizedSearchCV Tuning", value=False)
    if feature_config['enable_tuning']:
        feature_config['n_iter'] = st.sidebar.slider("Tuning Iterations", 5, 100, 20, 5)

nn_config = {}
st.sidebar.subheader("Neural Network Parameters")
nn_config['time_steps'] = st.sidebar.slider("Time steps (look-back)", 5, 120, 40, key="nn_ts_unified")
nn_config['hp_mode'] = st.sidebar.radio("NN Hyperparameter Mode", ["Manual", "Automatic (Tuner)"], index=0, key="nn_hp_mode_unified")

if nn_config['hp_mode'] == 'Automatic (Tuner)':
    st.sidebar.warning("⚠️ Automatic tuning is selected.")
    nn_config['tuner_epochs'] = st.sidebar.slider("Max Epochs for Tuning", 10, 100, 30, 10, key="nn_tuner_epochs")
    nn_config['num_trials'] = st.sidebar.slider("Number of Tuning Trials", 5, 50, 10, 5, key="nn_tuner_trials")
    nn_config['executions_per_trial'] = st.sidebar.number_input("Executions per Trial", 1, 3, 1, key="nn_tuner_exec")
    nn_config['force_retune'] = st.sidebar.checkbox("Force Retune (ignore saved results)", value=False, key="nn_tuner_force")
    nn_config['epochs'], nn_config['batch_size'], nn_config['lr'] = 50, 32, 0.001
    nn_config['manual_arch'] = {}
else: # Manual Mode
    st.sidebar.markdown("**Manual NN Training Parameters**")
    nn_config['epochs'] = st.sidebar.number_input("Epochs", 1, 200, 30, key="nn_epochs_unified")
    nn_config['batch_size'] = st.sidebar.number_input("Batch Size", 1, 256, 16, key="nn_batch_size_unified")
    nn_config['lr'] = st.sidebar.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f", key="nn_lr_unified")
    with st.sidebar.expander("NN Model Architecture (Manual)", expanded=True):
        nn_config['manual_arch'] = {}
        first_nn_model = next((m for m in selected_models if m in nn_models), None)

        if not first_nn_model:
            st.info("Select a Neural Network model (e.g., LSTM) to see its architecture options.")
        elif first_nn_model == 'LSTM':
            st.markdown("**LSTM Architecture**"); nn_config['manual_arch']['num_lstm_layers'] = st.slider("LSTM Layers", 1, 3, 2, key="nn_lstm_layers_unified")
            for i in range(nn_config['manual_arch']['num_lstm_layers']):
                nn_config['manual_arch'][f'lstm_units_{i+1}'] = st.slider(f"Layer {i+1} Units", 32, 256, 100, 16, key=f"nn_lstm_units_{i+1}_unified")
                nn_config['manual_arch'][f'dropout_{i+1}'] = st.slider(f"Layer {i+1} Dropout", 0.0, 0.5, 0.2, 0.05, key=f"nn_lstm_dropout_{i+1}_unified")
        elif first_nn_model == 'Transformer':
            st.markdown("**Transformer Architecture**"); nn_config['manual_arch']['num_transformer_blocks'] = st.slider("Transformer Blocks", 1, 4, 2, key="nn_trans_blocks_unified")
            nn_config['manual_arch']['num_heads'] = st.slider("Attention Heads", 1, 8, 4, key="nn_trans_heads_unified")
            nn_config['manual_arch']['ff_dim'] = st.slider("Feed Forward Dim", 16, 128, 32, 16, key="nn_trans_ff_dim_unified")
        elif first_nn_model == 'Hybrid (LSTM + Transformer)':
            st.markdown("**Hybrid Architecture**"); nn_config['manual_arch']['lstm_units'] = st.slider("LSTM Units", 32, 256, 64, 32, key="nn_hybrid_lstm_units_unified")
            nn_config['manual_arch']['lstm_dropout'] = st.slider("LSTM Dropout", 0.0, 0.5, 0.2, 0.05, key="nn_hybrid_lstm_dropout_unified")
            nn_config['manual_arch']['num_transformer_blocks'] = st.slider("Transformer Blocks", 1, 4, 1, key="nn_hybrid_trans_blocks_unified")
            nn_config['manual_arch']['num_heads'] = st.slider("Attention Heads", 1, 8, 2, key="nn_hybrid_trans_heads_unified")

# --- Main Application Logic ---
if not tickers_to_run or not synced_params:
    st.warning("No tickers/parameters found. Please run a Scanner and send tickers to this page.")
else:
    st.success(f"**{len(tickers_to_run)}** tickers received for **{BACKTEST_MODE}** backtest.")
    st.write(f"Tickers: `{', '.join(tickers_to_run)}`")

    if not selected_models:
        st.warning("Please select at least one model from the sidebar.")
    elif st.button(f"Run {BACKTEST_MODE} Backtest", type="primary"):
        all_results = []
        for ticker in tickers_to_run:
            st.markdown(f"### Processing: **{ticker}**")
            data = fetch_data_for_backtest(ticker, BACKTEST_MODE, synced_params, start_date, end_date)
            if data.empty:
                st.error(f"Could not fetch data for {ticker}."); continue

            for model_name in selected_models:
                result = None
                try:
                    if model_name in classical_models:
                        result = run_classical_ml_backtest(ticker, model_name, data, backtest_period, BACKTEST_MODE, synced_params, feature_config)
                    elif model_name in nn_models:
                        result = run_nn_backtest(ticker, model_name, data, backtest_period, BACKTEST_MODE, synced_params, feature_config, nn_config)
                    
                    if result:
                        all_results.append({'Ticker': ticker, 'Model': model_name, **result})
                        
                        # --- FIX: Display metrics for both modes ---
                        c1, c2, c3, c4 = st.columns(4)
                        if BACKTEST_MODE == "Intraday":
                            c1.metric("MAE", f"{result['MAE']:.4f}")
                            c2.metric("RMSE", f"{result['RMSE']:.4f}")
                            c3.metric("R²", f"{result['R2']:.2f}")
                            if result['Next Prediction'] is not None:
                                c4.metric("Next Price", f"{result['Next Prediction']:.2f}")
                        else: # Daily
                            c1.metric("MAE", f"${result['MAE']:.2f}")
                            c2.metric("RMSE", f"${result['RMSE']:.2f}")
                            c3.metric("R²", f"{result['R2']:.2f}")
                            if result['Next Prediction'] is not None:
                                c4.metric("Next Day Price", f"${result['Next Prediction']:.2f}")

                        st.plotly_chart(result['Figure'], use_container_width=True)
                        st.markdown("---")

                except Exception as e:
                    st.error(f"An error occurred while backtesting {ticker} with {model_name}: {e}"); st.exception(e)

        if all_results:
            st.markdown("---"); st.header("🏆 Compiled Backtest Summary")
            summary_df = pd.DataFrame([{k: v for k, v in res.items() if k != 'Figure'} for res in all_results])
            cols_to_display = ['Ticker', 'Model', 'MAE', 'RMSE', 'R2', 'Next Prediction', 'Best Params']
            formatters = {'MAE': '{:.4f}', 'RMSE': '{:.4f}', 'R2': '{:.2f}', 'Next Prediction': '{:.2f}'}
            
            final_cols = [col for col in cols_to_display if col in summary_df.columns]
            st.dataframe(summary_df[final_cols].style.format(formatters), use_container_width=True)