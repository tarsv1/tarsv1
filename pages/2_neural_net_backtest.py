# pages/2_neural_net_backtest.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import tensorflow as tf
from tensorflow.keras import backend as K
import os

# Assuming these files are correctly set up from previous steps
from config import TECHNICAL_INDICATORS_DEFAULTS, GLOBAL_MARKET_TICKERS
from data import fetch_stock_data, add_technical_indicators, add_macro_indicators
from utils import preprocess_data, create_sequences, plot_predictions, calculate_metrics
from model import (
    build_lstm_model, build_transformer_model, build_hybrid_model, train_model,
    run_hyperparameter_tuning, KERAS_TUNER_AVAILABLE
)
from signal_features import add_signal_features

st.title("Neural Network Backtester")
st.info("This page backtests tickers from the scanner using Deep Learning models.")

# --- Path for saving tuner results ---
BEST_MODELS_DIR = "best_models_backtest"
if not os.path.exists(BEST_MODELS_DIR):
    os.makedirs(BEST_MODELS_DIR)

# --- Sidebar Configuration ---
st.sidebar.header("Backtest Configuration")
start_date = st.sidebar.date_input("Start Date for History", value=date.today() - timedelta(days=5*365))
end_date = st.sidebar.date_input("End Date for History", value=date.today())

st.sidebar.subheader("Model Parameters")
training_window_size = st.sidebar.slider("Time steps (look-back days)", 5, 120, 60, key="nn_ts")
backtest_days = st.sidebar.slider("Days to Backtest", 30, 500, 90, 10)
model_type = st.sidebar.radio("Select Model Type", ('LSTM', 'Transformer', 'Hybrid (LSTM + Transformer)'), index=0, key="nn_model")

with st.sidebar.expander("Feature Selection"):
    st.write("**Technical Indicators**")
    selected_indicators = [
        indicator for indicator, (_, enabled) in TECHNICAL_INDICATORS_DEFAULTS.items()
        if st.checkbox(indicator, value=enabled, key=f"tech_{indicator}")
    ]
    st.write("**Macroeconomic Indicators**")
    selected_macro = [name for name, _ in GLOBAL_MARKET_TICKERS.items() if st.checkbox(name, value=False, key=f"macro_{name}")]

st.sidebar.subheader("Advanced Features")
enable_signal_features = st.sidebar.checkbox("Enable Scanner Signal Features", value=True, help="Create binary features based on scanner rules.")

st.sidebar.subheader("Hyperparameter Mode")
if not KERAS_TUNER_AVAILABLE:
    st.sidebar.error("Keras Tuner not installed. Falling back to Manual mode. Please run: pip install keras-tuner")
    hp_mode = "Manual"
else:
    hp_mode = st.sidebar.radio("Select Mode", ["Manual", "Automatic (Tuner)"])

force_retune = st.sidebar.checkbox("Force Hyperparameter Retune", value=False, help="If unchecked, will load saved tuning results if they exist.")

manual_params = {}
if hp_mode == "Automatic (Tuner)":
    st.sidebar.warning("⚠️ Automatic tuning is very time-consuming and will be run for each ticker.")
    epochs = st.sidebar.slider("Max Epochs for Tuning", 10, 100, 30, 10, key="hp_epochs")
    num_trials = st.sidebar.slider("Number of Tuning Trials", 5, 50, 10, 5, key="hp_trials")
    executions_per_trial = st.sidebar.number_input("Executions per Trial", 1, 3, 1, key="hp_exec")
    # Set default values, which will be overwritten by the tuner
    batch_size = 32
    learning_rate = 0.001
else: # Manual Mode
    st.sidebar.subheader("Manual Hyperparameters")
    epochs = st.sidebar.number_input("Epochs", 1, 200, 50, key="nn_epochs")
    batch_size = st.sidebar.number_input("Batch Size", 1, 256, 32, key="nn_batch_size")
    learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f", key="nn_lr")
    
    # --- THIS IS THE NEWLY ADDED SECTION ---
    st.sidebar.markdown("---")
    st.sidebar.write("**Model Architecture**")
    if model_type == 'LSTM':
        manual_params['num_lstm_layers'] = st.sidebar.slider("LSTM Layers", 1, 3, 2, key="nn_lstm_layers")
        for i in range(manual_params['num_lstm_layers']):
            manual_params[f'lstm_units_{i+1}'] = st.sidebar.slider(f"LSTM Layer {i+1} Units", 32, 256, 100, 16, key=f"nn_lstm_units_{i+1}")
            manual_params[f'dropout_{i+1}'] = st.sidebar.slider(f"LSTM Layer {i+1} Dropout", 0.0, 0.5, 0.2, 0.05, key=f"nn_lstm_dropout_{i+1}")
    elif model_type == 'Transformer':
        manual_params['num_transformer_blocks'] = st.sidebar.slider("Transformer Blocks", 1, 4, 2, key="nn_trans_blocks")
        manual_params['num_heads'] = st.sidebar.slider("Attention Heads", 1, 8, 4, key="nn_trans_heads")
        manual_params['ff_dim'] = st.sidebar.slider("Feed Forward Dim", 16, 128, 32, 16, key="nn_trans_ff_dim")
    elif model_type == 'Hybrid (LSTM + Transformer)':
        manual_params['lstm_units'] = st.sidebar.slider("LSTM Units (Hybrid)", 32, 256, 64, 32, key="nn_hybrid_lstm_units")
        manual_params['lstm_dropout'] = st.sidebar.slider("LSTM Dropout (Hybrid)", 0.0, 0.5, 0.2, 0.05, key="nn_hybrid_lstm_dropout")
        manual_params['num_transformer_blocks'] = st.sidebar.slider("Transformer Blocks (Hybrid)", 1, 4, 1, key="nn_hybrid_trans_blocks")
        manual_params['num_heads'] = st.sidebar.slider("Attention Heads (Hybrid)", 1, 8, 2, key="nn_hybrid_trans_heads")
    # --- END OF NEWLY ADDED SECTION ---

# In pages/2_neural_net_backtest.py

# (Keep all imports and the rest of the file the same)

# --- REPLACE the old run_nn_backtest function with this new one ---
def run_nn_backtest(ticker, full_data):
    """Performs a historical backtest for a single ticker using the selected neural network."""
    st.markdown(f"#### Backtesting: **{ticker}**")

    # This part remains the same
    synced_params = st.session_state.get('scanner_params', {})
    if synced_params:
        # (info message display code is unchanged)
        message_lines = ["**Using All Synced Scanner Parameters for TA & Signals:**"]
        message_lines.append(f"- **Trend:** MAs ({synced_params.get('ma_fast', 'N/A')}/{synced_params.get('ma_slow', 'N/A')}), ADX ({synced_params.get('adx_period', 'N/A')})")
        message_lines.append(f"- **Momentum:** RSI ({synced_params.get('rsi_period', 'N/A')}), MACD ({synced_params.get('macd_fast', 'N/A')}/{synced_params.get('macd_slow', 'N/A')}/{synced_params.get('macd_signal', 'N/A')})")
        message_lines.append(f"- **Volatility:** ATR ({synced_params.get('atr_period', 'N/A')})")
        message_lines.append(f"- **Volume Spike Multiplier:** {synced_params.get('vol_spike_mult', 'N/A')}")
        message_lines.append(f"- **Darvas Box:** Lookback ({synced_params.get('darvas_lookback', 'N/A')}), Min Bars ({synced_params.get('darvas_min_bars', 'N/A')})")
        info_msg = "\n".join(message_lines)
        st.info(info_msg)

    data = full_data.copy()
    if selected_indicators:
        data = add_technical_indicators(df=data, selected_indicators=selected_indicators, rsi_window=synced_params.get('rsi_period', 14), macd_fast=synced_params.get('macd_fast', 12), macd_slow=synced_params.get('macd_slow', 26), macd_signal=synced_params.get('macd_signal', 9), adx_window=synced_params.get('adx_period', 14), atr_window=synced_params.get('atr_period', 14))
    if selected_macro: data = add_macro_indicators(data, selected_macro)
    if enable_signal_features: data = add_signal_features(data, synced_params)
    data.dropna(inplace=True)

    if len(data) < training_window_size + backtest_days:
        st.warning(f"Not enough data for {ticker} to perform backtest ({len(data)} rows). Skipping.")
        return

    # --- START: REVISED PREPROCESSING LOGIC TO PREVENT DATA LEAKAGE ---
    from sklearn.preprocessing import MinMaxScaler
    
    # 1. Split data into train and test sets BEFORE scaling
    train_data = data.iloc[:-backtest_days]
    test_data = data.iloc[-backtest_days:]

    # 2. Fit the scaler ONLY on the training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)

    # 3. Transform both the full dataset for sequencing
    # The scaler uses parameters learned ONLY from the training set, which is correct.
    full_scaled_data = scaler.transform(data)
    
    # 4. Create sequences from the correctly scaled data
    close_col_index = data.columns.get_loc('Close')
    X, y = create_sequences(full_scaled_data, close_col_index, training_window_size, 1)

    # 5. Split the SEQUENCES into train and test sets
    train_end_idx = len(X) - backtest_days
    X_train, y_train = X[:train_end_idx], y[:train_end_idx]
    X_test, y_test_actual_scaled = X[train_end_idx:], y[train_end_idx:]
    # --- END: REVISED PREPROCESSING LOGIC ---

    if len(X_train) == 0:
        st.warning(f"Not enough training data for {ticker} after sequencing. Skipping.")
        return

    K.clear_session()
    input_shape = (X_train.shape[1], X_train.shape[2])
    model_builder = {'LSTM': build_lstm_model, 'Transformer': build_transformer_model, 'Hybrid (LSTM + Transformer)': build_hybrid_model}
    
    build_params = manual_params
    current_lr = learning_rate
    current_batch_size = batch_size

    if hp_mode == "Automatic (Tuner)" and KERAS_TUNER_AVAILABLE:
        with st.spinner(f"Running Hyperparameter Tuning for {ticker}..."):
            _, best_hps, tuner_params, _ = run_hyperparameter_tuning(
                model_type=model_type, input_shape=input_shape, output_dim=1,
                X_train=X_train, y_train=y_train, num_trials=num_trials,
                executions_per_trial=executions_per_trial,
                best_models_dir=os.path.join(BEST_MODELS_DIR, ticker),
                force_retune=force_retune, epochs=epochs
            )
        if best_hps:
            build_params = best_hps.values
            current_lr = tuner_params.get('learning_rate', learning_rate)
            st.caption(f"Tuner found best learning rate: {current_lr:.5f}")
            st.caption(f"Tuner found best params: {build_params}")
        else:
            st.warning(f"Tuning failed for {ticker}. Falling back to manual parameters.")

    with st.spinner(f"Training {model_type} model for {ticker}..."):
        model = model_builder[model_type](input_shape, 1, manual_params=build_params, learning_rate=current_lr)
        train_model(model, X_train, y_train, epochs, current_batch_size, current_lr, X_val=X_test, y_val=y_test_actual_scaled)
        predicted_prices_scaled = model.predict(X_test, verbose=0).flatten()

    # The inverse transform part now correctly uses the scaler that was fitted only on training data
    actual_dummy = np.zeros((len(y_test_actual_scaled), scaler.n_features_in_))
    predicted_dummy = np.zeros((len(predicted_prices_scaled), scaler.n_features_in_))
    actual_dummy[:, close_col_index] = y_test_actual_scaled.flatten()
    predicted_dummy[:, close_col_index] = predicted_prices_scaled
    actual_prices = scaler.inverse_transform(actual_dummy)[:, close_col_index]
    predicted_prices = scaler.inverse_transform(predicted_dummy)[:, close_col_index]
    backtest_dates = data.index[-backtest_days:]
    
    rmse, mae, r2 = calculate_metrics(actual_prices, predicted_prices)
    c1, c2, c3 = st.columns(3)
    c1.metric("Backtest MAE", f"${mae:.2f}")
    c2.metric("Backtest RMSE", f"${rmse:.2f}")
    c3.metric("Backtest R2", f"{r2:.2f}")
    
    fig = plot_predictions(
        actual_prices=data['Close'][-(backtest_days + 30):],
        predicted_prices=pd.Series(predicted_prices, index=backtest_dates),
        ticker=ticker
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")

# Main button to run the backtest
tickers_to_run = st.session_state.get('tickers_for_backtest', [])

if not tickers_to_run:
    st.warning("No tickers found. Please run the Scanner and send tickers to the backtester.")
else:
    st.success(f"**{len(tickers_to_run)}** tickers ready for backtest: `{', '.join(tickers_to_run)}`")
    if st.button("Run Neural Net Backtest", type="primary"):
        for ticker in tickers_to_run:
            try:
                data = fetch_stock_data(ticker, start_date, end_date)
                if not data.empty:
                    run_nn_backtest(ticker, data)
            except Exception as e:
                st.error(f"An error occurred while backtesting {ticker}: {e}")
        
        if 'next_page_to_run' in st.session_state:
            st.success("Neural Network backtest complete.")
            next_page_path = st.session_state['next_page_to_run']
            if st.button(f"Continue to Classical ML Backtester"):
                del st.session_state['next_page_to_run']
                st.switch_page(next_page_path)