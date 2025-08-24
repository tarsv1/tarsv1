# pages/3_Classic_ML_Backtest.py

import streamlit as st
import pandas as pd
from datetime import date, timedelta
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from config import MODEL_CHOICES, MODEL_TUNING_GRIDS, TECHNICAL_INDICATORS_DEFAULTS, FUNDAMENTAL_METRICS, GLOBAL_MARKET_TICKERS
from data import fetch_stock_data, add_technical_indicators, add_macro_indicators, add_fundamental_indicators
from utils import calculate_metrics, plot_predictions
from signal_features import add_signal_features

st.title("Classic Machine Learning Backtester")
st.info("This page backtests tickers using multiple models like XGBoost and LightGBM and provides a compiled summary.")

# --- Sidebar Configuration ---
st.sidebar.header("Backtest Configuration")
start_date = st.sidebar.date_input("Start Date for History", value=date.today() - timedelta(days=5*365))
end_date = st.sidebar.date_input("End Date for History", value=date.today())
backtest_days = st.sidebar.slider("Days to Backtest", 30, 500, 90, 10, key="ml_backtest_days")

st.sidebar.subheader("Model Selection")
compatible_models = [m for m in MODEL_CHOICES if m != 'Prophet']
model_choices = st.sidebar.multiselect(
    "Choose Model(s) to Run",
    options=compatible_models,
    default=['LightGBM', 'XGBoost', 'Random Forest']
)

with st.sidebar.expander("Feature Selection"):
    st.write("**Technical Indicators**")
    selected_tech = [ind for ind, (_, en) in TECHNICAL_INDICATORS_DEFAULTS.items() if st.checkbox(ind, en, key=f"ml_tech_{ind}")]
    st.write("**Fundamental Metrics**")
    selected_fund = [name for name, _ in FUNDAMENTAL_METRICS.items() if st.checkbox(name, False, key=f"ml_fund_{name}")]
    st.write("**Macroeconomic Indicators**")
    selected_macro = [name for name, _ in GLOBAL_MARKET_TICKERS.items() if st.checkbox(name, False, key=f"ml_macro_{name}")]

st.sidebar.subheader("Advanced Features")
enable_signal_features = st.sidebar.checkbox("Enable Scanner Signal Features", value=True, key="ml_signal_features")

st.sidebar.subheader("Hyperparameter Tuning")
enable_tuning = st.sidebar.checkbox("Enable RandomizedSearchCV Tuning", value=False)
if enable_tuning:
    n_iter = st.sidebar.slider("Number of Tuning Iterations", 5, 100, 20, 5)

def get_model(model_name):
    models = {
        'Random Forest': RandomForestRegressor(random_state=42), 'XGBoost': xgb.XGBRegressor(random_state=42),
        'LightGBM': lgb.LGBMRegressor(random_state=42), 'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'CatBoost': cb.CatBoostRegressor(random_state=42, verbose=0), 'Decision Tree': DecisionTreeRegressor(random_state=42),
        'KNN': KNeighborsRegressor(), 'Linear Regression': LinearRegression(),
    }
    return models.get(model_name)

def run_classic_ml_backtest(ticker, model_name, full_data):
    """
    Performs a backtest for a single ticker and a single model.
    Returns a dictionary with metrics and the plot figure.
    """
    synced_params = st.session_state.get('scanner_params', {})
    
    data = full_data.copy()
    if selected_tech:
        data = add_technical_indicators(df=data, selected_indicators=selected_tech, rsi_window=synced_params.get('rsi_period', 14), macd_fast=synced_params.get('macd_fast', 12), macd_slow=synced_params.get('macd_slow', 26), macd_signal=synced_params.get('macd_signal', 9), adx_window=synced_params.get('adx_period', 14), atr_window=synced_params.get('atr_period', 14))
    if selected_macro: data = add_macro_indicators(data, selected_macro)
    if selected_fund: data = add_fundamental_indicators(data, ticker, selected_fund)
    if enable_signal_features: data = add_signal_features(data, synced_params)
    
    data['target'] = data['Close'].shift(-1)
    data.dropna(inplace=True)

    if len(data) < backtest_days + 30:
        st.warning(f"Not enough data for {ticker} after feature engineering ({len(data)} rows). Skipping.")
        return None

    features = [col for col in data.columns if col not in ['target']]
    X, y = data[features], data['target']
    X_train, X_test, y_train, y_test = X.iloc[:-backtest_days], X.iloc[-backtest_days:], y.iloc[:-backtest_days], y.iloc[-backtest_days:]
    
    model = get_model(model_name)
    best_params_str = "Defaults"
    
    with st.spinner(f"Training {model_name} for {ticker}..."):
        if enable_tuning and model_name in MODEL_TUNING_GRIDS:
            tscv = TimeSeriesSplit(n_splits=3)
            random_search = RandomizedSearchCV(estimator=model, param_distributions=MODEL_TUNING_GRIDS[model_name], n_iter=n_iter, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42)
            random_search.fit(X_train, y_train)
            model = random_search.best_estimator_
            best_params_str = str(random_search.best_params_)
        else:
            model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    rmse, mae, r2 = calculate_metrics(y_test, predictions)
    
    fig = plot_predictions(
        actual_prices=full_data['Close'][-(backtest_days + 30):],
        predicted_prices=pd.Series(predictions, index=y_test.index),
        ticker=f"{ticker} ({model_name})"
    )
    
    return {
        'Ticker': ticker,
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Figure': fig,
        'Best Params': best_params_str
    }

# --- Main App Logic ---
tickers_to_run = st.session_state.get('tickers_for_backtest', [])

if not tickers_to_run:
    st.warning("No tickers found. Please run the Scanner and send tickers to the backtester.")
else:
    st.success(f"**{len(tickers_to_run)}** tickers ready for backtest: `{', '.join(tickers_to_run)}`")
    
    if not model_choices:
        st.warning("Please select at least one model from the sidebar.")
    elif st.button("Run Classic ML Backtest", type="primary"):
        all_results = []
        
        for ticker in tickers_to_run:
            st.markdown(f"### Processing Ticker: **{ticker}**")
            synced_params = st.session_state.get('scanner_params', {}) 
            if synced_params:
                st.info(f"Using synced scanner parameters (RSI: {synced_params.get('rsi_period', 'N/A')}, MACD: {synced_params.get('macd_fast', 'N/A')}/{synced_params.get('macd_slow', 'N/A')}, etc.)")

            try:
                data = fetch_stock_data(ticker, start_date, end_date)
                if data.empty:
                    st.error(f"Could not fetch data for {ticker}.")
                    continue
                
                for model_name in model_choices:
                    st.markdown(f"##### Running Model: **{model_name}**")
                    result = run_classic_ml_backtest(ticker, model_name, data)
                    if result:
                        all_results.append(result)
                        
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Backtest MAE", f"${result['MAE']:.2f}")
                        c2.metric("Backtest RMSE", f"${result['RMSE']:.2f}")
                        c3.metric("Backtest R2", f"{result['R2']:.2f}")
                        st.plotly_chart(result['Figure'], use_container_width=True)
                        st.markdown("---")

            except Exception as e:
                st.error(f"An error occurred while backtesting {ticker}: {e}")
        
        if all_results:
            st.markdown("---")
            st.header("🏆 Compiled Backtest Summary")
            
            summary_df = pd.DataFrame([{k: v for k, v in res.items() if k != 'Figure'} for res in all_results])
            
            st.dataframe(summary_df.style.format({
                'MAE': '{:.2f}',
                'RMSE': '{:.2f}',
                'R2': '{:.2f}'
            }), use_container_width=True)
            
            if 'next_page_to_run' in st.session_state:
                st.success("Classical ML backtest complete.")
                next_page_path = st.session_state['next_page_to_run']
                if st.button(f"Continue to Neural Network Backtester"):
                    del st.session_state['next_page_to_run']
                    st.switch_page(next_page_path)