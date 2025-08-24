# -*- coding: utf-8 -*-
# 📊 Watchlist — Accuracy Pro (full app, end-to-end)

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ML
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.base import clone

# Optional libs
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    lgb = None
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    xgb = None
    HAS_XGB = False

try:
    import yfinance as yf
    HAS_YF = True
except Exception:
    yf = None
    HAS_YF = False

# ========= Watchlist Management =========
WATCHLIST_FILE = "watchlist.txt"

def load_tickers():
    """Loads tickers from watchlist.txt, returns default if file not found."""
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, 'r') as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
            return tickers if tickers else ["AAPL", "MSFT", "NVDA"]
    return ["AAPL", "MSFT", "NVDA"]


def save_tickers(tickers):
    """Saves the given list of tickers to watchlist.txt."""
    with open(WATCHLIST_FILE, 'w') as f:
        for ticker in tickers:
            f.write(f"{ticker}\n")


# ========= Data Ingestion for Indices =========

def download_and_merge_indices(df_main, index_tickers, start, end):
    df_merged = df_main.copy()
    for ticker in index_tickers:
        try:
            df_index = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if not df_index.empty:
                cols_to_keep = {}
                if 'Close' in df_index.columns:
                    cols_to_keep['Close'] = f'idx_{ticker}_Close'
                    df_index[f'idx_{ticker}_Return'] = df_index['Close'].pct_change()
                if 'Volume' in df_index.columns:
                    cols_to_keep['Volume'] = f'idx_{ticker}_Volume'

                df_index = df_index[list(cols_to_keep.keys()) + [f'idx_{ticker}_Return']].rename(columns=cols_to_keep)
                df_merged = pd.merge(df_merged, df_index, on='Date', how='left')
        except Exception as e:
            st.warning(f"Could not download data for index {ticker}: {e}")

    df_merged.fillna(method='ffill', inplace=True)
    df_merged.fillna(method='bfill', inplace=True)
    df_merged.dropna(inplace=True)
    return df_merged


# ========= Technical Indicators =========

def RSI(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def EMA(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def SMA(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def MACD(series: pd.Series, fast, slow, signal):
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd = ema_fast - ema_slow
    signal_line = EMA(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist


def ATR(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return EMA(tr, period)


# ========= Feature Engineering (Now with controls) =========

def create_features(df: pd.DataFrame, indicator_params: dict, is_predict_mode: bool = False) -> pd.DataFrame:
    df_copy = df.copy()
    if 'Date' not in df_copy.columns:
        if isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy = df_copy.reset_index()
        else:
            df_copy['Date'] = pd.to_datetime(df_copy.index)

    df_copy.rename(columns={'index': 'Date'}, inplace=True, errors='ignore')
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.sort_values("Date", inplace=True)
    df_copy = df_copy.reset_index(drop=True)

    df_copy["Daily_Return"] = df_copy["Close"].pct_change()
    df_copy["Log_Return"] = np.log(df_copy["Close"] / df_copy["Close"].shift(1))

    if indicator_params['Lags']['enabled']:
        for col in ["Close", "Open", "High", "Low", "Volume"]:
            df_copy[f"{col}_lag1"] = df_copy[col].shift(1)
            df_copy[f"{col}_ret1"] = df_copy[col].pct_change()

    if indicator_params['Rolling_Stats']['enabled']:
        for window in indicator_params['Rolling_Stats']['windows']:
            for col in ["Close", "Open", "High", "Low", "Volume"]:
                df_copy[f"{col}_roll_mean_{window}"] = df_copy[col].rolling(window).mean()
                df_copy[f"{col}_roll_std_{window}"] = df_copy[col].rolling(window).std()

    if indicator_params['RSI']['enabled']:
        df_copy["RSI"] = RSI(df_copy["Close"], indicator_params['RSI']['period'])

    if indicator_params['MACD']['enabled']:
        p = indicator_params['MACD']
        macd, macd_sig, macd_hist = MACD(df_copy["Close"], p['fast'], p['slow'], p['signal'])
        df_copy["MACD"] = macd
        df_copy["MACD_signal"] = macd_sig
        df_copy["MACD_hist"] = macd_hist

    if indicator_params['Moving_Averages']['enabled']:
        for window in indicator_params['Moving_Averages']['windows']:
            df_copy[f"EMA{window}"] = EMA(df_copy["Close"], window)
            df_copy[f"SMA{window}"] = SMA(df_copy["Close"], window)

    if indicator_params['Bollinger']['enabled']:
        p = indicator_params['Bollinger']
        bb_mid = df_copy["Close"].rolling(p['period']).mean()
        bb_std = df_copy["Close"].rolling(p['period']).std()
        df_copy["BB_Upper"] = bb_mid + p['std_dev'] * bb_std
        df_copy["BB_Lower"] = bb_mid - p['std_dev'] * bb_std
        denom = (df_copy["BB_Upper"] - df_copy["BB_Lower"]).replace(0, np.nan)
        df_copy["BB_pct"] = (df_copy["Close"].squeeze() - df_copy["BB_Lower"]) / denom

    if indicator_params['Volatility']['enabled']:
        df_copy['ATR'] = ATR(df_copy, indicator_params['Volatility']['atr_period'])
        for window in indicator_params['Volatility']['hist_vol_windows']:
            df_copy[f'Hist_Vol_{window}d'] = df_copy['Log_Return'].rolling(window=window).std() * np.sqrt(252)

    if indicator_params['Calendar']['enabled']:
        df_copy['DayOfWeek'] = df_copy['Date'].dt.dayofweek
        df_copy['Month'] = df_copy['Date'].dt.month
        df_copy['WeekOfYear'] = df_copy['Date'].dt.isocalendar().week.astype(int)
        df_copy['DayOfYear'] = df_copy['Date'].dt.dayofyear

    df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)

    if not is_predict_mode:
        df_copy.dropna(inplace=True)
        df_copy.reset_index(drop=True, inplace=True)
        for H in [1, 3, 5, 10]:
            df_copy[f"Target_Return_t+{H}"] = df_copy["Close"].shift(-H) / df_copy["Close"] - 1.0
    else:
        for H in [1, 3, 5, 10]:
            if f"Target_Return_t+{H}" not in df_copy.columns:
                df_copy[f"Target_Return_t+{H}"] = np.nan

    return df_copy


# ========= Feature Utilities =========

def get_feature_list(df: pd.DataFrame, extra_exclude: list[str] | None = None) -> list[str]:
    exclude = {"Date", "Ticker"}
    if extra_exclude:
        exclude |= set(extra_exclude)
    exclude |= {c for c in df.columns if isinstance(c, str) and c.startswith("Target_Return_t+")}
    feats = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return feats


def make_model(model_name: str, params: dict = None):
    params = params or {}
    if model_name == "LightGBM" and HAS_LGB:
        return lgb.LGBMRegressor(random_state=42, **params)
    if model_name == "XGBoost" and HAS_XGB:
        return xgb.XGBRegressor(random_state=42, **params)
    if model_name == "Random Forest":
        return RandomForestRegressor(random_state=42, **params)
    if model_name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=42, **params)
    return LinearRegression(**params)


# ========= Importance & Pruning =========

def compute_importances(model, X, y):
    try:
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            imp = np.abs(np.ravel(model.coef_))
        else:
            r = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            imp = r.importances_mean
        s = imp.sum()
        return imp / (s if s else 1.0)
    except Exception:
        return np.ones(X.shape[1]) / max(1, X.shape[1])


def prune_features_by_importance(model, scaler, features, X_scaled, y, prune_fraction: float, min_keep: int):
    if len(features) <= min_keep or prune_fraction <= 0.0:
        return model, scaler, features, X_scaled, None
    imp = compute_importances(model, X_scaled, y)
    idx_desc = np.argsort(imp)[::-1]
    keep_n = max(min_keep, int(round(len(features) * (1.0 - prune_fraction))))
    keep_idx = np.sort(idx_desc[:keep_n])
    pruned_features = [features[i] for i in keep_idx]
    Xp = X_scaled[:, keep_idx]
    scaler_new = StandardScaler()
    Xp_sc = scaler_new.fit_transform(Xp)
    try:
        new_model = clone(model)
    except Exception:
        new_model = model
    new_model.fit(Xp_sc, y)
    return new_model, scaler_new, pruned_features, Xp_sc, imp


# ========= Walk-Forward CV & Adaptive Window =========

def walk_forward_mae(X: np.ndarray, y: np.ndarray, model_name: str, splits: int = 3) -> float:
    tscv = TimeSeriesSplit(n_splits=splits)
    maes = []
    for tr, te in tscv.split(X):
        model = make_model(model_name)
        model.fit(X[tr], y[tr])
        p = model.predict(X[te])
        maes.append(mean_absolute_error(y[te], p))
    return float(np.mean(maes)) if maes else np.inf


def choose_best_window(df_features: pd.DataFrame, model_name: str, candidate_windows: list[int]) -> tuple[int | None, float | None]:
    best_w, best_mae = None, None
    for w in candidate_windows:
        sub = df_features.tail(w)
        if len(sub) < 200:
            continue
        features = get_feature_list(sub)
        if not features:
            continue
        X = sub[features].values
        y = sub['Daily_Return'].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        mae = walk_forward_mae(Xs, y, model_name, splits=3)
        if mae is not None and (best_mae is None or mae < best_mae):
            best_mae, best_w = mae, w
    return best_w, best_mae


def tune_model(model_name, X_train, y_train, n_iterations):
    """Performs RandomizedSearchCV for a given model."""
    param_grid = {}
    if model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_name == 'XGBoost' and HAS_XGB:
        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9]
        }
    elif model_name == 'LightGBM' and HAS_LGB:
        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [20, 31, 40, 50]
        }
    else:
        return make_model(model_name), {}

    model = make_model(model_name)
    tscv = TimeSeriesSplit(n_splits=3)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iterations,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    st.caption(f"_{model_name} best params: {random_search.best_params_}_")
    return random_search.best_estimator_, random_search.best_params_


# ========= Base Models, Manual Stacking Ensemble, & Training Pipeline =========

def _train_single_model(df_train, model_name, enable_pruning, prune_frac, min_keep, tuning_mode, tuning_iterations, manual_params):
    features = get_feature_list(df_train)
    X = df_train[features].values
    y = df_train['Daily_Return'].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xs = np.nan_to_num(Xs)

    if tuning_mode == 'Automatic':
        model, _ = tune_model(model_name, Xs, y, tuning_iterations)
    elif tuning_mode == 'Manual':
        st.caption(f"_{model_name} using manual params: {manual_params.get(model_name, {})}_")
        model = make_model(model_name, params=manual_params.get(model_name, {}))
        model.fit(Xs, y)
    else:  # Off
        model = make_model(model_name)
        model.fit(Xs, y)

    if enable_pruning:
        model, scaler, features, Xs, _ = prune_features_by_importance(
            model, scaler, features, Xs, y, prune_frac, min_keep
        )

    residuals = y - model.predict(Xs)
    return {
        'name': model_name,
        'model': model,
        'scaler': scaler,
        'features': features,
        'residuals': residuals
    }


def train_models_pipeline(
    df_train, model_names, enable_pruning, prune_frac, min_keep,
    ensemble_method, tuning_mode, tuning_iterations, manual_params,
    meta_model_choices=None
):
    """Main training pipeline for all models or stacking models."""
    if ensemble_method == 'Stacking':
        st.write("##### Training Manual Stacking Ensemble(s)...")

        # 1) Build out-of-fold (OOF) meta-features using TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        n = len(df_train)
        meta_cols = model_names if model_names else ["Gradient Boosting"]
        meta_X = np.full((n, len(meta_cols)), np.nan, dtype=float)
        y_all = df_train['Daily_Return'].values

        base_models_per_fold = []  # store per fold info if needed for debug
        for fold, (tr_idx, te_idx) in enumerate(tscv.split(df_train)):
            fold_train = df_train.iloc[tr_idx].copy()
            fold_test = df_train.iloc[te_idx].copy()
            st.caption(f"Manual Stacking: building OOF preds (fold {fold+1}/3; train={len(tr_idx)}, test={len(te_idx)})")
            for j, name in enumerate(meta_cols):
                info_fold = _train_single_model(
                    fold_train, name, enable_pruning, prune_frac, min_keep,
                    tuning_mode, tuning_iterations, manual_params
                )
                # Predict returns on fold test
                X_df = fold_test.reindex(columns=info_fold['features'], fill_value=0)
                Xs = info_fold['scaler'].transform(X_df.values)
                preds = info_fold['model'].predict(np.nan_to_num(Xs))
                meta_X[te_idx, j] = preds

        # Handle any remaining NaNs (if any initial rows not covered)
        meta_X = pd.DataFrame(meta_X, columns=meta_cols).ffill().bfill().values

        # 2) Train final base models on full data (for future predictions)
        base_models_full = {}
        for name in meta_cols:
            st.write(f"Training base model for deployment: **{name}**")
            info_full = _train_single_model(
                df_train, name, enable_pruning, prune_frac, min_keep,
                tuning_mode, tuning_iterations, manual_params
            )
            base_models_full[name] = info_full

        # 3) Train one or many meta-models on OOF predictions
        selected_meta_models = meta_model_choices or ["Ridge"]

        models_out = {}
        for meta_choice in selected_meta_models:
            if meta_choice == "Ridge":
                meta_model = Ridge()
            elif meta_choice == "Lasso":
                meta_model = Lasso()
            elif meta_choice == "Random Forest":
                meta_model = RandomForestRegressor(random_state=42)
            elif meta_choice == "LightGBM" and HAS_LGB:
                meta_model = lgb.LGBMRegressor(random_state=42)
            elif meta_choice == "XGBoost" and HAS_XGB:
                meta_model = xgb.XGBRegressor(random_state=42)
            else:
                meta_model = LinearRegression()

            meta_model.fit(meta_X, y_all)

            key = f"Ensemble ({meta_choice})"
            models_out[key] = {
                'name': f'Stacking Ensemble ({meta_choice})',
                'model': meta_model,           # takes base model returns as features
                'scaler': None,                # not used for meta
                'features': meta_cols,         # base model names in order
                'base_models': base_models_full,
                'residuals': y_all - meta_model.predict(meta_X)
            }

        return models_out

    else:  # Original 'Weighted Average' method
        infos = {}
        for mname in (model_names if model_names else ["Gradient Boosting"]):
            st.write(f"##### Training model: **{mname}**")
            model_info = _train_single_model(
                df_train, mname, enable_pruning, prune_frac, min_keep,
                tuning_mode, tuning_iterations, manual_params
            )
            infos[mname] = model_info
        return infos


def compute_cv_mae_per_model(df_train: pd.DataFrame, model_names: list[str], n_splits: int = 3) -> dict:
    maes = {}
    features = get_feature_list(df_train)
    X = df_train[features].values
    y = df_train['Daily_Return'].values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for name in model_names:
        errs = []
        for tr, te in tscv.split(Xs):
            m = make_model(name)
            m.fit(Xs[tr], y[tr])
            p = m.predict(Xs[te])
            errs.append(mean_absolute_error(y[te], p))
        maes[name] = float(np.mean(errs)) if errs else 1.0
    return maes


# ========= Prediction, Blending, Forecast =========

def _predict_with_base_model(info, last_row_df):
    X_df = last_row_df.reindex(columns=info['features'], fill_value=0)
    X = X_df.values
    Xs = info['scaler'].transform(X)
    ret = float(info['model'].predict(np.nan_to_num(Xs))[0]) if not (np.isnan(Xs).any() or np.isinf(Xs).any()) else np.nan
    return ret

def _predict_with_stacking(info, last_row_df):
    # Gather base model return predictions
    base_returns = []
    for bname in info['features']:
        binfo = info['base_models'][bname]
        X_df_b = last_row_df.reindex(columns=binfo['features'], fill_value=0)
        Xs_b = binfo['scaler'].transform(X_df_b.values)
        ret_b = float(binfo['model'].predict(np.nan_to_num(Xs_b))[0])
        base_returns.append(ret_b)
    meta_input = np.array(base_returns, dtype=float).reshape(1, -1)
    ret = float(info['model'].predict(meta_input)[0])
    return ret

def predict_next_day(models_info: dict, last_row_df: pd.DataFrame, last_close: float):
    preds = []
    for name, info in models_info.items():
        price_pred = np.nan
        try:
            # Detect stacking vs base by presence of 'base_models'
            if isinstance(info, dict) and 'base_models' in info and info.get('features') is not None:
                ret = _predict_with_stacking(info, last_row_df)
                if np.isfinite(ret):
                    price_pred = last_close * (1.0 + ret)
            else:
                ret = _predict_with_base_model(info, last_row_df)
                if np.isfinite(ret):
                    price_pred = last_close * (1.0 + ret)
        except Exception:
            pass
        preds.append((name, price_pred))
    return preds


def ensemble_blend(preds: list[tuple[str, float]], maes: dict | None = None, inverse_weight: bool = True) -> float | None:
    valid_preds = [(name, p) for name, p in preds if p is not None and np.isfinite(p)]
    if not valid_preds:
        return None
    values = np.array([p for _, p in valid_preds], dtype=float)
    if inverse_weight and maes:
        weights = [1.0 / maes.get(name, 1.0) for name, _ in valid_preds if maes.get(name, 1.0) > 0]
        weights = np.array(weights)
        if weights.sum() > 0:
            return float(np.dot(weights / weights.sum(), values))
    return float(values.mean())


# ========= Iterative (recursive) Forecast =========

def generate_iterative_forecast(df_features, models_info, days, maes, inverse_weight, indicator_params, ensemble_method):
    work = df_features.copy()
    outs = []
    for d in range(1, days + 1):
        last_row = work.iloc[[-1]].copy()
        last_close = float(last_row['Close'].values[0])
        preds = predict_next_day(models_info, last_row, last_close)

        if ensemble_method == 'Stacking':
            # Average all stacking ensemble predictions to drive recursion
            stacking_values = [p for name, p in preds if name.startswith("Ensemble (")]
            ens_price = None
            if stacking_values:
                stacking_values = [v for v in stacking_values if v is not None and np.isfinite(v)]
                if stacking_values:
                    ens_price = float(np.mean(stacking_values))
        else:
            ens_price = ensemble_blend(preds, maes, inverse_weight)

        if ens_price is None or not np.isfinite(ens_price):
            st.warning(f"Recursive forecast stopped at day {d} due to invalid prediction.")
            break

        next_date = pd.to_datetime(last_row['Date'].values[0]) + pd.Timedelta(days=1)
        day_output = {'Date': next_date, 'Ensemble_Predicted_Price': ens_price}
        for model_name, pred_price in preds:
            day_output[f'{model_name}_Predicted_Price'] = pred_price
        outs.append(day_output)

        new_row = last_row.copy()
        new_row['Date'] = next_date
        for c in ['Open', 'High', 'Low', 'Close']:
            new_row[c] = ens_price
        for c in [c for c in work.columns if 'Volume' in str(c) or str(c).startswith('idx_')]:
            new_row[c] = last_row[c].values[0]

        work = pd.concat([work, new_row], ignore_index=True)
        work = create_features(work, indicator_params, is_predict_mode=True)
    return pd.DataFrame(outs)


# ========= Backtest =========

def backtest_last_n_days(df_features, models_info, n_days, maes, inverse_weight, ensemble_method):
    df = df_features.sort_values('Date').copy()
    res = []
    max_i = min(n_days, len(df) - 2)

    for i in range(1, max_i + 1):
        idx = -i - 1
        last_row = df.iloc[[idx]].copy()
        last_close = float(last_row['Close'].values[0])
        true_close = float(df['Close'].iloc[idx + 1])
        preds = predict_next_day(models_info, last_row, last_close)

        if ensemble_method == 'Stacking':
            stacking_values = [p for name, p in preds if name.startswith("Ensemble (")]
            ens = None
            if stacking_values:
                stacking_values = [v for v in stacking_values if v is not None and np.isfinite(v)]
                if stacking_values:
                    ens = float(np.mean(stacking_values))
        else:
            ens = ensemble_blend(preds, maes, inverse_weight)

        day_result = {'Date': df['Date'].iloc[idx + 1], 'True_Close': true_close}
        if ens is not None and np.isfinite(ens):
            day_result['Ensemble_Pred_Close'] = ens
            day_result['Ensemble_AbsError'] = abs(ens - true_close)
            day_result['Ensemble_Error'] = ens - true_close
        for model_name, pred_price in preds:
            if pred_price is not None and np.isfinite(pred_price):
                day_result[f'{model_name}_Pred_Close'] = pred_price
                day_result[f'{model_name}_AbsError'] = abs(pred_price - true_close)
                day_result[f'{model_name}_Error'] = pred_price - true_close
        res.append(day_result)

    if not res:
        return pd.DataFrame()
    bt = pd.DataFrame(res).sort_values('Date')
    for col in bt.columns:
        if col.endswith('_AbsError'):
            model_prefix = col.replace('_AbsError', '')
            bt[f'{model_prefix}_APE_%'] = 100 * bt[col] / bt['True_Close'].replace(0, np.nan)
    return bt


# ========= Streamlit UI =========
st.set_page_config(page_title="Watchlist — Accuracy Pro", layout="wide")
st.title("📊 Watchlist — Accuracy Pro")

if 'tickers' not in st.session_state:
    st.session_state.tickers = load_tickers()

st.sidebar.subheader("Manage Watchlist")
new_ticker = st.sidebar.text_input("Add Ticker", "").upper()
if st.sidebar.button("Add"):
    if new_ticker and new_ticker not in st.session_state.tickers:
        st.session_state.tickers.append(new_ticker)
        save_tickers(st.session_state.tickers)
        st.sidebar.success(f"Added {new_ticker}")

tickers_to_remove = st.sidebar.multiselect("Select Tickers to Remove", st.session_state.tickers)
if st.sidebar.button("Remove Selected"):
    if tickers_to_remove:
        st.session_state.tickers = [t for t in st.session_state.tickers if t not in tickers_to_remove]
        save_tickers(st.session_state.tickers)
        st.sidebar.success(f"Removed selected tickers.")

st.sidebar.subheader("Data")
tickers = st.session_state.tickers
st.sidebar.info(f"**Current Watchlist:** {', '.join(tickers)}")
start_date = st.sidebar.date_input("Start date", value=(datetime.today() - timedelta(days=1000)))
end_date = st.sidebar.date_input("End date", value=datetime.today())

with st.sidebar.expander("Global Indices (Features)"):
    indices_map = {"S&P 500": "^GSPC", "NIFTY 50": "^NSEI", "Gold": "GC=F", "Crude Oil": "CL=F"}
    selected_indices_names = st.multiselect("Select indices", list(indices_map.keys()), default=["S&P 500"])
    selected_indices_tickers = [indices_map[name] for name in selected_indices_names]

indicator_params = {}
with st.sidebar.expander("Feature Engineering"):
    c1, c2 = st.columns(2)
    p = {}
    p['enabled'] = c1.checkbox("RSI", value=True)
    p['period'] = c2.number_input("RSI Period", 5, 50, 14, 1)
    indicator_params['RSI'] = p

    p = {}
    p['enabled'] = c1.checkbox("MACD", value=True)
    p['fast'] = c2.number_input("MACD Fast", 5, 50, 12, 1)
    p['slow'] = c2.number_input("MACD Slow", 10, 100, 26, 1)
    p['signal'] = c2.number_input("MACD Signal", 5, 50, 9, 1)
    indicator_params['MACD'] = p

    p = {}
    p['enabled'] = c1.checkbox("Moving Averages", value=True)
    p['windows'] = c2.multiselect("MA Windows", [20, 50, 200], default=[20, 50])
    indicator_params['Moving_Averages'] = p

    p = {}
    p['enabled'] = c1.checkbox("Bollinger Bands", value=True)
    p['period'] = c2.number_input("BB Period", 10, 100, 20, 1)
    p['std_dev'] = c2.number_input("BB Std Dev", 1.0, 4.0, 2.0, 0.5)
    indicator_params['Bollinger'] = p

    indicator_params['Lags'] = {'enabled': c1.checkbox("Price Lags", value=True)}

    p = {}
    p['enabled'] = c1.checkbox("Rolling Stats", value=True)
    p['windows'] = c2.multiselect("Rolling Windows", [5, 10, 20], default=[5, 20])
    indicator_params['Rolling_Stats'] = p

    st.sidebar.markdown("---")
    p = {}
    p['enabled'] = st.sidebar.checkbox("Volatility Features", value=True)
    p['atr_period'] = st.sidebar.number_input("ATR Period", 7, 50, 14, 1)
    p['hist_vol_windows'] = st.sidebar.multiselect("Hist. Vol. Windows", [20, 50, 100], default=[20, 50])
    indicator_params['Volatility'] = p

    indicator_params['Calendar'] = {'enabled': st.sidebar.checkbox("Calendar Features", value=True)}

st.sidebar.subheader("Models & Training")
model_options = []
if HAS_LGB:
    model_options.append("LightGBM")
if HAS_XGB:
    model_options.append("XGBoost")
model_options += ["Random Forest", "Gradient Boosting"]
selected_models = st.sidebar.multiselect(
    "Base models (for ensemble)",
    model_options,
    default=model_options[:3] if len(model_options) >= 3 else model_options
)

ensemble_method = st.sidebar.selectbox("Ensemble Method", ("Weighted Average", "Stacking"))

# NEW: Stacking meta-model multiselect
meta_model_choices = st.sidebar.multiselect(
    "Stacking meta-models (choose one or many)",
    ["Ridge", "Lasso", "Random Forest", "LightGBM", "XGBoost", "Linear"],
    default=["Ridge"]
)

if ensemble_method == 'Weighted Average':
    use_inverse_mae_weights = st.sidebar.checkbox("Weight by CV MAE", value=True)
else:
    use_inverse_mae_weights = False


tuning_mode = st.sidebar.selectbox("Tuning Mode", ["Off (Use Defaults)", "Automatic (Random Search)", "Manual (Specify Params)"])

tuning_iterations = 10
manual_params = {}

if tuning_mode == "Automatic (Random Search)":
    tuning_iterations = st.sidebar.number_input(
        "Tuning Iterations",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="How many parameter combinations to test. More is better but slower."
    )

if tuning_mode == "Manual (Specify Params)":
    st.sidebar.write("---")
    st.sidebar.write("**Manual Hyperparameters**")
    for model_name in selected_models:
        with st.sidebar.expander(f"Parameters for {model_name}"):
            if model_name == "LightGBM":
                params = {
                    'n_estimators': st.number_input("n_estimators", 10, 1000, 100, 10, key="lgbm_n_est"),
                    'learning_rate': st.number_input("learning_rate", 0.001, 0.5, 0.05, 0.005, key="lgbm_lr", format="%.3f"),
                    'num_leaves': st.number_input("num_leaves", 10, 100, 31, 1, key="lgbm_leaves"),
                    'max_depth': st.number_input("max_depth", 3, 20, 7, 1, key="lgbm_depth")
                }
                manual_params[model_name] = params
            elif model_name == "XGBoost":
                params = {
                    'n_estimators': st.number_input("n_estimators", 10, 1000, 100, 10, key="xgb_n_est"),
                    'learning_rate': st.number_input("learning_rate", 0.001, 0.5, 0.05, 0.005, key="xgb_lr", format="%.3f"),
                    'max_depth': st.number_input("max_depth", 3, 20, 5, 1, key="xgb_depth"),
                    'subsample': st.slider("subsample", 0.5, 1.0, 0.8, 0.1, key="xgb_subsample")
                }
                manual_params[model_name] = params
            elif model_name == "Random Forest":
                params = {
                    'n_estimators': st.number_input("n_estimators", 10, 1000, 100, 10, key="rf_n_est"),
                    'max_depth': st.number_input("max_depth", 5, 50, 10, 1, key="rf_depth"),
                    'min_samples_split': st.number_input("min_samples_split", 2, 20, 2, 1, key="rf_split"),
                    'min_samples_leaf': st.number_input("min_samples_leaf", 1, 20, 1, 1, key="rf_leaf")
                }
                manual_params[model_name] = params

st.sidebar.subheader("Accuracy Boosters")
enable_feature_pruning = st.sidebar.checkbox("Feature pruning", value=True)
prune_fraction = st.sidebar.slider("Prune bottom fraction", 0.0, 0.9, 0.3, 0.05)
min_features_keep = st.sidebar.number_input("Min features to keep", 5, 200, 20, 1)

enable_adaptive_window = st.sidebar.checkbox("Adaptive rolling window", value=True)
candidate_windows = st.sidebar.multiselect("Candidate windows", [250, 500, 750, 1000], default=[250, 500, 750])

st.sidebar.subheader("Forecasting & Backtest")
recursive_days = st.sidebar.slider("Recursive forecast days", 1, 30, 7)
backtest_days = st.sidebar.slider("Backtest lookback (days)", 10, 120, 30)

run = st.sidebar.button("Run Watchlist")

# ========= Main =========
if run:
    if not HAS_YF:
        st.error("yfinance not available.")
    elif not tickers:
        st.warning("Your watchlist is empty.")
    else:
        ## NEW ## - Fix for NameError. Initialize the list here.
        all_detailed_forecasts = []

        # Convert tuning_mode string to a simpler keyword for the functions
        tuning_mode_keyword = "Off"
        if "Automatic" in tuning_mode:
            tuning_mode_keyword = "Automatic"
        if "Manual" in tuning_mode:
            tuning_mode_keyword = "Manual"

        for ticker in tickers:
            st.markdown(f"### 🔹 {ticker}")
            try:
                df_raw = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if df_raw.empty:
                    st.warning("No data returned.")
                    continue
                df_raw = df_raw.reset_index()
                if selected_indices_tickers:
                    df_raw = download_and_merge_indices(df_raw, selected_indices_tickers, start_date, end_date)
                df_raw["Ticker"] = ticker
                df_features = create_features(df_raw, indicator_params)
            except Exception as e:
                st.error(f"{ticker} data/feature error: {e}")
                continue

            if df_features.shape[0] < 50:
                st.warning(f"Not enough data for {ticker} after feature engineering ({df_features.shape[0]} rows).")
                continue

            train_df = df_features.copy()
            if enable_adaptive_window and candidate_windows:
                base_name = selected_models[0] if selected_models else "Gradient Boosting"
                with st.spinner(f'Finding best window for {ticker}...'):
                    best_w, best_mae = choose_best_window(train_df, base_name, candidate_windows)
                if best_w is not None:
                    st.caption(f"Adaptive window → **{best_w}d** (CV MAE≈{best_mae:.6f})")
                    train_df = train_df.tail(best_w)

            with st.spinner(f"Training models for {ticker}..."):
                models_info = train_models_pipeline(
                    train_df,
                    selected_models if selected_models else ["Gradient Boosting"],
                    enable_feature_pruning,
                    prune_fraction,
                    min_features_keep,
                    ensemble_method,
                    tuning_mode_keyword,
                    tuning_iterations,
                    manual_params,
                    meta_model_choices=meta_model_choices
                )

            maes = None
            if ensemble_method == 'Weighted Average':
                maes = compute_cv_mae_per_model(train_df, [mi for mi in models_info.keys()], n_splits=3)

            rec_df = generate_iterative_forecast(
                train_df,
                models_info,
                recursive_days,
                maes,
                use_inverse_mae_weights,
                indicator_params,
                ensemble_method
            )
            if not rec_df.empty:
                rec_df['Ticker'] = ticker
                all_detailed_forecasts.append(rec_df)

            bt = backtest_last_n_days(
                train_df,
                models_info,
                n_days=backtest_days,
                maes=maes,
                inverse_weight=use_inverse_mae_weights,
                ensemble_method=ensemble_method
            )

            if not bt.empty:
                st.markdown("##### Backtest Performance")
                summary_data = []
                error_cols = [c for c in bt.columns if c.endswith('_AbsError')]
                for col in error_cols:
                    model_name = col.replace('_AbsError', '')
                    mae = bt[col].mean()
                    mape = bt.get(f'{model_name}_APE_%', pd.Series(dtype=float)).mean()
                    error_col_name = col.replace('_AbsError', '_Error')
                    me_bias = bt.get(error_col_name, pd.Series(dtype=float)).mean()
                    summary_data.append({'Model': model_name, 'MAE': mae, 'MAPE (%)': mape, 'ME (Bias)': me_bias})

                if summary_data:
                    summary_df = pd.DataFrame(summary_data).sort_values('MAE').reset_index(drop=True)
                    st.caption(f"Model Performance (last {len(bt)} trading days)")
                    st.dataframe(
                        summary_df.style.format({'MAE': '{:,.4f}', 'MAPE (%)': '{:,.2f}', 'ME (Bias)': '{:,.4f}'}),
                        use_container_width=True
                    )
                with st.expander("View Daily Backtest Details"):
                    st.dataframe(bt, use_container_width=True)
            else:
                st.warning("Backtest could not be generated.")

        if all_detailed_forecasts:
            st.markdown("### Consolidated Future Forecasts")
            consolidated_df = pd.concat(all_detailed_forecasts, ignore_index=True)
            final_table = consolidated_df.pivot_table(index='Ticker', columns='Date')
            final_table.columns = final_table.columns.rename("Model", level=0)
            final_table.columns = final_table.columns.rename("Date", level=1)
            available_dates = final_table.columns.get_level_values('Date').unique().sort_values()
            selected_dates = st.multiselect("Filter forecast dates:", options=available_dates, default=available_dates)
            if selected_dates:
                filtered_table = final_table.loc[:, final_table.columns.get_level_values('Date').isin(selected_dates)]
                st.dataframe(filtered_table.style.format("{:,.2f}", na_rep="-"))

else:
    st.info("Configure the sidebar and click **Run Watchlist**")
