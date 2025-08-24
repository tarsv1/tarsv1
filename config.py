# config.py - Central configuration for the application

# Defines the technical indicators available and their default parameters.
# Each key is the indicator name, and the value is a tuple: (default_parameter, default_enabled_state)
TECHNICAL_INDICATORS_DEFAULTS = {
    # --- Core Scanner Indicators (Default to ON) ---
    'MA_WINDOWS': ([10, 20, 50, 100], True),
    'MACD_SHORT_WINDOW': (12, True),
    'MACD_LONG_WINDOW': (26, True),
    'MACD_SIGNAL_WINDOW': (9, True),
    'RSI_WINDOW': (14, True),
    'ADX_WINDOW': (14, True),
    'ATR_WINDOW': (14, True),
    'OBV': (True, True),

    # --- Experimental/Extra Indicators (Default to OFF) ---
    # These are not used by the scanner for its primary score, so they
    # should be disabled by default for a fair backtest. You can still
    # manually enable them in the sidebar to experiment.
    'LAG_FEATURES': ([1, 2, 3, 5, 10], False),      # CHANGED: Default to False
    'STOCH_K': (14, False),                         # CHANGED: Default to False
    'STOCH_D': (3, False),                          # CHANGED: Default to False
    'STD_WINDOWS': ([20], False),                   # CHANGED: Default to False
    'BB_WINDOW': (20, False),                       # CHANGED: Default to False
    'BB_STD_DEV': (2.0, False),                     # CHANGED: Default to False
    'ICHIMOKU': (True, False),                      # CHANGED: Default to False
    'FIBONACCI_RETRACEMENT': (True, False)          # CHANGED: Default to False
}

# (The rest of your config.py file remains the same)

# Defines the fundamental metrics available for feature selection.
FUNDAMENTAL_METRICS = {
    'Trailing P/E': 'trailingPE',
    'Forward P/E': 'forwardPE',
    'Price to Sales': 'priceToSalesTrailing12Months',
    'Price to Book': 'priceToBook',
    'Enterprise to Revenue': 'enterpriseToRevenue',
    'Enterprise to EBITDA': 'enterpriseToEbitda',
    'Profit Margins': 'profitMargins',
    'Return on Equity': 'returnOnEquity',
    'Debt to Equity': 'debtToEquity',
    'Dividend Yield': 'dividendYield',
    'Beta': 'beta'
}

# Defines the machine learning models available for prediction throughout the application.
MODEL_CHOICES = [
    'Random Forest',
    'XGBoost',
    'LightGBM',
    'Gradient Boosting',
    'CatBoost',
    'Decision Tree',
    'KNN',
    'Linear Regression',
    'Prophet',
]
# ... and so on for the rest of the file.

# New: Defines hyperparameter search spaces for RandomizedSearchCV
MODEL_TUNING_GRIDS = {
    'Random Forest': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [20, 31, 40, 50],
        'max_depth': [-1, 10, 20],
        'subsample': [0.7, 0.8, 0.9, 1.0],
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'subsample': [0.7, 0.8, 0.9, 1.0]
    },
    'CatBoost': {
        'iterations': [100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 8, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]
    }
}


# Defines the global market indices available for contextual analysis.
# The key is the user-friendly name, and the value is the Yahoo Finance ticker symbol.
GLOBAL_MARKET_TICKERS = {
    'S&P 500': '^GSPC',
    'Nifty 50': '^NSEI', # Added Nifty 50
    'Crude Oil': 'CL=F',
    'Gold': 'GC=F',
    'US Dollar Index': 'DX-Y.NYB',
    'VIX Volatility': '^VIX',
    'Bitcoin': 'BTC-USD',
    'US 10Y Treasury': '^TNX'
}

# Default settings for prediction and comparison pages.
DEFAULT_N_FUTURE_DAYS = 15
DEFAULT_RECENT_DATA_FOR_COMPARISON = 90
