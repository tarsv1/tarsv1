# signal_features.py

import pandas as pd
import numpy as np
from data import display_log # Use the same logger

def add_signal_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Adds binary signal features to the DataFrame based on scanner rules.
    This function checks for historical occurrences of scanner events.
    
    Args:
        df (pd.DataFrame): DataFrame with basic technical indicators already calculated.
        params (dict): The synced parameter dictionary from the scanner.

    Returns:
        pd.DataFrame: DataFrame with new binary signal columns.
    """
    display_log("⚡ Generating Historical Signal Features...", "info")
    df_copy = df.copy()

    # --- MA Cross Signals ---
    fast_ma_key = params.get('ma_fast', 20)
    slow_ma_key = params.get('ma_slow', 50)
    fast_ma = df_copy['Close'].rolling(window=fast_ma_key).mean()
    slow_ma = df_copy['Close'].rolling(window=slow_ma_key).mean()
    
    # Golden Cross: fast MA crosses ABOVE slow MA
    df_copy['signal_golden_cross'] = ((fast_ma.shift(1) < slow_ma.shift(1)) & (fast_ma > slow_ma)).astype(int)
    # Death Cross: fast MA crosses BELOW slow MA
    df_copy['signal_death_cross'] = ((fast_ma.shift(1) > slow_ma.shift(1)) & (fast_ma < slow_ma)).astype(int)

    # --- RSI State Signals ---
    # These signals are 1 if the condition is met, 0 otherwise.
    if 'RSI' in df_copy.columns:
        rsi_low = params.get('rsi_low', 30)
        rsi_high = params.get('rsi_high', 70)
        df_copy['signal_rsi_oversold'] = (df_copy['RSI'] < rsi_low).astype(int)
        df_copy['signal_rsi_overbought'] = (df_copy['RSI'] > rsi_high).astype(int)

    # --- MACD Cross Signals ---
    if 'MACD' in df_copy.columns and 'MACD_Signal' in df_copy.columns:
        # Bullish Cross: MACD crosses ABOVE signal line
        df_copy['signal_macd_bullish_cross'] = ((df_copy['MACD'].shift(1) < df_copy['MACD_Signal'].shift(1)) & (df_copy['MACD'] > df_copy['MACD_Signal'])).astype(int)
        # Bearish Cross: MACD crosses BELOW signal line
        df_copy['signal_macd_bearish_cross'] = ((df_copy['MACD'].shift(1) > df_copy['MACD_Signal'].shift(1)) & (df_copy['MACD'] < df_copy['MACD_Signal'])).astype(int)

    # Note: Implementing historical Darvas Box and Divergence signals is highly complex and
    # computationally intensive, so we focus on the most common cross/state signals here.

    display_log("✅ Signal features generated.", "info")
    return df_copy