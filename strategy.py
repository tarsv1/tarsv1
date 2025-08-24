# strategy.py

import pandas as pd
import numpy as np

def generate_signals(
    df: pd.DataFrame,
    scanner_signals: pd.DataFrame,
    predictions: pd.DataFrame,
    min_scanner_score: int,
    atr_multiplier: float
) -> pd.DataFrame:
    """
    Generates entry and exit signals based on a robust, multi-factor strategy.

    Args:
        df (pd.DataFrame): The original stock data DataFrame.
        scanner_signals (pd.DataFrame): DataFrame containing the scanner's signals (e.g., Score, MA Cross).
        predictions (pd.DataFrame): DataFrame with model predictions.
        min_scanner_score (int): The minimum score from the scanner to consider a trade.
        atr_multiplier (float): The multiplier for the ATR to set stop-loss.

    Returns:
        pd.DataFrame: A DataFrame with the trade signals and management logic.
    """
    if df.empty or scanner_signals.empty or predictions.empty:
        return pd.DataFrame()

    # Align all dataframes by index
    combined_df = df.copy()
    combined_df = combined_df.join(scanner_signals, how='left')
    combined_df = combined_df.join(predictions, how='left')

    # Calculate ATR for stop-loss
    combined_df['ATR'] = combined_df['High'].rolling(14, min_periods=1).mean()
    # Note: ATR calculation is simplified for demonstration; the full ta.atr is better

    # Initialize columns for trading logic
    combined_df['Position'] = 0.0 # 1 for Long, -1 for Short, 0 for None
    combined_df['Entry_Price'] = np.nan
    combined_df['Stop_Loss'] = np.nan
    combined_df['Exit_Signal'] = 0.0

    # Iterate through the DataFrame to apply strategy logic
    for i in range(1, len(combined_df)):
        # --- Entry Logic ---
        # A long entry signal is valid if:
        # 1. Scanner score is high enough.
        # 2. We are not already in a position.
        # 3. Model predicts a price increase.
        if (
            combined_df['Position'].iloc[i-1] == 0 and
            combined_df['Score'].iloc[i] >= min_scanner_score and
            combined_df['Predicted_Price'].iloc[i] > combined_df['Close'].iloc[i]
        ):
            combined_df['Position'].iloc[i] = 1
            combined_df['Entry_Price'].iloc[i] = combined_df['Close'].iloc[i]
            # Set the initial stop-loss
            if pd.notna(combined_df['ATR'].iloc[i]):
                combined_df['Stop_Loss'].iloc[i] = combined_df['Entry_Price'].iloc[i] - atr_multiplier * combined_df['ATR'].iloc[i]

        # --- Position Management Logic ---
        # If already in a position, carry forward the position and management levels
        elif combined_df['Position'].iloc[i-1] == 1:
            combined_df['Position'].iloc[i] = 1
            combined_df['Entry_Price'].iloc[i] = combined_df['Entry_Price'].iloc[i-1]
            combined_df['Stop_Loss'].iloc[i] = combined_df['Stop_Loss'].iloc[i-1]

            # --- Trailing Stop-Loss Logic ---
            # Move the stop-loss up if the price has moved favorably
            if combined_df['Close'].iloc[i] > combined_df['Entry_Price'].iloc[i]:
                new_stop = combined_df['Close'].iloc[i] - atr_multiplier * combined_df['ATR'].iloc[i]
                if new_stop > combined_df['Stop_Loss'].iloc[i]:
                    combined_df['Stop_Loss'].iloc[i] = new_stop

            # --- Exit Logic ---
            # Exit if the price hits the stop-loss
            if combined_df['Close'].iloc[i] <= combined_df['Stop_Loss'].iloc[i]:
                combined_df['Position'].iloc[i] = 0
                combined_df['Exit_Signal'].iloc[i] = -1 # Exit due to stop-loss

    return combined_df