import modin.pandas as pd
import pandas as pdu
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import talib
import math
from numba import njit

def load_data(filename1, filename2):
    """
    Load and merge two datasets based on start_time and align their start and end dates.

    Parameters:
        filename1 (str): Path to the first dataset (e.g., data source 1).
        filename2 (str): Path to the second dataset (e.g., data source 2).

    Returns:
        pd.DataFrame: A cleaned and aligned DataFrame containing data from both sources.
    """
    # Load the data from CSV files
    df_1 = pdu.read_csv(filename1)
    df_2 = pdu.read_csv(filename2)

    # Merge the two datasets on the 'start_time' column using asof
    df = pdu.merge_asof(df_1.sort_values('start_time'), df_2.sort_values('start_time'), on="start_time", direction="nearest")

    # Align the start and end dates
    start_date = max(df_1['start_time'].min(), df_2['start_time'].min())
    end_date = min(df_1['start_time'].max(), df_2['start_time'].max())

    # Filter the merged data to ensure alignment
    df = df[(df['start_time'] >= start_date) & (df['start_time'] <= end_date)]

    # Ensure data is in ascending order
    df = df.sort_values('start_time').reset_index(drop=True)
    return df

def model_calculation(df, rolling_window, threshold, model='zscore', factor='close'):
    series = df[factor]  # Keep as pandas Series
    epsilon = 1e-9

    if model == 'zscore':
        if rolling_window != 0:
            df[f"{model}_{factor}"] = (series - series.rolling(window=rolling_window).mean()) / \
                                      (series.rolling(window=rolling_window).std(ddof=0) + epsilon)
        else:
            df[f"{model}_{factor}"] = (series - series.mean()) / (series.std() + epsilon)

    elif model == 'minmax':
        if rolling_window != 0:
            roll = series.rolling(window=rolling_window)
            df[f'{model}_{factor}'] = 2 * (series - roll.min()) / (roll.max() - roll.min() + epsilon) - 1
        else:
            df[f'{model}_{factor}'] = 2 * (series - series.min()) / (series.max() - series.min() + epsilon) - 1

    elif model == 'sma_diff':
        sma = series.rolling(window=rolling_window).mean()
        df[f'{model}_{factor}'] = (series - sma) / (sma + epsilon)

    elif model == 'ewm':
        ewm_mean = series.ewm(span=rolling_window, adjust=False).mean()
        ewm_std = series.ewm(span=rolling_window, adjust=False).std()
        df[f'{model}_{factor}'] = (series - ewm_mean) / (ewm_std + epsilon)

    elif model == 'momentum':
        momentum = series - series.shift(rolling_window)
        log_momentum = np.log10(np.abs(momentum) + 1) * np.sign(momentum)
        df[f'{model}_{factor}'] = log_momentum - log_momentum.rolling(window=rolling_window).mean()

    elif model == 'volatility':
        rolling_std = series.rolling(window=rolling_window).std(ddof=0)
        df[f'{model}_{factor}'] = (series - series.shift(1)) / (rolling_std + epsilon)

    elif model == 'robust':
        if rolling_window != 0:
            roll = series.rolling(window=rolling_window)
            q1 = roll.quantile(0.25)
            q3 = roll.quantile(0.75)
            median = roll.median()
            iqr = q3 - q1
        else:
            q1, q3 = series.quantile([0.25, 0.75])
            median = series.median()
            iqr = q3 - q1

        df[f"{model}_{factor}"] = (series - median) / (iqr + epsilon)

    elif model == 'percentile':
        if rolling_window != 0:
            df[f"{model}_{factor}"] = series.rolling(window=rolling_window).rank(pct=True) * 2 - 1
        else:
            df[f"{model}_{factor}"] = 2 * (rankdata(series) / len(series)) - 1

    elif model == 'maxabs':
        if rolling_window != 0:
            roll_max_abs = series.rolling(window=rolling_window).apply(lambda x: np.max(np.abs(x)), raw=True)
            df[f'{model}_{factor}'] = series / (roll_max_abs + epsilon)
        else:
            df[f'{model}_{factor}'] = series / (series.abs().max() + epsilon)

    elif model == 'mean_norm':
        if rolling_window != 0:
            roll = series.rolling(window=rolling_window)
            df[f'{model}_{factor}'] = (series - roll.mean()) / (roll.max() - roll.min() + epsilon)
        else:
            df[f'{model}_{factor}'] = (series - series.mean()) / (series.max() - series.min() + epsilon)

    elif model == 'roc':
        df[f'{model}_{factor}'] = (series - series.shift(rolling_window)) / (series.shift(rolling_window) + epsilon)

    elif model == 'rsi':
        df[f'{model}_{factor}'] = (talib.RSI(series.values, timeperiod=rolling_window) - 50.0) / 50 * 3

    elif model == 'psy':
        up_days = np.where(np.diff(series, prepend=series.iloc[0]) > 0, 1, 0)
        df[f'{model}_{factor}'] = (pdu.Series(up_days).rolling(window=rolling_window, min_periods=1).mean() * 100 - 50) / 50 * 3
    
    elif model == 'rvi':
        delta = np.diff(series, prepend=np.nan)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pdu.Series(gain).rolling(window=rolling_window).mean()
        avg_loss = pdu.Series(loss).rolling(window=rolling_window).mean()
        rvi = ((avg_gain / (avg_gain + avg_loss + epsilon)) * 100 - 50) / 50 * 3
        df[f'{model}_{factor}'] = rvi

    elif model == 'mad':
        rolling_mean = series.rolling(window=rolling_window).mean()
        rolling_mad = series.rolling(window=rolling_window).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        df[f'{model}_{factor}'] = (series - rolling_mean) / (rolling_mad + epsilon)

    elif model == 'ma_ratio':
        ma = pdu.Series(series).rolling(window=rolling_window).mean()
        df[f'{model}_{factor}'] = (series / (ma + epsilon)) - 1

    return df

@njit
def position_calculation(signal, close, close_ema, backtest_mode:str, threshold:float):
    # Position Calculation Part
    position = np.zeros(len(signal))

    if backtest_mode == "Trend":
        # Trend position Entry
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → Go long
                position[i] = 1
            elif signal[i] <= -threshold:  # Signal below negative threshold → Go short
                position[i] = -1
            else:  # Hold position
                position[i] = position[i-1]
    elif backtest_mode == "Trend_Reverse":
        # Trend Reverse position Entry
        for i in range(len(signal)):
            if signal[i] >= threshold:
                position[i] = -1
            elif signal[i] <= -threshold:
                position[i] = 1
            else:   # Hold position
                position[i] = position[i-1]
    elif backtest_mode == "MR":
        # Mean Reversion position Entry, Exit at 0
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → Go long
                position[i] = 1
            elif signal[i] <= -threshold:  # Signal below negative threshold → Go short
                position[i] = -1
            elif (signal[i] <= 0 and position[i-1] == 1) or (signal[i] >= 0 and position[i-1] == -1):
                # Signal crosses back to 0 → Close position
                position[i] = 0
            else:  # Hold position if no entry/exit condition is met
                position[i] = position[i-1]
    elif backtest_mode == "MR_Reverse":
        # Mean Reversion Reverse position Entry, Exit at 0
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → Go short
                position[i] = -1
            elif signal[i] <= -threshold:  # Signal below negative threshold → Go long
                position[i] = 1
            elif (signal[i] <= 0 and position[i-1] == -1) or (signal[i] >= 0 and position[i-1] == 1):
                # Signal crosses back to 0 → Close position
                position[i] = 0
            else:  # Hold position if no entry/exit condition is met
                position[i] = position[i-1]
    elif backtest_mode == "Trend_NoHold":
        # Trend position Entry, Exit at Opposite Threshold
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → Go long
                position[i] = 1
            elif signal[i] <= -threshold:  # Signal below negative threshold → Go short
                position[i] = -1
            else:  # Do Not Hold position
                position[i] = 0
    elif backtest_mode == "Trend_emaFilter":
        # Trend position Entry, EMA filter input
        for i in range(len(signal)):
            if signal[i] >= threshold and close[i] >= close_ema[i]:
                position[i] = 1
            elif signal[i] <= -threshold and close[i] <= close_ema[i]:
                position[i] = -1
            else:   # Hold position
                position[i] = position[i-1]
    elif backtest_mode == "Trend_NoHold_emaFilter":
        # Trend position Entry, Exit at Opposite Threshold
        for i in range(len(signal)):
            if signal[i] >= threshold and close[i] >= close_ema[i]:
                position[i] = 1
            elif signal[i] <= -threshold and close[i] <= close_ema[i]:
                position[i] = -1
            else:  # Do Not Hold position
                position[i] = 0
    elif backtest_mode == "L_Trend":
        # Long-Only Trend position Entry
        for i in range(len(signal)):
            if signal[i] >= threshold:
                position[i] = 1
            elif signal[i] <= -threshold:
                position[i] = 0
            else:  # Hold position
                position[i] = position[i-1]
    elif backtest_mode == "S_Trend":
        # Short-Only Trend position Entry
        for i in range(len(signal)):
            if signal[i] >= threshold:
                position[i] = 0
            elif signal[i] <= -threshold:
                position[i] = -1
            else:  # Hold position
                position[i] = position[i-1]
    elif backtest_mode == "L_Trend_Reverse":
        # Long-Only Trend Reverse position Entry
        for i in range(len(signal)):
            if signal[i] >= threshold:
                position[i] = 0
            elif signal[i] <= -threshold:
                position[i] = 1
            else:  # Hold position
                position[i] = position[i-1]
    elif backtest_mode == "S_Trend_Reverse":
        # Short-Only Trend Reverse position Entry
        for i in range(len(signal)):
            if signal[i] >= threshold:
                position[i] = -1
            elif signal[i] <= -threshold:
                position[i] = 0
            else:  # Hold position
                position[i] = position[i-1]
    elif backtest_mode == "L_MR":
        # Long-Only Mean Reversion position Entry, Exit at 0
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → Go long
                position[i] = 1
            elif signal[i] <= -threshold:  # Signal below negative threshold → No Position
                position[i] = 0
            elif (signal[i] <= 0 and position[i-1] == 1) or (signal[i] >= 0 and position[i-1] == -1):
                # Signal crosses back to 0 → Close position
                position[i] = 0
            else:  # Hold position if no entry/exit condition is met
                position[i] = position[i-1]
    elif backtest_mode == "S_MR":
        # Short-Only Mean Reversion position Entry, Exit at 0
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → No Position
                position[i] = 0
            elif signal[i] <= -threshold:  # Signal below negative threshold → Go short
                position[i] = -1
            elif (signal[i] <= 0 and position[i-1] == 1) or (signal[i] >= 0 and position[i-1] == -1):
                # Signal crosses back to 0 → Close position
                position[i] = 0
            else:  # Hold position if no entry/exit condition is met
                position[i] = position[i-1]
    elif backtest_mode == "L_MR_Reverse":
        # Long-Only Mean Reversion Reverse position Entry, Exit at 0
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → No Short(Long Only)
                position[i] = 0
            elif signal[i] <= -threshold:  # Signal below negative threshold → Go long
                position[i] = 1
            elif (signal[i] <= 0 and position[i-1] == -1) or (signal[i] >= 0 and position[i-1] == 1):
                # Signal crosses back to 0 → Close position
                position[i] = 0
            else:  # Hold position if no entry/exit condition is met
                position[i] = position[i-1]
    elif backtest_mode == "S_MR_Reverse":
        # Short-Only Mean Reversion Reverse position Entry, Exit at 0
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → Go Short
                position[i] = -1
            elif signal[i] <= -threshold:  # Signal below negative threshold → No Long(Short Only)
                position[i] = 0
            elif (signal[i] <= 0 and position[i-1] == -1) or (signal[i] >= 0 and position[i-1] == 1):
                # Signal crosses back to 0 → Close position
                position[i] = 0
            else:  # Carry forward the previous position if no entry/exit condition is met
                position[i] = position[i-1]
    elif backtest_mode == "L_Trend_NoHold":
        # Long-Only Trend position Entry, Exit at Opposite Threshold
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → Go long
                position[i] = 1
            elif signal[i] <= -threshold:  # Signal below negative threshold → No short(Long Only)
                position[i] = 0
            else:  # Do Not Hold position
                position[i] = 0
    elif backtest_mode == "S_Trend_NoHold":
        # Short-Only Trend position Entry, Exit at Opposite Threshold
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → No Long(Short Only)
                position[i] = 0
            elif signal[i] <= -threshold:  # Signal below negative threshold → Go Short
                position[i] = -1
            else:  # Do Not Hold position
                position[i] = 0
    elif backtest_mode == "L_Trend_emaFilter":
        # Long-Only Trend position Entry, EMA filter input
        for i in range(len(signal)):
            if signal[i] >= threshold and close[i] >= close_ema[i]:
                position[i] = 1
            elif signal[i] <= -threshold and close[i] <= close_ema[i]:
                position[i] = 0
            else:   # Hold position
                position[i] = position[i-1]
    elif backtest_mode == "S_Trend_emaFilter":
        # Short-Only Trend position Entry, EMA filter input
        for i in range(len(signal)):
            if signal[i] >= threshold and close[i] <= close_ema[i]:
                position[i] = 0
            elif signal[i] <= -threshold and close[i] >= close_ema[i]:
                position[i] = -1
            else:   # Hold Position
                position[i] = position[i-1]
    elif backtest_mode == "L_Trend_NoHold_emaFilter":
        # Long-Only Trend position Entry, Exit at Opposite Threshold, EMA filter input
        for i in range(len(signal)):
            if signal[i] >= threshold and close[i] >= close_ema[i]:
                position[i] = 1
            elif signal[i] <= -threshold and close[i] <= close_ema[i]:
                position[i] = 0
            else:   # Do Not Hold
                position[i] = 0
    elif backtest_mode == "S_Trend_NoHold_emaFilter":
        # Short-Only Trend position Entry, Exit at Opposite Threshold, EMA filter input
        for i in range(len(signal)):
            if signal[i] >= threshold and close[i] <= close_ema[i]:
                position[i] = 0
            elif signal[i] <= -threshold and close[i] >= close_ema[i]:
                position[i] = -1
            else:   # Do Not Hold
                position[i] = 0

    return position

def data_processing(df: pd.DataFrame, method: str, column: str, mode: str = 'default') -> pd.DataFrame:
    """
    Apply transformation and preprocessing methods to a specific column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - method (str): Transformation method ('log', 'pct_change', 'diff', 'square', 'sqrt', 'cube', 'cbrt').
    - column (str): Column to apply the transformation on.
    - mode (str): 'default' for PnL plotting (modifies original df), 'sr' for SR Heatmap (uses df copy).

    Returns:
    - pd.DataFrame: DataFrame with the transformed column.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the DataFrame.")

    # ✅ Use a copy for SR calculations to preserve the original data
    if mode == 'sr':
        df_transformed = df.copy()
    else:
        df_transformed = df  # Direct modification for PnL calculations

    # Apply transformations
    if method == 'log':
        df_transformed[column] = np.log(df_transformed[column].replace(0, np.nan))
    elif method == 'pctChange':
        df_transformed[column] = df_transformed[column].pct_change()
    elif method == 'diff':
        df_transformed[column] = df_transformed[column].diff()
    elif method == 'square':
        df_transformed[column] = np.square(df_transformed[column])
    elif method == 'sqrt':
        df_transformed[column] = np.sqrt(df_transformed[column].clip(lower=0))
    elif method == 'cube':
        df_transformed[column] = np.power(df_transformed[column], 3)
    elif method == 'cbrt':
        df_transformed[column] = np.cbrt(df_transformed[column])
    else:
        raise ValueError(f"Invalid transformation method: {method}")

    # Handle NaN based on the mode
    if mode == 'sr':  # For SR Heatmap
        df_transformed[column] = df_transformed[column].ffill()  # Forward fill NaNs
    else:  # Default for PnL plotting
        df_transformed[column] = df_transformed[column].fillna(0)  # Replace NaNs with 0 to avoid PnL issues

    return df_transformed

def backtest(df:pd.DataFrame, rolling_window:int, threshold:float, preprocess_method="NONE", backtest_mode="Trend", annualizer=365, model='zscore', factor='close', interval='1d', plotsr='default'):
    # Preprocess the data if needed
    if preprocess_method != "none":
        df = data_processing(df, preprocess_method, factor, plotsr)

    df = model_calculation(df, rolling_window, threshold, model, factor)
    # Copy the zscore_btc column to a new column called signal
    df.loc[:,'signal'] = df[f"{model}_{factor}"]
    df.loc[:,'close_ema'] = talib.EMA(df['close'].values, timeperiod=52)

    # Position Calculation
    close = df['close'].values
    close_ema = df['close_ema'].values
    signal = df['signal'].values
    position = position_calculation(signal, close, close_ema, backtest_mode, threshold)

    # Metrics Calculation Part
    fee = 0.0006
    pos = position
    trades = np.abs(np.diff(pos, prepend=0))
    
    # Precompute shifted position for PNL
    shifted_pos = np.roll(pos, 1)
    shifted_pos[0] = 0
    
    # Calculate PNL using numpy vectorization
    pct_change = np.concatenate(([0], np.diff(close) / close[:-1]))
    pnl = pct_change * shifted_pos - trades * fee
    cumu_pnl = np.cumsum(pnl)
    cumu_max = np.maximum.accumulate(cumu_pnl)
    drawdown = cumu_pnl - cumu_max

    # Assign back to DataFrame
    df['pos'] = pos
    df['trades'] = trades
    df['pnl'] = pnl
    df['cumu_pnl'] = cumu_pnl
    df['drawdown'] = drawdown

    # Metrics Calculation
    pnl_std = np.std(pnl)
    pnl_mean = np.mean(pnl)
    max_drawdown = np.min(drawdown)

    sharpe_ratio = np.nan if pnl_std == 0 or np.isnan(pnl_std) else math.sqrt(annualizer) * pnl_mean / pnl_std
    calmar_ratio = np.nan if max_drawdown == 0 or np.isnan(max_drawdown) else annualizer * pnl_mean / abs(max_drawdown)

    avg_return = pnl_mean * annualizer
    total_return = cumu_pnl[-1]
    num_trades = np.sum(trades)
    trade_per_interval = num_trades / len(df)
    
    # Store SR into a dictionary
    performance_metrics = {
            "factor_name": "strategy_0001",
            "Data_Preprocess": preprocess_method,
            "backtest_mode": backtest_mode,
            "fees": fee,
            "interval": interval,
            "model": model,
            "rolling_window": int(rolling_window),
            "threshold": float(threshold),
            "Number_of_Trades": int(num_trades),
            "TR": float(total_return),
            "SR": float(sharpe_ratio),
            "CR": float(calmar_ratio),
            "MDD": float(max_drawdown),
            "AR": float(avg_return),
            "trade_per_interval": float(trade_per_interval)
        }
    
    return performance_metrics