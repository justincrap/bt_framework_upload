# import modin.pandas as pd
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import rankdata, boxcox, skew, kurtosis, linregress
import talib
import math
from numba import njit, jit

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
    df_1 = pd.read_csv(filename1)
    df_2 = pd.read_csv(filename2)

    # Merge the two datasets on the 'start_time' column using asof
    df = pd.merge_asof(df_1.sort_values('start_time'), df_2.sort_values('start_time'), on="start_time", direction="nearest")

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
        df[f'{model}_{factor}'] = (pd.Series(up_days).rolling(window=rolling_window, min_periods=1).mean() * 100 - 50) / 50 * 3
    
    elif model == 'rvi':
        delta = np.diff(series, prepend=np.nan)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=rolling_window).mean()
        avg_loss = pd.Series(loss).rolling(window=rolling_window).mean()
        rvi = ((avg_gain / (avg_gain + avg_loss + epsilon)) * 100 - 50) / 50 * 3
        df[f'{model}_{factor}'] = rvi

    elif model == 'mad':
        rolling_mean = series.rolling(window=rolling_window).mean()
        rolling_mad = series.rolling(window=rolling_window).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        df[f'{model}_{factor}'] = (series - rolling_mean) / (rolling_mad + epsilon)

    elif model == 'ma_ratio':
        ma = pd.Series(series).rolling(window=rolling_window).mean()
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
    elif method == 'boxcox':
        df_transformed[column], best_lambda = boxcox(df_transformed[column])
    else:
        raise ValueError(f"Invalid transformation method: {method}")

    # Handle NaN based on the mode
    if mode == 'sr':  # For SR Heatmap
        df_transformed[column] = df_transformed[column].ffill()  # Forward fill NaNs
    else:  # Default for PnL plotting
        df_transformed[column] = df_transformed[column].fillna(0)  # Replace NaNs with 0 to avoid PnL issues

    return df_transformed

def compute_drawdown_durations(cumu_pnl: np.ndarray,
                               include_last_incomplete: bool = True) -> list:
    """
    回傳所有「完整回撤區間」或包含未完成區間(可設定)的長度(單位: bar)。

    參數:
    ----------
    cumu_pnl : np.ndarray, 長度 N 的累積盈虧序列 (Equity Curve)。
    include_last_incomplete : bool
        True 表示將最後未結束的回撤也納入(計算到最後一筆)；
        False 表示只計算已完整回到新高的回撤。

    回傳:
    ----------
    dd_periods : list
        每一段回撤期間的長度(單位: bar)。
        例如 [10, 5, 32] 表示歷史上出現了三次回撤區間，分別長 10、5、32 bar。
    """
    if len(cumu_pnl) < 2:
        return []

    cumu_max = np.maximum.accumulate(cumu_pnl)
    in_drawdown = False
    start_idx = 0
    dd_periods = []

    for i in range(len(cumu_pnl)):
        if cumu_pnl[i] < cumu_max[i]:
            # 進入或持續在回撤狀態
            if not in_drawdown:
                in_drawdown = True
                start_idx = i
        else:
            # 不在回撤狀態
            if in_drawdown:
                # 剛剛結束回撤區間
                in_drawdown = False
                dd_periods.append(i - start_idx)

    # 若最後一段回撤尚未結束
    if in_drawdown and include_last_incomplete:
        dd_periods.append(len(cumu_pnl) - start_idx)

    return dd_periods

def backtest(df:pd.DataFrame, rolling_window:int, threshold:float, preprocess_method="NONE", backtest_mode="Trend", annualizer=365, model='zscore', factor='close', interval='1d', plotsr='default'):
    # Preprocess the data if needed
    if preprocess_method != "none":
        df = data_processing(df, preprocess_method, factor, plotsr)

    df = model_calculation(df, rolling_window, threshold, model, factor)
    # Copy the zscore_btc column to a new column called signal
    df['signal'] = df[f"{model}_{factor}"]
    

    # Position Calculation
    close = df['close'].values
    df['close_ema'] = talib.EMA(close, timeperiod=52)
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
    # Sharpe * Calmar
    sharpe_calmar = sharpe_ratio * calmar_ratio if not (np.isnan(sharpe_ratio) or np.isnan(calmar_ratio)) else np.nan

    avg_return = pnl_mean * annualizer if not np.isnan(pnl_mean) else np.nan
    total_return = cumu_pnl[-1] if len(cumu_pnl) > 0 else 0.0
    num_trades = np.sum(trades)
    trade_per_interval = num_trades / len(df)
    
    # Average and MAX Drawdown Duration (Bar)
    dd_periods = compute_drawdown_durations(cumu_pnl)
    avg_dd_bar = float(np.mean(dd_periods)) if len(dd_periods) > 0 else 0.0
    max_dd_bar = float(np.max(dd_periods)) if len(dd_periods) > 0 else 0.0
    
    # Equity Curve Slope
    if len(cumu_pnl) > 1:
        # Equity Curve Slope
        x_arr = np.arange(len(cumu_pnl))
        slope, _ = np.polyfit(x_arr, cumu_pnl, 1)

        # R-Square
        slope_, intercept_, r_value, p_value, std_err = linregress(np.arange(len(cumu_pnl)), cumu_pnl)
        r_square = r_value**2 if not np.isnan(r_value) else np.nan
    else:
        slope = np.nan
        r_square = np.nan

    # Sortino Ratio
    negative_pnl = pnl[pnl < 0]
    if len(negative_pnl) > 1:
        downside_deviation = np.std(negative_pnl) if np.std(negative_pnl) != 0 else np.nan
        sortino_ratio = (pnl_mean * annualizer) / (downside_deviation * math.sqrt(annualizer))
    else:
        sortino_ratio = np.nan
    
    # Skewness
    pnl_skewness = skew(pnl, bias=False) if len(pnl) > 1 else np.nan

    pnl_kurtosis = kurtosis(pnl, bias=False) if len(pnl) > 1 else np.nan

    # Tail Ratio
    if len(pnl) > 1:
        p95 = np.percentile(pnl, 95)
        p5 = np.percentile(pnl, 5)
        if p5 == 0 or np.isnan(p5) or np.isinf(p5):
            p5 = np.nan
        tail_ratio = (abs(p95) / abs(p5))
    else:
        tail_ratio = np.nan

    # Commission to Profit Ratio
    commission = fee * num_trades
    if total_return == 0.0:
        total_return = np.nan
    C2P_ratio = commission / total_return
    
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
            "Sortino_Ratio": float(sortino_ratio),
            "sharpe_calmar": float(sharpe_calmar),
            "MDD": float(max_drawdown),
            "AR": float(avg_return),
            "trade_per_interval": float(trade_per_interval),
            f"MAX_Drawdown_Duration({interval})": float(max_dd_bar),
            f"Average_Drawdown_Duration({interval})": float(avg_dd_bar),
            "Equity_Curve_Slope": float(slope),
            "skewness": float(pnl_skewness),
            "kurtosis": float(pnl_kurtosis),
            "R-Square": float(r_square),
            "Tail_Ratio": float(tail_ratio),
            "Commission_to_Profit_Ratio": float(C2P_ratio)
        }
    
    return performance_metrics