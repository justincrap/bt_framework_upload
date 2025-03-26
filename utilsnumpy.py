import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import rankdata, boxcox, skew, kurtosis, linregress
import talib
import pandas_ta
import math
from numba import njit, jit
import config as c
import sys

def load_single_data(filename, factor)->pd.DataFrame:
    df = pd.read_csv(filename)
    # get only start_time and factor
    df = df[['start_time', factor]]
    # df['Time'] = pd.to_datetime(df['start_time'], unit='ms')
    # put the data in this order df['Time', 'start_time', factor]
    df = df[['start_time', factor]]
    # Check duplicate and remove duplicate
    df = df.drop_duplicates(subset=['start_time'], keep='first')
    df = df.sort_values('start_time')
    df = df.reset_index(drop=True)
    return df

def chop_data(df, start, end):
    modified_data = df[df['start_time'] >= start]
    finalized_data = modified_data[modified_data['start_time'] <= end]

    return finalized_data

def load_all_data(candle_file, factor_file, factor2_file, factor, factor2):
    # Load data and factor
    candle_data = load_single_data(candle_file, 'Close')
    factor_data = load_single_data(factor_file, factor)
    factor2_data = None

    if c.operation != 'none':
        factor2_data = load_single_data(factor2_file, factor2)
        factor2_nan = nan_count(factor2_data[factor2])
        factor2_nan_percent = factor2_nan / len(factor2_data[factor2])
        if factor2_nan_percent > c.nan_perc:
            print(f"{c.factor2} NaN percentage: {factor2_nan:.3f}, skipping backtest.")
            # End the backtest directly
            sys.exit()  # 直接結束當前 Jupyter cell # 如果之後要修改做 Hopeless的python版本要修改一下
        else:
            print(f"{c.factor2} NaN percentage: {factor2_nan:.3f}, preceed with backtest.")

    factor_nan = nan_count(factor_data[factor])
    factor_nan_percent = factor_nan / len(factor_data[factor])
    if factor_nan_percent > c.nan_perc:
        print(f"{c.factor} NaN percentage: {factor_nan_percent:.3f}, skipping backtest.")
        # End the backtest directly
        sys.exit()
    else:
        print(f"{c.factor} NaN percentage: {factor_nan_percent:.3f}, proceeding with backtest.")

    factor_zero_count = factor_data[factor].eq(0).sum()
    factor_zero_percent = factor_zero_count / len(factor_data[factor])
    if factor_zero_percent > c.zero_perc:
        print(f"{c.factor} zero percentage: {factor_zero_percent:.3f}, skipping backtest.")
        sys.exit()
    else:
        print(f"{c.factor} zero percentage: {factor_zero_percent:.3f}.")

    if factor2_data is not None:
        min_date = max(candle_data['start_time'].min(), factor_data['start_time'].min(), factor2_data['start_time'].min())
        max_date = min(candle_data['start_time'].max(), factor_data['start_time'].max(), factor2_data['start_time'].max())
        chopped_factor2_data = chop_data(factor2_data, min_date, max_date).reset_index(drop=True)
        
    else:
        min_date = max(candle_data['start_time'].min(), factor_data['start_time'].min())
        max_date = min(candle_data['start_time'].max(), factor_data['start_time'].max())

    chopped_candle_data = chop_data(candle_data, min_date, max_date).reset_index(drop=True)
    chopped_factor_data = chop_data(factor_data, min_date, max_date).reset_index(drop=True)

    # merge factor_df and factor2_df, dropna at last
    if factor2_data is not None:
        merged_factor_data = pd.merge_asof(chopped_factor_data, chopped_factor2_data, direction='nearest', on='start_time')
        dropped_merged_factor_data = merged_factor_data.dropna()
        return chopped_candle_data, dropped_merged_factor_data
    else:
        chopped_factor_data = chopped_factor_data.dropna()
        return chopped_candle_data, chopped_factor_data

def nan_count(series):
    # Check if series have more then 3% NaN values
    nan_count = series.isna().sum()
    return nan_count

def split_data(raw_candle, raw_factor, years_for_training=3):
    """
    Split financial data correctly, with special handling for weekday-only data.
    
    Parameters:
    - raw_candle: DataFrame with candle data
    - raw_factor: DataFrame with factor data
    - years_for_training: Number of years to use for training (default: 3)
    
    Returns:
    - Dictionary containing train, test, and full datasets
    """
    # Make sure data is sorted by timestamp
    raw_candle = raw_candle.sort_values('start_time').reset_index(drop=True)
    raw_factor = raw_factor.sort_values('start_time').reset_index(drop=True)
    
    # Convert start_time to datetime for easier date manipulation
    candle_dates = pd.to_datetime(raw_candle['start_time'], unit='ms')
    factor_dates = pd.to_datetime(raw_factor['start_time'], unit='ms')
    
    # Detect if the data is weekday-only by checking for weekends in a sample
    # We'll check the factor data since that's the one likely to be weekday-only
    is_weekday_only = False
    
    # Check a sample of dates to see if there are any weekends
    # Using a sample size of min(100, len(factor_dates)) to avoid checking the entire dataset
    sample_size = min(100, len(factor_dates))
    sample_dates = factor_dates.sample(sample_size) if len(factor_dates) > sample_size else factor_dates
    weekend_count = sum(date.weekday() >= 5 for date in sample_dates)
    
    # If less than 5% of sampled dates are on weekends, it's likely weekday-only data
    if weekend_count / sample_size < 0.05:
        is_weekday_only = True
        print("Detected weekday-only data source - using business day adjustment for split")
    else:
        print("Detected 24/7 data source - using standard calendar split")
    
    # Get the earliest date in the dataset
    start_date = candle_dates.min()
    
    # Calculate the end date for training (start_date + years_for_training years)
    train_end_date = start_date + pd.DateOffset(years=years_for_training)
    
    # If weekday-only data and train_end_date falls on a weekend, move to next business day
    if is_weekday_only and train_end_date.weekday() >= 5:  # If it's a weekend
        days_to_add = 7 - train_end_date.weekday() + 0  # Move to next Monday
        train_end_date = train_end_date + pd.DateOffset(days=days_to_add)
        print(f"Adjusted split date from weekend to next business day: {train_end_date}")
    
    print(f"Training period: {start_date} to {train_end_date}")
    
    # Convert back to milliseconds timestamp for comparison
    train_end_timestamp = int(train_end_date.timestamp() * 1000)
    
    # Find the index where the split should occur
    split_indices = raw_candle[raw_candle['start_time'] >= train_end_timestamp].index
    
    if len(split_indices) == 0:
        print("Warning: No data points after the training period end date. Using all data for training.")
        candle_train = raw_candle.copy()
        factor_train = raw_factor.copy()
        candle_test = pd.DataFrame(columns=raw_candle.columns)
        factor_test = pd.DataFrame(columns=raw_factor.columns)
    else:
        split_idx = split_indices[0]
        
        # Get the actual timestamp at the split point to ensure it exists in our data
        split_timestamp = raw_candle.iloc[split_idx]['start_time']
        
        # Split the data
        candle_train = raw_candle.iloc[:split_idx].copy()
        factor_train = raw_factor[raw_factor['start_time'] < split_timestamp].copy()
        
        candle_test = raw_candle.iloc[split_idx:].copy()
        factor_test = raw_factor[raw_factor['start_time'] >= split_timestamp].copy()
    
    candle_full = raw_candle.copy()
    factor_full = raw_factor.copy()
    
    # Print some info about the splits
    print(f"Training data: {len(candle_train)} candles from {pd.to_datetime(candle_train['start_time'].min(), unit='ms')} to {pd.to_datetime(candle_train['start_time'].max(), unit='ms')}")
    
    if not candle_test.empty:
        print(f"Testing data: {len(candle_test)} candles from {pd.to_datetime(candle_test['start_time'].min(), unit='ms')} to {pd.to_datetime(candle_test['start_time'].max(), unit='ms')}")
        if factor_test.empty:
            print("Warning: No factor data available for testing period!")
        else:
            print(f"Testing factor data: {len(factor_test)} points from {pd.to_datetime(factor_test['start_time'].min(), unit='ms')} to {pd.to_datetime(factor_test['start_time'].max(), unit='ms')}")
    else:
        print("Testing data: None")
    
    return {
        'train': {'candle': candle_train, 'factor': factor_train},
        'test': {'candle': candle_test, 'factor': factor_test},
        'full': {'candle': candle_full, 'factor': factor_full}
    }


def combines_data(factor_df: pd.DataFrame, factor1: str, factor2: str, operation: str):
    """
    Combines two factors from a DataFrame using specified operation and handles infinite values.
    
    Parameters:
        factor_df (pd.DataFrame): DataFrame containing the factor columns
        factor1 (str): Name of the first factor column
        factor2 (str): Name of the second factor column
        operation (str): Operation to perform ('+', '-', '*', '/')
    
    Returns:
        tuple: (processed DataFrame, new factor name)
    """
    # Create the new factor name from the operation
    new_factor_name = f"{factor1}{operation}{factor2}"
    
    # Perform the requested operation and store directly in DataFrame
    if operation == '+':
        factor_df[new_factor_name] = factor_df[factor1] + factor_df[factor2]
    elif operation == '-':
        factor_df[new_factor_name] = factor_df[factor1] - factor_df[factor2]
    elif operation == '*':
        factor_df[new_factor_name] = factor_df[factor1] * factor_df[factor2]
    elif operation == '/':
        factor_df[new_factor_name] = factor_df[factor1] / factor_df[factor2]
    else:
        raise ValueError(f"不支援的運算: {operation}")
    
    # Replace any infinite values with NaN
    factor_df[new_factor_name].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Remove any rows with NaN values
    factor_df = factor_df.dropna()
    
    return factor_df, new_factor_name
    
def data_processing(factor_df, method, factor):
    """
    Apply a transformation method to a factor column in a DataFrame.
    
    Parameters:
        factor_df (pd.DataFrame): DataFrame containing the factor column
        method (str): Transformation method to apply
        factor (str): Name of the factor column
        
    Returns:
        pd.DataFrame: DataFrame with the new transformed column added
        str: Name of the new transformed column
    """
    # Create a copy to avoid modifying the original
    factor_df = factor_df.copy()
   
    # Create new column name for the transformed data
    new_factor_name = f"{factor}_{method}"
    
    # Apply transformation directly to the DataFrame
    try:      
        if method == 'log':
            factor_df[new_factor_name] = np.log(factor_df[factor])
        elif method == 'log10':
            factor_df[new_factor_name] = np.log10(factor_df[factor])
        elif method == 'pct_chg':
            factor_df[new_factor_name] = factor_df[factor].pct_change()
        elif method == 'diff':
            factor_df[new_factor_name] = factor_df[factor].diff()
        elif method == 'square':
            factor_df[new_factor_name] = np.square(factor_df[factor])
        elif method == 'sqrt':
            factor_df[new_factor_name] = np.sqrt(factor_df[factor])
        elif method == 'cube':
            factor_df[new_factor_name] = np.power(factor_df[factor], 3)
        elif method == 'cbrt':
            factor_df[new_factor_name] = np.cbrt(factor_df[factor])
        elif method == 'boxcox':
            factor_df[new_factor_name], _ = boxcox(factor_df[factor])
        else:
            raise ValueError(f"Invalid transformation method: {method}")
    except Exception as e:
        print(f"Error in {method} transformation: {str(e)}, Using Raw Data")
        factor_df[new_factor_name] = factor_df[factor].copy()
        new_factor_name = factor
    
    # Replace infinities with NaN
    factor_df[new_factor_name] = factor_df[new_factor_name].replace([np.inf, -np.inf], np.nan)
    
    # 修改这里: 从第二条数据开始 (skip first row) 并丢弃NaN值
    # 原先的代码: factor_df = factor_df.dropna(subset=[new_factor_name])
    factor_df = factor_df.dropna(subset=[new_factor_name])
    
    return factor_df, new_factor_name

def precompute_rolling_stats(series: pd.Series, windows: list) -> dict:
    """
    預先計算給定 series 在不同 rolling window 下的 mean 與 std，
    並存入字典以便後續查詢。
    
    Parameters:
        series (pd.Series): 要計算 rolling 指標的數據序列。
        windows (list): 各個 rolling window 的大小，例如 [5, 10, 20, 50, 100]。
    
    Returns:
        dict: { window_size: {'mean': np.array, 'std': np.array} }
    """
    rolling_stats = {}
    # 將 series 轉換為 numpy array
    arr = series.values.astype(np.float64)
    n = len(arr)

    for window in windows:
        if window > n:
            continue
        
        roll = series.rolling(window=window, min_periods=window)
        roll_mean = roll.mean().values
        roll_std = roll.std(ddof=0).values
        roll_min = roll.min().values
        roll_max = roll.max().values
        roll_q1 = roll.quantile(0.25).values
        roll_q3 = roll.quantile(0.75).values
        rolling_stats[window] = {'roll_mean': roll_mean, 'roll_std': roll_std, 'roll_min': roll_min, 'roll_max': roll_max, 'roll_q1': roll_q1, 'roll_q3': roll_q3}

    return rolling_stats

def model_calculation_cached(series, rolling_window, model='zscore', factor=None, rolling_stats=None): 
    if model == 'zscore':   # Sample Standard deviation
        roll = series.rolling(window=rolling_window, min_periods=rolling_window)
        roll_mean = roll.mean()
        roll_std = roll.std()
        result = (series - roll_mean) / roll_std

    elif model == 'zscorev1': # zscore 改成 zscorev1
        roll = series.rolling(window=rolling_window, min_periods=rolling_window)
        roll_mean = roll.mean()
        roll_std = roll.std(ddof=0)
        result = (series - roll_mean) / roll_std

    elif model == 'ezscore': # 以前的ezscore使用std(ddof=0), 為了對齊 bq的model就只使用std()
        ewm_mean = series.ewm(span=rolling_window,min_periods=rolling_window, adjust=False).mean()
        std = series.rolling(window=rolling_window, min_periods=rolling_window).std()
        result = (series - ewm_mean) / std

    elif model == 'ezscorev1':
        ewm_mean = series.ewm(span=rolling_window, min_periods=rolling_window, adjust=False).mean()
        ewm_std = series.ewm(span=rolling_window, min_periods=rolling_window, adjust=False).std()
        result = (series - ewm_mean) / ewm_std

    elif model == 'madzscore':
        median = series.rolling(window=rolling_window, min_periods=rolling_window).median()
        deviation = abs(series - median)
        mad = deviation.rolling(window=rolling_window, min_periods=rolling_window).median()
        result = 0.6745 * (series - median) / mad

    elif model == 'robustscaler':
        roll = series.rolling(window=rolling_window, min_periods=rolling_window)
        q1 = roll.quantile(0.25)
        q3 = roll.quantile(0.75)
        roll_median = roll.median()
        iqr = q3 - q1
        result = (series - roll_median) / iqr

    elif model == 'minmaxscaling':
        if rolling_window != 0:
            roll = series.rolling(window=rolling_window, min_periods=rolling_window)
            roll_min = roll.min()
            roll_max = roll.max()
            result = 2 * (series - roll_min) / (roll_max - roll_min) - 1
        else:
            result = 2 * (series - series.min()) / (series.max() - series.min()) - 1

    elif model == 'meannorm':
        roll = series.rolling(window=rolling_window, min_periods=rolling_window)
        roll_mean = roll.mean()
        roll_min = roll.min()
        roll_max = roll.max()
        result = (series - roll_mean) / (roll_max - roll_min)

    elif model == 'maxabs':
        roll_max_abs = series.abs().rolling(window=rolling_window, min_periods=rolling_window).max()
        result = series / roll_max_abs

    elif model == 'smadiffv2_noabs': #應該會是一個不會再用的舊model
        sma = series.rolling(window=rolling_window, min_periods=rolling_window).mean()
        result = (series - sma) / sma

    elif model == 'smadiffv2':
        sma = series.rolling(window=rolling_window, min_periods=rolling_window).mean()
        result = (series - sma) / sma.abs()

    elif model == 'emadiffv2_noabs': #應該會是一個不會再用的舊model
        ewm_mean = series.ewm(span=rolling_window, min_periods=rolling_window, adjust=False).mean()
        result = (series - ewm_mean) / ewm_mean

    elif model == 'emadiffv2':
        ewm_mean = series.ewm(span=rolling_window, min_periods=rolling_window, adjust=False).mean()
        result = (series - ewm_mean) / ewm_mean.abs()

    elif model == 'mediandiffv2_noabs': #應該會是一個不會再用的舊model
        median = series.rolling(window=rolling_window, min_periods=rolling_window).median()
        result = (series - median) / median

    elif model == 'mediandiffv2':
        median = series.rolling(window=rolling_window, min_periods=rolling_window).median()
        result = (series - median) / median.abs()

    elif model == 'smadiffv3': # 舊名稱mad 對齊結果後變更為 smadiffv3
        # Use vectorized computation of rolling mean absolute deviation
        # Compute rolling mean using pandas
        roll_mean = series.rolling(window=rolling_window, min_periods=rolling_window).mean()
        x = series.values
        n = len(x)
        if n >= rolling_window:
            try:
                # Use sliding_window_view if available (NumPy >=1.20)
                windows = sliding_window_view(x, window_shape=rolling_window)
            except ImportError:
                # Fallback to a list comprehension if sliding_window_view is not available.
                windows = np.array([x[i:i+rolling_window] for i in range(n - rolling_window + 1)])
            # Compute the mean for each window along axis 1
            window_means = windows.mean(axis=1, keepdims=True)
            # Compute the mean absolute deviation for each window
            mad_values = np.mean(np.abs(windows - window_means), axis=1)
            # Pad the beginning with NaNs to align with the original series length
            rolling_mad = np.concatenate((np.full(rolling_window - 1, np.nan), mad_values))
            result = (series - roll_mean) / rolling_mad
        else:
            result = np.nan

    elif model == 'srsi':
        delta = np.diff(series, prepend=np.nan)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain, index=series.index).rolling(window=rolling_window, min_periods=rolling_window).mean()
        avg_loss = pd.Series(loss, index=series.index).rolling(window=rolling_window, min_periods=rolling_window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        result = (rsi / 100 * 6) - 3

    elif model == 'ersi':
        delta = np.diff(series, prepend=np.nan)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain, index=series.index).ewm(span=rolling_window, min_periods=rolling_window, adjust=False).mean()
        avg_loss = pd.Series(loss, index=series.index).ewm(span=rolling_window, min_periods=rolling_window, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        result = (rsi / 100 * 6) - 3

    elif model == 'srsiv2':
        delta = series.pct_change()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain, index=series.index).rolling(window=rolling_window, min_periods=rolling_window).mean()
        avg_loss = pd.Series(loss, index=series.index).rolling(window=rolling_window, min_periods=rolling_window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        result = (rsi / 100 * 6) - 3

    elif model == 'ersiv2':
        delta = series.pct_change()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain, index=series.index).ewm(span=rolling_window, min_periods=rolling_window, adjust=False).mean()
        avg_loss = pd.Series(loss, index=series.index).ewm(span=rolling_window, min_periods=rolling_window, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        result = (rsi / 100 * 6) - 3

    elif model == 'rsi':
        # talib.RSI returns an array; ensure conversion to Series if needed.
        rsi_values = talib.RSI(series.values, timeperiod=rolling_window)
        result = (rsi_values / 100 * 6) - 3 # 將rsi範圍從從[0,100] 轉換到 [-3,3]

    elif model == 'rvi':    # 
        delta = np.diff(series, prepend=np.nan)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain, index=series.index).rolling(window=rolling_window, min_periods=rolling_window).mean()
        avg_loss = pd.Series(loss, index=series.index).rolling(window=rolling_window, min_periods=rolling_window).mean()
        rvi = ((avg_gain / (avg_gain + avg_loss)) * 100 - 50) / 50 * 3
        result = rvi

    elif model == 'percentilerank': # 已經由percentile 變更為 percentilerank, 算式完全一樣
        result = series.rolling(window=rolling_window, min_periods=rolling_window).rank(pct=True) * 2 - 1

    elif model == 'L2':
        @jit(nopython=True)
        def rolling_l2_norm(data, window):
            n = len(data)
            result = np.zeros(n)
            rolling_sum = 0.0
            
            # Calculate initial window
            for i in range(window):
                rolling_sum += data[i] * data[i]
            result[window-1] = np.sqrt(rolling_sum)
            
            # Sliding window calculation
            for i in range(window, n):
                rolling_sum += data[i] * data[i] - data[i-window] * data[i-window]
                result[i] = np.sqrt(rolling_sum)
    
            return result
        l2_norms = rolling_l2_norm(series.values, rolling_window)
        result = series / l2_norms

    elif model == 'kurtosis':
        result = pandas_ta.kurtosis(series, length=rolling_window)

    elif model == 'skew':
        result = pandas_ta.skew(series, length=rolling_window)

    elif model == 'cci':
        high = series.rolling(window=rolling_window, min_periods=rolling_window).max()
        low = series.rolling(window=rolling_window, min_periods=rolling_window).min()
        result = talib.CCI(high, low, series, timeperiod=rolling_window)
        # result = pandas_ta.cci(high, low, series, length=rolling_window)

    elif model == 'Weirdroc':   # Old name roc to Weirdroc
        shifted_series = series.shift(rolling_window)
        result = (series - shifted_series) / shifted_series

    elif model == 'roc_ratio':
        shifted_series = series.shift(rolling_window)
        result = (series - shifted_series) / series

    elif model == 'pn': # pn(percentile_norm)
        percentile25 = series.rolling(window=rolling_window, min_periods=rolling_window).quantile(0.25)
        percentile75 = series.rolling(window=rolling_window, min_periods=rolling_window).quantile(0.75)
        result = (series - percentile25) / (percentile75 - percentile25)

    elif model == 'pn_epsilon': # 舊名稱quantile 變更為 pn_epsilon 作為記錄, bq沒有記錄, 應該不會再使用
        epsilon = 1e-9
        percentile25 = series.rolling(window=rolling_window, min_periods=rolling_window).quantile(0.25)
        percentile75 = series.rolling(window=rolling_window, min_periods=rolling_window).quantile(0.75)
        result = (series - percentile25) / (percentile75 - percentile25 + epsilon)

    elif model == 'momentum_old':   # Our Old Momentum
        momentum = series - series.shift(rolling_window)
        log_momentum = np.log10(np.abs(momentum) + 1) * np.sign(momentum)
        roll_log_mom = log_momentum.rolling(window=rolling_window, min_periods=rolling_window)
        result = log_momentum - roll_log_mom.mean()

    elif model == 'momentum':
        momentum = series - series.shift(rolling_window)
        log_momentum = np.log10(np.abs(series) + 1) * np.sign(series)
        roll_log_mom = log_momentum.rolling(window=rolling_window, min_periods=rolling_window)
        result = log_momentum - roll_log_mom.mean()

    elif model == 'volatilityv0': #舊名稱 volatility 變成為 volatilityv0
        rolling_std = series.rolling(window=rolling_window, min_periods=rolling_window).std(ddof=0)
        result = (series - series.shift(1)) / rolling_std

    elif model == 'psy':
        # Compute up_days and use rolling mean
        up_days = np.where(np.diff(series, prepend=series.iloc[0]) > 0, 1, 0)
        up_days_series = pd.Series(up_days, index=series.index)
        result = (up_days_series.rolling(window=rolling_window, min_periods=rolling_window)
                                   .mean() * 100 - 50) / 50 * 3

    elif model == 'winsor':
        lower_bound = series.rolling(window=rolling_window, min_periods=rolling_window).quantile(0.05)
        upper_bound = series.rolling(window=rolling_window, min_periods=rolling_window).quantile(0.95)
        result = np.where(series < lower_bound, 
                          lower_bound, 
                          np.where(series > upper_bound, upper_bound, series))

    elif model == 'winsor_zscorev1': # 舊名 winsorized_zscore 變更為 winsor_zscorev1
        lower_bound = series.rolling(window=rolling_window, min_periods=rolling_window).quantile(0.05)
        upper_bound = series.rolling(window=rolling_window, min_periods=rolling_window).quantile(0.95)
        winsorized = series.clip(lower=lower_bound, upper=upper_bound)
        roll = winsorized.rolling(window=rolling_window, min_periods=rolling_window)
        roll_mean = roll.mean()
        roll_std = roll.std(ddof=0)
        result = (winsorized - roll_mean) / roll_std

    elif model == 'sigmoid':
        roll = series.rolling(window=rolling_window, min_periods=rolling_window)
        roll_mean = roll.mean()
        roll_std = roll.std(ddof=0)
        result = 2 / (1 + np.exp(-(series - roll_mean) / roll_std )) - 1

    elif model == 'robust_zscore':  # 應該要拋棄這個, 理論上跟 madzscore 一樣, 但是因為運算上的分別, 導致得出的結果不同
        arr = series.values.astype(np.float64)
        n = len(arr)
        
        medians = np.full(n, np.nan)
        mads = np.full(n, np.nan)
        if n >= rolling_window:
            # 使用 sliding window trick 處理充足長度的部分
            shape = (n - rolling_window + 1, rolling_window)
            strides = (arr.strides[0], arr.strides[0])
            windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
            medians[rolling_window - 1:] = np.median(windows, axis=1)
            mads[rolling_window - 1:] = np.median(np.abs(windows - medians[rolling_window - 1:, None]), axis=1)
        
        result = 0.6745 * (series - medians) / mads

    elif model == 'tanh':
        arr = series.values.astype(np.float64)
        n = len(arr)
        
        medians = np.full(n, np.nan)
        mads = np.full(n, np.nan)
        if n >= rolling_window:
            # 使用 sliding window trick 處理充足長度的部分
            shape = (n - rolling_window + 1, rolling_window)
            strides = (arr.strides[0], arr.strides[0])
            windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
            medians[rolling_window - 1:] = np.median(windows, axis=1)
            mads[rolling_window - 1:] = np.median(np.abs(windows - medians[rolling_window - 1:, None]), axis=1)

        result = np.tanh((series - medians) / mads)

    elif model == 'linearregressionslope':  # 舊名 slope 變更為 linearregressionslope
        arr = series.values
        n = len(arr)
        
        slopes = np.full(n, np.nan)
        if n >= rolling_window:

            windows = np.lib.stride_tricks.sliding_window_view(arr, window_shape=rolling_window)

            x = np.arange(rolling_window)
            x_mean = np.mean(x)
            denominator = np.sum((x - x_mean)**2)
            window_means = np.mean(windows, axis=1)
            slopes_vec = np.sum((x - x_mean) * (windows - window_means[:, None]), axis=1) / denominator            
            constant_mask = np.all(np.abs(windows - windows[:, 0][:, None]) < 1e-12, axis=1)
            slopes_vec[constant_mask] = 0
            
            slopes = np.concatenate((np.full(rolling_window - 1, np.nan), slopes_vec))
        result = slopes

    elif model == 'chg':
        shifted = series.shift(rolling_window)
        result = (series - shifted) / rolling_window
        
    return result

@njit(cache=True)
def position_calculation(signal, entry_logic:str, threshold:float):

    # Position Calculation Part
    position = np.zeros(len(signal))

    if entry_logic == "trend":
        # Trend position Entry
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → Go long
                position[i] = 1
            elif signal[i] <= -threshold:  # Signal below negative threshold → Go short
                position[i] = -1
            else:  # Hold position
                position[i] = position[i-1]
    elif entry_logic == "trend_reverse":
        # Trend Reverse position Entry
        for i in range(len(signal)):
            if signal[i] >= threshold:
                position[i] = -1
            elif signal[i] <= -threshold:
                position[i] = 1
            else:   # Hold position
                position[i] = position[i-1]
    elif entry_logic == "mr":
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
    elif entry_logic == "mr_reverse":
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
    elif entry_logic == "trend_long":
        # Long-Only Trend position Entry
        for i in range(len(signal)):
            if signal[i] >= threshold:
                position[i] = 1
            elif signal[i] <= -threshold:
                position[i] = 0
            else:  # Hold position
                position[i] = position[i-1]
    elif entry_logic == "trend_short":
        # Short-Only Trend position Entry
        for i in range(len(signal)):
            if signal[i] >= threshold:
                position[i] = 0
            elif signal[i] <= -threshold:
                position[i] = -1
            else:  # Hold position
                position[i] = position[i-1]
    elif entry_logic == "trend_reverse_long":
        # Long-Only Trend Reverse position Entry
        for i in range(len(signal)):
            if signal[i] >= threshold:
                position[i] = 0
            elif signal[i] <= -threshold:
                position[i] = 1
            else:  # Hold position
                position[i] = position[i-1]
    elif entry_logic == "trend_reverse_short":
        # Short-Only Trend Reverse position Entry
        for i in range(len(signal)):
            if signal[i] >= threshold:
                position[i] = -1
            elif signal[i] <= -threshold:
                position[i] = 0
            else:  # Hold position
                position[i] = position[i-1]
    elif entry_logic == "mr_long":
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
    elif entry_logic == "mr_short":
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
    elif entry_logic == "mr_reverse_long":
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
    elif entry_logic == "mr_reverse_short":
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
    elif entry_logic == "fast":
        # Trend position Entry, Exit at Same Threshold
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → Go long
                position[i] = 1
            elif signal[i] <= -threshold:  # Signal below negative threshold → Go short
                position[i] = -1
            else:  # Do Not Hold position
                position[i] = 0
    elif entry_logic == "fast_long":
        # Long-Only Trend position Entry, Exit at Same Threshold
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → Go long
                position[i] = 1
            elif signal[i] <= -threshold:  # Signal below negative threshold → No short(Long Only)
                position[i] = 0
            else:  # Do Not Hold position
                position[i] = 0
    elif entry_logic == "fast_short":
        # Short-Only Trend position Entry, Exit at Same Threshold
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → No Long(Short Only)
                position[i] = 0
            elif signal[i] <= -threshold:  # Signal below negative threshold → Go Short
                position[i] = -1
            else:  # Do Not Hold position
                position[i] = 0
    elif entry_logic == "fast_reverse":
        # Trend Reverse position Entry, Exit at Same Threshold
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → Go short
                position[i] = -1
            elif signal[i] <= -threshold:  # Signal below negative threshold → Go long
                position[i] = 1
            else:  # Do Not Hold position
                position[i] = 0
    elif entry_logic == "fast_reverse_long":
        # Long-Only Trend Reverse position Entry, Exit at Same Threshold
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → No Short(Long Only)
                position[i] = 0
            elif signal[i] <= -threshold:  # Signal below negative threshold → Go long
                position[i] = 1
            else:  # Do Not Hold position
                position[i] = 0
    elif entry_logic == "fast_reverse_short":
        # Short-Only Trend Reverse position Entry, Exit at Same Threshold
        for i in range(len(signal)):
            if signal[i] >= threshold:  # Signal exceeds threshold → Go Short
                position[i] = -1
            elif signal[i] <= -threshold:  # Signal below negative threshold → No Long(Short Only)
                position[i] = 0
            else:  # Do Not Hold position
                position[i] = 0
    elif entry_logic == 'trend_zero':
        long_opened = False
        short_opened = False
        for i in range(len(signal)):    # 每次穿0只會在單邊開倉1次, 直至穿0換邊
            if signal[i] > 0 and not long_opened:  # 首次穿越0且未開多倉, Signal exceeds 0 → Go long
                position[i] = 1
                long_opened = True
                short_opened = False
            elif signal[i] < 0 and not short_opened:  # 首次穿越0且未開空倉, Signal below 0 → Go short
                position[i] = -1
                short_opened = True
                long_opened = False
            elif signal[i] >= threshold and position[i-1] == 1: # 多倉平倉
                position[i] = 0
            elif signal[i] <= -threshold and position[i-1] == -1: # 空倉平倉
                position[i] = 0
            else: 
                position[i] = position[i-1] # 維持前一個時間點的倉位

    elif entry_logic == 'trend_zero_long':
        long_opened = False
        short_opened = False
        for i in range(len(signal)):    # 每次穿0只會在單邊開倉1次, 直至穿0換邊
            if signal[i] > 0 and not long_opened:  # 首次穿越0且未開多倉, Signal exceeds 0 → Go long
                position[i] = 1
                long_opened = True
                short_opened = False
            elif signal[i] < 0 and not short_opened:  # 首次穿越0且未開空倉, Signal below 0 → Go short
                position[i] = 0
                short_opened = True
                long_opened = False
            elif signal[i] >= threshold and position[i-1] == 1: # 多倉平倉
                position[i] = 0
            elif signal[i] <= -threshold and position[i-1] == -1: # 空倉平倉
                position[i] = 0
            else:
                position[i] = position[i-1] # 維持前一個時間點的倉位
    elif entry_logic == 'trend_zero_short':
        long_opened = False
        short_opened = False
        for i in range(len(signal)):    # 每次穿0只會在單邊開倉1次, 直至穿0換邊
            if signal[i] > 0 and not long_opened:  # 首次穿越0且未開多倉, Signal exceeds 0 → Go long
                position[i] = 0
                long_opened = True
                short_opened = False
            elif signal[i] < 0 and not short_opened:  # 首次穿越0且未開空倉, Signal below 0 → Go short
                position[i] = -1
                short_opened = True
                long_opened = False
            elif signal[i] >= threshold and position[i-1] == 1: # 多倉平倉
                position[i] = 0
            elif signal[i] <= -threshold and position[i-1] == -1: # 空倉平倉
                position[i] = 0
            else:
                position[i] = position[i-1] # 維持前一個時間點的倉位
    elif entry_logic == 'trend_zero_fast':
        for i in range(len(signal)):    # 0至threshold內持有多倉, >threshold出場, 0至-threshold內持有空倉, <-threshold出場
            if signal[i] > 0:  # Signal exceeds 0 → Go long
                position[i] = 1
            elif signal[i] < 0: # Signal below 0 → Go short
                position[i] = -1
            elif signal[i] >= threshold and position[i-1] == 1: # 多倉平倉
                position[i] = 0
            elif signal[i] <= -threshold and position[i-1] == -1: # 空倉平倉
                position[i] = 0
            else:
                position[i] = 0
    elif entry_logic == 'trend_zero_fast_long':
        for i in range(len(signal)):    # 只開多, 0至threshold內持有多倉, >threshold出場
            if signal[i] > 0:  # Signal exceeds 0 → Go long
                position[i] = 1
            elif signal[i] < 0: # Signal below 0 → Go short
                position[i] = 0
            elif signal[i] >= threshold and position[i-1] == 1: # 多倉平倉
                position[i] = 0
            elif signal[i] <= -threshold and position[i-1] == -1: # 空倉平倉
                position[i] = 0
            else:
                position[i] = 0 # 維持前一個時間點的倉位
    elif entry_logic == 'trend_zero_fast_short':
        for i in range(len(signal)):    # 只開空, 0至threshold內持有空倉, <-threshold出場
            if signal[i] > 0:  # Signal exceeds 0 → Go long
                position[i] = 0
            elif signal[i] < 0: # Signal below 0 → Go short
                position[i] = -1
            elif signal[i] >= threshold and position[i-1] == 1: # 多倉平倉
                position[i] = 0
            elif signal[i] <= -threshold and position[i-1] == -1: # 空倉平倉
                position[i] = 0
            else:
                position[i] = 0
    return position

def compute_drawdown_durations(cumu_pnl: np.ndarray,
                               include_last_incomplete: bool = True) -> list:
    """
    Return the durations (in bars) of drawdown periods from a cumulative PnL curve.
    
    Parameters
    ----------
    cumu_pnl : np.ndarray
        1D array representing the cumulative profit and loss (Equity Curve).
    include_last_incomplete : bool, optional
        If True, include the final drawdown period even if it hasn't fully recovered.
        If False, only complete drawdowns (that end when a new high is reached) are counted.
    
    Returns
    -------
    dd_periods : list
        A list of durations (in bars) for each drawdown period.
        For example, [10, 5, 32] indicates three drawdown periods lasting 10, 5, and 32 bars respectively.
    """
    # If the array is too short, return an empty list
    if cumu_pnl.size < 2:
        return []

    # Compute the cumulative maximum up to each point
    cumu_max = np.maximum.accumulate(cumu_pnl)
    # Create a boolean mask where the PnL is below the running max (i.e. in drawdown)
    mask = cumu_pnl < cumu_max

    # If there's no drawdown at all, return an empty list
    if not mask.any():
        return []

    # Pad the mask with False on both sides to capture transitions
    padded = np.concatenate(([False], mask, [False]))
    # Compute the difference to find where drawdowns start (False -> True) and end (True -> False)
    diff = np.diff(padded.astype(np.int8))
    # Start indices where diff equals 1 (transition from False to True)
    starts = np.where(diff == 1)[0]
    # End indices where diff equals -1 (transition from True to False)
    ends = np.where(diff == -1)[0]
    
    # The duration of each drawdown is the difference between the corresponding end and start indices.
    durations = ends - starts

    # If we do not want to include an incomplete drawdown at the end, remove it if necessary.
    if not include_last_incomplete and mask[-1]:
        durations = durations[:-1]

    return durations.tolist()

def backtest_cached(candle_df: pd.DataFrame, factor_df: pd.DataFrame, rolling_window: int, threshold: float, preprocess_method="NONE",
             entry_logic="Trend", annualizer=365, model='zscore', factor='close', interval='1d', date_range=None,
             rolling_stats=None): 
    log_msgs = []
    fee = 0.0006
    # Initialize 
    tail_ratio = np.nan

    # 1. 模型計算（使用 cache 版本)
    factor_df[f'{model}_{factor}'] = model_calculation_cached(factor_df[factor], rolling_window, model, factor, rolling_stats)

    factor_df['signal'] = factor_df[f'{model}_{factor}']

    # 2. Signal 3% NaN Check (大於3%就不回測, return None)
    # 2.1 將 inf 和 -inf 轉換成NaN
    factor_df['signal'] = factor_df['signal'].replace([np.inf, -np.inf], np.nan)
    # 2.2 3% NaN Check
    
    if factor_df['signal'].isna().sum() < (c.nan_perc * (len(factor_df['signal']) - rolling_window)) :
        # 2.3 將 NaN 值刪除
        signal_nan_count = nan_count(factor_df['signal'])
        msg = (f"{c.alpha_id}, window: {rolling_window}, threshold: {threshold:.2f},"
               f"{c.factor} NaN count: {signal_nan_count}, Dropping NaN and Keep Backtest,\n"
               f"Total candle count: {len(candle_df)}, Total signal count: {len(factor_df['signal'])}.")
        # log_msgs.append(msg)
        factor_df['signal'] = factor_df['signal'].dropna()
    else:
        msg = (f"{c.alpha_id}, window: {rolling_window}, threshold: {threshold:.2f}, "
               f"nan_count: {factor_df['signal'].isna().sum()}, {c.factor} NaN percentage,"
               f"{factor_df['signal'].isna().sum() / len(factor_df['signal']):.3f}, skipping backtest,\n"
               f"Total candle count: {len(candle_df)}, Total signal count: {len(factor_df['signal'])}.")
        log_msgs.append(msg)
        performance_metrics = {
        "factor_name": c.factor_name,
        "Data_Preprocess": preprocess_method,
        "entry_exit_logic": entry_logic,
        "fees": fee,
        "interval": interval,
        "model": model,
        "rolling_window": int(rolling_window),
        "threshold": float(threshold),
        "Number_of_Trades": 0,
        "TR": np.nan,
        "SR": np.nan,
        "CR": np.nan,
        "Sortino_Ratio": np.nan,
        "sharpe_calmar": np.nan,
        "MDD": np.nan,
        "AR": np.nan,
        "trade_per_interval": np.nan,
        f"MAX_Drawdown_Duration({interval})": np.nan,
        f"Average_Drawdown_Duration({interval})": np.nan,
        "Equity_Curve_Slope": np.nan,
        "skewness": np.nan,
        "kurtosis": np.nan,
        "R-Square": np.nan,
        "Tail_Ratio": np.nan,
        "Commission_to_Profit_Ratio": np.nan
    }
        if 'df' not in locals():
            df = pd.DataFrame()
        return performance_metrics, df, log_msgs

    # 3. Position 計算
    close = candle_df['close'].values
    signal = factor_df['signal'].values
    position = position_calculation(signal,  entry_logic, threshold)

    factor_df['pos'] = position

    # 重新索引並前向填充缺失值
    factor_df = factor_df.reindex(date_range)
    factor_df = factor_df.ffill()

    # 合併 candle_df 和 factor_df
    factor_df['start_time'] = factor_df.index.astype('int64') // 10**6
    df = pd.merge_asof(candle_df.sort_values('start_time'), factor_df.sort_values('start_time'), on='start_time', direction='nearest') #, tolerance=10*60*1000

    df.set_index('time', inplace=True)

    # 4. 損益與績效計算
    pos = df['pos'].values

    # # 檢查 index
    # print("df index preview before trimming:", df.index[:5])
    # print("pos length:", len(pos))
    # print("df length before trimming:", len(df))

    # 4. 計算損益
    trades = np.abs(np.diff(pos, prepend=0))
    shifted_pos = np.roll(pos, 1)
    shifted_pos[0] = 0

    close = df['close'].values
    pct_change = np.concatenate(([0], np.diff(close) / close[:-1]))

    pnl = pct_change * shifted_pos - trades * fee

    # 检查pnl中是否有NaN值
    if np.isnan(pnl).any():
        print(f"Warning: PnL contains {np.isnan(pnl).sum()} NaN values")
        # 填充NaN值为0
        pnl = np.nan_to_num(pnl, nan=0.0)

    if np.isnan(trades).any():
        print(f"Warning: Trades contains {np.isnan(trades).sum()} NaN values")
        # 填充NaN值为0
        trades = np.nan_to_num(trades, nan=0.0)

    cumu_pnl = np.cumsum(pnl)
    cumu_max = np.maximum.accumulate(cumu_pnl)
    drawdown = cumu_pnl - cumu_max

    df['pos'] = pos
    df['trades'] = trades
    df['pnl'] = pnl
    df['cumu_pnl'] = cumu_pnl
    df['drawdown'] = drawdown

    # Calculate metrics, handling potential NaN/division by zero
    pnl_std = np.std(pnl)
    pnl_mean = np.mean(pnl)
    max_drawdown = np.min(drawdown)

    # Handle potential NaN or division by zero
    sharpe_ratio = np.nan if pnl_std == 0 or np.isnan(pnl_std) else math.sqrt(annualizer) * pnl_mean / pnl_std
    calmar_ratio = np.nan if max_drawdown == 0 or np.isnan(max_drawdown) else annualizer * pnl_mean / abs(max_drawdown)
    sharpe_calmar = sharpe_ratio * calmar_ratio if not (np.isnan(sharpe_ratio) or np.isnan(calmar_ratio)) else np.nan
    avg_return = pnl_mean * annualizer if not np.isnan(pnl_mean) else np.nan
    total_return = cumu_pnl[-1] if len(cumu_pnl) > 0 else 0.0
    num_trades = np.sum(trades)
    trade_per_interval = num_trades / len(df) if len(df) > 0 else np.nan
    
    dd_periods = compute_drawdown_durations(cumu_pnl)
    avg_dd_bar = float(np.mean(dd_periods)) if len(dd_periods) > 0 else 0.0
    max_dd_bar = float(np.max(dd_periods)) if len(dd_periods) > 0 else 0.0
    
    if len(cumu_pnl) > 1:
        if np.all(cumu_pnl == cumu_pnl[0]):
            slope = np.nan
            r_square = np.nan
        else:
            x_arr = np.arange(len(cumu_pnl))
            slope, intercept = np.polyfit(x_arr, cumu_pnl, 1)
            r_value = np.corrcoef(x_arr, cumu_pnl)[0, 1]
            r_square = r_value**2

    else:
        slope = np.nan
        r_square = np.nan

    negative_pnl = pnl[pnl < 0]
    if len(negative_pnl) > 1:
        downside_deviation = np.std(negative_pnl) if np.std(negative_pnl) != 0 else np.nan
        sortino_ratio = (pnl_mean * annualizer) / (downside_deviation * math.sqrt(annualizer)) if downside_deviation != 0 else np.nan
    else:
        sortino_ratio = np.nan
    
    # Handle skewness and kurtosis calculation for small samples
    try:
        pnl_skewness = skew(pnl, bias=False) if len(pnl) > 2 else np.nan
    except:
        pnl_skewness = np.nan
        
    try:
        pnl_kurtosis = kurtosis(pnl, bias=False) if len(pnl) > 2 else np.nan
    except:
        pnl_kurtosis = np.nan

    # Calculate tail ratio safely
    if len(pnl) > 1:
        p95 = np.percentile(pnl, 95)
        p5 = np.percentile(pnl, 5)
        if p5 == 0 or np.isnan(p5) or np.isinf(p5) or p5 == 0:
            p5 = np.nan
        else:
            tail_ratio = abs(p95) / abs(p5)

    else:
        tail_ratio = np.nan

    commission = fee * num_trades
    if total_return == 0.0 or np.isnan(total_return) or total_return == 0:
        C2P_ratio = np.nan
    else:
        C2P_ratio = commission / total_return
    
    # Create performance metrics dictionary, making sure to handle all potential NaN values
    performance_metrics = {
        "factor_name": c.factor_name,
        "Data_Preprocess": preprocess_method,
        "entry_exit_logic": entry_logic,
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
    if 'df' not in locals():
        df = pd.DataFrame()
    return performance_metrics, df, log_msgs

def additional_metrics(
    alpha_id="None", symbol="None", factor="None", factor2="None",
    factor_operation="None", shift_candle_minite="None",
    backtest_mode="None", start_time="None", end_time="None"
):
    return {
        "alpha_id": alpha_id,
        "symbol": symbol,
        "factor": factor,
        "factor2": factor2,
        "factor_operation": factor_operation,
        "shift_candle_minite": shift_candle_minite,
        "backtest_mode": backtest_mode,
        "start_time": start_time,
        "end_time": end_time,
    }

def model_calculation_bq(series, rolling_window, model='zscore', factor='close', rolling_stats=None): 
    calc_df = pd.DataFrame()
    calc_df['data'] = series

    if model == 'zscorev1':
        calc_df['sma'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).mean()
        calc_df['std'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).std(ddof=0)
        result = (calc_df['data'] - calc_df['sma']) / (calc_df['std'])

    elif model == 'ezscore':
        calc_df['ema'] = calc_df['data'].ewm(span=rolling_window, min_periods=rolling_window, adjust=False).mean()
        calc_df['std'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).std()
        result = (calc_df['data'] - calc_df['ema']) / (calc_df['std'])

    elif model == 'ezscorev1':  # Previous ezscore to ezscorev1
        calc_df['ema'] = calc_df['data'].ewm(span=rolling_window, min_periods=rolling_window, adjust=False).mean()
        calc_df['ema_std'] = calc_df['data'].ewm(span=rolling_window, min_periods=rolling_window, adjust=False).std()
        result = (calc_df['data'] - calc_df['ema']) / (calc_df['ema_std'])

    elif model == 'madzscore':
        calc_df['median'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).median()
        calc_df['deviation'] = abs(calc_df['data'] - calc_df['median'])
        calc_df['mad'] = calc_df['deviation'].rolling(window=rolling_window, min_periods=rolling_window).median()
        result = 0.6745 * (calc_df['data'] - calc_df['median']) / calc_df['mad']

    elif model == 'robustscaler':
        calc_df['rolling_median'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).median()
        calc_df['rolling_iqr'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).quantile(0.75) - calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).quantile(0.25)
        result = (calc_df['data'] - calc_df['rolling_median']) / calc_df['rolling_iqr']

    elif model == 'minmaxscaling':
        calc_df['rolling_min'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).min()
        calc_df['rolling_max'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).max()
        result = 2 * (calc_df['data'] - calc_df['rolling_min']) / (calc_df['rolling_max'] - calc_df['rolling_min']) - 1

    elif model == 'meannorm':
        calc_df['rolling_mean'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).mean()
        calc_df['rolling_min'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).min()
        calc_df['rolling_max'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).max()
        result = (calc_df['data'] - calc_df['rolling_mean']) / (calc_df['rolling_max'] - calc_df['rolling_min'])

    elif model == 'maxabs':
        calc_df['abs_data'] = np.abs(calc_df['data'])
        rolling_max_abs = calc_df['abs_data'].rolling(window=rolling_window, min_periods=rolling_window).max()
        result = series / rolling_max_abs

    elif model == 'smadiffv2':
        calc_df['sma'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).mean()
        result = (calc_df['data'] - calc_df['sma']) / calc_df['sma'].abs()

    elif model == 'emadiffv2':
        calc_df['ema'] = calc_df['data'].ewm(span=rolling_window, min_periods=rolling_window, adjust=False).mean()
        result = (calc_df['data'] - calc_df['ema']) / calc_df['ema'].abs()
            
    elif model == 'mediandiffv2':
        calc_df['median'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).median()
        result = (calc_df['data'] - calc_df['median']) / calc_df['median'].abs()

    elif model == 'smadiffv3':
        calc_df['roll_mean'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).mean()
        calc_df['mad'] = (calc_df['data'].rolling(window=rolling_window,min_periods=rolling_window).apply(lambda x: np.mean(np.abs(x - x.mean()))))
        result = (calc_df['data'] - calc_df['roll_mean']) / calc_df['mad']

    elif model == 'srsi':
        calc_df['delta'] = calc_df['data'].diff()
        calc_df['gain'] = calc_df['delta'].where(calc_df['delta'] > 0, 0)
        calc_df['loss'] = -calc_df['delta'].where(calc_df['delta'] < 0, 0)
        calc_df['avg_gain'] = calc_df['gain'].rolling(window=rolling_window, min_periods=rolling_window).mean()
        calc_df['avg_loss'] = calc_df['loss'].rolling(window=rolling_window, min_periods=rolling_window).mean()
        calc_df['rs'] = calc_df['avg_gain'] / calc_df['avg_loss']
        calc_df['rsi'] = 100 - (100 / (1+ calc_df['rs']))
        result = (calc_df['rsi'] / 100 * 6) - 3 # 將rsi範圍從從[0,100] 轉換到 [-3,3]

    elif model == 'ersi':
        calc_df['delta'] = calc_df['data'].diff()
        calc_df['gain'] = calc_df['delta'].where(calc_df['delta'] > 0, 0)
        calc_df['loss'] = -calc_df['delta'].where(calc_df['delta'] < 0, 0)
        calc_df['avg_gain'] = calc_df['gain'].ewm(span=rolling_window, min_periods=rolling_window, adjust=False).mean()
        calc_df['avg_loss'] = calc_df['loss'].ewm(span=rolling_window, min_periods=rolling_window, adjust=False).mean()
        calc_df['ers'] = calc_df['avg_gain'] / calc_df['avg_loss']
        calc_df['ersi'] = 100 - (100 / (1+ calc_df['ers']))
        result = (calc_df['ersi'] / 100 * 6) - 3 # 將rsi範圍從從[0,100] 轉換到 [-3,3]

    elif model == 'srsiv2':
        calc_df['delta'] = calc_df['data'].pct_change()
        calc_df['gain'] = calc_df['delta'].where(calc_df['delta'] > 0, 0)
        calc_df['loss'] = -calc_df['delta'].where(calc_df['delta'] < 0, 0)
        calc_df['avg_gain'] = calc_df['gain'].rolling(window=rolling_window, min_periods=rolling_window).mean()
        calc_df['avg_loss'] = calc_df['loss'].rolling(window=rolling_window, min_periods=rolling_window).mean()
        calc_df['rs'] = calc_df['avg_gain'] / calc_df['avg_loss']
        calc_df['srsiv2'] = 100 - (100 / (1+ calc_df['rs']))
        result = (calc_df['srsiv2'] / 100 * 6) - 3 # 將rsi範圍從從[0,100] 轉換到 [-3,3]

    elif model == 'ersiv2':
        calc_df['delta'] = calc_df['data'].pct_change()
        calc_df['gain'] = calc_df['delta'].where(calc_df['delta'] > 0, 0)
        calc_df['loss'] = -calc_df['delta'].where(calc_df['delta'] < 0, 0)
        calc_df['avg_gain'] = calc_df['gain'].ewm(span=rolling_window, min_periods=rolling_window, adjust=False).mean()
        calc_df['avg_loss'] = calc_df['loss'].ewm(span=rolling_window, min_periods=rolling_window, adjust=False).mean()
        calc_df['ers'] = calc_df['avg_gain'] / calc_df['avg_loss']
        calc_df['ersiv2'] = 100 - (100 / (1+ calc_df['ers']))
        result = (calc_df['ersiv2'] / 100 * 6) - 3 # 將rsi範圍從從[0,100] 轉換到 [-3,3]

    elif model == 'rsi':
        # talib.RSI returns an array; ensure conversion to Series if needed.
        rsi_values = talib.RSI(series.values, timeperiod=rolling_window)
        result = (rsi_values / 100 * 6) - 3 # 將rsi範圍從從[0,100] 轉換到 [-3,3]

    elif model == 'percentilerank': # 已經由percentile 變更為 percentilerank, 算式完全一樣
        result = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).rank(pct=True) * 2 - 1

    elif model == 'L2':
        calc_df['rolling_l2'] = np.sqrt(calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).apply(lambda x: np.sum(x**2), raw=True))
        result = calc_df['data'] / calc_df['rolling_l2']

    elif model == 'kurtosis':
        result = pandas_ta.kurtosis(calc_df['data'], length=rolling_window)

    elif model == 'skew':
        result = pandas_ta.skew(calc_df['data'], length=rolling_window)
    
    elif model == 'cci':
        high = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).max()
        low = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).min()
        result = pandas_ta.cci(high, low, calc_df['data'], length=rolling_window)

    elif model == 'Weirdroc':    # Old name roc to Weirdroc
        calc_df['shifted_series'] = calc_df['data'].shift(rolling_window)
        result = (calc_df['data'] - calc_df['shifted_series']) / calc_df['shifted_series']

    # Got other ROC from bq that is not inside the model.
    elif model == 'roc_ratio':
        calc_df['shifted_series'] = calc_df['data'].shift(rolling_window)
        result = (calc_df['data'] - calc_df['shifted_series']) / calc_df['data']

    elif model == 'pn':   # pn(percentile_norm)
        calc_df['25th_percentile'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).quantile(0.25)
        calc_df['75th_percentile'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).quantile(0.75)
        result = (calc_df['data'] - calc_df['25th_percentile']) / (calc_df['75th_percentile'] - calc_df['25th_percentile'])
    
    elif model == 'pn_epsilon':   # 舊名稱quantile 變更為 pn_epsilon 作為記錄, bq沒有記錄, 應該不會再使用
        epsilon = 1e-9
        calc_df['25th_percentile'] = calc_df['data'].rolling(window=rolling_window).quantile(0.25)
        calc_df['75th_percentile'] = calc_df['data'].rolling(window=rolling_window).quantile(0.75)
        result = (calc_df['data'] - calc_df['25th_percentile']) / (calc_df['75th_percentile'] - calc_df['25th_percentile'] + epsilon)

    elif model == 'momentum_old':   # Our Old Momentum
        momentum = series - series.shift(rolling_window)
        log_momentum = np.log10(np.abs(momentum) + 1) * np.sign(momentum)
        roll_log_mom = log_momentum.rolling(window=rolling_window, min_periods=rolling_window)
        result = log_momentum - roll_log_mom.mean()

    elif model == 'momentum':
        calc_df['momentum'] = calc_df['data'] - calc_df['data'].shift(rolling_window)
        calc_df['log10_momentum'] = np.log10(np.abs(calc_df['data']) + 1) * np.sign(calc_df['data'])
        result = calc_df['log10_momentum'] - calc_df['log10_momentum'].rolling(window=rolling_window, min_periods=rolling_window).mean()

    elif model == 'volatilityv0': #舊名稱 volatility 變成為 volatilityv0
        calc_df['volatility'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).std(ddof=0)
        result = (calc_df['data'] - calc_df['data'].shift(1)) / calc_df['volatility']
        
    elif model == 'psy':   # Still output [-3,3]
        # Compute up_days and use rolling mean
        calc_df['up_days'] = np.where(np.diff(calc_df['data'], prepend=series.iloc[0]) > 0, 1, 0)
        result = (calc_df['up_days'].rolling(window=rolling_window, min_periods=rolling_window)
                                   .mean() * 6) - 3

    elif model == 'winsor': 
        calc_df['lower_bound'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).quantile(0.05)
        calc_df['upper_bound'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).quantile(0.95)
        result = np.where(calc_df['data'] < calc_df['lower_bound'], 
                          calc_df['lower_bound'], 
                          np.where(calc_df['data'] > calc_df['upper_bound'], calc_df['upper_bound'], calc_df['data']))
        
    elif model == 'winsor_zscorev1': # 舊名 winsorized_zscore 變更為 winsor_zscorev1
        calc_df['lower_bound'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).quantile(0.05)
        calc_df['upper_bound'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).quantile(0.95)
        calc_df['winsorized'] = calc_df['data'].clip(lower=calc_df['lower_bound'], upper=calc_df['upper_bound'])
        calc_df['winsorized_mean'] = calc_df['winsorized'].rolling(window=rolling_window, min_periods=rolling_window).mean()
        calc_df['winsorized_std'] = calc_df['winsorized'].rolling(window=rolling_window, min_periods=rolling_window).std(ddof=0)
        result = (calc_df['winsorized'] - calc_df['winsorized_mean']) / calc_df['winsorized_std']

    elif model == 'sigmoid':
        calc_df['roll_mean'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).mean()
        calc_df['roll_std'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window).std(ddof=0)
        result = 2 / (1 + np.exp(-(series - calc_df['roll_mean']) / (calc_df['roll_std'] ))) - 1

    elif model == 'robust_zscore':  # 應該要拋棄這個, 理論上跟 madzscore 一樣, 但是因為運算上的分別, 導致得出的結果不同
        calc_df['rolling_median'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window, center=False).median()
        calc_df['rolling_mad'] = calc_df['data'].rolling(rolling_window, min_periods=rolling_window, center=False).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)

        result = 0.6745 * (calc_df['data'] - calc_df['rolling_median']) / calc_df['rolling_mad']

    elif model == 'tanh':
        calc_df['rolling_median'] = calc_df['data'].rolling(window=rolling_window, min_periods=rolling_window, center=False).median()
        calc_df['rolling_mad'] = calc_df['data'].rolling(rolling_window, min_periods=rolling_window, center=False).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)

        result = np.tanh((calc_df['data'] - calc_df['rolling_median']) / calc_df['rolling_mad'])
    
    elif model == 'linearregressionslope':
        def rolling_regression_slope(series: pd.Series, rolling_window: int):
            x = np.arange(rolling_window)
            x_mean = x.mean()
            denominator = np.sum((x - x_mean) ** 2)

            rolling_mean = series.rolling(rolling_window).mean()

            # Compute slope using rolling apply
            def compute_slope(window):
                if window.isna().any():
                    return np.nan
                y = window.values
                y_mean = y.mean()
                numerator = np.sum((x - x_mean) * (y - y_mean))
                return numerator / denominator

            slopes = series.rolling(rolling_window).apply(compute_slope, raw=False)

            return slopes

        result = rolling_regression_slope(calc_df['data'], rolling_window)

    elif model == 'chg':
        result = (calc_df['data'] - calc_df['data'].shift(rolling_window)) / rolling_window

    return result
