ALL_MODELS = ['zscore', 'minmaxscaling', 'smadiffv2', 'emadiffv2', 'momentum', 'volatility', 'robustscaler', 'percentile', 
              'maxabs', 'meannorm', 'roc', 'rsi', 'psy', 'srsi', 'mad', 'quantile', 'winsorized_zscore', 'sigmoid',
              'robust_zscore', 'tanh', 'slope']

ALL_ENTRYS = ['trend', 'trend_reverse', 'mr', 'mr_reverse', 'fast', 'trend_emaFilter', 'fast_emaFilter', 
              'trend_long', 'trend_reverse_long', 'mr_long', 'mr_reverse_long', 'fast_long', 'trend_long_emaFilter', 'fast_long_emaFilter', 
              'trend_short', 'trend_reverse_short', 'mr_short', 'mr_reverse_short', 'fast_short', 'trend_short_emaFilter', 'fast_short_emaFilter']

candle_timeframe = '1h'
candle_delay = -5

model = 'winsorized_zscore'
entry = 'trend_long_emaFilter'

alpha_id='nfXDXD'
factor = 'long_liquidations'
factor2 = 'long_liquidations'
interval = '1h'
operation = 'none'
preprocess = 'diff'

window=65
threshold=2

save_plot = False

candle_file = f"./data/resample_bybit_btc_{interval}.csv"
factor_file = f"./data/cryptoquant_btc_liquidations_{interval}.csv"
factor2_file = f"./data/cryptoquant_btc_liquidations_{interval}.csv"
