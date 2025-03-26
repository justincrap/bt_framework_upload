# Datasource
datasource = 'cryptoquant'
category = 'market-data'
endpoint_name = 'coinbase-premium-index'
data_exchange = 'none' # # all_exchange / binance / none

models = [
    'zscore', 
    'ezscore', 
    'ezscorev1', 
    'madzscore', 
    'robustscaler', 
    'minmaxscaling', 
    'meannorm', 
    'maxabs', 
    # 'smadiffv2_noabs', 
    'smadiffv2', 
    # 'emadiffv2_noabs', 
    'emadiffv2', 
    # 'mediandiffv2_noabs', 
    'mediandiffv2', 
    'smadiffv3', 
    'srsi', 
    'ersi', 
    'srsiv2', 
    'ersiv2', 
    'rsi', 
    'rvi', 
    'percentilerank', 
    'L2', 
    'kurtosis', 
    'skew', 
    'cci', 
    'Weirdroc', 
    'roc_ratio', 
    'pn', 
    'pn_epsilon', 
    'momentum_old', 
    # 'momentum', 
    'volatilityv0', 
    'psy', 
    'winsor', 
    'winsor_zscorev1', 
    'sigmoid', 
    # 'robust_zscore', 
    # 'tanh', 
    # 'linearregressionslope', 
    'chg'
]

entrys = [
    "trend",
    "trend_reverse",
    "mr",
    "mr_reverse",
    "trend_long",
    "trend_short",
    "trend_reverse_long",
    "trend_reverse_short",
    "mr_long",
    "mr_short",
    "mr_reverse_long",
    "mr_reverse_short",
    "fast",
    "fast_long",
    "fast_short",
    "fast_reverse",
    "fast_reverse_long",
    "fast_reverse_short",
    "trend_zero",
    "trend_zero_long",
    "trend_zero_short",
    "trend_zero_fast",
    "trend_zero_fast_long",
    "trend_zero_fast_short"
]

model = 'smadiffv2' # models[-1]
entry = 'trend' # entrys[-1]

# 用於Resample candle data
candle_exchange='bybit'
shift_candle_minite = 5

USE_ALL_MODELS = False
symbol="BTC"
alpha_id='nf010_'
factor = 'coinbase_premium_gap'
factor2 = ''
factor_name = f'{factor}_001'
interval = '1h'
operation = 'none'
preprocess = 'direct'  # 可以輸入單個或多個, 單個的例子: 'direct', 多個的例子: ['direct', 'diff']
nan_perc = 0.03
zero_perc = 0.3

# 用於Heatmap Loop
window_end = 351
window_step = 20
threshold_end = 4.01
threshold_step = 0.2

save_plot = False
# Plot 指定heatmap用
window=65
threshold=2.2
highlight_window = window
highlight_threshold = threshold

candle_file = f"./data/resample_{candle_exchange}_{symbol.lower()}_{interval}_-{shift_candle_minite}m.csv"
factor_file = f"./data/cryptoquant_{symbol.lower()}_coinbase-premium-index_{interval}.csv"
factor2_file = f"" #./data/cryptoquant_{coin}_coinbase-premium-index_{interval}.csv