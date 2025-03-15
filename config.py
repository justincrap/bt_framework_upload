ALL_MODELS = [
    "zscore",
    "ezscore",
    "ezscorev1",
    "madzscore",
    "robustscaler",
    "minmaxscaling",
    "meannorm",
    "maxabs",
    # "smadiffv2_noabs",
    "smadiffv2",
    # "emadiffv2_noabs",
    "emadiffv2",
    # "mediandiffv2_noabs",
    "mediandiffv2",
    "mad",
    "srsi",
    "ersi",
    "srsiv2",
    "ersiv2",
    "rsi",
    "rvi",
    "percentile",
    "L2",
    "kurtosis",
    "skew",
    "cci",
    "Weirdroc",
    "roc_ratio",
    "pn",
    "pn_epsilon",
    # "momentum_old",
    "momentum",
    "volatility",
    "psy",
    "winsor",
    "winsorized_zscore",
    "sigmoid",
    "quantile",
    "robust_zscore",
    "tanh",
    "chg"
]

ALL_ENTRYS = [
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

# ALL_MODELS = ['zscore', 'smadiffv2']
# ALL_ENTRYS = ['trend', 'mr']

# 用於Resample candle data
candle_timeframe = '1h'
candle_delay = 15
exchange_name='binance'
coin='eth'

USE_ALL_MODELS = True
alpha_id='nf010_'
factor = 'coinbase_premium_index'
factor2 = 'coinbase_premium_gap'
interval = '1h'
operation = 'none'
preprocess = 'diff'  # 可以輸入單個或多個, 單個的例子: 'direct', 多個的例子: ['direct', 'diff']

model = 'zscore'
entry = 'trend'
window=265
threshold=1.6
output_csv_full_time=False
save_plot = False

candle_file = f"./data/resample_{exchange_name}_{coin}_{interval}_-{candle_delay}m.csv"
factor_file = f"./data/cryptoquant_{coin}_coinbase-premium-index_{interval}.csv"
factor2_file = f"./data/cryptoquant_{coin}_coinbase-premium-index_{interval}.csv"

symbol=coin.upper()
factor_name = f'{factor}_001'
# for endpoint
datasource = 'cryptoquant'
category = 'market-data'
endpoint_name = 'coinbase-premium-index'
exchange = 'none' # # all_exchange / binance / none
