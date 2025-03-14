## 2nd Month Update:
 - NanCheck
 - New Models (bq models,)(需要安裝pandas_ta, pip install pandas_ta)
 - New Entrys (trend_zero, trend_zero_fast)
 - 舊model 改名(ezscore -> ezscorev1, roc -> Weirdroc, quantile -> pn_epsilon, 舊sma/ema/mediandiffv2 -> smadiffv2_noabs)
 - 剩餘的model/entry 可以看config.py
## Python Module Installation

Assumed you already created a conda environment.
You can run these two line of command in cmd to install the required modules.

```python
pip install -r requirements.txt
conda install conda-forge::ta-lib
```
