import matplotlib.pyplot as plt
import numpy as np


ts = np.cumprod(np.random.normal(loc=0.0001, scale=0.01, size=1000) + 1)
monotonity_index = np.zeros_like(ts)
monotonity_index_ema = np.zeros_like(ts)

lookback_period = 100
ema_factor = 0.05

for i in range(len(monotonity_index)):
    if i <= lookback_period:
        continue
    target_price = ts[i]
    win_count = 0
    for j in range(i - lookback_period, i):
        if ts[j] < target_price:
            win_count += 1
    monotonity_index[i] = win_count / lookback_period
    monotonity_index_ema[i] = monotonity_index[i] * ema_factor + monotonity_index_ema[i - 1] * (1 - ema_factor)

plt.plot(ts)
plt.plot(monotonity_index)
plt.plot(monotonity_index_ema)
plt.show()

