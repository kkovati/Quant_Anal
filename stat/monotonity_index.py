import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


ts = np.cumprod(np.random.normal(loc=0.0001, scale=0.01, size=1000) + 1)
monotonity_index = np.zeros_like(ts) + 0.5
monotonity_index_ema = np.zeros_like(ts) + 0.5
monotonity_integral = np.zeros_like(ts) + 0.5

lookback_period = 100
ema_factor = 0.05
integral_lookback_period = 35

# Calculate the monotonity index and its EMA
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

# Calculate monotonity integral
for i in range(len(monotonity_index)):
    monotonity_integral[i] = np.mean(monotonity_index[max(0, i - integral_lookback_period):i + 1])

# Plotting

plt.style.use('dark_background')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 9),
                                gridspec_kw={'height_ratios': [1, 1]})
fig.suptitle('Monotonicity Analysis', fontsize=16, fontweight='bold', color='white')

# --- Price subplot ---
ax1.plot(ts, label='Price', color='#00BFFF', linewidth=1.2)
ax1.set_title('Price', color='white')
ax1.set_ylabel('Price', color='white')
ax1.legend(facecolor='#222222', edgecolor='gray')
ax1.grid(True, which='major', color='#444444', linestyle='-',  linewidth=0.8, alpha=0.9)
ax1.grid(True, which='minor', color='#333333', linestyle='--', linewidth=0.5, alpha=0.6)
ax1.minorticks_on()
ax1.xaxis.set_major_locator(ticker.MultipleLocator(25))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax1.set_facecolor('#111111')
ax1.tick_params(colors='white')
for spine in ax1.spines.values():
    spine.set_edgecolor('#555555')

# --- Monotonicity subplot ---
ax2.plot(monotonity_index,     label='Monotonicity Index',    color='#FFA500', linewidth=1.2)
ax2.plot(monotonity_index_ema, label='Monotonicity EMA',      color='#FF4444', linewidth=1.5)
ax2.plot(monotonity_integral,  label='Monotonicity Integral', color='#00FF88', linewidth=1.5)

# Reference lines
ax2.axhline(0.5,  color='white',  linestyle='--', linewidth=0.8, alpha=0.6, label='Neutral (0.5)')
ax2.axhline(0.75, color='#00FF88', linestyle=':',  linewidth=0.8, alpha=0.6, label='Strong up (0.75)')
ax2.axhline(0.25, color='#FF6666', linestyle=':',  linewidth=0.8, alpha=0.6, label='Strong down (0.25)')

# Fill above/below neutral
ax2.fill_between(range(len(monotonity_index)), monotonity_index, 0.5,
                 where=(monotonity_index >= 0.5), alpha=0.15, color='#00FF88', label='Bullish zone')
ax2.fill_between(range(len(monotonity_index)), monotonity_index, 0.5,
                 where=(monotonity_index <  0.5), alpha=0.15, color='#FF4444', label='Bearish zone')

ax2.set_title('Monotonicity Index', color='white')
ax2.set_ylabel('Index Value', color='white')
ax2.set_xlabel('Time', color='white')
ax2.set_ylim(0, 1)
ax2.legend(facecolor='#222222', edgecolor='gray', ncol=2, fontsize=8)
ax2.grid(True, which='major', color='#444444', linestyle='-',  linewidth=0.8, alpha=0.9)
ax2.grid(True, which='minor', color='#333333', linestyle='--', linewidth=0.5, alpha=0.6)
ax2.minorticks_on()
ax2.xaxis.set_major_locator(ticker.MultipleLocator(25))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax2.set_facecolor('#111111')
ax2.tick_params(colors='white')
for spine in ax2.spines.values():
    spine.set_edgecolor('#555555')

plt.tight_layout()
plt.show()

