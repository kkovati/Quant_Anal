import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from dataset_generation.hsm_dataset import OPEN, HIGH, LOW, CLOSE


def calc_profit(buy_price, post_interval, stop_loss, take_profit):
    """
    Calculates the investment return ratio of ...
    -------
    Parameters
    buy_price : float
        DESCRIPTION.
    post_interval : ndarray
        No. of days
    stop_loss : float
        Stop when option price falls below stop_loss ratio in percentage, 0% < stop_loss < 100%
    take_profit : float
    -------
    Returns
    profit: float
    """
    assert type(post_interval) is np.ndarray
    assert post_interval.ndim == 2
    assert post_interval.shape[0] == 4
    assert post_interval.shape[1] > 1
    assert post_interval.min() > 0
    assert 0 < buy_price
    assert 0 < stop_loss < 100
    assert 100 < take_profit

    tp_price = buy_price * take_profit / 100
    curr_sl_price = buy_price * stop_loss / 100
    delta_price = buy_price - curr_sl_price

    for high, low, close in zip(post_interval[HIGH], post_interval[LOW], post_interval[CLOSE]):
        # Pessimist calculation:
        # Assumes daily low goes lower than the updated stop loss (which calculated with the daily high)
        # before daily high goes higher than take profit
        if low <= curr_sl_price:
            return curr_sl_price * 100 / buy_price
        if high >= tp_price:
            return tp_price * 100 / buy_price

        # Update current stop loss price
        curr_sl_price = max(curr_sl_price, high - delta_price)

        # Check again if daily low was lower than updater stop loss
        # This check must come after take profit price check
        if low <= curr_sl_price:
            return curr_sl_price * 100 / buy_price

        last_price = close

    # Return profit
    return last_price * 100 / buy_price


def calc_trend():
    mu = .001
    sigma = .01
    start_price = 5

    n_timeseries = 5
    timeseries_length = 20

    y = []

    for i in range(n_timeseries):
        # np.random.seed(0)
        returns = np.random.normal(loc=mu, scale=sigma, size=timeseries_length)
        timeseries = start_price * (1 + returns).cumprod()
        timeseries -= timeseries[0]
        plt.plot(timeseries)
        y.append(timeseries)

    X = np.reshape(np.concatenate([np.arange(timeseries_length)] * n_timeseries), (-1, 1))
    y = np.reshape(np.concatenate(y), (-1, 1))

    reg = LinearRegression(fit_intercept=False).fit(X, y)

    print(reg.coef_)
    print(reg.intercept_)

    assert reg.coef_.shape == (1, 1)
    # assert reg.intercept_.shape == (1,)
    assert reg.intercept_ == 0

    approx_timeseries = (np.arange(timeseries_length) * reg.coef_[0, 0])  # + reg.intercept_[0]
    plt.plot(approx_timeseries)

    logging.getLogger('matplotlib.font_manager').disabled = True
    plt.show()


if __name__ == "__main__":
    calc_trend()
