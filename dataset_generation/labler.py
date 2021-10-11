import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from dataset_generation.hsm_dataset import OPEN, HIGH, LOW, CLOSE
from dataset_generation.random_timeseries import generate_random_interval
from dataset_generation.standardize import standardize


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
    assert 0 < stop_loss < 1
    assert 1 < take_profit

    tp_price = buy_price * take_profit
    sl_price = buy_price * stop_loss

    for high, low, close in zip(post_interval[HIGH], post_interval[LOW], post_interval[CLOSE]):
        if low <= sl_price and high >= tp_price:
            return (sl_price + tp_price) / 2 / buy_price
        if low <= sl_price:
            return sl_price / buy_price
        if high >= tp_price:
            return tp_price / buy_price

        last_price = close

    return last_price / buy_price


def calc_profit_with_trailing_loss(buy_price, post_interval, stop_loss, take_profit):
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
    assert 0 < stop_loss < 1
    assert 1 < take_profit

    tp_price = buy_price * take_profit
    curr_sl_price = buy_price * stop_loss
    delta_price = buy_price - curr_sl_price

    for high, low, close in zip(post_interval[HIGH], post_interval[LOW], post_interval[CLOSE]):
        # Pessimist calculation:
        # Assumes daily low goes lower than the updated stop loss (which calculated with the daily high)
        # before daily high goes higher than take profit
        if low <= curr_sl_price:
            return curr_sl_price / buy_price
        if high >= tp_price:
            return tp_price / buy_price

        # Update current stop loss price
        curr_sl_price = max(curr_sl_price, high - delta_price)

        # Check again if daily low was lower than updater stop loss
        # This check must come after take profit price check
        if low <= curr_sl_price:
            return curr_sl_price / buy_price

        last_price = close

    # Return profit
    return last_price / buy_price


def calc_trend(pre_interval, post_interval, debug=False):
    interval = np.concatenate((np.copy(pre_interval), np.copy(post_interval)), axis=1)
    assert interval.shape[0] == 4

    interval_std = standardize(interval)
    buy_price = interval_std[CLOSE, pre_interval.shape[1] - 1]
    post_interval_std = interval_std[:, pre_interval.shape[1]:]
    post_interval_std -= buy_price

    X = np.reshape(np.concatenate([np.arange(post_interval.shape[1])] * 4), (-1, 1))
    y = np.reshape(post_interval_std, (-1, 1))

    reg = LinearRegression(fit_intercept=False).fit(X, y)

    assert reg.coef_.shape == (1, 1)
    assert reg.intercept_ == 0

    if debug:
        return reg.coef_[0, 0], post_interval_std
    else:
        return reg.coef_[0, 0]

    
# TODO: check how the previous functions are applied to full datasets?
def calc_min_max(y_post_interval):
    assert isinstance(y_post_interval, np.ndarray)
    assert y_post_interval.ndim == 3
    assert y_post_interval.shape[1] == 4
    
    y_min = np.min(y_post_interval, axis=(1, 2))
    y_max = np.max(y_post_interval, axis=(1, 2))
    
    assert y_min == np.min(y_post_interval[:, LOW, :], axis=1)
    assert y_max == np.max(y_post_interval[:, HIGH, :], axis=1)

    return y_min, y_max
    

if __name__ == "__main__":
    # TODO: test calc_profit
    # TODO: test calc_min_max

    pre_length = 150
    post_length = 20
    interval = generate_random_interval(length=pre_length + post_length)
    pre_interval = interval[:, :pre_length]
    post_interval = interval[:, pre_length:]

    trend, post_interval_std = calc_trend(pre_interval, post_interval, debug=True)

    trendline = np.arange(post_length) * trend

    from plotly.graph_objects import Candlestick, Figure
    from plotly.offline import plot

    candlestick = Candlestick(x=np.arange(post_length),
                              open=post_interval_std[OPEN],
                              high=post_interval_std[HIGH],
                              low=post_interval_std[LOW],
                              close=post_interval_std[CLOSE])
    figure = Figure(data=[candlestick])
    figure.update_layout(title=f'Trend: {trend:.3f}', yaxis_title='Price')
    figure.add_shape(type="line", x0=0, y0=0, x1=post_length - 1, y1=trendline[-1],
                     line=dict(color='blue', width=2, dash="dot"))
    plot(figure)
