import numpy as np
import pandas as pd

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
