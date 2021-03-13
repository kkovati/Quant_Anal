import numpy as np
import pandas as pd


def calc_profit(buy_price, post_interval, stop_loss, take_profit):
    """
    Calculates the investment return ratio of ...
    Parameters
    ----------
    buy_price : float
        DESCRIPTION.
    post_series : pandas.Series
        DESCRIPTION.
    pre : int
        No. of days of the input interval for the prediction including
        prediction day
    post_interval : int
        No. of days
    stop_loss : float
        Stop when option price falls below stop_loss ratio in percentage, 0% < stop_loss < 100%
    take_profit : float
    Returns
    -------
    None.
    """
    assert type(post_interval) is np.ndarray
    assert post_interval.ndim == 1
    assert len(post_interval) > 1
    assert 0 < buy_price
    assert 0 < stop_loss < 100
    assert 100 < take_profit

    tp_price = buy_price * take_profit / 100
    curr_sl_price = buy_price * stop_loss / 100
    delta_price = buy_price - curr_sl_price

    for price in post_interval:
        if price >= tp_price:
            return tp_price, tp_price * 100 / buy_price
        if price <= curr_sl_price:
            return curr_sl_price, curr_sl_price * 100 / buy_price
        curr_sl_price = max(curr_sl_price, price - delta_price)
        last_price = price

    return last_price, last_price * 100 / buy_price


if __name__ == '__main__':
    # Test 1 - take profit
    post_ = np.array([101, 102, 99, 102, 110])
    sell_price, profit = calc_profit(buy_price=100, post_interval=post_, stop_loss=95, take_profit=105)
    print(sell_price, profit)

    # Test 2 - stop loss
    post_ = np.array([101, 102, 96, 102, 110])
    sell_price, profit = calc_profit(buy_price=100, post_interval=post_, stop_loss=95, take_profit=105)
    print(sell_price, profit)

    # # Test 3 - sell at the end
    post_ = np.array([101, 102, 102, 102, 103])
    sell_price, profit = calc_profit(buy_price=100, post_interval=post_, stop_loss=95, take_profit=105)
    print(sell_price, profit)
