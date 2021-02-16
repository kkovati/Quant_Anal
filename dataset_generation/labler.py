import pandas as pd


def calc_profit(buy_price, post_series, stop_loss):
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
    post : int
        No. of days
    stop_loss : int
        Stop when option price falls below stop_loss ratio in percentage, 0% < stop_loss < 100%
    Returns
    -------
    None.
    """
    assert type(post_series) is pd.core.series.Series
    assert 0 < buy_price
    assert 0 < stop_loss < 100

    curr_sl_price = buy_price * stop_loss / 100
    delta_price = buy_price - curr_sl_price

    for price in post_series:
        if price <= curr_sl_price:
            return curr_sl_price, curr_sl_price / buy_price
        curr_sl_price = max(curr_sl_price, price - delta_price)
        last_price = price

    return last_price, last_price / buy_price


if __name__ == '__main__':
    # Test 1
    df = pd.read_csv("../data/test_data/AAPL_240.csv").set_index("Date")

    interval_series = df['AAPL'].iloc[:20]  # Output of sampler.sample_interval()
    buy_p = interval_series.iloc[9]
    post_ser = interval_series[10:20]

    sell_price, inv_ret = calc_profit(buy_price=buy_p, post_series=post_ser, stop_loss=95)
    assert sell_price == 73.473342964
    assert inv_ret == 0.9524080627080262
    print("Test1 OK")
