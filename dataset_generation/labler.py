import pandas as pd


def investment_return(buy_price, post_dataframe, stop_loss):
    """
    Calculates the investment return ratio of ...
    Parameters
    ----------
    buy_price : float
        DESCRIPTION.
    post_dataframe : pandas.Series
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
    stop_loss
    assert 0 < stop_loss < 100

    curr_sl_price = buy_price * stop_loss / 100
    delta_price = buy_price - curr_sl_price
    print('buy_price: ', buy_price)
    print('curr_sl_price: ', curr_sl_price)

    for price in post_dataframe[post_dataframe.columns[0]]:
        print('----------')
        if price <= curr_sl_price:
            print('SELL')
            print('price:', price)
            return curr_sl_price, curr_sl_price / buy_price
        curr_sl_price = max(curr_sl_price, price - delta_price)
        last_price = price
        print('price:', price)
        print('curr_sl_price:', curr_sl_price)

    return last_price, last_price / buy_price


if __name__ == '__main__':
    # Test 1
    df = pd.read_csv("../test/test_data/AAPL_240.csv").set_index("Date")
    buy_p = df.iloc[9][0]
    post_df = df.iloc[10:20]

    investment_return(buy_price=buy_p, post_dataframe=post_df, stop_loss=95)
    # TODO testing seems to be fine, asserts must be inserted
    # print(investment_return(prices, 3, 6, stop_loss=0.1))
