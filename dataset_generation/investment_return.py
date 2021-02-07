import pandas as pd


def investment_return(prices, pre, post, sl):
    """    
    Calculates the investment return ratio of ...
    Parameters
    ----------
    prices : pandas.Series
        DESCRIPTION.
    pre : int
        No. of days of the input interval for the prediction including 
        prediction day
    post : int
        No. of days 
    sl : int
        Stop loss ratio, 0% < sl < 100%
    Returns
    -------
    None.

    """
    assert len(prices) == pre + post
    assert 0 < sl < 1

    buy_price = last_price = prices.iloc[pre - 1]
    curr_sl_price = buy_price * (1 - sl)
    print('buy_price: ', buy_price)
    print('curr_sl_price: ', curr_sl_price)

    for price in prices.iloc[pre:]:
        if price <= curr_sl_price:
            return (curr_sl_price, curr_sl_price / buy_price)
        if price > last_price:
            curr_sl_price = price * (1 - sl)
        last_price = price
        print('price:', price)
        print('curr_sl_price:', curr_sl_price)

    return (last_price, last_price / buy_price)


if __name__ == '__main__':
    # Test 1
    data = [136, 134, 133, 130, 135, 140, 142, 140, 10]
    index = [24, 25, 26, 27, 28, 29, 30, 31, 32]
    prices = pd.Series(data=data, index=index, dtype='float64', name='AAPL')
    print(prices)

    print(investment_return(prices, 3, 6, sl=0.1))
