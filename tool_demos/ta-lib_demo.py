import pandas_datareader.data as web
import pandas as pd
import numpy as np
from talib import RSI, BBANDS, STOCHRSI, STOCHF, STOCH
import matplotlib.pyplot as plt


# https://github.com/mrjbq7/ta-lib
# https://mrjbq7.github.io/ta-lib/install.html

# https://towardsdatascience.com/trading-strategy-technical-analysis-with-python-ta-lib-3ce9d6ce5614

def bbp(price):
    up, mid, low = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    bbp = (price['AdjClose'] - low) / (up - low)
    return bbp


if __name__ == '__main__':
    c = np.random.randn(100) + 2
    k, d = STOCHRSI(c)
    rsi = RSI(c)
    k, d = STOCHF(rsi, rsi, rsi)
    rsi = RSI(c)
    k, d = STOCH(rsi, rsi, rsi)

    #############x

    start = '2015-04-22'
    end = '2017-04-22'

    symbol = 'MCD'
    max_holding = 100
    price = web.DataReader(name=symbol, data_source='quandl', start=start, end=end)
    price = price.iloc[::-1]
    price = price.dropna()
    close = price['AdjClose'].values
    up, mid, low = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    rsi = RSI(close, timeperiod=14)
    print("RSI (first 10 elements)\n", rsi[14:24])
