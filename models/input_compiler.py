import pandas_datareader.data as web
import pandas as pd
import numpy as np
import talib
from talib import RSI, BBANDS, STOCHRSI, STOCHF, STOCH
import matplotlib.pyplot as plt

from dataset_generation.hsm_dataset import OPEN, HIGH, LOW, CLOSE
from dataset_generation.standardize import standardize


# https://github.com/mrjbq7/ta-lib
# https://mrjbq7.github.io/ta-lib/install.html

# https://towardsdatascience.com/trading-strategy-technical-analysis-with-python-ta-lib-3ce9d6ce5614

def calculate_tech_anal_array(pre_interval):
    # TODO:volume
    pre_interval = standardize(pre_interval)
    open = pre_interval[OPEN]
    high = pre_interval[HIGH]
    low = pre_interval[LOW]
    close = pre_interval[CLOSE]

    indicators = []
    X = []

    real = talib.ADX(high, low, close, timeperiod=14)
    indicators.append('ADX_14')
    X.append(real)

    real = talib.CCI(high, low, close, timeperiod=14)
    indicators.append('CCI_14')
    X.append(real)

    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    indicators.append('MACD_12_26_9')
    X.append(macd)
    indicators.append('MACDSIGNAL_12_26_9')
    X.append(macdsignal)
    indicators.append('MACDHIST_12_26_9')
    X.append(macdhist)

    real = talib.MOM(close, timeperiod=10)
    indicators.append('MOM_10')
    X.append(real)

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
