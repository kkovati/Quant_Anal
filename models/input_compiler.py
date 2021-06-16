import pandas_datareader.data as web
import pandas as pd
import numpy as np
import talib
from talib import RSI, BBANDS, STOCHRSI, STOCHF, STOCH
import matplotlib.pyplot as plt

from dataset_generation.hsm_dataset import OPEN, HIGH, LOW, CLOSE
from dataset_generation.random_timeseries import generate_random_interval
from dataset_generation.standardize import standardize


# https://github.com/mrjbq7/ta-lib
# https://mrjbq7.github.io/ta-lib/install.html
# https://mrjbq7.github.io/ta-lib/doc_index.html

# https://towardsdatascience.com/trading-strategy-technical-analysis-with-python-ta-lib-3ce9d6ce5614

def tech_anal_input_compiler(pre_interval):
    pre_interval = standardize(pre_interval)
    open = pre_interval[OPEN]
    high = pre_interval[HIGH]
    low = pre_interval[LOW]
    close = pre_interval[CLOSE]

    indicators = []
    X = []

    indicators.append('OPEN')
    X.append(open[-1])
    indicators.append('HIGH')
    X.append(high[-1])
    indicators.append('LOW')
    X.append(low[-1])
    indicators.append('CLOSE')
    X.append(close[-1])

    # Overlap Studies Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/overlap_studies.html

    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    indicators.append('BBANDS_5_2_2_0_upper')
    X.append(upperband[-1])
    indicators.append('BBANDS_5_2_2_0_middle')
    X.append(middleband[-1])
    indicators.append('BBANDS_5_2_2_0_lower')
    X.append(lowerband[-1])

    real = talib.DEMA(close, timeperiod=30)
    indicators.append('DEMA_30')
    X.append(real[-1])

    for i in (3, 5, 10, 15, 20, 25, 30, 35, 40):
        real = talib.EMA(close, timeperiod=i)
        indicators.append(f'EMA_{i}')
        X.append(real[-1])

    real = talib.HT_TRENDLINE(close)
    indicators.append('HT_TRENDLINE')
    X.append(real[-1])

    real = talib.MA(close, timeperiod=30, matype=0)
    indicators.append('MA_30_0')
    X.append(real[-1])

    # Momentum Indicator Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/momentum_indicators.html

    real = talib.ADX(high, low, close, timeperiod=14)
    indicators.append('ADX_14')
    X.append(real[-1])

    real = talib.BOP(open, high, low, close)
    indicators.append('BOP')
    X.append(real[-1])

    real = talib.CCI(high, low, close, timeperiod=14)
    indicators.append('CCI_14')
    X.append(real[-1])

    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    indicators.append('MACD_12_26_9')
    X.append(macd[-1])
    indicators.append('MACDSIGNAL_12_26_9')
    X.append(macdsignal[-1])
    indicators.append('MACDHIST_12_26_9')
    X.append(macdhist[-1])

    real = talib.MOM(close, timeperiod=10)
    indicators.append('MOM_10')
    X.append(real[-1])

    real = talib.ROC(close, timeperiod=10)
    indicators.append('ROC_10')
    X.append(real[-1])

    real = talib.RSI(close, timeperiod=14)
    indicators.append('RSI_14')
    X.append(real[-1])

    slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3,
                               slowd_matype=0)
    indicators.append('STOCH_5_3_0_3_0_slowk')
    X.append(slowk[-1])
    indicators.append('STOCH_5_3_0_3_0_slowd')
    X.append(slowd[-1])

    fastk, fastd = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    indicators.append('STOCHRSI_14_5_3_0_fastk')
    X.append(fastk[-1])
    indicators.append('STOCHRSI_14_5_3_0_fastd')
    X.append(fastd[-1])

    real = talib.TRIX(close, timeperiod=30)
    indicators.append('TRIX_30')
    X.append(real[-1])

    real = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    indicators.append('ULTOSC_7_14_28')
    X.append(real[-1])

    real = talib.WILLR(high, low, close, timeperiod=14)
    indicators.append('WILLR_14')
    X.append(real[-1])

    # Volume Indicator Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/volume_indicators.html

    # TODO: Volume Indicator Functions

    # Price Transform Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/price_transform.html

    real = talib.AVGPRICE(open, high, low, close)
    indicators.append('AVGPRICE')
    X.append(real[-1])

    real = talib.MEDPRICE(high, low)
    indicators.append('MEDPRICE')
    X.append(real[-1])

    real = talib.WCLPRICE(high, low, close)
    indicators.append('WCLPRICE')
    X.append(real[-1])

    # Cycle Indicator Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/cycle_indicators.html

    sine, leadsine = talib.HT_SINE(close)
    indicators.append('HT_SINE_sine')
    X.append(sine[-1])
    indicators.append('WCLPRICE_leadsine')
    X.append(leadsine[-1])

    # Pattern Recognition Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/pattern_recognition.html

    integer = talib.CDLDOJI(open, high, low, close)
    indicators.append('CDLDOJI')
    X.append(float(integer[-1]))

    # Statistic Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/statistic_functions.html

    real = talib.BETA(high, low, timeperiod=5)
    indicators.append('BETA_5')
    X.append(real[-1])

    real = talib.VAR(close, timeperiod=5, nbdev=1)
    indicators.append('VAR_5_1')
    X.append(real[-1])

    # Math Transform Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/math_transform.html

    # Math Operator Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/math_operators.html

    # Check X
    for x, i in zip(X, indicators):
        assert isinstance(x, float), f'{i} is not float'
        assert isinstance(i, str), f'{i} is not string'

    X = np.array(X)

    # assert isinstance(X.dtype, np.float64)

    return X, indicators


if __name__ == '__main__':
    interval = generate_random_interval(length=200)

    X, indicators = tech_anal_input_compiler(interval)

    for x, i in zip(X, indicators):
        print(i, x)
