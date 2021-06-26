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
    pre_interval = standardize(np.copy(pre_interval))
    open = pre_interval[OPEN]
    high = pre_interval[HIGH]
    low = pre_interval[LOW]
    close = pre_interval[CLOSE]

    # Indicators with Price Dimension
    name_ind_price_dim = []
    value_ind_price_dim = []

    # Dimensionless Indicators
    name_ind_dimless = []
    value_ind_dimless = []

    days = (3, 4, 5, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50)

    # Daily Prices
    for i in range(-1, -6, -1):
        name_ind_price_dim.append(f'OPEN_{i}')
        value_ind_price_dim.append(open[i])
        name_ind_price_dim.append(f'HIGH_{i}')
        value_ind_price_dim.append(high[i])
        name_ind_price_dim.append(f'LOW_{i}')
        value_ind_price_dim.append(low[i])
        name_ind_price_dim.append(f'CLOSE_{i}')
        value_ind_price_dim.append(close[i])

    # Overlap Studies Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/overlap_studies.html

    upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    # nbdevup/nbdevdn -> Deviation multiplier for upper/lower band
    name_ind_price_dim.append('BBANDS_5_2_2_0_upper')
    value_ind_price_dim.append(upperband[-1])
    name_ind_price_dim.append('BBANDS_5_2_2_0_middle')
    value_ind_price_dim.append(middleband[-1])
    name_ind_price_dim.append('BBANDS_5_2_2_0_lower')
    value_ind_price_dim.append(lowerband[-1])

    real = talib.DEMA(close, timeperiod=30)
    name_ind_price_dim.append('DEMA_30')
    value_ind_price_dim.append(real[-1])

    for i in days:
        real = talib.EMA(close, timeperiod=i)
        name_ind_price_dim.append(f'EMA_{i}')
        value_ind_price_dim.append(real[-1])

    real = talib.HT_TRENDLINE(close)
    name_ind_price_dim.append('HT_TRENDLINE')
    value_ind_price_dim.append(real[-1])

    real = talib.KAMA(close, timeperiod=30)
    name_ind_price_dim.append('KAMA_30')
    value_ind_price_dim.append(real[-1])

    real = talib.MA(close, timeperiod=30, matype=0)
    name_ind_price_dim.append('MA_30_0')
    value_ind_price_dim.append(real[-1])

    real = talib.SMA(close, timeperiod=30)
    name_ind_price_dim.append('SMA_30')
    value_ind_price_dim.append(real[-1])

    # Momentum Indicator Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/momentum_indicators.html

    real = talib.ADX(high, low, close, timeperiod=14)
    name_ind_dimless.append('ADX_14')
    value_ind_dimless.append(real[-1])

    aroondown, aroonup = talib.AROON(high, low, timeperiod=14)
    name_ind_dimless.append('AROON_14')
    value_ind_dimless.append(real[-1])
    # TODO: AARON UP and DOWN difference must be present (often 25 period) + crossover?

    real = talib.BOP(open, high, low, close)
    name_ind_dimless.append('BOP')
    value_ind_dimless.append(real[-1])

    real = talib.CCI(high, low, close, timeperiod=14)
    name_ind_dimless.append('CCI_14')
    value_ind_dimless.append(real[-1])

    macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    name_ind_dimless.append('MACD_12_26_9')
    value_ind_dimless.append(macd[-1])
    name_ind_dimless.append('MACDSIGNAL_12_26_9')
    value_ind_dimless.append(macdsignal[-1])
    name_ind_dimless.append('MACDHIST_12_26_9')
    value_ind_dimless.append(macdhist[-1])

    real = talib.MOM(close, timeperiod=10)
    name_ind_dimless.append('MOM_10')
    value_ind_dimless.append(real[-1])

    real = talib.ROC(close, timeperiod=10)
    name_ind_dimless.append('ROC_10')
    value_ind_dimless.append(real[-1])

    real = talib.RSI(close, timeperiod=14)
    name_ind_dimless.append('RSI_14')
    value_ind_dimless.append(real[-1])

    slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3,
                               slowd_matype=0)
    name_ind_dimless.append('STOCH_5_3_0_3_0_slowk')
    value_ind_dimless.append(slowk[-1])
    name_ind_dimless.append('STOCH_5_3_0_3_0_slowd')
    value_ind_dimless.append(slowd[-1])

    fastk, fastd = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    name_ind_dimless.append('STOCHRSI_14_5_3_0_fastk')
    value_ind_dimless.append(fastk[-1])
    name_ind_dimless.append('STOCHRSI_14_5_3_0_fastd')
    value_ind_dimless.append(fastd[-1])

    real = talib.TRIX(close, timeperiod=30)
    name_ind_dimless.append('TRIX_30')
    value_ind_dimless.append(real[-1])

    real = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    name_ind_dimless.append('ULTOSC_7_14_28')
    value_ind_dimless.append(real[-1])

    real = talib.WILLR(high, low, close, timeperiod=14)
    name_ind_dimless.append('WILLR_14')
    value_ind_dimless.append(real[-1])

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