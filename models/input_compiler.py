from logging import info as p
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import talib
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset_generation.hsm_dataset import generate_dataset
from dataset_generation.hsm_dataset import OPEN, HIGH, LOW, CLOSE
from dataset_generation.random_timeseries import generate_random_interval
from dataset_generation.standardize import standardize


# https://github.com/mrjbq7/ta-lib
# https://mrjbq7.github.io/ta-lib/install.html
# https://mrjbq7.github.io/ta-lib/doc_index.html

# https://towardsdatascience.com/trading-strategy-technical-analysis-with-python-ta-lib-3ce9d6ce5614

def compile_tech_anal_single_interval(pre_interval, names=False):
    assert isinstance(pre_interval, np.ndarray)
    assert pre_interval.ndim == 2
    assert pre_interval.shape[0] == 4

    # TODO: add option to disable name generation

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

    days = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 20, 25, 26, 30, 40, 50)

    # Daily Prices

    for i in range(-1, -8, -1):
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

    for i in days:
        upperband, middleband, lowerband = talib.BBANDS(close, timeperiod=i, nbdevup=2, nbdevdn=2, matype=0)
        # nbdevup/nbdevdn -> Deviation multiplier for upper/lower band
        name_ind_price_dim.append(f'BBANDS_{i}_2_2_0_upper')
        value_ind_price_dim.append(upperband[-1])
        name_ind_price_dim.append(f'BBANDS_{i}_2_2_0_middle')
        value_ind_price_dim.append(middleband[-1])
        name_ind_price_dim.append(f'BBANDS_{i}_2_2_0_lower')
        value_ind_price_dim.append(lowerband[-1])

    for i in days:
        real = talib.DEMA(close, timeperiod=i)
        name_ind_price_dim.append(f'DEMA_{i}')
        value_ind_price_dim.append(real[-1])

    for i in days:
        real = talib.EMA(close, timeperiod=i)
        name_ind_price_dim.append(f'EMA_{i}')
        value_ind_price_dim.append(real[-1])

    for i in days:
        real = talib.HT_TRENDLINE(close)
        name_ind_price_dim.append(f'HT_TRENDLINE_{i}')
        value_ind_price_dim.append(real[-1])

    for i in days:
        real = talib.KAMA(close, timeperiod=i)
        name_ind_price_dim.append(f'KAMA_{i}')
        value_ind_price_dim.append(real[-1])

    for i in days:
        real = talib.MA(close, timeperiod=i, matype=0)
        name_ind_price_dim.append(f'MA_{i}_0')
        value_ind_price_dim.append(real[-1])

    for i in days:
        real = talib.SMA(close, timeperiod=i)
        name_ind_price_dim.append(f'SMA_{i}')
        value_ind_price_dim.append(real[-1])

    for i in days:
        real = talib.TEMA(close, timeperiod=i)
        name_ind_price_dim.append(f'TEMA_{i}')
        value_ind_price_dim.append(real[-1])

    for i in days:
        real = talib.TRIMA(close, timeperiod=i)
        name_ind_price_dim.append(f'TRIMA_{i}')
        value_ind_price_dim.append(real[-1])

    for i in days:
        real = talib.WMA(close, timeperiod=i)
        name_ind_price_dim.append(f'WMA_{i}')
        value_ind_price_dim.append(real[-1])

    # Calculate deltas of indicators with price dimension
    assert len(name_ind_price_dim) == len(value_ind_price_dim)
    original_length = len(name_ind_price_dim)
    for i in range(original_length):
        for j in range(i + 1, original_length):
            name_ind_price_dim.append(f'{name_ind_price_dim[i]}-{name_ind_price_dim[j]}')
            value_ind_price_dim.append(value_ind_price_dim[i] - value_ind_price_dim[j])

    # Momentum Indicator Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/momentum_indicators.html

    for i in days:
        real = talib.ADX(high, low, close, timeperiod=i)
        name_ind_dimless.append(f'ADX_{i}')
        value_ind_dimless.append(real[-1])

    for i in range(len(days)):
        for j in range(i + 1, len(days)):
            real = talib.APO(close, fastperiod=days[i], slowperiod=days[j], matype=0)
            name_ind_dimless.append(f'APO_{i}_{j}')
            value_ind_dimless.append(real[-1])

    for i in days:
        aroondown, aroonup = talib.AROON(high, low, timeperiod=i)
        name_ind_dimless.append(f'AROON_DOWN_{i}')
        value_ind_dimless.append(aroondown[-1])
        name_ind_dimless.append(f'AROON_UP_{i}')
        value_ind_dimless.append(aroonup[-1])
        name_ind_dimless.append(f'AROON_UP_{i}-AROON_DOWN_{i}')
        value_ind_dimless.append(aroonup[-1] - aroondown[-1])
        # TODO: crossover?

    real = talib.BOP(open, high, low, close)
    name_ind_dimless.append('BOP')
    value_ind_dimless.append(real[-1])

    for i in days:
        real = talib.CCI(high, low, close, timeperiod=i)
        name_ind_dimless.append(f'CCI_{i}')
        value_ind_dimless.append(real[-1])

    for i in range(len(days)):
        for j in range(i + 1, len(days)):
            for k in range(j + 1, len(days)):
                macd, macdsignal, macdhist = talib.MACD(close, fastperiod=days[j], slowperiod=days[k],
                                                        signalperiod=days[i])
                name_ind_dimless.append(f'MACD_{days[j]}_{days[k]}_{days[i]}')
                value_ind_dimless.append(macd[-1])
                name_ind_dimless.append(f'MACDSIGNAL_{days[j]}_{days[k]}_{days[i]}')
                value_ind_dimless.append(macdsignal[-1])
                name_ind_dimless.append(f'MACDHIST_{days[j]}_{days[k]}_{days[i]}')
                value_ind_dimless.append(macdhist[-1])

    for i in days:
        real = talib.MOM(close, timeperiod=i)
        name_ind_dimless.append(f'MOM_{i}')
        value_ind_dimless.append(real[-1])

    for i in days:
        real = talib.ROC(close, timeperiod=i)
        name_ind_dimless.append(f'ROC_{i}')
        value_ind_dimless.append(real[-1])

    for i in days:
        real = talib.RSI(close, timeperiod=i)
        name_ind_dimless.append(f'RSI_{i}')
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

    for i in days:
        real = talib.TRIX(close, timeperiod=i)
        name_ind_dimless.append(f'TRIX_{i}')
        value_ind_dimless.append(real[-1])

    for i in range(len(days)):
        for j in range(i + 1, len(days)):
            for k in range(k + 1, len(days)):
                real = talib.ULTOSC(high, low, close, timeperiod1=i, timeperiod2=j, timeperiod3=k)
                name_ind_dimless.append(f'ULTOSC_{i}_{j}_{k}')
                value_ind_dimless.append(real[-1])

    for i in days:
        real = talib.WILLR(high, low, close, timeperiod=i)
        name_ind_dimless.append(f'WILLR_{i}')
        value_ind_dimless.append(real[-1])

    # Volume Indicator Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/volume_indicators.html

    # TODO: Volume Indicator Functions

    # Price Transform Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/price_transform.html

    # real = talib.AVGPRICE(open, high, low, close)
    # indicators.append('AVGPRICE')
    # X.append(real[-1])
    #
    # real = talib.MEDPRICE(high, low)
    # indicators.append('MEDPRICE')
    # X.append(real[-1])
    #
    # real = TYPPRICE(high, low, close)
    #
    # real = talib.WCLPRICE(high, low, close)
    # indicators.append('WCLPRICE')
    # X.append(real[-1])

    # Cycle Indicator Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/cycle_indicators.html

    # sine, leadsine = talib.HT_SINE(close)
    # indicators.append('HT_SINE_sine')
    # X.append(sine[-1])
    # indicators.append('WCLPRICE_leadsine')
    # X.append(leadsine[-1])

    # Pattern Recognition Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/pattern_recognition.html

    # integer = talib.CDLDOJI(open, high, low, close)
    # indicators.append('CDLDOJI')
    # X.append(float(integer[-1]))

    # Statistic Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/statistic_functions.html

    # real = talib.BETA(high, low, timeperiod=5)
    # indicators.append('BETA_5')
    # X.append(real[-1])
    #
    # real = talib.VAR(close, timeperiod=5, nbdev=1)
    # indicators.append('VAR_5_1')
    # X.append(real[-1])

    # Math Transform Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/math_transform.html

    # Math Operator Functions
    # https://mrjbq7.github.io/ta-lib/func_groups/math_operators.html

    # Check indicators with price dimension
    for name, value in zip(name_ind_price_dim, value_ind_price_dim):
        assert isinstance(name, str), f'{i} is not string'
        assert isinstance(value, float), f'{i} is not float'

    # Check dimensionless indicators
    for name, value in zip(name_ind_dimless, value_ind_dimless):
        assert isinstance(name, str), f'{i} is not string'
        assert isinstance(value, float), f'{i} is not float'

    value_ind = np.array(value_ind_price_dim + value_ind_dimless)
    name_ind = name_ind_price_dim + name_ind_dimless

    if names:
        return value_ind, name_ind
    else:
        return value_ind


def compile_dataset(X_pre_interval):
    assert isinstance(X_pre_interval, np.ndarray)
    assert X_pre_interval.ndim == 3
    assert X_pre_interval.shape[1] == 4

    ind, names = compile_tech_anal_single_interval(X_pre_interval[0], names=True)

    X_indicators = np.zeros((X_pre_interval.shape[0], ind.shape[0]))

    p('Compile input')
    for i, pre_interval in enumerate(tqdm(X_pre_interval)):
        X_indicators[i] = compile_tech_anal_single_interval(pre_interval, names=False)

    return X_indicators


if __name__ == '__main__':
    # TEST 1
    interval = generate_random_interval(length=200)
    ind, names = compile_tech_anal_single_interval(interval, names=True)

    # TEST 2
    pre_interval_train, post_interval_train, _, _ = generate_dataset(100, 1, 61, 20, debug=True)
    X_ind = compile_dataset(pre_interval_train)
