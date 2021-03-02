import numpy as np
import pandas as pd


# https://en.wikipedia.org/wiki/Technical_analysis

# https://en.wikipedia.org/wiki/Relative_strength_index
# https://en.wikipedia.org/wiki/Trix_(technical_analysis)

# https://en.wikipedia.org/wiki/MACD

# https://towardsdatascience.com/trading-toolbox-02-wma-ema-62c22205e2a9
# https://www.investopedia.com/terms/e/ema.asp
# https://en.wikipedia.org/wiki/Moving_average


def single_mov_avg(series, window):
    assert type(series) is pd.Series
    sma = series.rolling(window=window).mean()
    sma.name = series.name + f'_SMA{window}'
    return sma


def weighted_mov_avg(series, window):
    assert type(series) is pd.Series
    weights = np.arange(1, window + 1)
    wma = series.rolling(window=window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    wma.name = series.name + f'_WMA{window}'
    return wma


def exp_mov_avg(input, span):
    if type(input) is pd.Series:
        return exp_mov_avg_series(input, span)
    if type(input) is np.ndarray:
        return exp_mov_avg_ndarray(input, span)
    else:
        raise Exception('Wrong input type')


def exp_mov_avg_series(series, span):
    assert type(series) is pd.Series
    ema = series.ewm(span=span).mean()
    ema.name = f'EMA{span}({series.name})'
    return ema


def exp_mov_avg_ndarray(array, span):
    assert type(array) is np.ndarray
    assert array.ndim == 1
    assert len(array) > span * 1.5

    smoothing = 2
    q = smoothing / (1 + span)
    ema = np.zeros((len(array),))
    ema[0] = ema_yesterday = array[0]

    for i, value_today in enumerate(array[1:]):
        ema_today = value_today * q + ema_yesterday * (1 - q)
        ema[i + 1] = ema_today
        ema_yesterday = ema_today

    return ema


def _ema_test(series, span):
    assert type(series) is pd.Series
    smoothing = 2
    q = smoothing / (1 + span)
    ema = []
    ema_yesterday = 0

    for value_today in series:
        ema_today = value_today * q + ema_yesterday * (1 - q)
        ema.append(ema_today)
        ema_yesterday = ema_today

    ema = pd.Series(ema)
    ema.index = series.index
    ema.name = series.name + f'_EMA_test{span}'
    return ema


def add_multi_exp_mov_avg(dataframe, column, span_list):
    assert type(dataframe) is pd.DataFrame
    for span in span_list:
        ema = exp_mov_avg(dataframe[column], span)
        df[ema.name] = ema


if __name__ == '__main__':
    import plotly.express as px

    # Test 1
    df = pd.read_csv("../data/test_data/AAPL_240.csv").set_index("Date")
    ser = df[df.columns[0]]
    sma_5 = single_mov_avg(ser, 5)
    df[sma_5.name] = sma_5
    wma_5 = weighted_mov_avg(ser, 5)
    df[wma_5.name] = wma_5
    ema_5 = exp_mov_avg(ser, 5)
    df[ema_5.name] = ema_5
    ema_test_5 = _ema_test(ser, 5)
    df[ema_test_5.name] = ema_test_5
    fig = px.scatter(df).update_traces(mode='lines+markers').update_layout(title='Test1', hovermode="x unified").show()

    # Test 2
    df = pd.read_csv("../data/test_data/AAPL_240.csv").set_index("Date")
    ser = df[df.columns[0]]
    sma_30 = single_mov_avg(ser, 30)
    df[sma_30.name] = sma_30
    wma_30 = weighted_mov_avg(ser, 30)
    df[wma_30.name] = wma_30
    ema_30 = exp_mov_avg(ser, 30)
    df[ema_30.name] = ema_30
    ema_test_30 = _ema_test(ser, 30)
    df[ema_test_30.name] = ema_test_30
    fig = px.scatter(df).update_traces(mode='lines+markers').update_layout(title='Test2', hovermode="x unified").show()

    # Test 3
    df = pd.read_csv("../data/test_data/AAPL_240.csv").set_index("Date")
    sl = [2, 4, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40]
    add_multi_exp_mov_avg(df, df.columns[0], sl)
    fig = px.scatter(df).update_traces(mode='lines+markers').update_layout(title='Test3', hovermode="x unified").show()

    # Test 4
    df = pd.read_csv("../data/test_data/AAPL_240.csv").set_index("Date")
    ser = df[df.columns[0]]
    ema_30_series = exp_mov_avg(ser, 30)
    arr = ser.to_numpy()
    ema_30_ndarray = exp_mov_avg(arr, 30)
    fig = px.scatter(y=[ser, ema_30_series, ema_30_ndarray]).update_traces(mode='lines+markers').update_layout(title='Test4', hovermode="x unified").show()
    # FIXME - the two calculation methods differs
