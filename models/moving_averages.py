import numpy as np
import pandas as pd


# https://en.wikipedia.org/wiki/Technical_analysis

# https://en.wikipedia.org/wiki/Relative_strength_index
# https://en.wikipedia.org/wiki/Trix_(technical_analysis)

# https://en.wikipedia.org/wiki/MACD

# https://towardsdatascience.com/trading-toolbox-02-wma-ema-62c22205e2a9
# https://www.investopedia.com/terms/e/ema.asp
# https://en.wikipedia.org/wiki/Moving_average


def add_single_mov_avg(dataframe, window):
    column = dataframe.columns[0]
    sma = dataframe[column].rolling(window=window).mean()
    dataframe[f'SMA{window}'] = sma


def add_weighted_mov_avg(dataframe, window):
    column = dataframe.columns[0]
    weights = np.arange(1, window + 1)
    wma = dataframe[column].rolling(window=window).apply(lambda prices: np.dot(prices, weights) / weights.sum(),
                                                         raw=True)
    dataframe[f'WMA{window}'] = wma


def add_exp_mov_avg(dataframe, span):
    column = dataframe.columns[0]
    ema = dataframe[column].ewm(span=span).mean()
    dataframe[f'EMA{span}'] = ema


def _add_ema_test(dataframe, span):
    smoothing = 2
    q = smoothing / (1 + span)
    output = []
    ema_yesterday = 0

    for value_today in dataframe[dataframe.columns[0]]:
        ema_today = value_today * q + ema_yesterday * (1 - q)
        output.append(ema_today)
        ema_yesterday = ema_today

    dataframe[f'EMA_test{span}'] = pd.Series(output).values


if __name__ == '__main__':
    # Test 1
    df = pd.read_csv("../test/test_data/AAPL_240.csv").set_index("Date")
    add_single_mov_avg(df, 10)
    add_weighted_mov_avg(df, 10)
    add_exp_mov_avg(df, 10)
    _add_ema_test(df, 10)
    import plotly.express as px

    fig = px.scatter(df).update_traces(mode='lines+markers').update_layout(hovermode="x unified").show()
