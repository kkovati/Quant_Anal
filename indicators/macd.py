import numpy as np
import pandas as pd

import moving_averages as mavg


def subtract_series(series1, series2):
    diff_series = series1.subtract(series2)
    diff_series.name = series1.name + '-' + series2.name
    return diff_series


def extra_macd(series, ema_spans, diff_spans):
    assert type(series) is pd.Series
    assert len(series) > (max(ema_spans) + max(diff_spans)) * 1.2
    assert type(ema_spans) is list
    assert min(ema_spans) > 1
    assert type(diff_spans) is list
    assert min(diff_spans) > 1

    # ema_spans must be sorted incrementally b/c calculating differences want to be aligned with macd calculation
    ema_spans.sort()
    diff_spans.sort()

    simple_ema_series = []

    for span in ema_spans:
        simple_ema_series.append(mavg.exp_mov_avg(series, span))

    diff_series = []

    for i in range(len(simple_ema_series) - 1):
        for j in range(i + 1, len(simple_ema_series)):
            diff_series.append(subtract_series(simple_ema_series[i], simple_ema_series[j]))

    assert len(diff_series) == len(simple_ema_series) * (len(simple_ema_series) - 1) / 2

    ema_diff_series = []

    for span in diff_spans:
        for de in diff_series:
            ema_diff_series.append(mavg.exp_mov_avg(de, span))

    # Create Dataframe
    dataframe = pd.DataFrame()
    dataframe[series.name] = series
    for s in simple_ema_series + diff_series + ema_diff_series:
        dataframe[s.name] = s

    return dataframe


if __name__ == '__main__':
    es = [12, 26]
    ds = [9]

    # Test 1
    df = pd.read_csv("../data/test_data/AAPL_240.csv").set_index("Date")
    ser = df[df.columns[0]]

    df = extra_macd(series=ser, ema_spans=es, diff_spans=ds)

    import plotly.express as px
    fig = px.scatter(df).update_traces(mode='lines+markers').update_layout(title='Test1', hovermode="x unified").show()
