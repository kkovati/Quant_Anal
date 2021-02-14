import numpy as np
import pandas as pd

import labler
import sampler


def profit_distribution(dataframe, post, stop_loss, levels, n_sample=1000, plot=False):
    assert type(dataframe) is pd.core.frame.DataFrame
    assert len(levels) >= 3  # FIXME 1 is enough

    levels = np.sort((np.array(levels) + 100) / 100)
    classes = np.zeros(len(levels) + 1, dtype=int)

    for _ in range(n_sample):
        interval_series = sampler.sample_interval(dataframe, interval_length=post + 1)
        buy_p = interval_series[0]
        post_series = interval_series[1:]
        sell_price, profit = labler.calc_profit(buy_price=buy_p, post_series=post_series, stop_loss=stop_loss)

        classes[np.searchsorted(levels, profit)] += 1

    print(classes)
    print(classes / n_sample)
    classes = classes / n_sample

    if plot:
        # TODO searchsorted placement index left or right
        levels *= 100
        x_axis_labels = [f'<{levels[0]}%']
        for i in range(len(levels) - 1):
            x_axis_labels.append(f'{levels[i]}-{levels[i+1]}%')
        x_axis_labels.append(f'{levels[-1]}%<')
        print(x_axis_labels)

        import plotly.graph_objects as go
        fig = go.Figure([go.Bar(x=x_axis_labels, y=classes)])
        fig.show()

    return classes


if __name__ == '__main__':
    # Test 1
    df = pd.read_csv("../test/test_data/AAPL_BBPL_CCPL_240.csv").set_index("Date")
    # levels = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    levels = list(range(-11, 12))
    profit_distribution(df, post=10, stop_loss=95, levels=levels, n_sample=10000, plot=True)
    print("Test1 OK")
