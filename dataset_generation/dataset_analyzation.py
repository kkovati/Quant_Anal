import numpy as np
import pandas as pd

import labler
import sampler


def profit_distribution(dataframe, post, stop_loss, n_sample, classes):
    classes = np.sort((np.array(classes) + 100) / 100)
    print(classes)

    for _ in range(n_sample):
        interval_series = sampler.sample_interval(dataframe, interval_length=post + 1)
        buy_p = interval_series[0]
        post_series = interval_series[1:]
        sell_price, profit = labler.calc_profit(buy_price=buy_p, post_series=post_series, stop_loss=stop_loss)

        print(profit)


if __name__ == '__main__':
    # Test 1
    df = pd.read_csv("../test/test_data/AAPL_BBPL_CCPL_240.csv").set_index("Date")
    profit_distribution(df, post=10, stop_loss=90, n_sample=3, classes=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    print("Test1 OK")
