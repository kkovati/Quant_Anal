import numpy as np
import pandas as pd


def sample_interval(dataframe, interval_length):
    """

    Parameters
    ----------
    dataframe
    interval_length

    Returns
    -------
    interval_series : pd.Series
    """
    assert type(dataframe) is pd.core.frame.DataFrame

    max_sampling = 100

    for _ in range(max_sampling):
        column = np.random.choice(dataframe.columns)
        start = np.random.randint(len(dataframe.index))
        interval_series = dataframe[column].iloc[start: start + interval_length]

        if len(interval_series) != interval_length:
            continue

        if interval_series.isnull().values.any():
            continue

        if (interval_series <= 0).any():
            continue

        return interval_series

    raise Exception(f'Number of sampling exceeded {max_sampling}')


if __name__ == '__main__':
    # Test 1
    df = pd.read_csv("../data/test_data/AAPL_BBPL_CCPL_240.csv").set_index("Date")
    print(sample_interval(df, interval_length=10))
    print('Test1 OK')

    # Test 2
    try:
        sample_interval(df, interval_length=1000)
    except Exception as e:
        assert str(e) == 'Number of sampling exceeded 100'
    print('Test2 OK')
