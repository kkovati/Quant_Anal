import numpy as np
import pandas as pd


def sample(df, interval_length):
    max_sampling = 100

    for _ in range(max_sampling):
        column = np.random.choice(df.columns)
        start = np.random.randint(len(df.index))
        interval = df[column].iloc[start: start + interval_length]

        if len(interval) != interval_length:
            continue

        if interval.isnull().values.any():
            continue

        if (interval <= 0).any():
            continue

        return interval

    raise Exception(f'Number of sampling exceeded {max_sampling}')


if __name__ == '__main__':
    # Test 1
    df = pd.read_csv("../test/test_data/AAPL_BBPL_CCPL_240.csv").set_index("Date")
    print(sample(df, interval_length=10))
    print('Test1 OK')

    # Test 2
    try:
        sample(df, interval_length=1000)
    except Exception as e:
        assert str(e) == 'Number of sampling exceeded 100'
    print('Test2 OK')
