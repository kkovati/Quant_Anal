import datetime
import os
import numpy as np
import pandas as pd
import random


class HSMDataset:

    def __init__(self, debug=False):
        self.path = '../data/Huge_Stock_Market_Dataset/ETFs+Stocks'
        self.files = [f for f in os.listdir(self.path)]

        if debug:
            self.files = self.files[:20]
        else:
            self.files = self.files

        self.dataframes = [None] * len(self.files)

        for i, f in enumerate(self.files):
            if i % 200 == 199:
                print(f'{i + 1}/{len(self.files)} files loaded')
            # self.dataframes.append(self.open_file(f))
            self.dataframes[i] = self.open_file(f)

    def open_file(self, filename):
        path = self.path + '/' + filename
        dataframe = pd.read_csv(path).set_index("Date")
        dataframe.symbol = filename[:filename.index('.')]
        return dataframe

    def sample_dataframe(self):
        return random.choice(self.dataframes)

    def sample_interval(self, interval_len, interval_type):
        assert interval_len > 0
        assert interval_type in ['ohlc', 'close']

        max_sampling = 50
        for _ in range(max_sampling):

            dataframe = self.sample_dataframe()

            start = np.random.randint(len(dataframe.index))

            if interval_type == 'close':
                series = dataframe['Close'].iloc[start: start + interval_len]
                if self.is_series_valid(series, dataframe.symbol, interval_len, start):
                    return series
                else:
                    continue

            elif interval_type == 'ohlc':
                for column in ['Open', 'High', 'Low', 'Close']:
                    series = dataframe[column].iloc[start: start + interval_len]
                    if not self.is_series_valid(series, dataframe.symbol, interval_len, start):
                        continue
                return dataframe.iloc[start: start + interval_len]

            else:
                raise Exception()

        raise Exception(f'Number of sampling exceeded {max_sampling}')

    def is_series_valid(self, series, symbol, interval_len, start):
        assert type(series) is pd.Series

        if len(series) != interval_len:
            print('Sampling problem in file', symbol, 'at line', str(start), 'Too short interval')
            return False

        if series.isnull().values.any():
            print('Sampling problem in file', symbol, 'at line', str(start), 'Null value')
            return False

        if (series <= 0).any():
            print('Sampling problem in file', symbol, 'at line', str(start), 'Zero value')
            return False

        for i in range(len(series.index) - 1):
            day1 = series.index[i]
            day2 = series.index[i + 1]
            day1 = datetime.date(int(day1[0:4]), int(day1[5:7]), int(day1[8:10]))
            day2 = datetime.date(int(day2[0:4]), int(day2[5:7]), int(day2[8:10]))
            if (day2 - day1).days > 7:
                print('Sampling problem in file', symbol, 'at line', str(start), str((day2 - day1).days),
                      'days long break in the interval')
                return False

        return True

    def sample_datapoint(self, pre_len, post_len, interval_type, labler=None):
        assert pre_len > 0
        assert post_len > 0
        assert interval_type in ['ohlc', 'close']
        # assert callable(labler)

        if interval_type == 'ohlc':
            dataframe = self.sample_interval(interval_len=(pre_len + post_len), interval_type=interval_type)
            datapoint_dict = {'open': dataframe['Open'].to_numpy(),
                              'close': dataframe['Close'].to_numpy(),
                              'index': None,
                              'symbol': None}
            return datapoint_dict
        else:
            raise NotImplementedError()


if __name__ == '__main__':
    ds = HSMDataset(debug=True)

    # Test 1
    # for _ in range(3):
    #     print(ds.sample_interval(interval_len=10, interval_type='close'))

    # Test 2
    for _ in range(3):
        print(ds.sample_datapoint(pre_len=3, post_len=2, interval_type='ohlc'))
