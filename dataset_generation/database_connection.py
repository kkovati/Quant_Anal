import datetime
import os
import numpy as np
import pandas as pd
import random


class HSMDataset:

    def __init__(self, debug=False):
        self.path = 'D:\Kovacs_Attila/08_Programming\Python_projects\Quant_Anal\data\Huge_Stock_Market_Dataset\ETFs+Stocks'
        self.files = [f for f in os.listdir(self.path)]

        if debug:
            self.files = self.files[:20]
        else:
            self.files = self.files

        self.dataframes = [None] * len(self.files)

        for i, f in enumerate(self.files):
            if i % 200 == 199:
                print(f'{i + 1}/{len(self.files)} files loaded')
            self.dataframes[i] = self.open_file(f)

    def open_file(self, filename):
        path = self.path + '/' + filename
        dataframe = pd.read_csv(path).set_index("Date")
        dataframe.symbol = filename[:filename.index('.')]
        return dataframe

    def sample_random_dataframe(self):
        dataframe = random.choice(self.dataframes)
        return dataframe

    def sample_random_interval(self, interval_len):
        assert interval_len > 0

        max_sampling = 50
        for _ in range(max_sampling):
            dataframe = self.sample_random_dataframe()
            symbol = dataframe.symbol

            start = np.random.randint(len(dataframe.index))

            interval = dataframe.iloc[start: start + interval_len]

            if self.is_interval_valid(interval, symbol, interval_len, start):
                interval.symbol = symbol
                return interval

        raise Exception(f'Number of sampling exceeded {max_sampling}')

    def is_interval_valid(self, interval, symbol, interval_len, start):
        assert type(interval) is pd.DataFrame

        # Check interval length
        if len(interval) != interval_len:
            # print('Sampling problem in file', symbol, 'at line', str(start), 'Too short interval')
            return False

        # Check NaN
        if interval.isnull().values.any():
            # print('Sampling problem in file', symbol, 'at line', str(start), 'Null value')
            return False

        # Check zero or negative values
        flag = False
        for column in ['Open', 'High', 'Low', 'Close']:
            if (interval[column] <= 0).any():
                flag = True
        if flag:
            # print('Sampling problem in file', symbol, 'at line', str(start), 'Zero value')
            return False

        # Check for too long breaks between interval days
        for i in range(len(interval.index) - 1):
            day1 = interval.index[i]
            day2 = interval.index[i + 1]
            day1 = datetime.date(int(day1[0:4]), int(day1[5:7]), int(day1[8:10]))
            day2 = datetime.date(int(day2[0:4]), int(day2[5:7]), int(day2[8:10]))
            if (day2 - day1).days > 7:
                # print('Sampling problem in file', symbol, 'at line', str(start), str((day2 - day1).days),
                #       'days long break in the interval')
                return False

        return True

    def sample_datapoint(self, pre_len, post_len, return_type='df'):
        assert pre_len > 0
        assert post_len > 0
        assert return_type in ('df', 'np')

        interval = self.sample_random_interval(interval_len=(pre_len + post_len))
        symbol = interval.symbol

        if return_type == 'df':
            datapoint = interval.iloc[:pre_len]
            datapoint.symbol = symbol

            label = interval.iloc[pre_len:pre_len + post_len]

            return datapoint, label

        elif return_type == 'np':
            datapoint = np.zeros((4, pre_len))
            label = np.zeros((4, post_len))

            datapoint[0] = interval['Open'].iloc[:pre_len]
            datapoint[1] = interval['High'].iloc[:pre_len]
            datapoint[2] = interval['Low'].iloc[:pre_len]
            datapoint[3] = interval['Close'].iloc[:pre_len]

            label[0] = interval['Open'].iloc[pre_len:pre_len + post_len]
            label[1] = interval['High'].iloc[pre_len:pre_len + post_len]
            label[2] = interval['Low'].iloc[pre_len:pre_len + post_len]
            label[3] = interval['Close'].iloc[pre_len:pre_len + post_len]

            # TODO - some info dict can be returned also if needed
            # datapoint_dict = {'open': interval['Open'].to_numpy(),
            #                   'close': interval['Close'].to_numpy(),
            #                   'index': None,
            #                   'symbol': None}
            # return datapoint_dict

            return datapoint, label


if __name__ == '__main__':
    ds = HSMDataset(debug=True)

    # Test 1
    for _ in range(2):
        print(ds.sample_datapoint(pre_len=3, post_len=2, return_type='df'))

    # Test 2
    for _ in range(2):
        print(ds.sample_datapoint(pre_len=3, post_len=2, return_type='np'))

    # Test 3 - NaN value
    dummy_df = pd.DataFrame({'Open': [2, 1, 4],
                             'High': [2, 1, 4],
                             'Low': [2, 1, 4],
                             'Close': [2, 4, np.nan]},
                            index=['2020-01-01', '2020-01-02', '2020-01-03'])
    print(dummy_df)
    ds.is_interval_valid(dummy_df, 'DUMMY', 3, 1234)

    # Test 4 - negative value
    dummy_df = pd.DataFrame({'Open': [2, 1, 4],
                             'High': [2, 1, 4],
                             'Low': [2, 1, 4],
                             'Close': [2, 4, 0]},
                            index=['2020-01-01', '2020-01-02', '2020-01-03'])
    print(dummy_df)
    ds.is_interval_valid(dummy_df, 'DUMMY', 3, 1234)

    # Test 5 - too long breaks between interval days
    dummy_df = pd.DataFrame({'Open': [2, 1, 4],
                             'High': [2, 1, 4],
                             'Low': [2, 1, 4],
                             'Close': [2, 4, 1]},
                            index=['2020-01-01', '2020-01-02', '2020-01-10'])
    print(dummy_df)
    ds.is_interval_valid(dummy_df, 'DUMMY', 3, 1234)
