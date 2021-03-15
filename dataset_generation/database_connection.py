import datetime
import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Constants for ndarray indexing
OPEN, HIGH, LOW, CLOSE = 0, 1, 2, 3


class HSMDataset:

    def __init__(self, test_size, debug=False):
        self.path = 'D:/Kovacs_Attila/08_Programming/Python_projects/Quant_Anal/data/Huge_Stock_Market_Dataset/Stocks'
        self.files = [f for f in os.listdir(self.path)]

        if debug:
            self.files = self.files[:200]
        else:
            self.files = self.files

        dataframes = [None] * len(self.files)

        print('Loading files')
        for i, f in enumerate(tqdm(self.files)):
            dataframes[i] = self.open_file(f)

        self.train_dataframes, self.test_dataframes = train_test_split(dataframes, test_size=test_size, shuffle=True)

    def open_file(self, filename):
        path = self.path + '/' + filename
        dataframe = pd.read_csv(path).set_index("Date")
        dataframe.symbol = filename[:filename.index('.')]
        return dataframe

    def sample_random_dataframe(self, subset):
        assert subset in ('train', 'test')

        if subset == 'train':
            return random.choice(self.train_dataframes)
        else:
            return random.choice(self.test_dataframes)

    def sample_random_interval(self, interval_len, subset):
        assert interval_len > 0

        max_sampling = 100
        for _ in range(max_sampling):
            dataframe = self.sample_random_dataframe(subset)
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

    def sample_datapoint(self, pre_len, post_len, subset, return_type='df'):
        assert pre_len > 0
        assert post_len > 0
        assert return_type in ('df', 'np')

        full_interval = self.sample_random_interval(interval_len=(pre_len + post_len), subset=subset)
        symbol = full_interval.symbol

        if return_type == 'df':
            # TODO - need to filter columns?
            pre_interval = full_interval.iloc[:pre_len]
            pre_interval.symbol = symbol

            post_interval = full_interval.iloc[pre_len:pre_len + post_len]

            return pre_interval, post_interval

        elif return_type == 'np':
            pre_interval = np.zeros((4, pre_len))
            post_interval = np.zeros((4, post_len))

            pre_interval[0] = full_interval['Open'].iloc[:pre_len]
            pre_interval[1] = full_interval['High'].iloc[:pre_len]
            pre_interval[2] = full_interval['Low'].iloc[:pre_len]
            pre_interval[3] = full_interval['Close'].iloc[:pre_len]

            post_interval[0] = full_interval['Open'].iloc[pre_len:pre_len + post_len]
            post_interval[1] = full_interval['High'].iloc[pre_len:pre_len + post_len]
            post_interval[2] = full_interval['Low'].iloc[pre_len:pre_len + post_len]
            post_interval[3] = full_interval['Close'].iloc[pre_len:pre_len + post_len]

            # TODO - some info dict can be returned also if needed
            # datapoint_dict = {'open': full_interval['Open'].to_numpy(),
            #                   'close': full_interval['Close'].to_numpy(),
            #                   'index': None,
            #                   'symbol': None}
            # return datapoint_dict

            return pre_interval, post_interval

    def sample_train_datapoint(self, pre_len, post_len, return_type='df'):
        return self.sample_datapoint(pre_len, post_len, subset='train', return_type=return_type)

    def sample_test_datapoint(self, pre_len, post_len, return_type='df'):
        return self.sample_datapoint(pre_len, post_len, subset='test', return_type=return_type)


if __name__ == '__main__':
    ds = HSMDataset(test_size=0.1, debug=True)

    # Test 1
    for _ in range(2):
        print(ds.sample_train_datapoint(pre_len=3, post_len=2, return_type='df'))

    # Test 2
    for _ in range(2):
        print(ds.sample_test_datapoint(pre_len=3, post_len=2, return_type='np'))

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
