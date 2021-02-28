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

    def sample_interval_series(self, interval_length):
        # TODO rename
        max_sampling = 50
        for _ in range(max_sampling):

            dataframe = self.sample_dataframe()

            start = np.random.randint(len(dataframe.index))
            interval_series = dataframe['Close'].iloc[start: start + interval_length]

            if len(interval_series) != interval_length:
                print('Sampling problem in file', dataframe.symbol, 'at line', str(start), 'Too short interval')
                continue

            if interval_series.isnull().values.any():
                print('Sampling problem in file', dataframe.symbol, 'at line', str(start), 'Null value')
                continue

            if (interval_series <= 0).any():
                print('Sampling problem in file', dataframe.symbol, 'at line', str(start), 'Zero value')
                continue

            flag = False
            for i in range(len(interval_series.index) - 1):
                day1 = interval_series.index[i]
                day2 = interval_series.index[i + 1]
                day1 = datetime.date(int(day1[0:4]), int(day1[5:7]), int(day1[8:10]))
                day2 = datetime.date(int(day2[0:4]), int(day2[5:7]), int(day2[8:10]))
                if (day2 - day1).days > 7:
                    print('Sampling problem in file', dataframe.symbol, 'at line', str(start), str((day2 - day1).days),
                          'days long break in the interval')
                    flag = True
                    break
            if flag:
                continue

            return interval_series

        raise Exception(f'Number of sampling exceeded {max_sampling}')


if __name__ == '__main__':
    ds = HSMDataset(debug=True)

    # Test 1
    while True:
        print(ds.sample_interval_series(interval_length=10))
