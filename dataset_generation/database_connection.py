import datetime
import os
import numpy as np
import pandas as pd


class HSMDataset:

    def __init__(self):
        self.path = '../data/Huge_Stock_Market_Dataset/ETFs+Stocks'
        self.files = [f for f in os.listdir(self.path)]

        for f in self.files:
            if f[-7:] != '.us.txt':
                raise Exception(f'Filename problem: ' + f)

    def sample_file(self):
        return np.random.choice(self.files)

    def open_file(self, filename):
        p = self.path + '/' + filename
        return pd.read_csv(p).set_index("Date")

    def sample_interval(self, interval_length):

        max_sampling = 50
        for _ in range(max_sampling):

            filename = self.sample_file()
            print(filename)
            dataframe = self.open_file(filename)

            start = np.random.randint(len(dataframe.index))
            interval_series = dataframe['Close'].iloc[start: start + interval_length]

            if len(interval_series) != interval_length:
                print('Sampling problem in file', filename, 'at line', str(start), 'Too short interval')
                continue

            if interval_series.isnull().values.any():
                print('Sampling problem in file', filename, 'at line', str(start), 'Null value')
                continue

            if (interval_series <= 0).any():
                print('Sampling problem in file', filename, 'at line', str(start), 'Zero value')
                continue

            flag = False
            for i in range(len(interval_series.index) - 1):
                day1 = interval_series.index[i]
                day2 = interval_series.index[i + 1]
                day1 = datetime.date(int(day1[0:4]), int(day1[5:7]), int(day1[8:10]))
                day2 = datetime.date(int(day2[0:4]), int(day2[5:7]), int(day2[8:10]))
                if (day2 - day1).days > 7:
                    print('Sampling problem in file', filename, 'at line', str(start), str((day2 - day1).days),
                          'days long break in the interval')
                    flag = True
                    break
            if flag:
                continue

            return interval_series

        raise Exception(f'Number of sampling exceeded {max_sampling}')


if __name__ == '__main__':
    ds = HSMDataset()

    # Test 1
    while True:
        print(ds.sample_interval(interval_length=10))
