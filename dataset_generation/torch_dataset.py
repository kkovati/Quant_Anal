import pandas as pd
import torch


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class RandomSampledDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, interval_length):
        self.dataframe = pd.read_csv(csv_file).set_index("Date")
        self.interval_length = interval_length

    def __getitem__(self, _):
        sample(self.dataframe, self.interval_length)

    def __len__(self):
        # TODO decide virtual length of dataset
        return 1000


if __name__ == '__main__':
    # Test 1
    rsd = RandomSampledDataset("../test/test_data/AAPL_BBPL_CCPL_240.csv")