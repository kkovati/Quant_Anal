import pandas as pd
import torch

import sampler
import labler


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class RandomSampledDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, pre, post, stop_loss):
        self.dataframe = pd.read_csv(csv_file).set_index("Date")
        self.pre = pre
        self.post = post
        self.stop_loss = stop_loss

    def __getitem__(self, _):
        sampled_df = sampler.sample_interval(self.dataframe, self.pre + self.post)
        pre_df = sampled_df[0:self.pre]
        post_df = sampled_df[self.pre:]

        return labler.calc_profit(post_df)

    def __len__(self):
        # TODO decide virtual length of dataset
        return 1000


if __name__ == '__main__':
    # Test 1
    rsds = RandomSampledDataset("../test/test_data/AAPL_BBPL_CCPL_240.csv",
                                pre=3,
                                post=2,
                                stop_loss=90)

    # dl = torch.utils.data.DataLoader(rsds, batch_size=1)

    rsds[1]
