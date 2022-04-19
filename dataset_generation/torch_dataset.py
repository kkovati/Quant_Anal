import pandas as pd
import torch

import hsm_dataset as dc
import labler


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class RandomSampledDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, pre, post, stop_loss):
        self.dataset = dc.HSMDataset()
        self.pre = pre
        self.post = post
        self.stop_loss = stop_loss

    def __getitem__(self, _):
        sampled_interval = self.dataset.sample_random_interval(self.pre + self.post)
        pre_df = sampled_interval[0:self.pre]
        post_df = sampled_interval[self.pre:]

        y = labler.calc_profit(post_df)

    def __len__(self):
        # TODO decide virtual length of dataset
        return 1000


if __name__ == '__main__':
    # Test 1
    rsds = RandomSampledDataset("../data/test_data/AAPL_BBPL_CCPL_240.csv",
                                pre=3,
                                post=2,
                                stop_loss=90)

    # dl = torch.utils.data.DataLoader(rsds, batch_size=1)

    rsds[1]
