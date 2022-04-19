import numpy as np

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from dataset_generation.hsm_dataset import OPEN, HIGH, LOW, CLOSE


class KnnStartegy(Strategy):
    model = None
    hyper_dict = None

    def init(self):
        close = self.data.Close
        self.sma = self.I(SMA, close, self.hyper_dict['pre_len_max'])  # for denying the warm-up days

    def next(self):
        interval = np.zeros((4, self.hyper_dict['pre_len']))

        o = self.data.Open[-self.hyper_dict['pre_len'] - 1: -1]
        h = self.data.High[-self.hyper_dict['pre_len'] - 1: -1]
        l = self.data.Low[-self.hyper_dict['pre_len'] - 1: -1]
        c = self.data.Close[-self.hyper_dict['pre_len'] - 1: -1]

        i = np.concatenate((o, h, l, c))

        pred = self.model.predict([i])[0]

        price = self.data.Close[-1]

        if pred == 1:
            self.buy(size=0.01, tp=self.hyper_dict['take_profit'] * price, sl=self.hyper_dict['stop_loss'] * price)
        else:
            assert pred == 0
            # self.sell()
