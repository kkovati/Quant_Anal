import numpy as np
import unittest

import dataset_generation.labler as labler


class TestLabler(unittest.TestCase):

    def test_labler_close_price(self):
        # Sell at close price on last day
        post_interval = np.array([[100, 100, 100, 100, 100],  # OPEN
                                  [100, 100, 100, 100, 100],  # HIGH
                                  [100, 100, 100, 100, 100],  # LOW
                                  [100, 100, 100, 100, 102]])  # CLOSE
        profit = labler.calc_profit(buy_price=100,
                                    post_interval=post_interval,
                                    stop_loss=95,
                                    take_profit=105)
        self.assertEqual(profit, 102)

    def test_labler_take_profit(self):
        # Sell at take profit price
        post_interval = np.array([[100, 100, 100, 100, 100],  # OPEN
                                  [100, 100, 100, 106, 100],  # HIGH
                                  [100, 100, 100, 100, 100],  # LOW
                                  [100, 100, 100, 100, 100]])  # CLOSE
        profit = labler.calc_profit(buy_price=100,
                                    post_interval=post_interval,
                                    stop_loss=95,
                                    take_profit=105)
        self.assertEqual(profit, 105)

    def test_labler_stop_loss(self):
        # Sell at stop loss price
        post_interval = np.array([[100, 100, 100, 100, 100],  # OPEN
                                  [100, 100, 100, 100, 100],  # HIGH
                                  [100, 100, 94, 100, 100],  # LOW
                                  [100, 100, 100, 100, 100]])  # CLOSE
        profit = labler.calc_profit(buy_price=100,
                                    post_interval=post_interval,
                                    stop_loss=95,
                                    take_profit=105)
        self.assertEqual(profit, 95)


if __name__ == '__main__':
    unittest.main()


