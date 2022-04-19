from logging import info as p
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import talib
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset_generation.hsm_dataset import generate_dataset
from dataset_generation.hsm_dataset import OPEN, HIGH, LOW, CLOSE
from dataset_generation.random_timeseries import generate_random_interval
from dataset_generation.standardize import standardize


class GridTrader():

    def __init__(self, init_price, grid_step):
        self.init_price = init_price
        self.grid_step = grid_step
        self.grid = [init_price]
        next_grid_price_up = next_grid_price_down = init_price
        for i in range(100):
            next_grid_price_up *= grid_step
            next_grid_price_down *= grid_step
            self.grid.append(next_grid_price_up)
            self.grid.append(next_grid_price_down)
        self.grid.sort()
