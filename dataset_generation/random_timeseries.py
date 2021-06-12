import numpy as np

from dataset_generation.hsm_dataset import OPEN, HIGH, LOW, CLOSE


def generate_random_timeseries(length, mu=.001, sigma=.01, start_price=5):
    returns = np.random.normal(loc=mu, scale=sigma, size=length)
    timeseries = start_price * (1 + returns).cumprod()
    timeseries += (start_price - timeseries[0])
    return timeseries


def generate_random_interval(length):
    unit = 10
    timeseries = generate_random_timeseries(unit * length)
    interval = np.zeros((4, length))
    for i in range(length):
        timeseries_unit = timeseries[i * unit: (i + 1) * unit]
        interval[OPEN, i] = timeseries_unit[0]
        interval[HIGH, i] = max(timeseries_unit)
        interval[LOW, i] = min(timeseries_unit)
        interval[CLOSE, i] = timeseries_unit[unit - 1]
    return interval


if __name__ == '__main__':
    # np.random.seed(0)
    generate_random_timeseries(length=20)
    generate_random_interval(length=20)
