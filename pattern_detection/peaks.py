import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks

from dataset_generation.database_connection import HSMDataset

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html



if __name__ == '__main__':
    ds = HSMDataset(debug=True)

    series = ds.sample_interval_series(interval_length=100)

    peaks, _ = find_peaks(series)

