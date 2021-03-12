import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from dataset_generation.database_connection import HSMDataset
from dataset_generation.labler import calc_profit
from dataset_generation.standardize import standardize

if __name__ == '__main__':
    ds = HSMDataset(debug=True)

    size = 10
    n_days = 7
    X_train = np.empty((size, n_days))
    y_train = np.empty((size,))

    for i in range(size):
        datapoint, label_data = ds.sample_datapoint(pre_len=n_days, post_len=3, interval_type='ohlc')
        price, profit = calc_profit(buy_price=datapoint[3, -1], post=label_data[3], stop_loss=95)
        X_train[i] = standardize(datapoint)[3]
        y_train[i] = 1 if profit > 0.05 else 0

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
