import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from dataset_generation.database_connection import HSMDataset




if __name__ == '__main__':
    ds = HSMDataset(debug=True)

    size = 10
    n_days = 7
    X_train = np.empty((size, n_days))
    y_train = np.empty((size,))

    for i in range(size):
        datapoint_dict = ds.sample_datapoint(pre_len=n_days, post_len=3, interval_type='ohlc')
        X_train[i] =

    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')