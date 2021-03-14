import numpy as np
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from dataset_generation.database_connection import HSMDataset
from dataset_generation.labler import calc_profit
from dataset_generation.standardize import standardize


def run_knn(ds, pre_len, post_len, profit_threshold, stop_loss, n_neighbors, weights, minkowski_p):
    trainset_size = 10000
    testset_size = 1000

    # Generate train set
    X_train = np.empty((trainset_size, pre_len))
    y_train = np.empty((trainset_size,))
    X_train_full_interval = np.empty((trainset_size, pre_len + post_len))
    print('Process train set')
    for i in tqdm(range(trainset_size)):
        pre_interval, post_interval = ds.sample_datapoint(pre_len=pre_len, post_len=post_len, return_type='np')
        _, profit = calc_profit(buy_price=pre_interval[3, -1], post_interval=post_interval[3], stop_loss=95,
                                take_profit=105.5)
        X_train[i] = standardize(pre_interval.copy())[3]
        y_train[i] = 1 if profit > 105 else 0
        full_interval = np.concatenate((pre_interval, post_interval), axis=1)
        X_train_full_interval[i] = standardize(full_interval)[3]

    # Fit KNN model
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=1)
    knn.fit(X_train, y_train)

    # Generate test set
    X_test = np.empty((testset_size, pre_len))
    y_test = np.empty((testset_size,))
    X_test_full_interval = np.empty((testset_size, pre_len + post_len))
    print('Process train set')
    for i in tqdm(range(testset_size)):
        pre_interval, post_interval = ds.sample_datapoint(pre_len=pre_len, post_len=post_len, return_type='np')
        _, profit = calc_profit(buy_price=pre_interval[3, -1], post_interval=post_interval[3], stop_loss=95,
                                take_profit=105.5)
        X_test[i] = standardize(pre_interval.copy())[3]
        y_test[i] = 1 if profit > 105 else 0
        full_interval = np.concatenate((pre_interval, post_interval), axis=1)
        X_test_full_interval[i] = standardize(full_interval)[3]

    # Evaluate
    print(f'Accuracy: {knn.score(X_test, y_test)}')

    # Show first test sample and closest neighbours
    _, neigh_indicies = knn.kneighbors(X_test)

    # pre_intervals = [X_test[0]]
    # for i in range(5):
    #     pre_intervals.append(X_train[neigh_indicies[0, i]])

    full_intervals = [X_test_full_interval[0]]
    for i in range(5):
        full_intervals.append(X_train_full_interval[neigh_indicies[0, i]])

    fig = px.scatter(np.transpose(full_intervals))
    fig.update_traces(mode='lines+markers')
    fig.update_layout(title='Input and closest neighbour', hovermode="x unified").show()


if __name__ == '__main__':
    ds_ = HSMDataset(debug=True)

    run_knn(ds=ds_,
            pre_len=30,
            post_len=10,
            profit_threshold=105,
            stop_loss=95,
            n_neighbors=5,
            weights='distance',
            minkowski_p=1)
