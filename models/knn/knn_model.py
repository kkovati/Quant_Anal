import numpy as np
import plotly.express as px
from sklearn.neighbors import KNeighborsClassifier

from dataset_generation.database_connection import HSMDataset
from dataset_generation.labler import calc_profit
from dataset_generation.standardize import standardize

if __name__ == '__main__':
    ds = HSMDataset(debug=True)

    pre_len = 30
    post_len = 7
    trainset_size = 10000
    testset_size = 1000

    # Generate train set
    X_train = np.empty((trainset_size, pre_len))
    y_train = np.empty((trainset_size,))
    for i in range(trainset_size):
        if i % 1000 == 999:
            print(f'Train set processed: {i + 1}/{trainset_size}')
        datapoint, label_data = ds.sample_datapoint(pre_len=pre_len, post_len=3, return_type='np')
        price, profit = calc_profit(buy_price=datapoint[3, -1], post_interval=label_data[3], stop_loss=95,
                                    take_profit=105.5)
        X_train[i] = standardize(datapoint)[3]
        y_train[i] = 1 if profit > 105 else 0

    # Fit KNN model
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=1)
    knn.fit(X_train, y_train)

    # Generate test set
    X_test = np.empty((testset_size, pre_len))
    y_test = np.empty((testset_size,))
    for i in range(testset_size):
        if i % 1000 == 999:
            print(f'Test set processed: {i + 1}/{trainset_size}')
        datapoint, label_data = ds.sample_datapoint(pre_len=pre_len, post_len=3, return_type='np')
        price, profit = calc_profit(buy_price=datapoint[3, -1], post_interval=label_data[3], stop_loss=95,
                                    take_profit=105.5)
        X_test[i] = standardize(datapoint)[3]
        y_test[i] = 1 if profit > 105 else 0

    # Evaluate
    print(f'Accuracy: {knn.score(X_test, y_test)}')

    # Show first test sample and closest neighbours
    _, neigh_indicies = knn.kneighbors(X_test)
    pre_intervals = [X_test[0],
                     X_train[neigh_indicies[0, 0]],
                     X_train[neigh_indicies[0, 1]],
                     X_train[neigh_indicies[0, 2]],
                     X_train[neigh_indicies[0, 3]],
                     X_train[neigh_indicies[0, 4]]]
    fig = px.scatter(np.transpose(pre_intervals))
    fig.update_traces(mode='lines+markers')
    fig.update_layout(title='Input and closest neighbour', hovermode="x unified").show()
