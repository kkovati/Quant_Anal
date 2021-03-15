import numpy as np
import pandas as pd
import plotly.express as px
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from dataset_generation.database_connection import HSMDataset
from dataset_generation.database_connection import OPEN, HIGH, LOW, CLOSE
from dataset_generation.labler import calc_profit
from dataset_generation.standardize import standardize


def fit_knn(X_train, y_train, hyper_dict, visualization=False):
    # Fit KNN model
    knn = KNeighborsClassifier(n_neighbors=hyper_dict['n_neighbors'],
                               weights=hyper_dict['weights'],
                               metric='minkowski',
                               p=hyper_dict['minkowski_p'])
    knn.fit(X_train, y_train)

    # Evaluate
    # print(f'Accuracy: {knn.score(X_test, y_test)}')

    if visualization:
        pass
        # Show first test sample and closest neighbours
        # _, neigh_indicies = knn.kneighbors(X_test)

        # pre_intervals = [X_test[0]]
        # for i in range(5):
        #     pre_intervals.append(X_train[neigh_indicies[0, i]])

        # full_intervals = [X_test_full_interval[0]]
        # for i in range(5):
        #     full_intervals.append(X_train_full_interval[neigh_indicies[0, i]])
        #
        # fig = px.scatter(np.transpose(full_intervals))
        # fig.update_traces(mode='lines+markers')
        # fig.update_layout(title='Input and closest neighbour', hovermode="x unified").show()

    return knn


def generate_dataset(trainset_size, testset_size, hyper_dict, return_full_interval=False, debug=False):
    ds = HSMDataset(test_size=0.1, debug=debug)

    # Generate train set
    X_train = np.empty((trainset_size, hyper_dict['pre_len']))
    y_train = np.empty((trainset_size,))
    X_train_full_interval = np.empty((trainset_size, hyper_dict['pre_len'] + hyper_dict['post_len']))

    print('Process train set')
    for i in tqdm(range(trainset_size)):
        pre_interval, post_interval = ds.sample_train_datapoint(pre_len=hyper_dict['pre_len'],
                                                                post_len=hyper_dict['post_len'],
                                                                return_type='np')

        _, profit = calc_profit(buy_price=pre_interval[CLOSE, -1], post_interval=post_interval[CLOSE],
                                stop_loss=hyper_dict['stop_loss'],
                                take_profit=hyper_dict['profit_threshold'])

        X_train[i] = standardize(pre_interval.copy())[CLOSE]
        y_train[i] = 1 if profit >= hyper_dict['profit_threshold'] else 0
        full_interval = np.concatenate((pre_interval, post_interval), axis=1)
        X_train_full_interval[i] = standardize(full_interval)[CLOSE]

    # Generate test set
    X_test = np.empty((testset_size, hyper_dict['pre_len']))
    y_test = np.empty((testset_size,))
    X_test_full_interval = np.empty((testset_size, hyper_dict['pre_len'] + hyper_dict['post_len']))

    print('Process test set')
    for i in tqdm(range(testset_size)):
        pre_interval, post_interval = ds.sample_test_datapoint(pre_len=hyper_dict['pre_len'],
                                                               post_len=hyper_dict['post_len'],
                                                               return_type='np')

        _, profit = calc_profit(buy_price=pre_interval[CLOSE, -1], post_interval=post_interval[CLOSE],
                                stop_loss=hyper_dict['stop_loss'],
                                take_profit=hyper_dict['profit_threshold'])

        X_test[i] = standardize(pre_interval.copy())[CLOSE]
        y_test[i] = 1 if profit >= hyper_dict['profit_threshold'] else 0
        full_interval = np.concatenate((pre_interval, post_interval), axis=1)
        X_test_full_interval[i] = standardize(full_interval)[CLOSE]

    if return_full_interval:
        return X_train, y_train, X_train_full_interval, X_test, y_test, X_test_full_interval
    else:
        return X_train, y_train, X_test, y_test


def init_random_hyperparameters():
    return {'pre_len': np.random.randint(10, 61),
            'post_len': np.random.randint(3, 20),
            'profit_threshold': round(np.random.uniform(101, 108)),
            'stop_loss': round(np.random.uniform(90, 99)),
            'n_neighbors': np.random.choice(np.arange(1, 21)),
            'weights': np.random.choice(('uniform', 'distance')),
            'minkowski_p': np.random.choice((1, 2))}


def hyperparameter_tuner(trainset_size, testset_size, n_tune_iteration, debug=False):
    hyper_dict = init_random_hyperparameters()
    result_columns = ['TP/TN/FN/FP', 'ACC', 'F1']
    columns = list(hyper_dict) + result_columns

    X_train, y_train, X_test, y_test = generate_dataset(trainset_size, testset_size, hyper_dict,
                                                        return_full_interval=False, debug=debug)

    df = pd.DataFrame(columns=columns)

    print('Hyperparameter tuning')
    for _ in tqdm(range(n_tune_iteration)):
        hyper_dict = init_random_hyperparameters()

        knn = fit_knn(X_train, y_train, hyper_dict)

        y_pred = knn.predict(X_test)
        y_pred_proba = knn.predict_proba(X_test)

        hyper_dict['ACC'] = metrics.accuracy_score(y_test, y_pred)
        cm = metrics.confusion_matrix(y_test, y_pred)
        hyper_dict['TP/TN/FN/FP'] = f'{str(cm[1, 1])}/{str(cm[0, 0])}/{str(cm[1, 0])}/{str(cm[0, 1])}'

        df = df.append(hyper_dict, ignore_index=True, verify_integrity=True)

    print(df)

    df.to_excel('df.xlsx')


if __name__ == '__main__':
    hyperparameter_tuner(trainset_size=10000,
                         testset_size=1000,
                         n_tune_iteration=10,
                         debug=False)
