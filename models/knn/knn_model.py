from logging import info as p
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from dataset_generation.hsm_dataset import generate_dataset
from dataset_generation.hsm_dataset import OPEN, HIGH, LOW, CLOSE
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


def preprocess_dataset(X_train_raw, y_train_raw, X_test_raw, y_test_raw, hyper_dict, return_full_interval=False,
                       debug=False):
    assert len(X_train_raw) == len(y_train_raw)
    assert len(X_test_raw) == len(y_test_raw)

    # Copy
    X_train_cpy = X_train_raw.copy()
    y_train_raw = y_train_raw.copy()
    X_test_cpy = X_test_raw.copy()
    y_test_raw = y_test_raw.copy()

    # Slice X sets
    X_train_sliced = X_train_cpy[:, :, -(hyper_dict['pre_len'] + 1): -1]
    X_test_sliced = X_test_cpy[:, :, -(hyper_dict['pre_len'] + 1): -1]

    assert X_train_sliced.shape[2] == hyper_dict['pre_len']
    assert X_test_sliced.shape[2] == hyper_dict['pre_len']

    # Init preprocessed dataset arrays
    X_train = np.zeros((len(X_train_raw), hyper_dict['pre_len']))
    X_test = np.zeros((len(X_test_raw), hyper_dict['pre_len']))
    y_train = np.zeros((len(y_train_raw),))
    y_test = np.zeros((len(y_test_raw),))
    y_profit = np.zeros((len(y_test_raw),))

    # need concat, thats not OK
    # X_train_full_interval = np.empty((trainset_size, hyper_dict['pre_len'] + hyper_dict['post_len']))
    # X_test_full_interval = np.empty((testset_size, hyper_dict['pre_len'] + hyper_dict['post_len']))

    p('Prepare train set')
    for i in tqdm(range(len(X_train_raw))):
        # Calculate profit
        profit = calc_profit(buy_price=X_train_sliced[i, CLOSE, -1],
                             post_interval=y_train_raw[i],
                             stop_loss=hyper_dict['stop_loss'],
                             take_profit=hyper_dict['take_profit'])

        # Standardize chart
        X_train[i] = standardize(X_train_sliced[i])[CLOSE]

        # Labeling
        y_train[i] = 1 if profit >= hyper_dict['profit_threshold'] else 0

        # concat, thats not OK
        # full_interval = np.concatenate((pre_interval, post_interval), axis=1)
        # X_train_full_interval[i] = standardize(full_interval)[CLOSE]

    p('Prepare test set')
    for i in tqdm(range(len(X_test_raw))):
        # Calculate profit
        profit = calc_profit(buy_price=X_test_sliced[i, CLOSE, -1],
                             post_interval=y_train_raw[i],
                             stop_loss=hyper_dict['stop_loss'],
                             take_profit=hyper_dict['take_profit'])

        # Standardize chart
        X_test[i] = standardize(X_test_raw[i])[CLOSE]

        # Labeling
        y_test[i] = 1 if profit >= hyper_dict['profit_threshold'] else 0

        # Save profit of each test datapoint
        y_profit[i] = profit

        # concat, thats not OK
        # full_interval = np.concatenate((pre_interval, post_interval), axis=1)
        # X_test_full_interval[i] = standardize(full_interval)[CLOSE]

    if return_full_interval:
        # return X_train, y_train, X_train_full_interval, X_test, y_test, X_test_full_interval
        return X_train, y_train, X_test, y_test, y_profit
    else:
        return X_train, y_train, X_test, y_test, y_profit


def init_random_hyperparameters():
    hyperdict = {'pre_len': np.random.randint(10, 61),
                 'post_len': np.random.randint(3, 20),
                 'profit_threshold': round(np.random.uniform(101, 108)),
                 'take_profit': 0,
                 'stop_loss': round(np.random.uniform(90, 99)),
                 'n_neighbors': np.random.choice(np.arange(1, 21)),
                 'weights': np.random.choice(('uniform', 'distance')),
                 'minkowski_p': np.random.choice((1, 2))}

    hyperdict['take_profit'] = round(np.random.uniform(hyperdict['profit_threshold'] + 1, 110))

    return hyperdict


def hyperparameter_tuner(trainset_size, testset_size, n_tune_iteration, debug=False):
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = generate_dataset(trainset_size, testset_size, 61, 20)

    df = pd.DataFrame()

    p('Start Hyperparameter tuning')
    for i in range(n_tune_iteration):
        p(f'{i + 1}/{n_tune_iteration} Hyperparameter settings')
        hyper_dict = init_random_hyperparameters()

        returned_tuple = preprocess_dataset(X_train_raw, y_train_raw, X_test_raw, y_test_raw, hyper_dict,
                                            return_full_interval=False, debug=debug)

        X_train, y_train, X_test, y_test, y_profit = returned_tuple

        knn = fit_knn(X_train, y_train, hyper_dict)

        y_pred = knn.predict(X_test)
        y_pred_proba = knn.predict_proba(X_test)

        assert len(y_pred) == testset_size
        assert len(y_profit) == testset_size

        # Calculate average profit
        sum_profit = 0
        for pred, profit in zip(y_pred, y_profit):
            print(profit)
            if pred == 1:
                sum_profit += profit
            else:
                assert pred == 0
                sum_profit += 100
        avg_profit = sum_profit / testset_size

        hyper_dict['AVG_PROFIT'] = avg_profit
        hyper_dict['ACC'] = metrics.accuracy_score(y_test, y_pred)
        hyper_dict['F1'] = metrics.f1_score(y_test, y_pred)
        hyper_dict['MCC'] = metrics.matthews_corrcoef(y_test, y_pred)
        hyper_dict['PREC'] = metrics.precision_score(y_test, y_pred)
        hyper_dict['REC'] = metrics.recall_score(y_test, y_pred)
        cm = metrics.confusion_matrix(y_test, y_pred)
        cond_pos = cm[1, 1] + cm[1, 0]
        cond_neg = cm[0, 0] + cm[0, 1]
        hyper_dict['P/N'] = round(cond_pos / (cond_pos + cond_neg), 2)
        hyper_dict['TP/TN/FN/FP'] = f'{str(cm[1, 1])}/{str(cm[0, 0])}/{str(cm[1, 0])}/{str(cm[0, 1])}'

        df = df.append(hyper_dict, ignore_index=True, verify_integrity=True)

    print(df)

    df.to_excel('df.xlsx')


if __name__ == '__main__':
    hyperparameter_tuner(trainset_size=100,
                         testset_size=100,
                         n_tune_iteration=20,
                         debug=True)
