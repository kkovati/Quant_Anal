from logging import info as p
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from dataset_generation.dataset_analyzation import find_threshold
from dataset_generation.hsm_dataset import generate_dataset
from dataset_generation.hsm_dataset import OPEN, HIGH, LOW, CLOSE
from dataset_generation.labler import calc_profit, calc_trend
from dataset_generation.standardize import standardize


def init_random_hyperparameters():
    hyperdict = {'pre_len': np.random.randint(10, 61),
                 'post_len': np.random.randint(3, 20),
                 'trend_threshold': np.random.uniform(.01, .4),
                 'profit_threshold': np.random.uniform(1.01, 1.08),
                 'take_profit': 0,
                 'stop_loss': np.random.uniform(.85, .99),
                 'n_neighbors': np.random.randint(1, 21),
                 'weights': np.random.choice(('uniform', 'distance')),
                 'minkowski_p': np.random.choice((1, 2))}

    hyperdict['take_profit'] = np.random.uniform(hyperdict['profit_threshold'] + .01, 1.1)

    return hyperdict


def preprocess_dataset(pre_interval, post_interval, hyper_dict, trend_thres=None):
    assert len(pre_interval) == len(post_interval)

    # Slice intervals sets (note: slicing does NOT make a copy, it refers to original array)
    pre_interval_sliced = np.copy(pre_interval)[:, :, -(hyper_dict['pre_len'] + 1): -1]
    post_interval_sliced = np.copy(post_interval)[:, :, -(hyper_dict['post_len'] + 1): -1]
    assert pre_interval_sliced.shape[2] == hyper_dict['pre_len']
    assert post_interval_sliced.shape[2] == hyper_dict['post_len']

    # Init preprocessed dataset arrays
    X = np.zeros((len(pre_interval), 4, hyper_dict['pre_len']))
    y_trend = np.zeros((len(post_interval),))
    y_trend_thres = np.zeros((len(post_interval),))
    y_profit = np.zeros((len(post_interval),))

    p('Prepare set')
    for i in tqdm(range(len(pre_interval))):
        # Calculate profit
        y_profit[i] = calc_profit(buy_price=pre_interval_sliced[i, CLOSE, -1],
                                  post_interval=post_interval_sliced[i],
                                  stop_loss=hyper_dict['stop_loss'],
                                  take_profit=hyper_dict['take_profit'])

        # Calc trend label
        y_trend[i] = calc_trend(pre_interval_sliced[i], post_interval_sliced[i])

        # Standardize chart
        X[i] = standardize(pre_interval_sliced[i])

    if trend_thres is None:
        trend_thres = find_threshold(y_trend, hyper_dict['trend_threshold'])

    p('Threshold trend labels')
    for i in tqdm(range(len(pre_interval))):
        # Threshold trend labels
        y_trend_thres[i] = 1 if y_trend[i] >= trend_thres else 0

    # Flatten X
    X = np.concatenate((X[:, OPEN], X[:, HIGH], X[:, LOW], X[:, CLOSE]), axis=1)
    assert X.ndim == 2

    return X, y_trend_thres, y_profit, trend_thres


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


def hyperparameter_tuner(trainset_size, testset_size, n_tune_iteration, debug=False):
    retval = generate_dataset(trainset_size, testset_size, 61, 20, debug=debug)
    pre_interval_train, post_interval_train, pre_interval_test, post_interval_test = retval

    df = pd.DataFrame()

    p('Start Hyperparameter tuning')
    for i in range(n_tune_iteration):
        p(f'{i + 1}/{n_tune_iteration} Hyperparameter settings')
        hyper_dict = init_random_hyperparameters()

        retval = preprocess_dataset(pre_interval_train, post_interval_train, hyper_dict)
        X_train, y_trend_thres_train, y_profit_train, trend_thres = retval

        knn = fit_knn(X_train, y_trend_thres_train, hyper_dict)

        retval = preprocess_dataset(pre_interval_test, post_interval_test, hyper_dict, trend_thres)
        X_test, y_trend_thres_test, y_profit_test, trend_thres = retval

        y_pred = knn.predict(X_test)
        assert len(y_pred) == testset_size
        y_pred_proba = knn.predict_proba(X_test)
        neigh_ind = knn.kneighbors(X_test)

        # Calculate average profit
        sum_profit = 0
        for pred, profit in zip(y_pred, y_profit_test):
            # print(profit)
            if pred == 1:
                sum_profit += profit
            else:
                assert pred == 0
                sum_profit += 100
        avg_profit = sum_profit / testset_size

        hyper_dict['AVG_PROFIT'] = avg_profit
        hyper_dict['ACC'] = metrics.accuracy_score(y_trend_thres_test, y_pred)
        hyper_dict['F1'] = metrics.f1_score(y_trend_thres_test, y_pred)
        hyper_dict['MCC'] = metrics.matthews_corrcoef(y_trend_thres_test, y_pred)
        hyper_dict['PREC'] = metrics.precision_score(y_trend_thres_test, y_pred)
        hyper_dict['REC'] = metrics.recall_score(y_trend_thres_test, y_pred)
        cm = metrics.confusion_matrix(y_trend_thres_test, y_pred)
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
