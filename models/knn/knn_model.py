from logging import info as p
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
import time
from tqdm import tqdm

from dataset_generation.dataset_analyzation import find_threshold
from dataset_generation.hsm_dataset import generate_dataset
from dataset_generation.hsm_dataset import OPEN, HIGH, LOW, CLOSE
from dataset_generation.labler import calc_profit, calc_trend
from dataset_generation.standardize import standardize


def init_random_hyperparameters():
    hyperdict = {'pre_len': np.random.randint(10, 61),
                 'post_len': np.random.randint(3, 20),
                 'trend_threshold': np.random.uniform(.2, .6),
                 'take_profit': np.random.uniform(1.02, 1.15),
                 'stop_loss': np.random.uniform(.9, .99),
                 'n_neighbors': np.random.randint(1, 21),
                 'weights': np.random.choice(('uniform', 'distance')),
                 'minkowski_p': np.random.choice((1, 2))}
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

    for i in range(len(pre_interval)):
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
        p(f'\n--- {i + 1}/{n_tune_iteration} Hyperparameter settings ---')
        hyper_dict = init_random_hyperparameters()

        p('Preprocess train set')
        retval = preprocess_dataset(pre_interval_train, post_interval_train, hyper_dict)
        X_train, y_trend_thres_train, y_profit_train, trend_thres = retval

        knn = fit_knn(X_train, y_trend_thres_train, hyper_dict)

        p('Preprocess test set')
        retval = preprocess_dataset(pre_interval_test, post_interval_test, hyper_dict, trend_thres)
        X_test, y_trend_thres_test, y_profit_test, trend_thres = retval

        start_time = time.time()
        p('Predict test set')
        y_pred = knn.predict(X_test)
        # y_pred_proba = knn.predict_proba(X_test)
        # neigh_ind = knn.kneighbors(X_test)
        p('Prediction time: ' + str(round(time.time() - start_time)) + ' sec')

        # Calculate average profit
        avg_profit, avg_win, avg_loose, avg_miss = 0.0, 0.0, 0.0, 0.0
        net_profit, net_miss = 1.0, 1.0
        trade_counter, win_counter, loose_counter, miss_counter = 0, 0, 0, 0
        assert len(y_pred) == len(y_profit_test) == testset_size
        for pred, profit in zip(y_pred, y_profit_test):
            if pred == 1:
                avg_profit += profit
                net_profit *= profit
                trade_counter += 1
                if profit > 1:
                    avg_win += profit
                    win_counter += 1
                else:
                    avg_loose += profit
                    loose_counter += 1
            else:
                assert pred == 0
                avg_miss += profit
                net_miss *= profit
                miss_counter += 1
        assert np.count_nonzero(y_pred == 1) == trade_counter
        assert np.count_nonzero(y_pred == 0) == miss_counter
        if trade_counter != 0:
            avg_profit /= trade_counter
        if win_counter != 0:
            avg_win /= win_counter
        if loose_counter != 0:
            avg_loose /= loose_counter
        if miss_counter != 0:
            avg_miss /= miss_counter
        hyper_dict['01_AVG_PROFIT'] = round(avg_profit, 4)
        hyper_dict['01_AVG_WIN'] = round(avg_win, 4)
        hyper_dict['01_AVG_LOOSE'] = round(avg_loose, 4)
        hyper_dict['01_NET_PROFIT'] = round(net_profit, 4)
        hyper_dict['01_AVG_MISS'] = round(avg_miss, 4)
        hyper_dict['01_NET_MISS'] = round(net_miss, 4)

        if y_trend_thres_test.min() == y_trend_thres_test.max() and y_pred.min() == y_pred.max():
            tn, fp, fn, tp = testset_size, 0, 0, 0
        else:
            tn, fp, fn, tp = metrics.confusion_matrix(y_trend_thres_test, y_pred).ravel()

        # Predicted condition positive (PP) over all instances
        hyper_dict['02_PP/ALL'] = round((tp + fp) / (tn + fp + fn + tp), 2)
        # Positive predictive value (PPV)
        if tp + fp > 0:
            hyper_dict['03_TP/PP'] = round(tp / (tp + fp), 2)
        else:
            hyper_dict['03_TP/PP'] = np.nan
        # Actual condition positive (P) over all instances
        hyper_dict['04_P/ALL'] = round((tp + fn) / (tn + fp + fn + tp), 2)
        hyper_dict['05_TP'] = tp
        hyper_dict['06_FP'] = fp
        hyper_dict['07_FN'] = fn
        hyper_dict['08_TN'] = tn

        hyper_dict['09_ACC'] = round(metrics.accuracy_score(y_trend_thres_test, y_pred), 2)
        hyper_dict['10_F1'] = round(metrics.f1_score(y_trend_thres_test, y_pred), 2)
        # hyper_dict['MCC'] = metrics.matthews_corrcoef(y_trend_thres_test, y_pred)
        # hyper_dict['PREC'] = metrics.precision_score(y_trend_thres_test, y_pred)
        # hyper_dict['REC'] = metrics.recall_score(y_trend_thres_test, y_pred)

        df = df.append(hyper_dict, ignore_index=True, verify_integrity=True)

    print(df)

    df.to_excel('df.xlsx')


if __name__ == '__main__':
    start_time = time.time()

    hyperparameter_tuner(trainset_size=500,
                         testset_size=500,
                         n_tune_iteration=10,
                         debug=True)

    # hyperparameter_tuner(trainset_size=100000,
    #                      testset_size=5000,
    #                      n_tune_iteration=50,
    #                      debug=False)

    print('\nTotal running time: ', round(time.time() - start_time), ' sec')
