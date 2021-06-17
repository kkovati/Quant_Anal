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


def preprocess_dataset(X_train_raw, y_train_raw, X_test_raw, y_test_raw, hyper_dict):
    assert len(X_train_raw) == len(y_train_raw)
    assert len(X_test_raw) == len(y_test_raw)

    # Copy (note: slicing does NOT make a copy, it refers to original array
    X_train_cpy = np.copy(X_train_raw)
    X_test_cpy = np.copy(X_test_raw)

    # Slice X sets
    X_train_sliced = X_train_cpy[:, :, -(hyper_dict['pre_len'] + 1): -1]
    X_test_sliced = X_test_cpy[:, :, -(hyper_dict['pre_len'] + 1): -1]
    assert X_train_sliced.shape[2] == hyper_dict['pre_len']
    assert X_test_sliced.shape[2] == hyper_dict['pre_len']

    # Init preprocessed dataset arrays
    X_train = np.zeros((len(X_train_raw), 4, hyper_dict['pre_len']))
    X_test = np.zeros((len(X_test_raw), 4, hyper_dict['pre_len']))
    y_train_trend = np.zeros((len(y_train_raw),))
    y_test_trend = np.zeros((len(y_test_raw),))
    y_train_trend_thres = np.zeros((len(y_train_raw),))
    y_test_trend_thres = np.zeros((len(y_test_raw),))
    y_train_profit = np.zeros((len(y_train_raw),))
    y_test_profit = np.zeros((len(y_test_raw),))

    p('Prepare train set')
    for i in tqdm(range(len(X_train_raw))):
        # Calculate profit
        y_train_profit[i] = calc_profit(buy_price=X_train_sliced[i, CLOSE, -1],
                                        post_interval=y_train_raw[i],
                                        stop_loss=hyper_dict['stop_loss'],
                                        take_profit=hyper_dict['take_profit'])

        # Calc trend label
        y_train_trend[i] = calc_trend(X_train_raw, y_train_raw)

        # Standardize chart
        X_train[i] = standardize(X_train_sliced[i])

    trend_thres = find_threshold(y_train_trend, hyper_dict['trend_threshold'] / 100)

    p('Threshold train trend labels')
    for i in tqdm(range(len(X_train_raw))):
        # Threshold trend labels
        y_train_trend_thres[i] = 1 if y_train_trend[i] >= trend_thres else 0

    p('Prepare test set')
    for i in tqdm(range(len(X_test_raw))):
        # Calculate profit
        y_test_profit[i] = calc_profit(buy_price=X_test_sliced[i, CLOSE, -1],
                                       post_interval=y_test_raw[i],
                                       stop_loss=hyper_dict['stop_loss'],
                                       take_profit=hyper_dict['take_profit'])

        # Calc trend label
        y_test_trend[i] = calc_trend(X_test_raw, y_test_raw)

        # Standardize chart
        X_test[i] = standardize(X_test_sliced[i])

    p('Threshold test trend labels')
    for i in tqdm(range(len(X_test_raw))):
        # Threshold trend labels
        y_test_trend_thres[i] = 1 if y_test_trend[i] >= trend_thres else 0

    return X_train, y_train_trend_thres, y_train_profit, X_test, y_test_trend_thres, y_test_profit


def init_random_hyperparameters():
    hyperdict = {'pre_len': np.random.randint(10, 61),
                 'post_len': np.random.randint(3, 20),
                 # 'profit_threshold': round(np.random.uniform(101, 108)),
                 'trend_threshold': np.random.randint(3, 40),
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

        retval = preprocess_dataset(X_train_raw, y_train_raw, X_test_raw, y_test_raw, hyper_dict)

        X_train, y_train_trend_thres, y_train_profit, X_test, y_test_trend_thres, y_test_profit = retval

        knn = fit_knn(X_train, y_train_trend_thres, hyper_dict)

        y_pred = knn.predict(X_test)
        y_pred_proba = knn.predict_proba(X_test)

        neigh_ind = knn.kneighbors(X_test)

        assert len(y_pred) == testset_size

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
