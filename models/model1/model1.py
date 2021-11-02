import numpy as np
import pandas as pd
from sklearn.svm import LinearSVR, SVR

from dataset_generation.hsm_dataset import generate_dataset
from dataset_generation.hsm_dataset import OPEN, HIGH, LOW, CLOSE
from dataset_generation.labler import calc_min_max
from models.input_compiler import compile_dataset


class Model1:

    def __init__(self):
        pass

    def compile_input(self, series):
        pass


if __name__ == '__main__':
    X_pre_interval_train, y_post_interval_train, _, _ = generate_dataset(1000, 1, 61, 20, debug=True)
    X_indicator, names = compile_dataset(X_pre_interval_train)
    y_min, y_max = calc_min_max(X_pre_interval_train[:, CLOSE, -1], y_post_interval_train)

    X_indicator = np.nan_to_num(X_indicator, nan=0.0, posinf=10000, neginf=-10000)

    svr_min = LinearSVR()
    svr_max = LinearSVR()

    # svr_temp = SVR(kernel='linear')
    # svr_temp.fit(X_indicator, y_max)

    svr_min.fit(X_indicator, y_min)
    svr_max.fit(X_indicator, y_max)

    svr_min_coef_idx = np.abs(svr_min.coef_).argsort()[-10:][::-1]

    for idx in svr_min_coef_idx:
        print(names[idx])

    a = 1