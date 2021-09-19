import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

from dataset_generation.hsm_dataset import generate_dataset
from dataset_generation.hsm_dataset import OPEN, HIGH, LOW, CLOSE
from models.input_compiler import compile_dataset


class Model1:

    def __init__(self):
        pass

    def compile_input(self, series):
        pass


if __name__ == '__main__':
    X_pre_interval_train, y_post_interval_train, _, _ = generate_dataset(100, 1, 61, 20, debug=True)
    X_indicator = compile_dataset(X_pre_interval_train)

    svc = LinearSVC()

    svc.fit(X_indicator)

    pass

