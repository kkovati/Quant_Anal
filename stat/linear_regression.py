import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def main():
    df = pd.read_csv('../data/BTCUSDT-1m-2021.csv')
    # TODO use high/low values instead close
    ts = df.iloc[:, 4].to_numpy()

    interval = ts[2000:4000]

    lr = LinearRegression()
    lr.fit(np.arange(len(interval)), interval)
    print(lr.coef_)
    print(lr.intercept_)

    plt.plot(interval)
    plt.show()


if __name__ == '__main__':
    main()
