import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm


def random_timeseries():
    n_points = 1000
    timeseries = []
    # value = float(random.randint(10, 100))
    # mu = value * (random.random() - 0.5) / 200
    # sigma = value / 100
    value = 1000
    mu = random.random() - 0.5
    sigma = 10
    timeseries.append(value)
    for i in range(n_points - 1):
        value += random.gauss(mu, sigma)
        timeseries.append(value)
    return timeseries


def calc_monotonicity(timeseries):
    # greater_than_list = []
    n_greater_than = 0
    sum_return = 0
    for i in range(len(timeseries) - 1):
        for j in range(i + 1, len(timeseries)):
            n_greater_than += int(timeseries[j] >= timeseries[i]) * 2 - 1
            sum_return += (timeseries[j] / timeseries[i]) - 1
            # greater_than_list.append(timeseries[j] >= timeseries[i])
    # assert len(greater_than_list) == len(timeseries) * (len(timeseries) - 1) / 2
    monotonicity = n_greater_than / (len(timeseries) * (len(timeseries) - 1) / 2)
    avg_return = sum_return / (len(timeseries) * (len(timeseries) - 1) / 2)
    return monotonicity, avg_return


def main():
    timeseries_list = []
    monotonicity_list = []
    avg_return_list = []
    dist_list = []
    for _ in tqdm(range(100)):
        timeseries = random_timeseries()
        monotonicity, avg_return = calc_monotonicity(timeseries)
        timeseries_list.append(timeseries)
        monotonicity_list.append(monotonicity)
        avg_return_list.append(avg_return)
        dist_list.append(monotonicity ** 2 + avg_return ** 2 if monotonicity > 0 and avg_return > 0 else 0.0)

    plt.subplot(1, 4, 1)
    plt.scatter(monotonicity_list, avg_return_list)
    plt.title("Mon / Avg ret")
    plt.xlabel("Mon")
    plt.ylabel("Avg ret")

    plt.subplot(1, 4, 2)
    best_idx = np.argmax(dist_list)
    plt.plot(timeseries_list[best_idx])
    plt.title(f'Mon: {monotonicity_list[best_idx]:.2} Avg ret: {avg_return_list[best_idx] * 100:.2f} %')

    plt.subplot(1, 4, 3)
    best_idx = np.argmax(monotonicity_list)
    plt.plot(timeseries_list[best_idx])
    plt.title(f'Mon: {monotonicity_list[best_idx]:.2} Avg ret: {avg_return_list[best_idx] * 100:.2f} %')

    plt.subplot(1, 4, 4)
    best_idx = np.argmax(avg_return_list)
    plt.plot(timeseries_list[best_idx])
    plt.title(f'Mon: {monotonicity_list[best_idx]:.2} Avg ret: {avg_return_list[best_idx] * 100:.2f} %')

    plt.show()


if __name__ == '__main__':
    main()
