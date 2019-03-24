import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn import metrics
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection

from util import Timer


def rp_car():
    dataset = pd.read_csv("datasets/car/car.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, :-4]
    y = data.values[:, -4:]

    for n in range(1, 15):
        with Timer() as t:
            result = GaussianRandomProjection(n_components=n+1, random_state=10)
            result.fit(X)

        print('n_components={}, time to fit={}s'.format(n+1, t.interval))

        colors = list()

        X_rp = result.fit_transform(X)

        plt.figure()
        plt.title('RP (car.dataset, n_components={})'.format(n+1))

        for i, x in enumerate(X_rp):
            color = 'black'

            if np.array_equal(y[i], [1, 0, 0, 0]):
                color = 'red'
            elif np.array_equal(y[i], [0, 1, 0, 0]):
                color = 'blue'
            elif np.array_equal(y[i], [0, 0, 1, 0]):
                color = 'green'
            elif np.array_equal(y[i], [0, 0, 0, 1]):
                color = 'yellow'

            colors.append(color)

        plt.scatter(X_rp[:, 0], X_rp[:, 1], c=colors)
        plt.autoscale()
        plt.legend()
        plt.savefig('out/rp/car-rp-comp-{}.png'.format(n+1))


def rp_cancer():
    dataset = pd.read_csv("datasets/breastcancer/breast-cancer-wisconsin.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, 1:-1]  # strip sample id
    y = data.values[:, -1:]

    for n in range(1, 15):
        with Timer() as t:
            result = GaussianRandomProjection(n_components=n+1, random_state=10)
            result.fit(X)

        print('n_components={}, time to fit={}s'.format(n+1, t.interval))

        colors = list()

        X_rp = result.fit_transform(X)

        plt.figure()
        plt.title('RP (cancer.dataset, n_components={})'.format(n+1))

        for i, x in enumerate(X_rp):
            color = 'black'

            if np.array_equal(y[i], [0]):
                color = 'red'
            elif np.array_equal(y[i], [1]):
                color = 'blue'

            colors.append(color)

        plt.scatter(X_rp[:, 0], X_rp[:, 1], c=colors)
        plt.autoscale()
        plt.legend()
        plt.savefig('out/rp/cancer-rp-comp-{}.png'.format(n+1))


def rp(options):
    print('car')
    rp_car()
    print()
    print('cancer')
    rp_cancer()
    print('done')
