import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn import metrics
from sklearn.decomposition import FastICA

from util import Timer


def ica_car():
    dataset = pd.read_csv("datasets/car/car.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, :-4]
    y = data.values[:, -4:]

    for n in range(1, 15):
        with Timer() as t:
            result = FastICA(n_components=n+1, random_state=10)
            result.fit(X)

        print('n_components={}, time to fit={}s'.format(n+1, t.interval))

        colors = list()

        X_ica = result.fit_transform(X)

        plt.figure()
        plt.title('ICA (car.dataset, n_components={})'.format(n+1))

        for i, x in enumerate(X_ica):
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

        plt.scatter(X_ica[:, 0], X_ica[:, 1], c=colors)
        plt.autoscale()
        plt.legend()
        plt.savefig('out/ica/car-ica-comp-{}.png'.format(n+1))


def ica_cancer():
    dataset = pd.read_csv("datasets/breastcancer/breast-cancer-wisconsin.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, 1:-1]  # strip sample id
    y = data.values[:, -1:]

    for n in range(1, 15):
        with Timer() as t:
            result = FastICA(n_components=n+1, random_state=10)
            result.fit(X)

        print('n_components={}, time to fit={}s'.format(n+1, t.interval))

        colors = list()

        X_ica = result.fit_transform(X)

        plt.figure()
        plt.title('ICA (cancer.dataset, n_components={})'.format(n+1))

        for i, x in enumerate(X_ica):
            color = 'black'

            if np.array_equal(y[i], [0]):
                color = 'red'
            elif np.array_equal(y[i], [1]):
                color = 'blue'

            colors.append(color)

        plt.scatter(X_ica[:, 0], X_ica[:, 1], c=colors)
        plt.autoscale()
        plt.legend()
        plt.savefig('out/ica/cancer-ica-comp-{}.png'.format(n+1))


def ica(options):
    print('car')
    ica_car()
    print()
    print('cancer')
    ica_cancer()
    print('done')
