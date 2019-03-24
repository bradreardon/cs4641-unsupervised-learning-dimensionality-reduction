import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score

from util import Timer


def pca_car():
    dataset = pd.read_csv("datasets/car/car.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, :-4]
    y = data.values[:, -4:]

    for n in range(1, 15):
        with Timer() as t:
            result = PCA(n_components=n+1, random_state=10)
            result.fit(X)

        print('n_components={}, time to fit={}s'.format(n+1, t.interval))
        print('n_components={}, score={}s'.format(n+1, result.score(X, y)))

        colors = list()

        X_pca = result.fit_transform(X)

        plt.figure()
        plt.title('PCA (car.dataset, n_components={})'.format(n+1))

        for i, x in enumerate(X_pca):
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

        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors)
        plt.autoscale()
        plt.legend()
        plt.savefig('out/pca/car-pca-comp-{}.png'.format(n+1))


def pca_cancer():
    dataset = pd.read_csv("datasets/breastcancer/breast-cancer-wisconsin.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, 1:-1]  # strip sample id
    y = data.values[:, -1:]

    for n in range(1, 15):
        with Timer() as t:
            result = PCA(n_components=n+1, random_state=10)
            result.fit(X)

        print('n_components={}, time to fit={}s'.format(n+1, t.interval))
        print('n_components={}, score={}s'.format(n+1, result.score(X, y)))

        colors = list()

        X_pca = result.fit_transform(X)

        plt.figure()
        plt.title('PCA (cancer.dataset, n_components={})'.format(n+1))

        for i, x in enumerate(X_pca):
            color = 'black'

            if np.array_equal(y[i], [0]):
                color = 'red'
            elif np.array_equal(y[i], [1]):
                color = 'blue'

            colors.append(color)

        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors)
        plt.autoscale()
        plt.legend()
        plt.savefig('out/pca/cancer-pca-comp-{}.png'.format(n+1))


def pca(options):
    print('car')
    pca_car()
    print()
    print('cancer')
    pca_cancer()
    print('done')
