import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.decomposition import FastICA, PCA
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.random_projection import GaussianRandomProjection
import scikitplot as skplt

from util import Timer


def nn_dr_car(nn, dr_cls):
    dataset = pd.read_csv("datasets/car/car.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, :-4]
    y = data.values[:, -4:]

    _run(X, y, nn, dr_cls, 'car')


def nn_dr_cancer(nn, dr_cls):
    dataset = pd.read_csv("datasets/breastcancer/breast-cancer-wisconsin.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, :-1]
    y = data.values[:, -1:]

    _run(X, y, nn, dr_cls, 'cancer')

DR_MAP = {
    PCA: 'pca',
    FastICA: 'ica',
    GaussianRandomProjection: 'rp',
    FeatureAgglomeration: 'fa',
}


def _run(X, y, nn, dr, dname):
    with Timer() as t:
        if dr is None:
            nn.fit(X, y)
            X_dr = None
        else:
            X_dr = dr.fit_transform(X)
            nn = nn.fit(X_dr, y)

    print(dname, type(dr).__name__)
    print('\tTime: {0:.3f}'.format(t.interval))
    if X_dr is not None:
        score = nn.score(X_dr, y)
    else:
        score = nn.score(X, y)
    print('\tScore: {0:.3f}'.format(score))

    if dr is None:
        title = "Learning Curve: NN ({}.dataset)".format(dname)
        skplt.estimators.plot_learning_curve(nn, X, y, title=title, cv=5)
        plt.savefig('out/nn_dr/{}-learning.png'.format(dname))
    else:
        title = "Learning Curve: NN + {} ({}.dataset)".format(DR_MAP[type(dr)], dname)
        skplt.estimators.plot_learning_curve(nn, X_dr, y, title=title, cv=5)
        plt.savefig('out/nn_dr/{}-{}-learning.png'.format(dname, DR_MAP[type(dr)]))


def nn_dr(options):

    car_dr = [
        None,
        PCA(n_components=5, random_state=10),
        FastICA(n_components=5, random_state=10, max_iter=500),
        # GaussianRandomProjection(n_components=5, random_state=10),
    ]

    for _dr in car_dr:
        nn_dr_car(
            MLPClassifier(
                solver='adam', warm_start=True, max_iter=1000
            ),
            _dr)

    cancer_dr = [
        None,
        PCA(n_components=7, random_state=10),
        FastICA(n_components=7, random_state=10, max_iter=500),
        # GaussianRandomProjection(n_components=5, random_state=10),
    ]

    for _dr in cancer_dr:
        nn_dr_cancer(
            MLPClassifier(
                solver='adam', warm_start=True, max_iter=1000
            ),
            _dr)
