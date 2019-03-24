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


def nn_cluster_car(nn, clf):
    dataset = pd.read_csv("datasets/car/car.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, :-4]
    y = data.values[:, -4:]

    _run(X, y, nn, clf, 'car')


def nn_cluster_cancer(nn, clf):
    dataset = pd.read_csv("datasets/breastcancer/breast-cancer-wisconsin.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, :-1]
    y = data.values[:, -1:]

    _run(X, y, nn, clf, 'cancer')


CLF_MAP = {
    KMeans: 'kmeans',
    GaussianMixture: 'em'
}


def _run(X, y, nn, clf, dname):
    with Timer() as t:
        if clf is None:
            nn.fit(X, y)
            X_cluster = None
        else:
            X_cluster = clf.fit_transform(X)
            nn = nn.fit(X_cluster, y)

    print(dname, type(clf).__name__)
    print('\tTime: {0:.3f}'.format(t.interval))
    if X_cluster is not None:
        score = nn.score(X_cluster, y)
    else:
        score = nn.score(X, y)
    print('\tScore: {0:.3f}'.format(score))

    if clf is None:
        title = "Learning Curve: NN ({}.dataset)".format(dname)
        skplt.estimators.plot_learning_curve(nn, X, y, title=title, cv=5)
        plt.savefig('out/nn_cluster/{}-learning.png'.format(dname))
    else:
        title = "Learning Curve: NN + {} ({}.dataset)".format(CLF_MAP[type(clf)], dname)
        skplt.estimators.plot_learning_curve(nn, X_cluster, y, title=title, cv=5)
        plt.savefig('out/nn_cluster/{}-{}-learning.png'.format(dname, CLF_MAP[type(clf)]))


def nn_cluster(options):

    car_clf = [
        None,
        KMeans(n_init=20, n_clusters=4, random_state=10, max_iter=500),
        # GaussianMixture(n_components=4, random_state=10, max_iter=500)
    ]

    for _clf in car_clf:
        nn_cluster_car(
            MLPClassifier(
                solver='adam', warm_start=True, max_iter=1000
            ),
            _clf)

    cancer_clf = [
        None,
        KMeans(n_init=20, n_clusters=2, random_state=10, max_iter=500),
        # GaussianMixture(n_components=3, random_state=10, max_iter=500)
    ]

    for _clf in cancer_clf:
        nn_cluster_cancer(
            MLPClassifier(
                solver='adam', warm_start=True, max_iter=1000
            ),
            _clf)
