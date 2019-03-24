import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.decomposition import FastICA, PCA
from sklearn.mixture import GaussianMixture
from sklearn.random_projection import GaussianRandomProjection
import scikitplot as skplt

from util import Timer


def cluster_dr_car(cluster_clf, dr_cls):
    dataset = pd.read_csv("datasets/car/car.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, :-4]
    y = data.values[:, -4:]

    _run(X, y, cluster_clf, dr_cls, 'car')


def cluster_dr_cancer(cluster_clf, dr_cls):
    dataset = pd.read_csv("datasets/breastcancer/breast-cancer-wisconsin.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, :-1]
    y = data.values[:, -1:]

    _run(X, y, cluster_clf, dr_cls, 'cancer')


CLF_MAP = {
    KMeans: 'kmeans',
    GaussianMixture: 'em'
}

DR_MAP = {
    PCA: 'pca',
    FastICA: 'ica',
    GaussianRandomProjection: 'rp',
    FeatureAgglomeration: 'fa',
}


def _run(X, y, clf, dr, dname):
    X_dr = None
    with Timer() as t:
        X_dr = dr.fit_transform(X)
        cluster_alg = clf.fit(X_dr)

    if isinstance(clf, KMeans):
        pass
    elif isinstance(clf, GaussianMixture):
        pass

    print(dname, type(clf).__name__, type(dr).__name__)
    print('\tTime: {0:.3f}'.format(t.interval))
    score = cluster_alg.score(X_dr, y)
    print('\tScore: {0:.3f}'.format(score))

    title = "Learning Curve: {} + {} ({}.dataset)".format(CLF_MAP[type(clf)], DR_MAP[type(dr)], dname)

    args = {}
    if isinstance(clf, GaussianMixture):
        args['train_sizes'] = np.linspace(.1, 1.0, 10)

    if X_dr is None:
        skplt.estimators.plot_learning_curve(clf, X, y, title=title, cv=5, **args)
    else:
        skplt.estimators.plot_learning_curve(clf, X_dr, y, title=title, cv=5, **args)

    # if isinstance(clf, GaussianMixture):
    #     plt.yscale('symlog')

    plt.savefig('out/cluster_dr/{}-{}-{}-learning.png'.format(dname, CLF_MAP[type(clf)], DR_MAP[type(dr)]))


def cluster_dr(options):
    car_clf = [
        KMeans(n_init=20, n_clusters=4, random_state=10, max_iter=500),
        GaussianMixture(n_components=4, random_state=10, max_iter=500)
    ]

    car_dr = [
        PCA(n_components=5, random_state=10),
        FastICA(n_components=5, random_state=10, max_iter=500),
        # GaussianRandomProjection(n_components=5, random_state=10),
    ]

    for _clf in car_clf:
        for _dr in car_dr:
            cluster_dr_car(_clf, _dr)

    cancer_clf = [
        KMeans(n_init=20, n_clusters=2, random_state=10, max_iter=500),
        GaussianMixture(n_components=3, random_state=10, max_iter=500)
    ]

    cancer_dr = [
        PCA(n_components=3, random_state=10),
        FastICA(n_components=3, random_state=10, max_iter=500),
        # GaussianRandomProjection(n_components=5, random_state=10),
    ]

    for _clf in cancer_clf:
        for _dr in cancer_dr:
            cluster_dr_cancer(_clf, _dr)
