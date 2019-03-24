import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score

from util import Timer


def em_car():
    dataset = pd.read_csv("datasets/car/car.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, :-4]
    y = data.values[:, -4:]

    x = list()
    y_train = list()
    y_cross = list()

    for i in range(15):
        with Timer() as t:
            clf = GaussianMixture(n_components=i+1, random_state=10, max_iter=500)
            clf.fit(X)

        print('n_components={}, time to fit={}s'.format(i+1, t.interval))

        train_score = clf.score(X, y)
        cross_val = cross_val_score(clf, X, y, cv=5)
        x.append(i+1)
        y_train.append(train_score)
        y_cross.append(np.mean(cross_val))

    plt.figure()
    plt.title('Scores for various n_components (car.dataset)')
    plt.xlabel('n_components')
    plt.ylabel('Score')
    plt.plot(x, y_train, label='Log-likelihood score')
    plt.plot(x, y_cross, label='Cross-validation score')
    plt.yscale('symlog')
    plt.legend()
    plt.savefig('out/em/car-components-testing.png')

    clf = GaussianMixture(n_components=4, random_state=10, max_iter=500)
    clf.fit(X)

    skplt.estimators.plot_learning_curve(
        clf, X, y, title="Learning Curve: EM (car.dataset, n_components=4)", cv=5,
        train_sizes=np.linspace(.1, 1.0, 10), n_jobs=-1)

    plt.yscale('symlog')
    plt.savefig('out/em/car-learning.png')


def em_cancer():
    dataset = pd.read_csv("datasets/breastcancer/breast-cancer-wisconsin.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, 1:-1]  # strip sample id
    y = data.values[:, -1:]

    x = list()
    y_train = list()
    y_cross = list()

    for i in range(15):
        with Timer() as t:
            clf = GaussianMixture(n_components=i+1, random_state=10, max_iter=500)
            clf.fit(X)

        print('n_components={}, time to fit={}s'.format(i+1, t.interval))

        train_score = clf.score(X, y)
        cross_val = cross_val_score(clf, X, y, cv=5)
        x.append(i+1)
        y_train.append(train_score)
        y_cross.append(np.mean(cross_val))

    plt.figure()
    plt.title('Scores for various n_components (cancer.dataset)')
    plt.xlabel('n_components')
    plt.ylabel('Score')
    plt.plot(x, y_train, label='Log-likelihood score')
    plt.plot(x, y_cross, label='Cross-validation score')
    plt.yscale('symlog')
    plt.legend()
    plt.savefig('out/em/cancer-components-testing.png')

    clf = GaussianMixture(n_components=3, random_state=10, max_iter=500)
    clf.fit(X)

    skplt.estimators.plot_learning_curve(
        clf, X, y, title="Learning Curve: EM (cancer.dataset, n_components=3)", cv=5,
        train_sizes=np.linspace(.1, 1.0, 10), n_jobs=-1)
    plt.yscale('symlog')
    plt.savefig('out/em/cancer-learning.png')


def em(options):
    print('car')
    em_car()
    print()
    print('cancer')
    em_cancer()
    print('done')
