import numpy as np
import pandas as pd
import scikitplot as skplt
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from util import Timer


def kmeans_car():
    dataset = pd.read_csv("datasets/car/car.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, :-4]
    y = data.values[:, -4:]

    for n_clusters in [2, 3, 4, 5, 6]:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)
        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        with Timer() as t:
            clusterer = KMeans(n_init=20, n_clusters=n_clusters, random_state=10, max_iter=500)
            cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters)
        print("\tThe average silhouette_score is: {0:.3}".format(silhouette_avg))
        print("\tScore: {0:.3}".format(clusterer.score(X, y)))
        print("\tTraining time: {0:.3}s".format(t.interval))

        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("Silhouette plot")
        ax1.set_xlabel("Silhouette coefficient")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("Clustered data")
        ax2.set_xlabel("Feature space, 1st feature")
        ax2.set_ylabel("Feature space, 2nd feature")

        plt.suptitle("Silhouette analysis (car.dataset, n_clusters={})".format(n_clusters),
                     fontsize=14, fontweight='bold')

        plt.savefig('out/kmeans/car-{}-clusters.png'.format(n_clusters), dpi=150)

    clf = KMeans(n_init=20, n_clusters=4, random_state=10, max_iter=500)
    clf.fit(X)

    skplt.estimators.plot_learning_curve(
        clf, X, y, title="Learning Curve: k-means (car.dataset, n_clusters=4)", cv=5)
    plt.savefig('out/kmeans/car-learning.png')


def kmeans_cancer():

    dataset = pd.read_csv("datasets/breastcancer/breast-cancer-wisconsin.data", sep=',', header=None, low_memory=False)
    data = pd.get_dummies(dataset)

    X = data.values[:, 1:-1]  # strip sample id
    y = data.values[:, -1:]

    for n_clusters in [2, 3, 4, 5, 6]:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        ax1.set_xlim([-0.1, 1])
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        with Timer() as t:
            clusterer = KMeans(n_init=20, n_clusters=n_clusters, random_state=10, max_iter=500)
            cluster_labels = clusterer.fit_predict(X)

        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters)
        print("\tThe average silhouette_score is: {0:.3}".format(silhouette_avg))
        print("\tScore: {0:.3}".format(clusterer.score(X, y)))
        print("\tTraining time: {0:.3f}s".format(t.interval))

        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("Silhouette plot")
        ax1.set_xlabel("Silhouette coefficient")
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        centers = clusterer.cluster_centers_
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("Clustered data")
        ax2.set_xlabel("Feature space, 1st feature")
        ax2.set_ylabel("Feature space, 2nd feature")

        plt.suptitle("Silhouette analysis (breastcancer.dataset, n_clusters={})".format(n_clusters),
                     fontsize=14, fontweight='bold')

        plt.savefig('out/kmeans/cancer-{}-clusters.png'.format(n_clusters), dpi=150)

    clf = KMeans(n_init=20, n_clusters=2, random_state=10, max_iter=500)
    clf.fit(X)

    skplt.estimators.plot_learning_curve(
        clf, X, y, title="Learning Curve: k-means (breastcancer.dataset, n_clusters=2)", cv=5)
    plt.savefig('out/kmeans/cancer-learning.png')


def kmeans(options):
    print('car')
    kmeans_car()

    print()
    print('cancer')
    kmeans_cancer()

    print("done")
