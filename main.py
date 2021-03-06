#!/usr/bin/env python3
import argparse

from algorithms.clustering_with_dr import cluster_dr
from algorithms.kmeans import kmeans
from algorithms.em import em
from algorithms.nn_with_clustering import nn_cluster
from algorithms.nn_with_dr import nn_dr
from algorithms.pca import pca
from algorithms.ica import ica
from algorithms.rp import rp
from algorithms.neural_net import neural_net


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Performs unsupervised learning and dimensionality reduction")
    subparsers = parser.add_subparsers()
    parser.set_defaults(func=lambda x: parser.print_help())

    # # Neural net parser
    # parser_neural_net = subparsers.add_parser(
    #     'neural_net', help='Runs supervised learning using neural networks.')
    # parser_neural_net.set_defaults(func=neural_net)

    parser_kmeans = subparsers.add_parser(
        'kmeans', help='Runs unsupervised learning using k-means clustering.')
    parser_kmeans.set_defaults(func=kmeans)

    parser_em = subparsers.add_parser(
        'em', help='Runs unsupervised learning using expectation maximization.')
    parser_em.set_defaults(func=em)

    parser_pca = subparsers.add_parser(
        'pca', help='Runs dimensionality reduction using PCA.')
    parser_pca.set_defaults(func=pca)

    parser_ica = subparsers.add_parser(
        'ica', help='Runs dimensionality reduction using ICA.')
    parser_ica.set_defaults(func=ica)

    parser_rp = subparsers.add_parser(
        'rp', help='Runs dimensionality reduction using Randomized Projections.')
    parser_rp.set_defaults(func=rp)

    parser_cluster_dr = subparsers.add_parser(
        'cluster_dr', help='Runs dimensionality reduction, then clustering.')
    parser_cluster_dr.set_defaults(func=cluster_dr)

    parser_nn_dr = subparsers.add_parser(
        'nn_dr', help='Runs dimensionality reduction, then a neural net.')
    parser_nn_dr.set_defaults(func=nn_dr)

    parser_nn_cluster = subparsers.add_parser(
        'nn_cluster', help='Runs clustering, then a neural net.')
    parser_nn_cluster.set_defaults(func=nn_cluster)

    # Parse args and jump into correct algorithm
    options = parser.parse_args()
    options.func(options)
