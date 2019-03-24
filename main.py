#!/usr/bin/env python3
import argparse

from algorithms.kmeans import kmeans
from algorithms.em import em
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

    # Parse args and jump into correct algorithm
    options = parser.parse_args()
    options.func(options)
