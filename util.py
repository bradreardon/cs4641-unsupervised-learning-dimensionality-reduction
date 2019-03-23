import pickle
import subprocess
import time

from sklearn import tree


def load_data_set(name):
    with open(f'datasets/{name}/{name}.dataset', 'rb') as f:
        d = pickle.load(f)

    return d


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
