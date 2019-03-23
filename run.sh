#!/bin/bash
export PYENV_VERSION=cs4641
mkdir -p out/{kmeans,em}

python main.py kmeans
python main.py em