#!/bin/bash
export PYENV_VERSION=cs4641
mkdir -p out/{kmeans,em}

./main.py kmeans > out/kmeans/out.log
./main.py em > out/em/out.log
