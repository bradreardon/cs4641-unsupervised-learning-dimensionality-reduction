#!/bin/bash
export PYENV_VERSION=cs4641
mkdir -p out/{kmeans,em,pca,ica,rp,cluster_dr}

#./main.py kmeans > out/kmeans/out.log
#./main.py em > out/em/out.log

#./main.py pca > out/pca/out.log
#./main.py ica > out/ica/out.log
#./main.py rp > out/rp/out.log

./main.py cluster_dr > out/cluster_dr/out.log
