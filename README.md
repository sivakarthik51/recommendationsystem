# Evaluating Graph-based Recommender Systems Effectiveness across Datasets in Similar Domains
## CS6240 Web Search Text Mining Project

### Environment Setup

      conda create -n recsys
      conda activate recsys
      pip install requirements.txt 
### Light GCN

How to run:

1. Train LightGCN model for MovieLens data 
    
        cd LightGCN-PyTorch/code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="movielens" --topks="[20]" --recdim=64

2. Train LightGCN model for GoodReads data

        cd LightGCN-PyTorch/code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="goodreads" --topks="[20]" --recdim=64


### IGMC

How to run: Refer to instructions at IGMC/README.md

## Credit

1. LightGCN - Base code was used from https://github.com/gusye1234/LightGCN-PyTorch
2. IGMC - Base code was used from https://github.com/muhanzhang/IGMC