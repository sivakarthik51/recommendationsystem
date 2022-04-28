# Evaluating Graph-based Recommender Systems Effectiveness across Datasets in Similar Domains
## CS6240 Web Search Text Mining Project

### Light GCN

How to run:

1. Train LightGCN model for MovieLens data 
    
        cd LightGCN-PyTorch/code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="movielens" --topks="[20]" --recdim=64

2. Train LightGCN model for GoodReads data

        cd LightGCN-PyTorch/code && python main.py --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --dataset="goodreads" --topks="[20]" --recdim=64


### IGMC

How to run:

1. Train IGMC model for MovieLens Data

2. Train LightGCN model for GoodReads data

## Credit

1. LightGCN - Base code was used from https://github.com/gusye1234/LightGCN-PyTorch
2. IGMC - Base code was used from  