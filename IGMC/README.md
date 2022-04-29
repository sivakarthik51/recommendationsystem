IGMC -- Inductive Graph-based Matrix Completion
===============================================================================

![alt text](https://github.com/muhanzhang/IGMC/raw/master/overall2.svg?sanitize=true "Illustration of IGMC")

https://github.com/muhanzhang/IGMC is the model this is based off of. We added recall@k and precision@k metrics and add a separate dataloader which loads directly from file. 

Requirements
------------

Stable version: Python 3.8.1 + PyTorch 1.4.0 + PyTorch_Geometric 1.4.2. If your PyG version is higher than this.

If you use latest PyTorch/PyG versions, you may also refer to the [latest](https://github.com/muhanzhang/IGMC/tree/latest) branch.

Install [PyTorch](https://pytorch.org/)

Install [PyTorch_Geometric](https://rusty1s.github.io/pytorch_geometric/build/html/notes/installation.html)

Other required python libraries: numpy, scipy, pandas, h5py, networkx, tqdm etc.

Usages
------





### MovieLens-100K and MovieLens-1M

To train on MovieLens-1M, type:
    
    python Main.py --data-name ml_1m_stratified --save-appendix _mnhp100 --data-appendix _mnph100 --max-nodes-per-hop 100 --testing --epochs 40 --save-interval 5 --adj-dropout 0 --lr-decay-step-size 20  --dynamic-train

where the --max-nodes-per-hop argument specifies the maximum number of neighbors to sample for each node during the enclosing subgraph extraction, whose purpose is to limit the subgraph size to accomodate large datasets. The --dynamic-train option makes the training enclosing subgraphs dynamically generated rather than generated in a preprocessing step and saved in disk, whose purpose is to reduce memory consumption. However, you may remove the option to generate a static dataset for future reuses. Append "--dynamic-test" to make the test dataset also dynamic. The default batch size is 50, if a batch cannot fit into your GPU memory, you can reduce batch size by appending "--batch-size 25" to the above command.

The results will be saved in "results/ml_1m\_stratified\_mnph200\_testmode/". The processed enclosing subgraphs will be saved in "data/ml\_1m\_stratified/testmode/" if you do not use dynamic datasets. 

    

To train on Goodreads, type:
    
    python Main.py --data-name goodreads_stratified --save-appendix _mnhp100 --data-appendix _mnph100 --max-nodes-per-hop 100 --testing --epochs 40 --save-interval 5 --adj-dropout 0 --lr-decay-step-size 20  --dynamic-train



Modify --ratio 0.2 to change the sparsity ratios. Attach --ensemble and run again to get the ensemble test results.

### Transfer learning

To repeat the transfer learning experiment in the paper (transfer the model trained previously on MovieLens-100K to Flixster, Douban, and YahooMusic), use the provided script by typing:

    ./run_transfer_exps.sh DATANAME

Replace DATANAME with flixster, douban and yahoo_music to transfer to each dataset. The results will be attached to each dataset's original "log.txt" file.

### Visualization

After training a model on a dataset, to visualize the testing enclosing subgraphs with the highest and lowest predicted ratings, type the following (we use Flixster as an example):

    python Main.py --data-name ml_1m_stratified --epochs 40 --testing --no-train --visualize

It will load "results/flixster\_testmode/model\_checkpoint40.pth" and save the visualization in "results/flixster\_testmode/visualization_flixster_prediction.pdf".

Check "Main.py" and "Main2.py" for more options to play with. Check "models.py" for the graph neural network used.

Reference
---------

If you find the code useful, please cite our paper.

    @inproceedings{
      Zhang2020Inductive,
      title={Inductive Matrix Completion Based on Graph Neural Networks},
      author={Muhan Zhang and Yixin Chen},
      booktitle={International Conference on Learning Representations},
      year={2020},
      url={https://openreview.net/forum?id=ByxxgCEYDS}
    }

