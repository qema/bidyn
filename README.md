# Bipartite Dynamic Representations (BiDyn)

Bipartite Dynamic Representations (BiDyn) is a general approach to transductive node classification on dynamic bipartite graphs, with additional components for the task of detecting abusive behavior in online communities.

It consists of three main components:
1. A neural architecture for handling dynamic networks
2. A memory-efficient training scheme for transductive node classification
3. A self-supervised pretraining task for abuse detection

## Problem setup
Given a dynamic bipartite graph and a subset of nodes with labeled classes, the model predicts the labels of the remaining nodes in the graph.

The dynamic bipartite graph consists of user nodes and item nodes connected by dynamic edges (edges with an associated timestamp). Each node and dynamic edge has an associated arbitrary feature vector.

The algorithm propagates signals through the graph via alternating updates of user and item node representations, alternating application of RNN and GNN layers. See paper and website for detailed explanation of the algorithm.

## Setup
1. Run `pip3 install -r requirements.txt` to install required libraries
2. Download the datasets:
```
wget -P data/ http://snap.stanford.edu/jodie/wikipedia.csv
wget -P data/ http://snap.stanford.edu/jodie/reddit.csv
wget -P data/ http://snap.stanford.edu/jodie/mooc.csv
```

## Usage
From the root project directory, run `python3 -m alt_batch.train --dataset=[dataset] --agg=[agg] --objective=[objective]` to train the model under different experimental conditions.

| Option | Possible values |
| ------------- | ------------- |
| dataset | "wikipedia", "reddit", "mooc" |
| agg | "mean" or "sum" graph convolution aggregation  |
| objective | "abuse" for node classification or "pretrain-link" for self-supervised pretraining followed by node classification |

Further configuration options can be found in `alt_batch/config.py`.

### Baseline: TGAT
From the root directory, run `python3 -m baselines.tgat_transductive [args]` where the command-line arguments follow the same format as the main script. Use `alt_batch/config.py` to adjust the configuration of TGAT as well.

### Baseline: DyRep, JODIE, TGN
From the `baselines/tgn` directory, run the same commands as in https://github.com/twitter-research/tgn to run DyRep, JODIE and TGN, but replace `train_supervised.py` with `train_supervised_transductive.py`. Note that the self-supervised training phase (before the main transductive task) is used in the experiments in the paper.

## Custom Datasets
To train and test on custom datasets, pass `--dataset=[dataset]` as the command-line option, and place your dataset in the file `data/[dataset].csv`. See https://github.com/srijankr/jodie, as well as the example datasets, for the dataset format. Note that unlike JODIE, BiDyn is designed for transductive node classification, i.e. each node in the graph gets a single label rather than a label per time step. Hence the label field for edges corresponding to a given node should have the same value across all its time steps. (The program will use the label of the edge at the last time step.)
