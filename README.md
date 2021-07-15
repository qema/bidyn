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
| dataset  | "wikipedia", "reddit", "mooc" |
| agg | "mean" or "sum" graph convolution aggregation  |
| objective | "abuse" for node classification or "pretrain-link" for self-supervised pretraining followed by node classification |

Available configuration options can be found in `alt_batch/config.py`.
