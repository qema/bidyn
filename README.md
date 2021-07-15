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

### Setup
1. Run `pip3 install -r requirements.txt` to install required libraries
2. Download the datasets:

    wget -P data/ http://snap.stanford.edu/jodie/wikipedia.csv
    wget -P data/ http://snap.stanford.edu/jodie/reddit.csv
    wget -P data/ http://snap.stanford.edu/jodie/mooc.csv

### Train the model
From the root project directory, run `python3 -m alt_batch.train --dataset=wikipedia --agg=sum` to train the model on abuse classification on the Wikipedia dataset.

### Usage
The module `python3 -m subgraph_matching.alignment.py [--query_path=...] [--target_path=...]` provides a utility to obtain all pairs of corresponding matching scores, given a pickle file of the query and target graphs in networkx format. Run the module without these arguments for an example using random graphs. 
If exact isomorphism mapping is desired, a conflict resolution algorithm can be applied on the
alignment matrix (the output of alignment.py). 
Such algorithms are available in recent works. For example: [Deep Graph Matching
Consensus](https://arxiv.org/abs/2001.09621) and [Convolutional Set Matching for Graph
Similarity](https://arxiv.org/abs/1810.10866).

Both synthetic data (`common/combined_syn.py`) and real-world data (`common/data.py`) can be used to train the model.
One can also train with synthetic data, and transfer the learned model to make inference on real
data (see `subgraph_matching/test.py`).
The `neural_matching` folder contains an encoder that uses GNN to map the query and target into the
embedding space and make subgraph predictions.

Available configurations can be found in `subgraph_matching/config.py`.


## Frequent Subgraph Mining
This package also contains an implementation of SPMiner, a graph neural network based framework to extract frequent subgraph patterns from an input graph dataset.

Running the pipeline consists of training the encoder on synthetic data, then running the decoder on the dataset from which to mine patterns.

Full configuration options can be found in `subgraph_matching/config.py` and `subgraph_mining/config.py`.

### Run SPMiner
To run SPMiner to identify common subgraph pattern, the prerequisite is to have a checkpoint of
trained subgraph matching model (obtained by training the GNN encoder).
The config argument `args.model_path` (`subgraph_matching/config.py`) specifies the location of the
saved checkpoint, and is shared for both the `subgraph_matching` and `subgraph_mining` models.
1. `python3 -m subgraph_mining.decoder --dataset=enzymes --node_anchored`

Full configuration options can be found in `decoder/config.py`. SPMiner also shares the
configurations of NeuroMatch `subgraph_matching/config.py` since it's used as a subroutine.

## Analyze results
- Analyze the order embeddings after training the encoder: `python3 -m analyze.analyze_embeddings --node_anchored`
- Count the frequencies of patterns generated by the decoder: `python3 -m analyze.count_patterns --dataset=enzymes --out_path=results/counts.json --node_anchored`
- Analyze the raw output from counting: `python3 -m analyze.analyze_pattern_counts --counts_path=results/`

## Dependencies
The library uses PyTorch and [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) to implement message passing graph neural networks (GNN). 
It also uses [DeepSNAP](https://github.com/snap-stanford/deepsnap), which facilitates easy use
of graph algorithms (such as subgraph operation and matching operation) to be performed during training for every iteration, 
thanks to its synchronization between an internal graph object (such as a NetworkX object) and the Pytorch Geometric Data object.

Detailed library requirements can be found in requirements.txt
