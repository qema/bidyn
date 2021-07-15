from collections import defaultdict
import pickle
import random

import torch
import torch_scatter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind
from scipy.sparse import csr_matrix
import networkx as nx
from tqdm import tqdm, trange
import torch.optim as optim

from common import data
from common import util

from alt_batch.config import parse_args
from alt_batch.train import *
from baselines.tgat.module import TGAN
from baselines.tgat.graph import NeighborFinder
from baselines.tgat.utils import EarlyStopMonitor, RandEdgeSampler

def train(args, dataset):
    (us_to_edges, vs_to_edges, u_labels, v_labels, train_us, u_train_mask,
        u_val_mask, u_test_mask, feat_dim, event_counts_u, event_counts_v,
        u_to_idx, v_to_idx, mat_flat, u_feats, v_feats) = dataset
    # for tgat numbering, u's come first, then v's
    v_start_idx = len(us_to_edges)
    dataset_name = args.dataset
    device = torch.device(args.device)
    emb_dim = args.emb_dim
    batch_size_u = args.batch_size_u
    batch_size_v = args.batch_size_v
    max_time = max([e[0] for l in us_to_edges for e in l])
    use_inductive = False
    method = "tgat"
    print(dataset_name)
    print(method)

    # make data structure for neighbor finder
    adj_list = []
    eidx = 0
    edge_feats = []
    for l in us_to_edges:
        new_l = []
        for t, v, f in l:
            edge_feats.append(f)
            new_l.append((v + v_start_idx, eidx, t))
            eidx += 1
        adj_list.append(new_l)
    for l in vs_to_edges:
        new_l = []
        for t, u, f in l:
            edge_feats.append(f)
            new_l.append((u, eidx, t))
            eidx += 1
        adj_list.append(new_l)

    n_nodes = len(adj_list)
    n_edges = eidx

    ngh_finder = NeighborFinder(adj_list, uniform=True)

    if u_feats is None:
        node_feats = torch.zeros((n_nodes, len(edge_feats[0])))
    else:
        if v_feats is None:
            v_feats = torch.zeros((len(vs_to_edges), len(u_feats[0])))

        node_feats = torch.cat((u_feats, v_feats), dim=0)

    edge_feats = np.stack(edge_feats)
    print(node_feats.shape, edge_feats.shape)
    if len(edge_feats[0]) < len(node_feats[0]):
        edge_feats = np.concatenate((edge_feats, np.zeros((len(edge_feats),
            len(node_feats[0]) - len(edge_feats[0])))), axis=1)
        print(edge_feats.shape)

    model = TGAN(ngh_finder, node_feats.numpy(), edge_feats,
        num_layers=2, use_time="time", agg_method="attn",
        attn_mode="prod", seq_len=60, n_head=2,
        drop_out=0.1, node_dim=100, time_dim=100, device=device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    u_criterion = nn.NLLLoss()

    best_val_loss, best_test_auroc = float("inf"), 0
    task_schedule = util.make_task_schedule(args.objective, args.n_epochs)
    for epoch, tasks in enumerate(task_schedule):
        task = tasks[0]   # only use training task from schedule (always eval on abuse)
        train_loss_total, train_logp, train_labels = 0, [], []
        val_loss_total, val_logp, val_labels = 0, [], []
        test_logp, test_labels = [], []
        train_logp_vs, train_labels_vs = [], []
        val_logp_vs, val_labels_vs = [], []
        for group in ["train", "val"]:
            if group == "train":
                model.train()
            else:
                model.eval()
            with torch.set_grad_enabled(group == "train"):
                if task == "abuse":
                    sides_cfg = [("u", us_to_edges, vs_to_edges)]
                    for (side_name, side_to_edges, opp_side_to_edges) in sides_cfg:
                        batch_size = batch_size_u if side_name == "u" else batch_size_v
                        n_batches = int(np.ceil(len(side_to_edges) / batch_size))
                        all_mse = []
                        batch_pts = np.random.permutation(len(side_to_edges))
                        for batch_n in tqdm(range(n_batches)):
                            s_idx = batch_n * batch_size
                            e_idx = min((batch_n+1)*batch_size, len(side_to_edges))
                            batch_idxs = batch_pts[s_idx:e_idx]
                            batch = [side_to_edges[idx] for idx in batch_idxs]
                            lengths = [len(l) for l in batch]

                            if group == "train":
                                opt.zero_grad()

                            ts = np.array([max_time]*len(batch_idxs))
                            logits = model.binary_predict(batch_idxs, ts,
                                model.num_layers, 60)
                            logp = F.log_softmax(logits, dim=1)
                            if side_name == "u":
                                train_mask = u_train_mask[batch_idxs]
                                val_mask = u_val_mask[batch_idxs]
                                test_mask = u_test_mask[batch_idxs]
                                labels = u_labels[batch_idxs].to(device)
                                if group == "train":
                                    train_loss = u_criterion(logp[train_mask],
                                        labels[train_mask])
                                    train_logp.append(logp[train_mask].detach().cpu())
                                    train_labels.append(labels[train_mask].detach().cpu())
                                    if torch.sum(train_mask) > 0:
                                        train_loss_total += train_loss.item()
                                else:
                                    val_loss = u_criterion(logp[val_mask], labels[val_mask])
                                    val_logp.append(logp[val_mask].detach().cpu())
                                    val_labels.append(labels[val_mask].detach().cpu())
                                    if torch.sum(val_mask) > 0:
                                        val_loss_total += val_loss.item()
                                    test_logp.append(logp[test_mask].detach().cpu())
                                    test_labels.append(labels[test_mask].detach().cpu())

                                if group == "train":
                                    train_loss.backward()
                                    opt.step()

                    if side_name == "v":
                        print("Loss: {:.4f}".format(np.mean(all_mse)))

        # get scores
        train_logp = torch.cat(train_logp, dim=0).numpy()
        train_labels = torch.cat(train_labels, dim=0).numpy()
        try:
            train_auroc = roc_auc_score(train_labels, train_logp[:,1])
        except:
            train_auroc = 0
        if val_logp:
            val_logp = torch.cat(val_logp, dim=0).numpy()
            val_labels = torch.cat(val_labels, dim=0).numpy()
            val_auroc = roc_auc_score(val_labels, val_logp[:,1])
            test_logp = torch.cat(test_logp, dim=0).numpy()
            test_labels = torch.cat(test_labels, dim=0).numpy()
            test_auroc = roc_auc_score(test_labels, test_logp[:,1])
            if val_loss_total < best_val_loss:
                best_val_loss = val_loss_total
                best_test_auroc = test_auroc
                print("Best validation loss")
            print("Train loss: {:.4f}. Val loss: {:.4f}. ".format(
                train_loss_total, val_loss_total))
            print("Train AUROC: {:.4f}. Val AUROC: {:.4f}. "
                "Test AUROC: {:.4f}".format(train_auroc, val_auroc, test_auroc))
            if args.v_objective == "clf":
                train_labels_vs = torch.cat(train_labels_vs, dim=0).numpy()
                train_logp_vs = torch.cat(train_logp_vs, dim=0).numpy()
                v_train_auroc = roc_auc_score(train_labels_vs,
                    train_logp_vs[:,1])
                val_labels_vs = torch.cat(val_labels_vs, dim=0).numpy()
                val_logp_vs = torch.cat(val_logp_vs, dim=0).numpy()
                v_val_auroc = roc_auc_score(val_labels_vs, val_logp_vs[:,1])
                print("v train AUROC: {:.4f}. v val AUROC: {:.4f}".format(v_train_auroc, v_val_auroc))
        else:
            print("Train loss: {:.4f}. Train AUROC: {:.4f}".format(
                train_loss_total, train_auroc))

    print("Test AUROC with best validation model: {:.4f}".format(best_test_auroc))
    return best_test_auroc

if __name__ == "__main__":
    args = parse_args()

    aurocs = []
    for trial_n in range(args.n_trials):
        print(trial_n)
        dataset = load_dataset(args)
        auroc = train(args, dataset)
        aurocs.append(auroc)
    print(args)
    print(aurocs)
    print(np.mean(aurocs), np.std(aurocs, ddof=1))

