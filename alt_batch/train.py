from collections import defaultdict, Counter
import os
import pickle
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_ind
from scipy.sparse import csr_matrix
import torch
import torch_scatter
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from alt_batch.config import parse_args
from common import data
from common import util

class BipartiteAltBatcher(nn.Module):
    def __init__(self, feat_dim, hidden_dim, emb_dim, n_layers, dropout=0.0,
        device=torch.device("cuda"), u_feat_dim=0, v_feat_dim=0, v_objective=None):
        super().__init__()
        self.u_encoder = nn.LSTM(feat_dim + emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.v_encoder = nn.LSTM(feat_dim + emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.v_decoder = nn.LSTM(feat_dim + emb_dim, hidden_dim, n_layers, dropout=dropout)

        self.link_pred = nn.Sequential(nn.Linear(2 * emb_dim + 2*32, 128),
            nn.ReLU(), nn.Linear(128, 2))

        self.u_fc1 = nn.Linear(hidden_dim + u_feat_dim, emb_dim)
        self.u_fc2 = nn.Linear(emb_dim, 2)
        self.v_decoder_fc1 = nn.Linear(hidden_dim + v_feat_dim, emb_dim)
        self.v_decoder_fc2 = nn.Linear(emb_dim, feat_dim + emb_dim)
        self.v_decoder_fc2_clf = nn.Linear(emb_dim, 2)
        self.device = device

        self.u_pre_mlp = nn.Sequential(nn.Linear(feat_dim + emb_dim,
            128), nn.ReLU(), nn.Linear(128, feat_dim + emb_dim))
        self.v_pre_mlp = nn.Sequential(nn.Linear(feat_dim + emb_dim,
            128), nn.ReLU(), nn.Linear(128, feat_dim + emb_dim))

    def forward(self, seqs, feats, lengths, select="u", teacher_forcing=0.5,
        h0=None):
        device = self.device
        if h0 is not None:
            h0 = torch.stack((h0, torch.zeros_like(h0).to(device)), dim=0)
            h0 = (h0, torch.zeros_like(h0))

        packed_seqs = nn.utils.rnn.pack_padded_sequence(seqs, lengths,
            enforce_sorted=False).to(device)
        if select == "u":
            _, (ht, ct) = self.u_encoder(packed_seqs, h0)
            h = ht[-1].squeeze(0)
            if feats is not None:
                h = torch.cat((h, feats), axis=-1)
            embs = self.u_fc1(h)
            return embs, self.u_fc2(embs)
        elif select in ["v-mean", "v-link", "v-gcnprop", "v-imppool"]:
            _, (ht, ct) = self.v_encoder(packed_seqs)
            embs = self.v_decoder_fc1(ht[-1].squeeze(0))
            return embs, self.v_decoder_fc2(embs)
        elif select == "v-clf":
            _, (ht, ct) = self.v_encoder(packed_seqs, h0)
            h = ht[-1].squeeze(0)
            if feats is not None:
                h = torch.cat((h, feats), axis=-1)
            embs = self.v_decoder_fc1(h.squeeze(0))
            return embs, self.v_decoder_fc2_clf(embs)
        elif select == "v-autoenc-labels":
            out, (ht, ct) = self.u_encoder(packed_seqs, h0)
            out, _ = nn.utils.rnn.pad_packed_sequence(out)
            out = self.v_decoder_fc2_clf(out)
            h = ht[-1].squeeze(0)
            embs = self.u_fc1(h)
            return embs, out
        elif select == "v-autoenc":
            _, (h, c) = self.u_encoder(packed_seqs)
            embs = self.u_fc1(h[-1].squeeze(0))

            outputs = torch.zeros_like(seqs).to(device)
            inp = torch.zeros((seqs.shape[1], seqs.shape[2])).to(device)
            for t in range(seqs.shape[0]):
                out, (h, c) = self.v_decoder(inp.unsqueeze(0), (h, c))
                out_proc = self.v_decoder_fc2(self.v_decoder_fc1(out[-1]))
                outputs[t] = out_proc

                inp = (seqs[t] if random.random() < teacher_forcing else out_proc)

            return embs, outputs

def load_dataset(args):
    dataset = data.load_dataset(args.dataset, get_edges=True)

    u_to_idx, v_to_idx, us_to_edges, vs_to_edges = data.get_edge_lists(dataset)

    u_labels = torch.zeros(len(us_to_edges), dtype=torch.long)
    n_pos_labels = 0
    for u, i in u_to_idx.items():
        u_labels[i] = dataset["labels"][u]
        if dataset["labels"][u] == 1:
            n_pos_labels += 1
    print("NUMBER OF POSITIVE LABELS:", n_pos_labels)
    random.seed(2020)
    us = set(range(len(u_to_idx)))
    train_amt = args.train_amt
    train_us = set(random.sample(range(len(us)), int(len(us)*train_amt)))
    val_us = set(random.sample([x for x in range(len(us)) if x not in
        train_us], int(len(us)*(1-train_amt)/2)))
    test_us = (us - train_us) - val_us
    u_train_mask = torch.tensor([u in train_us for u in range(len(us))])
    u_val_mask = torch.tensor([u in val_us for u in range(len(us))])
    u_test_mask = torch.tensor([u in test_us for u in range(len(us))])
    feat_dim = len(us_to_edges[0][0][2])

    if args.use_discrete_time_batching:
        mats = dataset["mats"]
        max_time = max(list(mats.keys()))
        ts = np.arange(1, max_time + 1)
        event_counts_raw = np.zeros((mats[1].shape[0], max_time))
        for t in ts:
            event_counts_raw[:,t-1] = mats[t].sum(axis=1).flatten()
        event_counts_u = np.zeros((len(u_to_idx), max_time))
        event_counts_v = np.zeros((len(v_to_idx), max_time))
        print("getting event counts")
        for k, v in tqdm(u_to_idx.items()):
            event_counts_u[v] = event_counts_raw[k]
        for k, v in tqdm(v_to_idx.items()):
            event_counts_v[v] = event_counts_raw[k]
        event_counts_u = torch.from_numpy(event_counts_u)
        event_counts_v = torch.from_numpy(event_counts_v)
    else:
        # make feats into tensors for faster batching later
        us_to_edges = [[(x, y, torch.tensor(z)) for (x, y, z) in l]
            for l in us_to_edges]
        vs_to_edges = [[(x, y, torch.tensor(z)) for (x, y, z) in l]
            for l in vs_to_edges]
        # time encode each event
        def one_eye(x):
            v = torch.zeros(max(len(us_to_edges), len(vs_to_edges)))
            v[x] = 1
            return v
        max_time = max([e[0] for l in us_to_edges for e in l])
        us_to_edges = [[(x, y, torch.cat((z.float(), time_encode(x /
            max_time).view(-1))))
            for (x, y, z) in l] for l in us_to_edges]
        vs_to_edges = [[(x, y, torch.cat((z.float(), time_encode(x /
            max_time).view(-1))))
            for (x, y, z) in l] for l in vs_to_edges]
        feat_dim = len(us_to_edges[0][0][2])

        event_counts_u, event_counts_v = None, None

    if args.objective in ["pretrain-link", "link"]:
        mat_flat = dataset["mat_flat"]
        mat_flat = mat_flat[sorted(u_to_idx.keys())][:,sorted(v_to_idx.keys())]
    else:
        mat_flat = None

    u_feats, v_feats = None, None
    if args.v_objective == "clf":
        v_labels = torch.tensor([0]*len(vs_to_edges))
        for k, v in v_to_idx.items():
            v_labels[v] = dataset["labels"][k]
    else:
        v_labels = None

    print(feat_dim, "EDGE FEATURE DIM")
    return (us_to_edges, vs_to_edges, u_labels, v_labels, train_us, u_train_mask,
        u_val_mask, u_test_mask, feat_dim, event_counts_u, event_counts_v,
        u_to_idx, v_to_idx, mat_flat, u_feats, v_feats)

max_time_cache = None
def get_batch(args, model, batch, batch_idxs, lengths,
    feat_dim, emb_dim, side_name, side_to_edges, side_embs, opp_side_to_edges,
    opp_side_embs, event_counts_u, event_counts_v, side_feats,
    masked_edges=None):
    global max_time_cache
    device = args.device

    if masked_edges is not None:
        batch = [[(t, v, f) for t, v, f in l if t < mask_t] for u, l, mask_t in
            zip(batch_idxs, batch, masked_edges)]
        lengths = [max(len(l), 1) for l in batch]

    if args.use_discrete_time_batching:
        feats_block = torch.zeros((max(lengths), len(batch), feat_dim))
        neigh_idxs = torch.tensor([[l[j][1] if j < len(l)
            else -1 for i, l in enumerate(batch)] for j in range(max(lengths))])

        embs_block = opp_side_embs[neigh_idxs.view(-1)].view(-1,
            len(batch), emb_dim)
        events = torch.cat((feats_block, embs_block), dim=-1)
        events = events.to(device)
        if max_time_cache is None:
            max_time_cache = max([[ll[0] for l in batch for ll in l]])
        max_time = max_time_cache

        # last index is a dummy index to put masked-out events
        time_idxs = torch.tensor([[l[j][0] - 1 if j < len(l)
            else max_time for i, l in enumerate(batch)]
            for j in range(max(lengths))]).to(device)
        batch_t = torch_scatter.scatter_mean(events,
            time_idxs, dim=0, dim_size=max_time+1)
        batch_t = batch_t[:-1,:,:]   # remove masked events

        lengths = [max_time]*len(batch)

        if side_name == "u":
            batch_t[:,:,0] = event_counts_u[batch_idxs].T
        else:
            batch_t[:,:,0] = event_counts_v[batch_idxs].T
        batch_t = batch_t.to(device)
    else:
        feats_block = torch.stack([torch.stack([l[j][2] if j < len(l) else
            torch.zeros(feat_dim) for j in range(max(lengths))]) for l in
            batch]).type(torch.float)
        feats_block = feats_block.permute(1, 0, 2)  # (time, batch, feats)
        neigh_idxs = torch.tensor([[l[j][1] if j < len(l)
            else -1 for i, l in enumerate(batch)]
            for j in range(max(lengths))])

        embs_block = opp_side_embs[neigh_idxs.view(-1)].view(-1,
            len(batch), emb_dim)
        batch_t = torch.cat((feats_block, embs_block), dim=-1)
        batch_t = batch_t.to(device)

    side_feats_batch = (side_feats[batch_idxs].to(device) if side_feats is not
        None else None)

    return batch_t, side_feats_batch, lengths

def is_edge_present_batch(us_to_edges, us, vs, ts, labels_flat):
    out = np.zeros(len(us), dtype=int)
    for i, (u, v, (t_start, t_end)) in enumerate(zip(us, vs, ts)):
        if labels_flat[i] == 1:
            for t, w, _ in us_to_edges[u]:
                if t >= t_start and t <= t_end and w == v:
                    out[i] = 1
                    break
                if t > t_end:
                    break
    return out

def time_encode(x):
    if type(x) == float:
        x = torch.tensor(x).unsqueeze(0)
    x *= 100
    x = x.unsqueeze(1)
    pe = torch.zeros((len(x), 32))
    coef = torch.exp((torch.arange(0, 32, 2, dtype=torch.float) *
        -(np.log(10000.0) / 32)))
    pe[:,0::2] = torch.sin(x * coef)
    pe[:,1::2] = torch.cos(x * coef)
    return pe

def train(args, dataset):
    (us_to_edges, vs_to_edges, u_labels, v_labels, train_us, u_train_mask,
        u_val_mask, u_test_mask, feat_dim, event_counts_u, event_counts_v,
        u_to_idx, v_to_idx, mat_flat, u_feats, v_feats) = dataset
    dataset_name = args.dataset
    device = torch.device(args.device)
    emb_dim = args.emb_dim
    batch_size_u = args.batch_size_u
    batch_size_v = args.batch_size_v
    use_inductive = False
    method = args.method
    print(dataset_name)
    print(method)

    u_embs = torch.zeros((len(us_to_edges) + 1, emb_dim)) # last entry is 0
    v_embs = torch.zeros((len(vs_to_edges) + 1, emb_dim)) # last entry is 0

    if args.objective in ["pretrain-link", "link"]:
        max_time = max([e[0] for l in us_to_edges for e in l])

    # preprocessing for specific v tasks
    if args.v_objective == "autoenc-labels":
        print("Making v target sequences")
        if args.use_discrete_time_batching:
            max_time = max([e[0] for l in us_to_edges for e in l])
            v_target_seqs = torch.zeros(len(vs_to_edges), max_time)
            for v, edges in tqdm(enumerate(vs_to_edges)):
                for t, u, feats in edges:
                    v_target_seqs[v, t - 1] += u_labels[u]
        else:
            max_len = max([len(l) for l in vs_to_edges])
            v_target_seqs = torch.zeros(len(vs_to_edges), max_len,
                dtype=torch.long)
            for v, edges in tqdm(enumerate(vs_to_edges)):
                for i, (t, u, feats) in enumerate(edges):
                    v_target_seqs[v, i] = u_labels[u]
    elif args.v_objective == "clf":
        vs = set(range(len(v_to_idx)))
        train_vs = set(random.sample(range(len(vs)), int(len(vs)*0.8)))
        val_vs = set(random.sample([x for x in range(len(vs)) if x not in
            train_vs], int(len(vs)*0.2)))
        v_train_mask = torch.tensor([v in train_vs for v in range(len(vs))])
        v_val_mask = torch.tensor([v in val_vs for v in range(len(vs))])

    model = BipartiteAltBatcher(feat_dim, 64, args.emb_dim, 2,
        dropout=args.dropout, device=args.device, u_feat_dim=len(u_feats[0]) if
        u_feats is not None else 0, v_objective=args.v_objective,
        v_feat_dim=len(v_feats[0]) if v_feats is not None else 0)
    model.to(device)

    link_criterion = nn.NLLLoss(weight=torch.tensor([0.001, 0.999]).to(device))
    opt = {"u": optim.Adam(model.parameters(), lr=1e-3),
        "v": optim.Adam(model.parameters(), lr=1e-3)}

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
                    sides_cfg = [("u", us_to_edges, vs_to_edges, u_embs, v_embs)]
                    if method == "alt-batch" and group == "train":
                        sides_cfg += [("v", vs_to_edges, us_to_edges, v_embs, u_embs)]
                    for (side_name, side_to_edges, opp_side_to_edges,
                        side_embs, opp_side_embs) in sides_cfg:
                        batch_size = batch_size_u if side_name == "u" else batch_size_v
                        all_mse = []
                        batch_pts = np.random.permutation(len(side_to_edges))
                        n_batches = int(np.ceil(len(batch_pts) / batch_size))
                        for batch_n in tqdm(range(n_batches)):
                            s_idx = batch_n * batch_size
                            e_idx = min((batch_n+1)*batch_size, len(batch_pts))
                            batch_idxs = batch_pts[s_idx:e_idx]
                            batch = [side_to_edges[idx] for idx in batch_idxs]
                            lengths = [len(l) for l in batch]

                            if group == "train":
                                opt[side_name].zero_grad()

                            batch_t, side_feats_batch, lengths = get_batch(
                                args, model, batch, batch_idxs, lengths,
                                feat_dim, emb_dim, side_name, side_to_edges, side_embs,
                                opp_side_to_edges, opp_side_embs, event_counts_u, event_counts_v,
                                u_feats if side_name == "u" else v_feats)
                            select = "u" if side_name == "u" else "v-{}".format(args.v_objective)
                            if not (side_name == "v" and args.v_objective in
                                ["mean", "gcnprop", "imppool"]):
                                embs, out = model(batch_t, side_feats_batch,
                                    lengths, select=select)
                            if side_name == "u":
                                train_mask = u_train_mask[batch_idxs]
                                val_mask = u_val_mask[batch_idxs]
                                test_mask = u_test_mask[batch_idxs]
                                logp = F.log_softmax(out, dim=1)
                                labels = u_labels[batch_idxs].to(device)
                                if group == "train":
                                    train_loss = F.nll_loss(logp[train_mask],
                                        labels[train_mask])
                                    train_logp.append(logp[train_mask].detach().cpu())
                                    train_labels.append(labels[train_mask].detach().cpu())
                                    if torch.sum(train_mask) > 0:
                                        train_loss_total += train_loss.item()
                                else:
                                    val_loss = F.nll_loss(logp[val_mask], labels[val_mask])
                                    val_logp.append(logp[val_mask].detach().cpu())
                                    val_labels.append(labels[val_mask].detach().cpu())
                                    if torch.sum(val_mask) > 0:
                                        val_loss_total += val_loss.item()
                                    test_logp.append(logp[test_mask].detach().cpu())
                                    test_labels.append(labels[test_mask].detach().cpu())

                                if group == "train":
                                    train_loss.backward()
                                    opt[side_name].step()
                                    side_embs[batch_idxs] = embs.detach().cpu()
                            elif side_name == "v":
                                assert group == "train"
                                if args.v_objective == "mean":
                                    if args.use_discrete_time_batching:
                                        weights = batch_t[:,:,0]
                                        batch_t *= weights.to(device).unsqueeze(
                                            -1).expand(-1, -1, feat_dim)
                                    targets = torch.sum(batch_t[:,:,feat_dim:], axis=0)
                                    if args.agg == "sum":
                                        embs = targets.detach().cpu()
                                        embs = (side_embs[batch_idxs]*0.9 +
                                            targets.detach().cpu()*0.1)
                                    else:
                                        targets = (targets.T / torch.tensor([len(l)
                                            for l in batch]).to(device)).T
                                        embs = (side_embs[batch_idxs]*0.9 +
                                            targets.detach().cpu()*0.1)
                                    train_loss = torch.tensor(0)
                                elif args.v_objective == "autoenc":
                                    max_len = max(lengths)
                                    length_mask = torch.tensor([[True]*l + [False]*(max_len-l)
                                        for l in lengths]).to(device).T
                                    train_loss = F.mse_loss(out[length_mask], batch_t[length_mask])
                                    train_loss.backward()
                                    opt[side_name].step()
                                elif args.v_objective == "autoenc-labels":
                                    max_len = max(lengths)
                                    target_seq = v_target_seqs[batch_idxs,:max_len].to(device).T
                                    length_mask = torch.tensor([[True]*l + [False]*(max_len-l)
                                        for l in lengths]).to(device).T
                                    train_loss = F.nll_loss(
                                        F.log_softmax(out[length_mask], dim=-1),
                                        target_seq[length_mask])
                                    train_loss.backward()
                                    opt[side_name].step()
                                elif args.v_objective == "clf":
                                    logp = F.log_softmax(out, dim=1)
                                    labels = v_labels[batch_idxs].to(device)
                                    train_mask = v_train_mask[batch_idxs]
                                    val_mask = v_val_mask[batch_idxs]
                                    train_loss = F.nll_loss(logp[train_mask],
                                        labels[train_mask])
                                    train_logp_vs.append(logp[train_mask].detach().cpu())
                                    train_labels_vs.append(labels[train_mask].detach().cpu())
                                    val_logp_vs.append(logp[val_mask].detach().cpu())
                                    val_labels_vs.append(labels[val_mask].detach().cpu())
                                    if torch.sum(train_mask) > 0:
                                        train_loss_total += train_loss.item()

                                    train_loss.backward()
                                    opt[side_name].step()
                                elif args.v_objective == "link":
                                    # NOTE: this is link pred for v side only!!
                                    us_pos = [random.choice([u for t, u, f in l])
                                        for l in batch]
                                    us_neg = [random.randint(0,
                                        len(us_to_edges)-1) for _ in us_pos]
                                    us = us_pos + us_neg
                                    u_batch = [us_to_edges[idx] for idx in us]
                                    u_lengths = [len(l) for l in u_batch]
                                    u_batch_t, u_feats_batch, u_lengths = get_batch(args, model,
                                        u_batch, us, u_lengths,
                                        feat_dim, emb_dim, "u", us_to_edges, u_embs,
                                        vs_to_edges, v_embs, event_counts_u, event_counts_v,
                                        u_feats)

                                    u_embs_batch, out = model(u_batch_t,
                                        u_feats_batch, u_lengths,
                                        select="u")
                                    embs_dbl = torch.cat((embs, embs))
                                    sims = torch.sum(u_embs_batch * embs_dbl, dim=-1)
                                    logp = torch.stack((torch.log(1-torch.sigmoid(sims)),
                                        torch.log(torch.sigmoid(sims)))).T
                                    labels = torch.tensor([1]*len(us_pos) +
                                        [0]*len(us_neg)).to(device)

                                    train_loss = F.nll_loss(logp,
                                        labels)
                                    if torch.sum(train_mask) > 0:
                                        train_loss_total += train_loss.item()

                                    train_loss.backward()
                                    opt[side_name].step()

                                side_embs[batch_idxs] = embs.detach().cpu()
                                all_mse.append(train_loss.item())
                    if side_name == "v":
                        print("Loss: {:.4f}".format(np.mean(all_mse)))
                elif task == "link" and args.pretrain_variant == "raq":
                    if group != "train": continue
                    embs_sides, idxs_sides = {}, {"u": [], "v": []}
                    labels = torch.zeros(len(us_to_edges), dtype=torch.long)
                    ts = torch.zeros((len(us_to_edges), 2))
                    idxs_sides["u"] = np.random.permutation(len(us_to_edges))
                    masked_edges = []
                    for i, u in enumerate(idxs_sides["u"]):
                        rnd = random.random()
                        t, v, _ = random.choice(us_to_edges[u])
                        a = np.random.randint(0, max_time)
                        b = np.random.randint(0, max_time)
                        if a > b: a, b = b, a
                        ts[i, 0], ts[i, 1] = a, b
                        if rnd < 0.5:   # positive example
                            labels[i] = 1 if t >= a and t <= b else 0
                        else:   # negative example (perturb v)
                            v = random.randint(0, len(vs_to_edges) - 1)
                        idxs_sides["v"].append(v)
                        masked_edges.append(a)

                    sides_cfg = [("u", us_to_edges, u_embs, vs_to_edges, v_embs),
                        ("v", vs_to_edges, v_embs, us_to_edges, u_embs)]

                    batch_size = batch_size_u
                    n_batches = int(np.ceil(len(us_to_edges) / batch_size))
                    pbar = tqdm(range(n_batches))
                    batch_idxs = {}
                    for batch_n in pbar:
                        s_idx = batch_n * batch_size
                        e_idx = min((batch_n+1)*batch_size, len(us_to_edges))

                        opt["u"].zero_grad()
                        for (side_name, side_to_edges, side_embs,
                            opp_side_to_edges, opp_side_embs) in sides_cfg:
                            batch_idxs[side_name] = idxs_sides[side_name][s_idx:e_idx]
                            batch = [side_to_edges[idx] for idx in
                                batch_idxs[side_name]]
                            masked_edges_batch = masked_edges[s_idx:e_idx]
                            lengths = [len(l) for l in batch]
                            batch_t, feats, lengths = get_batch(args, model, batch,
                                batch_idxs[side_name], lengths, feat_dim, emb_dim,
                                side_name, side_to_edges, side_embs,
                                opp_side_to_edges, opp_side_embs, event_counts_u, event_counts_v,
                                u_feats if side_name == "u" else v_feats,
                                masked_edges=masked_edges_batch)

                            embs, out = model(batch_t, feats, lengths, select="u" if
                                side_name == "u" else "v-clf")

                            embs_sides[side_name] = embs
                        us_l = embs_sides["u"]
                        vs_l = embs_sides["v"]
                        ts_enc = time_encode(ts[s_idx:e_idx].view(-1) /
                            max_time).to(device).view(len(us_l), 2, -1)
                        logp = F.log_softmax(model.link_pred(torch.cat((us_l,
                            vs_l, ts_enc[:,0], ts_enc[:,1]), dim=1)).squeeze(-1), dim=-1)
                        train_loss = link_criterion(logp,
                            labels[s_idx:e_idx].to(device))
                        train_logp.append(logp.detach().cpu())
                        train_labels.append(labels[s_idx:e_idx].detach().cpu())

                        train_loss_total += train_loss.item()
                        pbar.set_description("Loss: {:.4f}".format(train_loss.item()))
                        train_loss.backward()
                        opt["u"].step()
                        u_embs[batch_idxs["u"]] = embs_sides["u"].detach().cpu()
                        v_embs[batch_idxs["v"]] = embs_sides["v"].detach().cpu()
                elif task == "link" and args.pretrain_variant == "static":
                    if group != "train": continue
                    pbar = tqdm(range(10))
                    for batch_n in pbar:
                        opt["u"].zero_grad()
                        sides_cfg = [("u", us_to_edges, u_embs, vs_to_edges, v_embs),
                            ("v", vs_to_edges, v_embs, us_to_edges, u_embs)]
                        embs_sides, idxs_sides = {}, {}
                        for (side_name, side_to_edges, side_embs,
                            opp_side_to_edges, opp_side_embs) in sides_cfg:
                            batch_size = batch_size_u if side_name == "u" else batch_size_v
                            batch_idxs = np.random.choice(np.arange(len(side_to_edges)),
                                batch_size)
                            batch = [side_to_edges[idx] for idx in batch_idxs]
                            lengths = [len(l) for l in batch]
                            batch_t, feats, lengths = get_batch(args, model, batch,
                                batch_idxs, lengths, feat_dim, emb_dim,
                                side_name, side_to_edges, side_embs,
                                opp_side_to_edges, opp_side_embs, event_counts_u, event_counts_v,
                                u_feats if side_name == "u" else v_feats)

                            embs, out = model(batch_t, feats, lengths, select="u" if
                                side_name == "u" else "v-clf")

                            embs_sides[side_name] = embs
                            idxs_sides[side_name] = batch_idxs
                        us_l = embs_sides["u"].unsqueeze(1).expand(
                            -1, batch_size_v, -1).reshape(-1, emb_dim)
                        vs_l = embs_sides["v"].unsqueeze(0).expand(
                            batch_size_u, -1, -1).reshape(-1, emb_dim)
                        a = np.random.randint(0, max_time, size=len(us_l))
                        b = np.random.randint(0, max_time, size=len(us_l))
                        to_flip = a > b
                        a[to_flip], b[to_flip] = b[to_flip], a[to_flip]
                        ts = torch.zeros((len(us_l), 2))
                        ts[:,0] = torch.from_numpy(a)
                        ts[:,1] = torch.from_numpy(b)
                        ts_enc = time_encode(ts.view(-1) /
                            max_time).to(device).view(len(us_l), 2, -1)
                        logp = F.log_softmax(model.link_pred(torch.cat((us_l,
                            vs_l, ts_enc[:,0], ts_enc[:,1]), dim=1)).squeeze(-1), dim=-1)
                        labels = torch.from_numpy(
                            mat_flat[idxs_sides["u"]][:,idxs_sides["v"]].toarray()
                            > 0).type(torch.long).view(-1).to(device)
                        # test: event time
                        labels = torch.from_numpy(
                            is_edge_present_batch(us_to_edges,
                                np.repeat(idxs_sides["u"], batch_size_v),
                                np.tile(idxs_sides["v"], batch_size_u), ts,
                                labels)
                            ).type(torch.long).view(-1).to(device)
                        train_loss = link_criterion(logp, labels)
                        train_logp.append(logp.detach().cpu())
                        train_labels.append(labels.detach().cpu())

                        train_loss_total += train_loss.item()
                        pbar.set_description("Loss: {:.4f}".format(train_loss.item()))
                        train_loss.backward()
                        opt["u"].step()
                        u_embs[idxs_sides["u"]] = embs_sides["u"].detach().cpu()
                        v_embs[idxs_sides["v"]] = embs_sides["v"].detach().cpu()

        # save embs
        with open(args.out_embs_path, "wb") as f:
            torch.save((u_embs, v_embs), f)

        # get scores
        is_best = False
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
                is_best = True
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

        # analyze embs
        if args.analyze and (task == "link" or is_best):
            from sklearn.manifold import TSNE
            embs = torch.cat((u_embs[:-1], v_embs[:-1])).detach().cpu().numpy()
            embs_2d = TSNE().fit_transform(embs)
            xs, ys = zip(*embs_2d)
            xs, ys = list(xs), list(ys)
            colors = ["red" if l == 1 else "blue" for l in u_labels]
            colors += ["green"]*(len(v_embs)-1)
            plt.scatter(xs, ys, color=colors, alpha=0.3)
            xrs = [x for x, l in zip(xs, u_labels) if l == 1]
            yrs = [y for y, l in zip(ys, u_labels) if l == 1]
            plt.scatter(xrs, yrs, color="red", alpha=0.9)

            if task == "link":
                fn = "link-embs.png"
            else:
                fn = "clf-embs.png"
            plt.savefig(fn, bbox_inches="tight")
            plt.close()
            print("saved", fn)

    print("Test AUROC with best validation model: {:.4f}".format(best_test_auroc))
    return best_test_auroc

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_embs_path), exist_ok=True)
    print("Saving embeddings to", args.out_embs_path)

    aurocs = []
    for trial_n in range(args.n_trials):
        print(trial_n)
        dataset = load_dataset(args)
        auroc = train(args, dataset)
        aurocs.append(auroc)
    print(args)
    print(aurocs)
    print(np.mean(aurocs), np.std(aurocs, ddof=1))
