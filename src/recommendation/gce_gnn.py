import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from config import *
from layers import *
from utils import *


class GCEGNN(nn.Module):
    """_summary_: implementation of GCEGNN
    reference: Ziyang Wang, Wei Wei, Gao Cong, Xiao-Li Li, Xianling Mao, and Minghui Qiu. 2020. Global Context Enhanced Graph Neural Networks for Session-based Recommendation. In SIGIR. 169–178.
    code reference: https://github.com/CCIIPLab/GCE-GNN
    """

    def __init__(self, opt, *args, **kwargs):
        super(GCEGNN, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.max_len = opt.max_len
        self.dim = opt.hidden_dim
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.n_iter = opt.n_iter
        self.sample_num = opt.n_sample_model
        self.clip = opt.clip
        self.mtl = opt.mtl

        self.n_nodes = kwargs["n_nodes"]
        if isinstance(self.n_nodes, int):
            self.n_nodes = {SEQUENCE: self.n_nodes}
        adjs = kwargs.get("adjs", dict())
        nums = kwargs.get("nums", dict())

        self.adjs = dict()
        self.nums = dict()
        assert len(adjs) == len(nums)
        for key in self.n_nodes:
            adj = adjs.get(key, None)
            num = nums.get(key, None)
            if isinstance(adj, np.ndarray):
                self.adjs[key] = trans_to_cuda(torch.from_numpy(adj))
            elif isinstance(adj, torch.Tensor):
                self.adjs[key] = trans_to_cuda(adj.clone())
            elif isinstance(adj, (tuple, list)) and len(adj) == 2 and isinstance(adj[0], torch.Tensor):
                self.adjs[key] = trans_to_cuda(to_dense_adj(adj[0], None, adj[1])[0])
            else:
                self.adjs[key] = trans_to_cuda(torch.zeros((self.n_nodes[key], 1), dtype=torch.long))

            if isinstance(num, np.ndarray):
                self.nums[key] = trans_to_cuda(torch.from_numpy(num).float())
            elif isinstance(num, torch.Tensor):
                self.nums[key] = trans_to_cuda(num.clone().float())
            elif isinstance(num, (tuple, list)) and len(num) == 2 and isinstance(num[0], torch.Tensor):
                self.nums[key] = trans_to_cuda(to_dense_adj(num[0], None, num[1])[0].float())
            else:
                self.nums[key] = trans_to_cuda(torch.zeros((self.n_nodes[key], 1), dtype=torch.float))

        # Aggregator
        self.local_agg = LocalAggregator(self.dim, dropout=0.0)
        self.global_aggs = nn.ModuleList()
        for i in range(self.n_iter):
            if opt.activate == 'relu':
                agg = GlobalAggregator(self.dim, opt.dropout_gnn, act=torch.relu)
            else:
                agg = GlobalAggregator(self.dim, opt.dropout_gnn, act=torch.tanh)
            self.global_aggs.append(agg)

        # Item representation
        self.embeddings = nn.ModuleDict()
        for key in self.n_nodes:
            self.embeddings[key] = nn.Embedding(self.n_nodes[key], self.dim, padding_idx=0)

        # Predictor
        self.predictors = nn.ModuleDict()
        for key in self.n_nodes:
            self.predictors[key] = Predictor(self.dim, 1, self.max_len, dropout=0.0)

        # Optimization
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = LinearWarmupScheduler(
            self.optimizer,
            num_warmup_steps=opt.lr_dc_step,
            num_schedule_steps=opt.lr_dc_step * opt.epoch,
            min_percent=opt.lr_dc**2
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for name, param in self.named_parameters():
            if name.endswith("bias"):
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -stdv, stdv)

    def forward(self, input, mask=None, items=None, adj=None, alias=None, shortcut=None, heteitems=None, heteadj=None):
        """_summary_: forward propagation of GCE-GNN.
        It learns two levels of item representations from session graphs and global graphs 
        and aggregates two-level embeddings with the soft attention mechanism.

        :param input: input data, including sequence and attributes
        :type input: dict
        :param mask: mask for valid data
        :type mask: torch.Tensor
        :param items: sequence of row numbers aligned with adjacency matrix
        :type items: torch.Tensor
        :param adj: adjacency matrix
        :type adj: torch.Tensor
        :param alias: real item ids that can map "items" to real item ids
        :type alias: torch.Tensor
        :return: hidden states for sequence and attributes
        :rtype: dict
        """

        batch_size, seq_len = items[SEQUENCE].shape[:2]
        batch_flat_offset = seq_len * torch.arange(
            batch_size, dtype=input[SEQUENCE].dtype, device=input[SEQUENCE].device
        ).unsqueeze(1)
        sorted_keys = sorted(self.embeddings.keys())
        assert len(set(sorted_keys) - set(input.keys())) == 0
        seq_idx = sorted_keys.index(SEQUENCE)
        sorted_keys[0], sorted_keys[seq_idx] = sorted_keys[seq_idx], sorted_keys[0]

        h_graphs = dict()
        for key in sorted_keys:
            h_graphs[key] = self.embeddings[key](items[key])
        h_graph = torch.stack([h_graphs[key] for key in sorted_keys]).mean(0)

        # local
        h_local = self.local_agg(h_graph, adj[SEQUENCE], mask=mask)
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)

        # global
        item_neighbors = [items[SEQUENCE]]
        weight_neighbors = [
        ]  # [(alias[SEQUENCE][:, :, None] == alias[SEQUENCE][:, None, :]).float().sum(dim=-1).masked_fill(~mask, 0)]

        for i in range(1, self.n_iter + 1):
            item_sample_i, weight_sample_i = sample_neighbors(
                self.adjs[SEQUENCE], self.nums[SEQUENCE], item_neighbors[-1].view(-1), self.sample_num
            )
            item_neighbors.append(item_sample_i.view(batch_size, -1))
            weight_neighbors.append(weight_sample_i.view(batch_size, -1))

        entity_vectors = [self.embeddings[SEQUENCE](nei) for nei in item_neighbors]
        weight_vectors = weight_neighbors

        session_info = []
        item_emb = self.embeddings[SEQUENCE](input[SEQUENCE]) * mask.float().unsqueeze(-1)

        # mean
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask.float(), -1).unsqueeze(-1)

        # sum
        # sum_item_emb = torch.sum(item_emb, 1)

        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.n_iter):
            session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1))

        for n_hop in range(self.n_iter):
            entity_vectors_next_iter = []
            shape = [batch_size, -1, self.sample_num, self.dim]
            aggregator = self.global_aggs[n_hop]
            for hop in range(self.n_iter - n_hop):
                vector = aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vector=entity_vectors[hop + 1].view(shape),
                    mask=None,
                    batch_size=batch_size,
                    neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num),
                    extra_vector=session_info[hop]
                )
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        h_global = entity_vectors[0].view(batch_size, seq_len, self.dim)
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)

        # combine
        combine = h_local + h_global

        if not exists(alias):
            alias = dict()

        if self.mtl:
            output = {key: None for key in sorted_keys}
        else:
            output = {SEQUENCE: None}
        for key in output:
            if key in alias:
                output[key] = combine.view((batch_size * seq_len),
                                           -1)[(alias[key] + batch_flat_offset).view(-1)].view(batch_size, seq_len, -1)
            else:
                output[key] = combine

        return output

    def compute_scores(self, hidden, mask=None):
        """_summary_: compute inner-product scores for ranking

        :param hidden: hidden states for seqeuence and attributes
        :type hidden: dict
        :param mask: mask for valid data, defaults to None
        :type mask: torch.Tensor, optional
        :return: scores for sequence and attributes
        :rtype: dict
        """

        scores = dict()
        for key in hidden.keys():
            select = self.predictors[key](hidden[key], mask=mask)
            b = self.embeddings[key].weight  # n_nodes x latent_size
            scores[key] = torch.matmul(select, b.transpose(1, 0))
        return scores

    def compute_loss(self, scores, target, coefs=None):
        if self.mtl:
            if coefs is None:
                coefs = {key: 1.0 for key in scores.keys()}
            loss = torch.stack([coef * self.loss_function(scores[key], target[key])
                                for key, coef in coefs.items()]).sum()
        else:
            loss = self.loss_function(scores[SEQUENCE], target[SEQUENCE])
        return loss
