import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import *
from utils import *


class EOPA(nn.Module):
    def __init__(
        self, dim, batch_norm=False, dropout=0.0, act=None
    ):
        super().__init__()
        self.dim = dim
        self.batch_norm = nn.BatchNorm1d(dim) if batch_norm else None
        self.dropout = dropout
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.fc_self = nn.Linear(dim, dim, bias=False)
        self.fc_neigh = nn.Linear(dim, dim, bias=False)
        self.act = act

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for name, param in self.named_parameters():
            if name.endswith("bias"):
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -stdv, stdv)
        if exists(self.batch_norm):
            self.batch_norm.reset_parameters()

    def forward(self, x, A, edgeorder=None, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)
        if exists(self.batch_norm):
            shape = x.shape
            x = self.batch_norm(x.view(-1, shape[-1])).view(*shape)
        x = F.dropout(x, p=self.dropout, training=self.training)

        neigh = x[:, None, :, :].repeat(1, seq_len, 1, 1).masked_fill((A.transpose(1, 2) != 1).unsqueeze(-1), 0.0) # bsz, dst, src, d
        if exists(mask):
            neigh = neigh.masked_fill(~mask[:, None, :, None], 0.0)

        if exists(edgeorder):
            assert edgeorder.size(0) == A.size(0)
            assert edgeorder.size(1) == A.size(1)
            assert edgeorder.size(2) == A.size(2)
            index0 = torch.arange(A.size(0)).to(edgeorder)[:, None, None]
            index1 = edgeorder.transpose(1, 2)
            index2 = torch.arange(A.size(1)).to(edgeorder)[None, None, :]
            neigh = neigh[index0, index1, index2]

        # neigh = self.gru(neigh.view(batch_size * seq_len, seq_len, -1))[1]
        neigh = neigh.max(dim=-2)[0]
        neigh = neigh.view(batch_size, seq_len, -1) # bsz, seq_len, d
        out = self.fc_self(x) + self.fc_neigh(neigh) # bsz, seq_len, d
        if exists(self.act):
            shape = out.shape
            out = self.act(out.view(-1, shape[-1])).view(*shape)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.dim}, batch_norm={exists(self.batch_norm)}, aggr=gru, act={self.act})')


class SGAT(nn.Module):
    def __init__(
        self,
        dim,
        batch_norm=False,
        dropout=0.0,
        act=None,
    ):
        super().__init__()
        self.dim = dim
        self.batch_norm = nn.BatchNorm1d(dim) if batch_norm else None
        self.dropout = dropout
        self.fc_q = nn.Linear(dim, dim, bias=True)
        self.fc_k = nn.Linear(dim, dim, bias=False)
        self.fc_v = nn.Linear(dim, dim, bias=False)
        self.fc_e = nn.Linear(dim, 1, bias=False)
        self.act = act
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for name, param in self.named_parameters():
            if name.endswith("bias"):
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -stdv, stdv)
        if exists(self.batch_norm):
            self.batch_norm.reset_parameters()
    
    def forward(self, x, A, mask=None):
        if exists(self.batch_norm):
            shape = x.shape
            x = self.batch_norm(x.view(-1, shape[-1])).view(*shape)
        x = F.dropout(x, p=self.dropout, training=self.training)
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)

        e = k[:, :, None, :] + q[:, None, :, :]
        e = e.masked_fill((A != 1).unsqueeze(-1), 0)
        e = self.fc_e(F.sigmoid(e)) # bsz, seq_len, seq_len, 1
        if exists(mask):
            _inf = -(torch.finfo(e.dtype).max)
            e.masked_fill((~mask)[:, :, None, None], _inf)
        a = stable_softmax(e, 1)
        out = (a * v[:, :, None, :]).sum(1)
        if exists(self.act):
            shape = out.shape
            out = self.act(out.view(-1, shape[-1])).view(*shape)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.dim}, batch_norm={exists(self.batch_norm)}, aggr=gru, act={self.act})')


class AttnReadout(nn.Module):
    def __init__(
        self,
        dim,
        batch_norm=False,
        dropout=0.0,
        act=None,
    ):
        super().__init__()
        self.dim = dim
        self.batch_norm = nn.BatchNorm1d(dim) if batch_norm else None
        self.dropout = dropout
        self.fc_u = nn.Linear(dim, dim, bias=True)
        self.fc_v = nn.Linear(dim, dim, bias=False)
        self.fc_e = nn.Linear(dim, 1, bias=False)
        self.fc_out = None
        self.act = act
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for name, param in self.named_parameters():
            if name.endswith("bias"):
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -stdv, stdv)
        if exists(self.batch_norm):
            self.batch_norm.reset_parameters()

    def forward(self, x, last_nodes=None):
        # x: bsz x seq_len x d
        # last_nodes: bsz
        if exists(last_nodes):
            assert x.size(0) == last_nodes.size(0)
        batch_size = x.size(0)

        if exists(self.batch_norm):
            shape = x.shape
            x = self.batch_norm(x.view(-1, shape[-1])).view(*shape)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_u = self.fc_u(x)
        if exists(last_nodes):
            index0 = torch.arange(batch_size).to(last_nodes)
            index1 = last_nodes
            x_v = self.fc_v(x[index0, index1])[:, None, :]
        else:
            x_v = self.fc_v(x[:, -1])[:, None, :]
        e = self.fc_e(F.sigmoid(x_u + x_v)) # bsz x seq_len x 1
        alpha = stable_softmax(e, dim=-2)
        out = (x * alpha).sum(-2) # # bsz x d

        if exists(self.fc_out):
            out = self.fc_out(out)
        if exists(self.act):
            shape = out.shape
            out = self.act(out.view(-1, shape[-1])).view(*shape)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(batch_norm={exists(self.batch_norm)}, act={self.act})')


class LESSR(nn.Module):
    """_summary_: implementation of LESSR
    reference: Tianwen Chen, and Raymong Chi-Wing Wong. 2020. Handling Information Loss of Graph Neural Networks for Session-based Recommendation. In SIGKDD. 1172-1180.
    code reference: https://github.com/twchen/lessr/lessr.py
    """

    def __init__(self, opt, *args, **kwargs):
        super(LESSR, self).__init__()
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

        # Aggregator
        self.layers = nn.ModuleDict()
        for key in self.n_nodes:
            layers = nn.ModuleList()
            for i in range(self.n_iter):
                if i % 2 == 0:
                    layers.append(EOPA(self.dim, dropout=self.dropout_local, act=nn.PReLU(self.dim)))
                else:
                    layers.append(SGAT(self.dim, dropout=self.dropout_local, act=nn.PReLU(self.dim)))
            self.layers[key] = layers

        # Item representation
        self.embeddings = nn.ModuleDict()
        for key in self.n_nodes:
            self.embeddings[key] = nn.Embedding(self.n_nodes[key], self.dim, padding_idx=0)

        # Predictor
        self.readout = AttnReadout(self.dim, dropout=self.dropout_local, act=nn.PReLU(self.dim))
        self.batch_norm = nn.BatchNorm1d(2 * self.dim)
        self.fc_sr = nn.Linear(2 * self.dim, self.dim, bias=False)

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
        if exists(self.batch_norm):
            self.batch_norm.reset_parameters()

    def forward(self, input, mask=None, items=None, adj=None, alias=None, shortcut=None, heteitems=None, heteadj=None):
        """_summary_: forward propagation of SR-GNN.
        It transforms the session data into a direct unweighted graph and 
        utilizes gated GNN to learn the representation of the item transitions graph.

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
        :param shortcut: shortcut matrix
        :type shortcut: torch.Tensor
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
            for i, layer in enumerate(self.layers[key]):
                if i % 2 == 0:
                    h_graphs[key] = layer(h_graphs[key], (adj[key] == 2).float())
                else:
                    h_graphs[key] = layer(h_graphs[key], (shortcut[key] == 1).float() + (shortcut[key] == 2).float() + (shortcut[key] == 4).float())
        h_graph = torch.stack([h_graphs[key] for key in sorted_keys], dim=0).mean(0)
        # last_nodes = (items[SEQUENCE] != 0).float().cumsum(1).argmax(1)
        # sr_g = self.readout(h_graph, last_nodes)
        # sr_l = h_graph[torch.arange(batch_size).to(last_nodes), last_nodes]
        sr_g = self.readout(h_graph)
        sr_l = h_graph[:, -1]
        sr = torch.cat([sr_g, sr_l], dim=1)

        if exists(self.batch_norm):
            shape = sr.shape
            sr = self.batch_norm(sr.view(-1, shape[-1])).view(*shape)
        sr = self.fc_sr(F.dropout(sr, p=self.dropout_global, training=self.training))
        
        if not exists(alias):
            alias = dict()

        if self.mtl:
            output = {key: None for key in sorted_keys}
        else:
            output = {SEQUENCE: None}
        for key in output:
            output[key] = sr

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
            select = hidden[key]
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
