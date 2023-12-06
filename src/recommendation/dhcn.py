import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import *
from utils import *


class HyperConv(nn.Module):
    """_summary_: implementation of hypergraph convolution for sequence graphs
    """

    def __init__(self, hidden_size, steps=1):
        super(HyperConv, self).__init__()
        self.hidden_size = hidden_size
        self.steps = steps

    def forward(self, hidden, adjacency):
        """_summary_: forward propagation of hypergraph convolution
        It is a mean of multiple graph convolutions

        :param hidden: hidden states of session graph
        :type hidden: torch.Tensor
        :param adjacency: adjacency matrix of session graph
        :type adjacency: torch.Tensor
        :return: hidden states of session graph
        :rtype: torch.Tensor
        """

        final = [hidden]
        for i in range(self.steps):
            final.append(torch.sparse.mm(adjacency, final[-1]))
        item_hiddens = torch.stack(final, dim=0).sum(0) / (self.steps + 1)
        return item_hiddens


class LineConv(nn.Module):
    """_summary_: implementation of graph convolutional network for batch data
    """

    def __init__(self, hidden_size=100, steps=1):
        super(LineConv, self).__init__()
        self.hidden_size = hidden_size
        self.steps = steps

    def forward(self, hidden, DA, mask=None):
        """_summary_: forward propagation of graph convolutional network
        It is a mean of multiple graph convolutions for line graphs of batch data

        :param hidden: hidden states of sessions in the same batch
        :type hidden: torch.Tensor
        :param DA: adjacency matrix of line graphs of sessions in the same batch
        :type DA: torch.Tensor
        :param mask: mask of sessions, defaults to None
        :type mask: torch.Tensor, optional
        :return: hidden states of sessions in the same batch
        :rtype: torch.Tensor
        """

        if mask is not None:
            session_emb_lgcn = torch.sum(hidden.masked_fill(~mask.unsqueeze(-1), 0), dim=1) / \
                torch.sum(mask.float(), 1, keepdim=True)
        else:
            session_emb_lgcn = torch.mean(hidden, 1)
        session = [session_emb_lgcn]  # batch_size * hidden_size
        for i in range(self.steps):
            session.append(torch.mm(DA, session[-1]))
        session_emb_lgcn = torch.stack(session, dim=0).sum(0) / (self.steps + 1)
        return session_emb_lgcn


class DHCN(nn.Module):
    """_summary_: implementation of DHCN
    reference: Xin Xia, Hongzhi Yin, Junliang Yu, Qinyong Wang, Lizhen Cui, and Xiangliang Zhang. 2021. Self-Supervised Hypergraph Convolutional Networks for Sessionbased Recommendation. In AAAI. 4503–4511.
    code reference: https://github.com/xiaxin1998/DHCN
    """

    def __init__(self, opt, *args, **kwargs):
        super(DHCN, self).__init__()
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

        self.adjacencies = dict()
        assert len(adjs) == len(nums)
        for key in self.n_nodes:
            adj = adjs.get(key, None)
            num = nums.get(key, None)
            if isinstance(adj, np.ndarray):
                adj = torch.from_numpy(adj)
            elif isinstance(adj, torch.Tensor):
                adj = adj.clone()
            elif isinstance(adj, (tuple, list)) and len(adj) == 2 and isinstance(adj[0], torch.Tensor):
                adj = to_dense_adj(adj[0], None, adj[1])[0]
            else:
                adj = torch.zeros((self.n_nodes[key], 1), dtype=torch.long)

            if isinstance(num, np.ndarray):
                num = torch.from_numpy(num).float()
            elif isinstance(num, torch.Tensor):
                num = num.clone().float()
            elif isinstance(num, (tuple, list)) and len(num) == 2 and isinstance(num[0], torch.Tensor):
                num = to_dense_adj(num[0], None, num[1])[0].float()
            else:
                num = torch.zeros((self.n_nodes[key], 1), dtype=torch.float)

            self.adjacencies[key] = trans_to_cuda(
                torch.sparse_coo_tensor(
                    indices=torch.stack(
                        [torch.arange(adj.shape[0], dtype=torch.long).repeat_interleave(adj.shape[1]),
                         adj.view(-1)], 0
                    ),
                    values=num.view(-1),
                    size=torch.Size([adj.shape[0], adj.shape[0]])
                )
            )

        # Aggregator
        self.hyper_conv = HyperConv(self.dim, self.n_iter)
        self.line_conv = LineConv(self.dim, self.n_iter)

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
        """_summary_: forward propagation of DHCN.
        It transforms the session data into hyper-graph and line-graph and uses GCN 
        to encode them to enhance the representation of the current session.

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
        h_local = torch.stack([h_graphs[key] for key in sorted_keys]).mean(0)

        # local
        # get batch overlap
        seq_b = input[SEQUENCE].transpose(0, 1)
        mask_t = mask.transpose(0, 1)
        overlap = (seq_b[:, :, None] == seq_b[:, None, :])
        overlap.masked_fill_(~mask_t.unsqueeze(1), 0)
        overlap.masked_fill_(~mask_t.unsqueeze(2), 0)
        overlap = overlap.float().sum(0)  # batch_size * batch_size
        A = overlap / (mask_t.sum(0) * 2 - overlap)
        A = A + torch.eye(batch_size, dtype=A.dtype, device=A.device)
        D = A.sum(1)
        DA = A / D.unsqueeze(1)
        h_lg = self.line_conv(h_local, DA, mask=mask)
        h_lg = F.dropout(h_lg, self.dropout_local, training=self.training)

        # global
        h_global = self.hyper_conv(self.embeddings[SEQUENCE].weight, self.adjacencies[SEQUENCE])
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)

        h_local = F.embedding(items[SEQUENCE], h_global)

        if not exists(alias):
            alias = dict()

        if self.mtl:
            output = {key: (h_lg, h_local, h_global) for key in sorted_keys}
        else:
            output = {SEQUENCE: (h_lg, h_local, h_global)}

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
            h_lg, h_local, h_global = hidden[key]
            select = self.predictors[key](h_local + h_lg.unsqueeze(1), mask=mask)
            b = h_global
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
