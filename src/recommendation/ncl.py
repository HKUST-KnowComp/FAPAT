import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import *
from utils import *


class HyperConv(nn.Module):
    """_summary_: implementation of HyperConv for batch data
    details: mean of multiple graph convolutions
    """

    def __init__(self, hidden_size, steps=1):
        super(HyperConv, self).__init__()
        self.hidden_size = hidden_size
        self.steps = steps

    def forward(self, hidden, adjacency):
        final = [hidden]
        for i in range(self.steps):
            final.append(torch.sparse.mm(adjacency, final[-1]))
        item_hiddens = torch.stack(final, dim=0).sum(0) / (self.steps + 1)
        return item_hiddens


class NCL(nn.Module):
    """_summary_: implementation of NCL
    reference: Zihan Lin, Changxin Tian, Yupeng Hou, and Wayne Xin Zhao. 2022. Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning. In WWW. 2320–2329.
    code reference: https://github.com/RUCAIBox/NCL
    """

    def __init__(self, opt, *args, **kwargs):
        super(NCL, self).__init__()
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
        self.adjacencies = dict()
        assert len(adjs) == len(nums)
        for key in self.n_nodes:
            adj = adjs.get(key, None)
            num = nums.get(key, None)
            if isinstance(adj, np.ndarray):
                adj = torch.from_numpy(adj)
                adj = torch.cat(
                    [
                        torch.cat([adj, torch.full((adj.shape[0], 1), adj.shape[0], dtype=torch.long)], dim=1),
                        torch.zeros((1, adj.shape[1] + 1), dtype=torch.long)  # 1 for anonymous user
                    ],
                    dim=0
                )
            elif isinstance(adj, torch.Tensor):
                adj = torch.cat(
                    [
                        torch.cat([adj, torch.full((adj.shape[0], 1), adj.shape[0], dtype=torch.long)], dim=1),
                        torch.zeros((1, adj.shape[1] + 1), dtype=torch.long)  # 1 for anonymous user
                    ],
                    dim=0
                )
            elif isinstance(adj, (tuple, list)) and len(adj) == 2 and isinstance(adj[0], torch.Tensor):
                adj = to_dense_adj(adj[0], None, adj[1])[0]
                adj = torch.cat(
                    [
                        torch.cat([adj, torch.full((adj.shape[0], 1), adj.shape[0], dtype=torch.long)], dim=1),
                        torch.zeros((1, adj.shape[1] + 1), dtype=torch.long)  # 1 for anonymous user
                    ],
                    dim=0
                )
            else:
                adj = torch.zeros((self.n_nodes[key] + 1, 1), dtype=torch.long)

            if isinstance(num, np.ndarray):
                num = torch.from_numpy(num).float()
                num = torch.cat(
                    [
                        torch.cat(
                            [num, torch.full((num.shape[0], 1), 1 / max(num.shape[0], 1), dtype=torch.float)], dim=1
                        ),
                        torch.zeros((1, num.shape[1] + 1), dtype=torch.float)  # 1 for anonymous user
                    ],
                    dim=0
                )
            elif isinstance(num, torch.Tensor):
                num = num.float()
                num = torch.cat(
                    [
                        torch.cat(
                            [num, torch.full((num.shape[0], 1), 1 / max(num.shape[0], 1), dtype=torch.float)], dim=1
                        ),
                        torch.zeros((1, num.shape[1] + 1), dtype=torch.float)  # 1 for anonymous user
                    ],
                    dim=0
                )
            elif isinstance(num, (tuple, list)) and len(num) == 2 and isinstance(num[0], torch.Tensor):
                num = to_dense_adj(num[0], None, num[1])[0].float()
                num = torch.cat(
                    [
                        torch.cat(
                            [num, torch.full((num.shape[0], 1), 1 / max(num.shape[0], 1), dtype=torch.float)], dim=1
                        ),
                        torch.zeros((1, num.shape[1] + 1), dtype=torch.float)  # 1 for anonymous user
                    ],
                    dim=0
                )
            else:
                num = torch.zeros((self.n_nodes[key] + 1, 1), dtype=torch.float)

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

        # Item representation
        self.embeddings = nn.ModuleDict()
        for key in self.n_nodes:
            self.embeddings[key] = nn.Embedding(self.n_nodes[key] + 1, self.dim, padding_idx=0)  # 1 for anonymous user

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
        """_summary_: forward propagation of NCL.
        It reduces the influence of session data sparsity by 
        graph collaborative filtering with neighborhood-enriched contrastive learning.

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

        # global
        h_global = self.hyper_conv(self.embeddings[SEQUENCE].weight, self.adjacencies[SEQUENCE])
        h_global = F.dropout(h_global, self.dropout_global, training=self.training)

        user_emb = h_global[-1].unsqueeze(0).repeat(batch_size, 1)
        item_emb = F.embedding(input[SEQUENCE], self.embeddings[SEQUENCE].weight)

        if not exists(alias):
            alias = dict()

        if self.mtl:
            output = {key: (user_emb, item_emb, h_global[:-1]) for key in sorted_keys}
        else:
            output = {SEQUENCE: (user_emb, item_emb, h_global[:-1])}

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
            user_emb, item_emb, h_global = hidden[key]
            select = (item_emb.masked_fill(~mask.unsqueeze(-1), 0.0)).sum(1) / mask.float().sum(1, keepdim=True) * user_emb
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
