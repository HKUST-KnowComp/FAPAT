import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import *
from utils import *


class GRAPHFORMER(nn.Module):
    """_summary_: implementation of GRAPHFORMER
    reference: Junhan Yang, Zheng Liu, Shitao Xiao, Chaozhuo Li, Defu Lian, Sanjay Agrawal, Amit Singh, Guangzhong Sun, and Xing Xie. 2021. GraphFormers: GNN-nested Transformers for Representation Learning on Textual Graph. In NeurIPS. 28798–28810.
    code reference: https://github.com/microsoft/GraphFormers
    """

    def __init__(self, opt, *args, **kwargs):
        super().__init__()

        self.opt = opt

        self.batch_size = opt.batch_size
        self.max_len = opt.max_len
        self.dim = opt.hidden_dim
        self.n_iter = opt.n_iter
        self.mem_type = opt.mem_type
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.clip = opt.clip
        self.mtl = opt.mtl

        self.n_nodes = kwargs["n_nodes"]
        if isinstance(self.n_nodes, int):
            self.n_nodes = {SEQUENCE: self.n_nodes}

        # Transformer
        self.transformers = nn.ModuleList()
        for i in range(self.n_iter):
            self.transformers.append(
                Transformer(
                    dim=opt.hidden_dim,
                    depth=1,
                    dim_head=opt.hidden_dim // opt.n_head,
                    heads=opt.n_head,
                    buckets=opt.n_bucket,
                    bidirectional=False,
                    attn_dropout=opt.dropout_attn,
                    ff_mult=4,
                    ff_dropout=opt.dropout_ff,
                )
            )

        # Aggregator
        self.local_aggs = nn.ModuleList()
        for i in range(self.n_iter):
            local_aggs = nn.ModuleDict()
            for key in self.n_nodes:
                local_aggs[key] = LocalAggregator(self.dim, dropout=self.dropout_local)
            self.local_aggs.append(local_aggs)

        # Item representation & Position representation
        self.pos_embedding = nn.Embedding(self.max_len + 2, self.dim)
        self.embeddings = nn.ModuleDict()
        for key in self.n_nodes:
            self.embeddings[key] = nn.Embedding(self.n_nodes[key], self.dim, padding_idx=0)

        # Parameters
        self.layer_norm = nn.LayerNorm(self.dim)
        self.cls = nn.Parameter(torch.Tensor(1, self.dim))
        self.mask = nn.Parameter(torch.Tensor(1, self.dim))
        self.to_qs = nn.ModuleDict()
        self.to_ks = nn.ModuleDict()
        self.predictors = nn.ModuleDict()
        for key in self.n_nodes:
            self.to_qs[key] = nn.Linear(opt.hidden_dim, opt.hidden_dim, bias=False)
            self.to_ks[key] = nn.Linear(opt.hidden_dim, opt.hidden_dim // opt.n_head, bias=False)
            self.predictors[key] = Predictor(self.dim, 2, self.max_len, dropout=0.0)

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
            if name.startswith("transformer"):
                continue
            if name.startswith("local_agg"):
                continue
            if name.endswith("bias"):
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -stdv, stdv)
        nn.init.ones_(self.layer_norm.weight)
        nn.init.constant_(self.layer_norm.bias, 0.0)

    def forward(self, input, mask=None, items=None, adj=None, alias=None, shortcut=None, heteitems=None, heteadj=None):
        """_summary_: forward propagation of GraphFormer.
        It uses GraphFormer to extract local context and global information.

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

        h_seqs = dict()
        h_graphs = dict()
        for key in sorted_keys:
            h_seqs[key] = self.embeddings[key](input[key])
            h_graphs[key] = self.embeddings[key](items[key])

        h_locals = h_graphs.copy()
        h_trans = torch.cat(
            [
                self.cls.unsqueeze(0).expand(batch_size, -1, -1),
                torch.stack([h_seqs[key] for key in sorted_keys]).mean(0),
                self.mask.unsqueeze(0).expand(batch_size, -1, -1)
            ],
            dim=1
        )
        mask_trans = torch.cat(
            [
                torch.ones(batch_size, 1, device=mask.device, dtype=torch.bool),
                mask,
                torch.ones(batch_size, 1, device=mask.device, dtype=torch.bool),
            ],
            dim=1
        )
        pos_idx = (torch.cumsum(mask_trans.long(), dim=1) - 1).clamp_min(0)
        pos_emb = self.pos_embedding(pos_idx)
        for i in range(self.n_iter):
            # local aggregation
            for key in sorted_keys:
                h_locals[key] = self.local_aggs[i][key](h_locals[key], adj[key], mask=mask)

            if not exists(alias):
                alias = dict()
            else:
                # h_locals = {key: torch.stack([h_locals[key][idx][alias[key][idx]] for idx in range(batch_size)]) for key in h_locals}
                h_locals = {
                    key:
                    h_locals[key].view((batch_size * seq_len),
                                       -1)[(alias[key] + batch_flat_offset).view(-1)].view(batch_size, seq_len, -1)
                    for key in h_locals
                }

            h_local = torch.stack([h_locals[key] for key in sorted_keys]).mean(0)

            # h_trans = self.layer_norm(h_trans + pos_emb)
            h_trans[:, 1:-1] += h_local
            if i == 0:
                h_trans += pos_emb
            h_trans = self.transformers[i](h_trans, mask=mask_trans)

        if self.mtl:
            output = {key: None for key in sorted_keys}
        else:
            output = {SEQUENCE: None}
        for key in output:
            q = self.to_qs[key](h_trans)
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.opt.n_head)
            k = self.to_ks[key](h_seqs[key])
            v = h_seqs[key]
            sim = einsum('b h i d, b j d -> b h i j', q, k)
            sim = sim.masked_fill(~mask[:, None, None, :], -torch.finfo(q.dtype).max)
            attn = stable_softmax(sim)
            o = einsum('b h i j, b j d -> b i d', attn, v)
            g = F.dropout(o, self.dropout_local, training=self.training)

            output[key] = (h_local, g[:, 1:-1], g[:, -1])

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
            select = self.predictors[key](hidden[key][:2], hidden[key][2], mask=mask)
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
