import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import *
from utils import *


class M2TREC(nn.Module):
    """_summary_: implementation of M2TREC
    reference: Walid Shalaby, Sejoon Oh, Amir Afsharinejad, Srijan Kumar, and Xiquan Cui. 2022. M2TRec: Metadata-aware Multi-task Transformer for Large-scale and Cold-start free Session-based Recommendations. In RecSys. 573–578.
    code reference: NA
    """

    def __init__(self, opt, *args, **kwargs):
        super().__init__()

        self.opt = opt

        self.batch_size = opt.batch_size
        self.max_len = opt.max_len
        self.dim = opt.hidden_dim
        self.n_iter = opt.n_iter
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.clip = opt.clip
        self.mtl = opt.mtl

        self.n_nodes = kwargs["n_nodes"]
        if isinstance(self.n_nodes, int):
            self.n_nodes = {SEQUENCE: self.n_nodes}

        # Transformer
        self.transformer = Transformer(
            dim=opt.hidden_dim,
            depth=opt.n_iter,
            dim_head=opt.hidden_dim // opt.n_head,
            heads=opt.n_head,
            buckets=opt.n_bucket,
            bidirectional=False,
            attn_dropout=opt.dropout_attn,
            ff_mult=4,
            ff_dropout=opt.dropout_ff,
        )

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
            self.predictors[key] = nn.Linear(self.dim, self.n_nodes[key])

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
            if name.endswith("bias"):
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -stdv, stdv)
        nn.init.ones_(self.layer_norm.weight)
        nn.init.constant_(self.layer_norm.bias, 0.0)

    def forward(self, input, mask=None, items=None, adj=None, alias=None, shortcut=None, heteitems=None, heteadj=None):
        """_summary_: forward propagation of M2TRec.
        It learns session sequence representations via Transformer by multi-task learning.

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

        batch_size, seq_len = input[SEQUENCE].shape[:2]
        sorted_keys = sorted(self.embeddings.keys())
        assert len(set(sorted_keys) - set(input.keys())) == 0
        seq_idx = sorted_keys.index(SEQUENCE)
        sorted_keys[0], sorted_keys[seq_idx] = sorted_keys[seq_idx], sorted_keys[0]

        h_seqs = dict()
        for key in sorted_keys:
            h_seqs[key] = self.embeddings[key](input[key])
        h_seq = torch.stack([h_seqs[key] for key in sorted_keys]).mean(0)

        h_trans = torch.cat(
            [
                self.cls.unsqueeze(0).expand(batch_size, -1, -1), h_seq,
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
        h_trans = self.layer_norm(h_trans + pos_emb)
        h_trans = self.transformer(h_trans, mask=mask_trans)

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
            output[key] = (g[:, 1:-1], g[:, -1])

        return output

    def compute_scores(self, hidden, mask=None):
        """_summary_: compute logit scores for ranking

        :param hidden: hidden states for seqeuence and attributes
        :type hidden: dict
        :param mask: mask for valid data, defaults to None
        :type mask: torch.Tensor, optional
        :return: scores for sequence and attributes
        :rtype: dict
        """

        scores = dict()
        for key in hidden.keys():
            scores[key] = self.predictors[key](hidden[key][1])
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