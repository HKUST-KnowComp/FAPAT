import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import *
from utils import *


class GRU4Rec(nn.Module):
    """_summary_: implementation of GRU4Rec
    reference: Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, and Domonkos Tikk. 2016. Session-based Recommendations with Recurrent Neural Networks. In ICLR.
    code reference: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/gru4rec.py
    """

    def __init__(self, opt, **kwargs):
        super(GRU4Rec, self).__init__()

        self.opt = opt

        self.batch_size = opt.batch_size
        self.max_len = opt.max_len
        self.dim = opt.hidden_dim
        self.n_iter = opt.n_iter
        self.rnn_type = opt.rnn_type
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.clip = opt.clip
        self.mtl = opt.mtl

        self.n_nodes = kwargs["n_nodes"]
        if isinstance(self.n_nodes, int):
            self.n_nodes = {SEQUENCE: self.n_nodes}
        adjs = kwargs.get("adjs", dict())
        nums = kwargs.get("nums", dict())
        patterns = kwargs.get("patterns", dict())

        # Item representation & Position representation
        self.embeddings = nn.ModuleDict()
        for key in self.n_nodes:
            self.embeddings[key] = nn.Embedding(self.n_nodes[key], self.dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_local)

        self.gru = nn.GRU(self.dim, self.dim, self.n_iter, batch_first=True)

        self.denses = nn.ModuleDict()
        for key in self.n_nodes:
            self.denses[key] = nn.Linear(self.dim, self.n_nodes[key])

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
        """_summary_: forward propagation of GRU4Rec.
        It uses GRU to extract local context and memorize global information.

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
        h_seq = self.emb_dropout(h_seq)

        gru_out, hidden = self.gru(h_seq)

        if self.mtl:
            output = {key: gru_out for key in sorted_keys}
        else:
            output = {SEQUENCE: gru_out}
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
            gru_out = hidden[key]
            select = gru_out[:, -1]
            b = self.embeddings[key].weight
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
