import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import *
from utils import *


class NARM(nn.Module):
    """_summary_: implementation of NARM
    reference:  Jing Li, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Tao Lian, and Jun Ma. 2017. Neural Attentive Session-based Recommendation. In CIKM. 1419–1428.
    code reference: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/narm.py
    """

    def __init__(self, opt, **kwargs):
        super(NARM, self).__init__()

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

        self.gru = nn.GRU(self.dim, self.dim, self.n_iter, batch_first=True)

        # Item representation & Position representation
        self.embeddings = nn.ModuleDict()
        for key in self.n_nodes:
            self.embeddings[key] = nn.Embedding(self.n_nodes[key], self.dim, padding_idx=0)
        self.emb_dropout = nn.Dropout(self.dropout_local)

        self.a_1 = nn.Linear(self.dim, self.dim, bias=False)
        self.a_2 = nn.Linear(self.dim, self.dim, bias=False)
        self.v_t = nn.Linear(self.dim, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout_local)
        self.b = nn.Linear(self.dim, 2 * self.dim, bias=False)

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
        """_summary_: forward propagation of NARM.
        It uses RNN to extract local context and memorize global information followed by attention.

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

        ht = hidden[-1]

        if self.mtl:
            output = {key: (gru_out, ht) for key in sorted_keys}
        else:
            output = {SEQUENCE: (gru_out, ht)}
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
            gru_out, ht = hidden[key]

            c_global = ht
            q1 = self.a_1(gru_out.contiguous().view(-1, self.dim)).view(gru_out.size())
            q2 = self.a_2(ht)
            q2_expand = q2.unsqueeze(1).expand_as(q1)
            if exists(mask):
                q2_expand = q2_expand * mask.unsqueeze(2).expand_as(q1).float()

            alpha = self.v_t(torch.sigmoid(q1 + q2_expand).view(-1, self.dim)).view(mask.size())
            c_local = torch.sum(alpha.unsqueeze(2).expand_as(gru_out) * gru_out, 1)

            select = torch.cat([c_local, c_global], 1)
            select = self.ct_dropout(select)
            b = self.b(self.embeddings[key].weight)  # n_nodes x 2 * latent_size
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