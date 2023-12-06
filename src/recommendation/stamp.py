import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import *
from utils import *


class STAMP(nn.Module):
    """_summary_: implementation of STAMP
    reference: Qiao Liu, Yifu Zeng, Refuoe Mokhosi, and Haibin Zhang. 2018. STAMP: ShortTerm Attention/Memory Priority Model for Session-based Recommendation. In KDD. 1831–1839.
    code reference: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/stamp.py
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

        # Item representation & Position representation
        self.embeddings = nn.ModuleDict()
        for key in self.n_nodes:
            self.embeddings[key] = nn.Embedding(self.n_nodes[key], self.dim, padding_idx=0)

        self.w1 = nn.Linear(self.dim, self.dim, bias=False)
        self.w2 = nn.Linear(self.dim, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.dim, bias=False)
        self.w0 = nn.Linear(self.dim, 1, bias=False)
        self.b_a = nn.Parameter(torch.zeros(self.dim), requires_grad=True)
        self.mlp_a = nn.Linear(self.dim, self.dim, bias=True)
        self.mlp_b = nn.Linear(self.dim, self.dim, bias=True)

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
        nn.init.constant_(self.b_a, 0.0)

    def forward(self, input, mask=None, items=None, adj=None, alias=None, shortcut=None, heteitems=None, heteadj=None):
        """_summary_: forward propagation of STAMP.
        It adopts attention mechanism between the last item 
        to previous histories to represent users’ short-term interests.

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

        last_h = h_seq[:, -1, :]

        mask = mask.unsqueeze(-1).float()
        ms = (h_seq * mask).sum(1) / mask.sum(1)

        if self.mtl:
            output = {key: None for key in sorted_keys}
        else:
            output = {SEQUENCE: None}

        for key in output:
            output[key] = (h_seqs[key], last_h, ms)

        return output

    def count_alpha(self, context, aspect, output):
        r"""This is a function that count the attention weights
        Args:
            context(torch.FloatTensor): Item list embedding matrix, shape of [batch_size, time_steps, emb]
            aspect(torch.FloatTensor): The embedding matrix of the last click item, shape of [batch_size, emb]
            output(torch.FloatTensor): The average of the context, shape of [batch_size, emb]
        Returns:
            torch.Tensor:attention weights, shape of [batch_size, time_steps]
        """
        timesteps = context.size(1)
        aspect_3dim = aspect.unsqueeze(1).repeat(1, timesteps, 1)
        output_3dim = output.unsqueeze(1).repeat(1, timesteps, 1)
        res_ctx = self.w1(context)
        res_asp = self.w2(aspect_3dim)
        res_output = self.w3(output_3dim)
        res_sum = res_ctx + res_asp + res_output + self.b_a
        res_act = self.w0(F.sigmoid(res_sum))
        alpha = res_act.squeeze(2)
        return alpha

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
            (context, last_h, ms) = hidden[key]
            alpha = self.count_alpha(context, last_h, ms)
            vec = torch.matmul(alpha.unsqueeze(1), context)
            ma = vec.squeeze(1) + ms
            hs = F.tanh(self.mlp_a(ma))
            ht = F.tanh(self.mlp_b(last_h))
            select = hs * ht
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
