import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import *
from utils import *


class BPRLoss(nn.Module):
    """_summary_: implementation of BPRLoss based on positive and negative scores
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class FPMC(nn.Module):
    """_summary_: implementation of FPMC
    reference: Steffen Rendle, Christoph Freudenthaler, and Lars Schmidt-Thieme. 2010. Factorizing personalized Markov chains for next-basket recommendation. In WWW. 811–820.
    code reference: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/fpmc.py
    """

    def __init__(self, opt, *args, **kwargs):
        super().__init__()

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

        # user embedding matrix
        self.UI_emb = nn.Parameter(torch.Tensor(1, self.dim))
        # label embedding matrix
        self.IU_embebeddings = nn.ModuleDict()
        # last click item embedding matrix
        self.LI_embebeddings = nn.ModuleDict()
        # label embedding matrix
        self.IL_embebeddings = nn.ModuleDict()

        for key in self.n_nodes:
            self.IU_embebeddings[key] = nn.Embedding(self.n_nodes[key], self.dim, padding_idx=0)
            self.LI_embebeddings[key] = nn.Embedding(self.n_nodes[key], self.dim, padding_idx=0)
            self.IL_embebeddings[key] = nn.Embedding(self.n_nodes[key], self.dim, padding_idx=0)

        # Optimization
        # self.loss_function = BPRLoss()
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
        """_summary_: forward propagation of FPMC.
        It learns the representation of session via Markov-chain based method.

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
        sorted_keys = sorted(self.IU_embebeddings.keys())
        assert len(set(sorted_keys) - set(input.keys())) == 0
        seq_idx = sorted_keys.index(SEQUENCE)
        sorted_keys[0], sorted_keys[seq_idx] = sorted_keys[seq_idx], sorted_keys[0]

        user_emb = self.UI_emb.expand(batch_size, -1)  # batch_size x dim

        if self.mtl:
            output = {
                key: (self.LI_embebeddings[key](input[key][:, -self.n_iter:]).sum(1), user_emb)
                for key in sorted_keys
            }
        else:
            output = {
                SEQUENCE: (self.LI_embebeddings[SEQUENCE](input[SEQUENCE][:, -self.n_iter:]).sum(1), user_emb)
            }
        return output

    def compute_scores(self, hidden, mask=None):
        """_summary_: compute matrix-factorization and Markov-chain scores for ranking

        :param hidden: hidden states for seqeuence and attributes
        :type hidden: dict
        :param mask: mask for valid data, defaults to None
        :type mask: torch.Tensor, optional
        :return: scores for sequence and attributes
        :rtype: dict
        """

        scores = dict()
        for key in hidden.keys():
            item_seq_emb, user_emb = hidden[key]
            iu_emb = self.IU_embebeddings[key].weight  # n_nodes x dim
            il_emb = self.IL_embebeddings[key].weight  # n_nodes x dim
            mf = torch.matmul(user_emb, iu_emb.transpose(1, 0))  # batch_size x n_nodes
            fmc = torch.matmul(item_seq_emb, il_emb.transpose(1, 0))  # batch_size x n_nodes
            scores[key] = (mf + fmc)
        return scores

    def compute_loss(self, scores, target, coefs=None, n_neg=100):
        # BPRLoss
        # ind = torch.arange(target[SEQUENCE].shape[0], device=scores[SEQUENCE].device).unsqueeze(1)
        # if self.mtl:
        #     if coefs is None:
        #         coefs = {key: 1.0 for key in scores.keys()}
        #     loss = 0.0
        #     for key, coef in coefs.items():
        #         pos_score = scores[key].gather(1, target[key].unsqueeze(1))
        #         random = torch.randint(0, self.n_nodes[key], (target[key].shape[0], n_neg), device=target[key].device)
        #         random = random + (random >= target[SEQUENCE].unsqueeze(1)).long()
        #         neg_score = scores[key][ind, random]
        #         loss += coef * self.loss_function(pos_score, neg_score)
        # else:
        #     # print(scores[SEQUENCE].size(), target[SEQUENCE].size())
        #     pos_score = scores[SEQUENCE].gather(1, target[SEQUENCE].unsqueeze(1))
        #     random = torch.randint(0, self.n_nodes[SEQUENCE] - 1, (target[SEQUENCE].shape[0], n_neg), device=target[SEQUENCE].device)
        #     random = random + (random >= target[SEQUENCE].unsqueeze(1)).long()
        #     neg_score = scores[SEQUENCE][ind, random]
        #     loss = self.loss_function(pos_score, neg_score)
        # return loss

        # CrossEntropyLoss
        if self.mtl:
            if coefs is None:
                coefs = {key: 1.0 for key in scores.keys()}
            loss = torch.stack([coef * self.loss_function(scores[key], target[key])
                                for key, coef in coefs.items()]).sum()
        else:
            loss = self.loss_function(scores[SEQUENCE], target[SEQUENCE])
        return loss