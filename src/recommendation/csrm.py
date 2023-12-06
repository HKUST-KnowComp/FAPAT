import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import *
from utils import *


class CSRM(nn.Module):
    """_summary_: implementation of CSRM
    reference: Meirui Wang, Pengjie Ren, Lei Mei, Zhumin Chen, Jun Ma, and Maarten de Rijke. 2019. A Collaborative Session-based Recommendation Approach with Parallel Memory Modules. In SIGIR. 345–354.
    code reference: https://github.com/wmeirui/CSRM_SIGIR2019
    """

    def __init__(self, opt, **kwargs):
        super(CSRM, self).__init__()

        self.opt = opt

        self.batch_size = opt.batch_size
        self.max_len = opt.max_len
        self.dim = opt.hidden_dim
        self.n_iter = opt.n_iter
        self.mem_type = opt.mem_type
        self.mem_size = opt.mem_size
        self.mem_share = opt.mem_share
        self.topk = opt.topk
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

        self.gru_local = nn.GRU(self.dim, self.dim, self.n_iter, batch_first=True)
        self.gru_gloabl = nn.GRU(self.dim, self.dim, self.n_iter, batch_first=True)

        self.encoder = nn.Linear(self.dim, self.dim, bias=False)
        self.decoder = nn.Linear(self.dim, self.dim, bias=False)
        self.bl_vector = nn.Parameter(torch.Tensor(self.dim, 1))
        self.inner_encoder = nn.Linear(self.dim, 1, bias=False)
        self.state_encoder = nn.Linear(self.dim, 1, bias=False)
        self.outer_encoder = nn.Linear(self.dim, 1, bias=False)

        self.bili = nn.ModuleDict()
        for key in self.n_nodes:
            self.bili[key] = nn.Linear(2 * self.dim, self.dim)

        self.memory = None

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
        """_summary_: forward propagation of CSRM.
        It engage an inner memory encoder and external memory network to capture correlations 
        between neighborhood sessions to enrich the collaborative representation of the current session.

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

        # global
        h_global, last_global = self.gru_gloabl(h_seq)
        global_sess_rep = last_global[-1]

        # local
        h_local, last_local = self.gru_local(h_seq)

        tmp1 = self.encoder(h_local)
        tmp2 = self.decoder(last_local[-1])
        tmp3 = F.sigmoid(tmp1 + tmp2.unsqueeze(1))
        alpha = torch.matmul(tmp3, self.bl_vector)
        alpha = alpha.masked_fill(~mask.unsqueeze(2), -1e9)
        att = stable_softmax(alpha, -1)
        att_sess_rep = (h_local * att).sum(1)

        # memory
        if self.memory is None:
            # copy
            self.memory = att_sess_rep.detach()

            mem_read = att_sess_rep
        else:
            # update
            cos_sim = torch.matmul(att_sess_rep, self.memory.transpose(1, 0))
            if self.memory.size(0) > self.topk:
                gate_top_k_val, gate_top_k_idx = torch.topk(cos_sim, k=self.topk, dim=-1, largest=True, sorted=False)
                gate_top_k_val = stable_softmax(gate_top_k_val, dim=-1)
                top_k_memory = F.embedding(gate_top_k_idx, self.memory)
                mem_read = (gate_top_k_val.unsqueeze(2) * top_k_memory).sum(1)

            else:
                gate = stable_softmax(cos_sim, dim=-1)
                mem_read = torch.matmul(gate, self.memory)

            if self.training:
                self.memory = torch.cat([self.memory, att_sess_rep.detach()], dim=0)[-self.mem_size:]

        att_std, att_mean = torch.std_mean(att_sess_rep, dim=1, keepdim=True)
        att_sess_rep = (att_sess_rep - att_mean) / (att_std + 1e-9)
        global_std, global_mean = torch.std_mean(global_sess_rep, dim=1, keepdim=True)
        global_sess_rep = (global_sess_rep - global_mean) / (global_std + 1e-9)
        mem_std, mem_mean = torch.std_mean(mem_read, dim=1, keepdim=True)
        mem_read = (mem_read - mem_mean) / (mem_std + 1e-9)

        new_gate = F.sigmoid(
            self.inner_encoder(att_sess_rep) + self.state_encoder(global_sess_rep) + self.outer_encoder(mem_read)
        )

        narm_rep = torch.cat([att_sess_rep, global_sess_rep], dim=1)
        mem_rep = mem_read.repeat(1, 2)
        final_rep = new_gate * narm_rep + (1 - new_gate) * mem_rep

        if self.mtl:
            output = {key: final_rep for key in sorted_keys}
        else:
            output = {SEQUENCE: final_rep}
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
            final_rep = hidden[key]
            select = self.bili[key](final_rep)
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
