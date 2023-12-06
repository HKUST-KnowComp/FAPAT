import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import *
from utils import *


class GNN(nn.Module):
    """_summary_: implementation of GNN layer
    """

    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = hidden_size * 3
        self.w_ih = nn.Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(self.gate_size))
        self.b_hh = nn.Parameter(torch.Tensor(self.gate_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # parameters initialization
        self._reset_parameters()

    def _reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, A, hidden):
        for i in range(self.step):
            input_in = torch.matmul(A[:, :, :A.size(1)], self.linear_edge_in(hidden))
            input_out = torch.matmul(A[:, :, A.size(1):2 * A.size(1)], self.linear_edge_out(hidden))
            # [batch_size, max_session_len, embedding_size * 2]
            inputs = torch.cat([input_in, input_out], 2)

            # gi.size equals to gh.size, shape of [batch_size, max_session_len, embedding_size * 3]
            gi = F.linear(inputs, self.w_ih, self.b_ih)
            gh = F.linear(hidden, self.w_hh, self.b_hh)
            # (batch_size, max_session_len, embedding_size)
            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)
            reset_gate = torch.sigmoid(i_r + h_r)
            input_gate = torch.sigmoid(i_i + h_i)
            new_gate = torch.tanh(i_n + reset_gate * h_n)
            hidden = (1 - input_gate) * hidden + input_gate * new_gate

        return hidden


class GCSAN(nn.Module):
    """_summary_: implementation of GCSAN
    reference: Chengfeng Xu, Pengpeng Zhao, Yanchi Liu, Victor S. Sheng, Jiajie Xu, Fuzhen Zhuang, Junhua Fang, and Xiaofang Zhou. 2019. Graph Contextualized SelfAttention Network for Session-based Recommendation. In IJCAI. 3940–3946.
    code reference: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/gcsan.py
    """

    def __init__(self, opt, *args, **kwargs):
        super(GCSAN, self).__init__()
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
        self.gnns = nn.ModuleDict()
        for key in self.n_nodes:
            self.gnns[key] = GNN(self.dim, step=self.n_iter)
        self.self_attn = Transformer(
            dim=opt.hidden_dim,
            depth=opt.n_iter,
            dim_head=opt.hidden_dim // opt.n_head,
            heads=opt.n_head,
            buckets=opt.n_bucket,
            bidirectional=True,
            attn_dropout=opt.dropout_attn,
            ff_mult=4,
            ff_dropout=opt.dropout_ff,
        )

        # Item representation
        self.embeddings = nn.ModuleDict()
        for key in self.n_nodes:
            self.embeddings[key] = nn.Embedding(self.n_nodes[key], self.dim, padding_idx=0)

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

    def forward(self, input, mask=None, items=None, adj=None, alias=None, shortcut=None, heteitems=None, heteadj=None):
        """_summary_: forward propagation of GC-SAN.
        It uses gated GNN to extract local context information and 
        then employs the self-attention mechanism to obtain the global representation.

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

        As = dict()
        # normalize
        for key in adj:
            A = (adj[key] == 2).float() + (adj[key] == 4).float()
            A_in = A / A.sum(1, keepdim=True).clamp(min=1)
            A_out = A.transpose(1, 2) / A.sum(2, keepdim=True).clamp(min=1)
            As[key] = torch.cat([A_in, A_out], 1).transpose(1, 2)

        h_graphs = dict()
        for key in sorted_keys:
            h_graphs[key] = self.embeddings[key](items[key])
            h_graphs[key] = self.gnns[key](As[key], h_graphs[key])

        h_graph = torch.stack([h_graphs[key] for key in sorted_keys], dim=0).mean(0)

        if not exists(alias):
            alias = dict()

        if SEQUENCE in alias:
            h_graph = h_graph.view((batch_size * seq_len),
                                   -1)[(alias[SEQUENCE] + batch_flat_offset).view(-1)].view(batch_size, seq_len, -1)

        h_attn = self.self_attn(h_graph, mask=mask)
        output = h_attn[:, -1, :] + h_graph[:, -1, :]

        if self.mtl:
            output = {key: output for key in sorted_keys}
        else:
            output = {SEQUENCE: output}

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
