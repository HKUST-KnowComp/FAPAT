import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax as sparse_softmax
# from torch_sparse import sum as sparse_sum, matmul as sparse_matmul, mul as sparse_mul
from config import *
from layers import *
from utils import *


class SemanticExpander(nn.Module):
    
    def __init__(self, dim, order=1, reducer="concat"):
        super().__init__()
        self.dim = dim
        self.order = order
        self.reducer = reducer
        self.grus = nn.ModuleList()
        for i in range(self.order):
            self.grus.append(nn.GRU(dim, dim, 1, batch_first=True, bias=True))
    
        if self.reducer == 'concat':
            self.Ws = nn.ModuleList()
            for i in range(1, self.order):
                self.Ws.append(nn.Linear(dim * (i+1), dim))
        elif self.reducer in ["max", "mean"]:
            pass
        else:
            raise NotImplementedError

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for name, param in self.named_parameters():
            if name.endswith("bias"):
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -stdv, stdv)
        
    def forward(self, x):
        # bsz x seq_len x order x d
        shape = list(x.shape)
        if len(shape) < 4:
            return x
        assert shape[-2] - 2 < self.order
        if self.reducer == 'mean':
            invar = torch.mean(x, dim=-2)
        elif self.reducer == 'max':
            invar =  torch.max(x, dim=-2)[0]
        elif self.reducer == 'concat':
            invar =  self.Ws[shape[-2] - 2](x.view(*(shape[:-2] + [shape[-2] * shape[-1]])))

        var = self.grus[shape[-2] - 2](x.view(-1, shape[-2], shape[-1]))[1].view(*(shape[:-2] + [shape[-1]]))

        # return invar + var
        return 0.5 * invar + 0.5 * var

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.dim}, reducer={self.reducer}, order={self.order})')


class GAT(nn.Module):
    def __init__(
        self,
        dim,
        heads=1,
        add_self_loops=True,
        project=True,
        bias=True,
        dropout=0.0
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.add_self_loops = add_self_loops
        self.dropout = dropout

        head_dim = dim // heads
        if project:
            self.project = nn.Linear(dim, dim, bias=bias)
        else:
            self.project = None
        self.att_src = nn.Parameter(torch.Tensor(1, heads, head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, head_dim))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim // self.heads)
        if exists(self.project):
            nn.init.uniform_(self.project.weight, -stdv, stdv)
            if exists(self.project.bias):
                nn.init.constant_(self.project.bias, 0.0)
        nn.init.uniform_(self.att_src, -stdv, stdv)
        nn.init.uniform_(self.att_dst, -stdv, stdv)
        if exists(self.bias):
            nn.init.constant_(self.bias, 0.0)
    
    def forward(self, x, A, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)
        if exists(self.project):
            x = self.project(x)
        
        x_src = x_dst = x.view(batch_size, seq_len, self.heads, -1)
        alpha_src = (x_src * self.att_src).sum(dim=-1)[:, :, None, :]
        alph_dst = (x_dst * self.att_dst).sum(dim=-1)[:, None, :, :]
        alpha = alpha_src + alph_dst
        alpha = alpha.masked_fill((A != 1).unsqueeze(-1), 0)
        alpha = F.leaky_relu(alpha, 0.2)
        if exists(mask):
            _inf = -(torch.finfo(alpha.dtype).max)
            alpha = alpha.masked_fill(mask[:, :, None, None], _inf)
            alpha = alpha.masked_fill(mask[:, None, :, None], _inf)
        alpha = stable_softmax(alpha, dim=1) # bsz, seq_len, seq_len, heads
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = (alpha.unsqueeze(-1) * x_src.unsqueeze(2)).sum(1) # bsz, seq_len, heads, dim
        out = out.view(batch_size, seq_len, -1)

        if self.add_self_loops:
            out = out + x

        if exists(self.bias):
            out += self.bias

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.dim}, heads={self.heads}, add_self_loops={self.add_self_loops}, project={exists(self.project)}, bias={exists(self.bias)})')


class MSHGNN(nn.Module):
    
    def __init__(self, dim, heads=1, order=1, add_self_loops=True, dropout=0.0, act=None):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.order = order
        self.add_self_loops = add_self_loops
        self.dropout = dropout
        self.act = act

        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        for i in range(self.order):
            self.conv1.append(GAT(dim, heads=heads, add_self_loops=add_self_loops, dropout=dropout))
            self.conv2.append(GAT(dim, heads=heads, add_self_loops=add_self_loops, dropout=dropout))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim // self.heads)
        for name, param in self.named_parameters():
            if name.endswith("bias"):
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -stdv, stdv)
        
    def forward(self, x, A, mask=None):
        assert x.size(1) <= self.order

        # x: bsz x order x seq_len x d
        # A: bsz x order x seq_len x seq_len
        A1 = (A == 2).float() + (A == 4).float()
        A2 = (A == 3).float() + (A == 4).float()

        h = []
        for i in range(x.size(1)):
            x_i = x[:, i]
            A1_i = A1[:, i]
            A2_i = A2[:, i]
            h1_i = self.conv1[i](x_i, A1_i)
            h2_i = self.conv2[i](x_i, A2_i)
            h_i = h1_i + h2_i
            h.append(h_i)
        h = torch.stack(h, dim=1)
        h_mean = h.mean(1, keepdim=True)
        out = h + h_mean # bsz x order x seq_len x d
        out = F.dropout(out, p=self.dropout, training=self.training)

        if exists(self.act):
            shape = out.shape
            out = self.act(out.view(-1, shape[-1])).view(*shape)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.dim}, heads={self.heads}, add_self_loops={self.add_self_loops}, dropout={self.dropout})')


class AttnReadout(nn.Module):
    def __init__(
        self,
        dim,
        order=1,
        dropout=0.0,
        act=None,
    ):
        super().__init__()
        self.dim = dim
        self.order = order
        self.dropout = dropout

        self.fc_u = nn.ModuleList()
        self.fc_v = nn.ModuleList()
        self.fc_e = nn.ModuleList()
        for i in range(self.order):
            self.fc_u.append(nn.Linear(dim, dim, bias=True))
            self.fc_v.append(nn.Linear(dim, dim, bias=False))
            self.fc_e.append(nn.Linear(dim, 1, bias=False))
        self.fc_out = None
        self.act = act
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for name, param in self.named_parameters():
            if name.endswith("bias"):
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x, last_nodes=None):
        # x: bsz x order x seq_len x d
        # last_nodes: bsz x order
        if exists(last_nodes):
            assert x.size(0) == last_nodes.size(0)
            assert x.size(1) == last_nodes.size(1)
        assert x.size(1) == self.order
        batch_size = x.size(0)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x_all = x.view(batch_size, -1, x.size(-1))

        if exists(last_nodes):
            index0 = torch.arange(batch_size).to(last_nodes)
            x_last = torch.stack([x[:, i][index0, last_nodes[:, i]] for i in range(self.order)], dim=1) # bsz x order x dim
        else:
            x_last = x[:, :, -1]

        out = []
        for i in range(self.order):
            x_u = self.fc_u[i](x)
            x_v = self.fc_v[i](x_last[:, i])[:, None, None, :]
            e = self.fc_e[i](F.sigmoid(x_u + x_v)) # bsz x order x seq_len x 1
            e = e.view(batch_size, -1, 1) # bsz x (order x seq_len) x 1
            alpha = stable_softmax(e, dim=-2)
            o = (x_all * alpha).sum(-2) # bsz x d

            if exists(self.fc_out):
                o = self.fc_out(o)
            if exists(self.act):
                o = self.act(o)
            
            out.append(o)

        out = torch.stack(out, dim=1) # bsz x order x d

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(order={self.order}, act={self.act})')


class MSGIFSR(nn.Module):
    """_summary_: implementation of MSGIFSR
    reference: Jiayan Guo, Yaming Yang, Xiangchen Song, Yuan Zhang, Yujing Wang, Jing Bai, and Yan Zhang. 2022. Learning Multi-granularity Consecutive User Intent Unit for Session-based Recommendation. In WSDM. 343-352.
    code reference: https://github.com/SpaceLearner/SessionRec-pytorch/blob/main/src/models/msgifsr.py
    """

    def __init__(self, opt, *args, **kwargs):
        super(MSGIFSR, self).__init__()
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
        order = 2

        self.n_nodes = kwargs["n_nodes"]
        if isinstance(self.n_nodes, int):
            self.n_nodes = {SEQUENCE: self.n_nodes}
        adjs = kwargs.get("adjs", dict())
        nums = kwargs.get("nums", dict())

        # Aggregator
        self.layers = nn.ModuleDict()
        for key in self.n_nodes:
            layers = nn.ModuleList()
            for i in range(self.n_iter):
                layers.append(MSHGNN(self.dim, heads=opt.n_head, order=order, dropout=self.dropout_local, act=nn.PReLU(self.dim)))
            self.layers[key] = layers

        # Item representation
        self.embeddings = nn.ModuleDict()
        for key in self.n_nodes:
            self.embeddings[key] = nn.Embedding(self.n_nodes[key], self.dim, padding_idx=0)
        self.expander = SemanticExpander(self.dim, order=order, reducer="mean")

        # Predictor
        self.alpha = nn.Parameter(torch.Tensor(order))
        self.readout = AttnReadout(self.dim, order=order, dropout=self.dropout_local, act=None)
        self.fc_sr = nn.ModuleList()
        for i in range(order):
            self.fc_sr.append(nn.Linear(2 * self.dim, self.dim, bias=False))

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
        with torch.no_grad():
            self.alpha.data.fill_(0)
            self.alpha.data[0].fill_(1)

    def forward(self, input, mask=None, items=None, adj=None, alias=None, shortcut=None, heteitems=None, heteadj=None):
        """_summary_: forward propagation of SR-GNN.
        It transforms the session data into a direct unweighted graph and 
        utilizes gated GNN to learn the representation of the item transitions graph.

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
        :param shortcut: shortcut matrix
        :type shortcut: torch.Tensor
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

        # order = 1
        h_graphs = dict()
        for key in sorted_keys:
            h_graphs[key] = self.expander(self.embeddings[key](items[key]))

        # order = 2
        h_hete_graphs = dict()
        for key in sorted_keys:
            h_hete_graphs[key] = self.expander(self.embeddings[key](heteitems[key]))

        # multi-granularity
        h_multi_graphs = dict()
        A_multi_adj = dict()
        for key in sorted_keys:
            h_multi_graphs[key] = torch.stack([h_graphs[key], h_hete_graphs[key]], dim=1)
            A_multi_adj[key] = torch.stack([adj[key], heteadj[key]], dim=1)
            for i, layer in enumerate(self.layers[key]):
                h_multi_graphs[key] = layer(h_multi_graphs[key], A_multi_adj[key])
        h_graph = torch.stack([h_multi_graphs[key] for key in sorted_keys], dim=0).mean(0)
        h_graph = F.normalize(h_graph, dim=-1) # bsz x order x seq_len x d

        # last_nodes = (items[SEQUENCE] != 0).float().cumsum(1).argmax(1)
        # sr_g = self.readout(h_graph, last_nodes)
        # sr_l = h_graph[torch.arange(batch_size).to(last_nodes), last_nodes]
        sr_g = self.readout(h_graph) # bsz x order x d
        sr_l = h_graph[:, :, -1] # bsz x order x d
        sr = torch.cat([sr_g, sr_l], dim=2)  # bsz x order x 2d
        sr = F.dropout(sr, p=self.dropout_global, training=self.training)
        sr = torch.stack([self.fc_sr[i](sr_) for i, sr_ in enumerate(torch.unbind(sr, dim=1))], dim=1) # bsz x order x d
        
        if not exists(alias):
            alias = dict()

        if self.mtl:
            output = {key: None for key in sorted_keys}
        else:
            output = {SEQUENCE: None}
        for key in output:
            output[key] = sr

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
            select = hidden[key] # bsz x order x latent_size
            b = self.embeddings[key].weight  # n_nodes x latent_size
            alpha = stable_softmax(self.alpha, dim=-1)
            scores[key] = torch.matmul(select, b.transpose(1, 0))
            scores[key] = (stable_softmax(scores[key] * 12, dim=-1) * alpha[None, :, None]).sum(1)
            scores[key] = torch.log(scores[key])
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
