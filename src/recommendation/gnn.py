import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse
import torch_geometric.nn.conv as gconv
from functools import partial
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.pool import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops, softmax as sparse_softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from torch_sparse import fill_diag
from torch_sparse import sum as sparse_sum, matmul as sparse_matmul, mul as sparse_mul
from config import *
from layers import *
from utils import *


class GCNConv(gconv.MessagePassing):
    """_summary_: implementation of GCN layer
    """

    def __init__(
        self,
        dim,
        normalize=True,
        add_self_loops=False,
        aggr="add",
        project=True,
        bias=True,
        **kwargs
    ):
        super(GCNConv, self).__init__(aggr=aggr, **kwargs)

        self.dim = dim
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        if project:
            self.project = nn.Linear(dim, dim, bias=bias)
        else:
            self.project = None

        if bias:
            self.bias = nn.Parameter(torch.Tensor(dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim)
        if exists(self.project):
            nn.init.uniform_(self.project.weight, -stdv, stdv)
            if exists(self.project.bias):
                nn.init.constant_(self.project.bias, 0.0)
        if exists(self.bias):
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, edge_index, edge_type=None, edge_weight=None, size=None):
        num_nodes = maybe_num_nodes(edge_index, x.size(self.node_dim))

        if self.normalize:
            if isinstance(edge_index, torch.Tensor):
                edge_index, edge_weight = gcn_norm(
                    edge_index, edge_weight, num_nodes, add_self_loops=self.add_self_loops
                )

            elif isinstance(edge_index, torch_sparse.SparseTensor):
                edge_index = gcn_norm(edge_index, edge_weight, num_nodes, add_self_loops=self.add_self_loops)
        else:
            if isinstance(edge_index, torch.Tensor):
                edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 1, num_nodes)
            elif isinstance(edge_index, torch_sparse.SparseTensor):
                edge_index = fill_diag(edge_index, 1)

        if exists(self.project):
            x = self.project(x)

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)

        if exists(self.bias):
            out += self.bias

        return out

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return sparse_matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return f'{self.__class__.__name__}(normalize={self.normalize}, add_self_loops={self.add_self_loops}, aggr={self.aggr}, project={exists(self.project)}, bias={exists(self.bias)})'


class GINConv(gconv.MessagePassing):
    """_summary_: implementation of GIN layer
    """

    def __init__(
        self,
        dim,
        init_eps=0.0,
        train_eps=True,
        mlp_layers=0,
        actication="relu",
        batch_norm=True,
        aggr="add",
        project=True,
        bias=True,
        **kwargs
    ):
        super(GINConv, self).__init__(aggr=aggr, **kwargs)

        self.dim = dim

        if project:
            self.project = nn.Linear(dim, dim, bias=bias)
        else:
            self.project = None

        if mlp_layers > 0:
            mlp = []
            for i in range(mlp_layers):
                mlp.append(nn.Linear(dim, dim, bias=bias))
                mlp.append(get_activation(actication))
                if batch_norm:
                    mlp.append(nn.BatchNorm1d(dim))
            self.mlp = nn.Sequential(*mlp)
        else:
            self.mlp = None

        if bias:
            self.bias = nn.Parameter(torch.Tensor(dim))
        else:
            self.register_parameter('bias', None)

        self.init_eps = init_eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([init_eps]))
        else:
            self.register_buffer('eps', torch.Tensor([init_eps]))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim)
        if exists(self.project):
            nn.init.uniform_(self.project.weight, -stdv, stdv)
            if exists(self.project.bias):
                nn.init.constant_(self.project.bias, 0.0)
        if exists(self.mlp):
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -stdv, stdv)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
        if exists(self.bias):
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, edge_index, edge_type=None, edge_weight=None, size=None):
        if isinstance(x, torch.Tensor):
            if exists(self.project):
                x_src = self.project(x)
            else:
                x_src = x
            x_dst = x
        else:
            x_src, x_dst = x
            if exists(self.project):
                x_src = self.project(x_src)

        out = self.propagate(edge_index, x=x_src, edge_weight=edge_weight, size=size)

        out += (1 + self.eps) * x_dst

        if exists(self.mlp):
            out = self.mlp(out)

        if exists(self.bias):
            out += self.bias

        return out

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return sparse_matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return f'{self.__class__.__name__}(eps={self.eps.detach().item()}, mlp={self.mlp}, aggr={self.aggr}, project={exists(self.project)}, bias={exists(self.bias)})'


class SAGEConv(gconv.MessagePassing):
    """_summary_: implementation of SAGE layer
    """

    def __init__(
        self,
        dim,
        normalize=False,
        add_self_loops=True,
        aggr="max",
        project=True,
        bias=True,
        **kwargs
    ):
        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', dim)
            kwargs['aggr_kwargs'].setdefault('out_channels', dim)

        super(SAGEConv, self).__init__(aggr, **kwargs)

        self.dim = dim
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        if project:
            self.project = nn.Linear(dim, dim, bias=bias)
        else:
            self.project = None

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = nn.LSTM(dim, dim, batch_first=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(dim)
        else:
            aggr_out_channels = dim

        self.linear_src = nn.Linear(aggr_out_channels, dim, bias=bias)
        if self.add_self_loops:
            self.linear_self = nn.Linear(dim, dim, bias=False)
        else:
            self.linear_self = None

        if bias:
            self.bias = nn.Parameter(torch.Tensor(dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim)
        if exists(self.project):
            nn.init.uniform_(self.project.weight, -stdv, stdv)
            if exists(self.project.bias):
                nn.init.constant_(self.project.bias, 0.0)
        nn.init.uniform_(self.linear_src.weight, -stdv, stdv)
        if exists(self.linear_src.bias):
            nn.init.constant_(self.linear_src.bias, 0.0)
        if exists(self.linear_self):
            nn.init.uniform_(self.linear_self.weight, -stdv, stdv)
        if exists(self.bias):
            nn.init.constant_(self.bias, 0.0)
        self.aggr_module.reset_parameters()

    def forward(self, x, edge_index, edge_type=None, edge_weight=None, size=None):
        if isinstance(x, torch.Tensor):
            x_src = x
            x_dst = x
        else:
            x_src, x_dst = x

        if exists(self.project):
            x_src = self.project(x_src)

        out = self.propagate(edge_index, x=(x_src, x_dst), edge_weight=edge_weight, size=size)
        out = self.linear_src(out)

        if self.add_self_loops:
            out += self.linear_self(x_dst)

        if exists(self.bias):
            out += self.bias

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        adj_t = adj_t.set_value(None, layout=None)
        return sparse_matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return f'{self.__class__.__name__}(normalize={self.normalize}, add_self_loops={self.add_self_loops}, aggr={self.aggr}, project={exists(self.project)}, bias={exists(self.bias)})'


class GATConv(gconv.MessagePassing):
    """_summary_: implementation of GAT layer
    """

    def __init__(
        self,
        dim,
        heads=1,
        edge_dim=None,
        add_self_loops=True,
        aggr="add",
        project=True,
        bias=True,
        dropout=0.0,
        **kwargs
    ):
        super(GATConv, self).__init__(aggr=aggr, node_dim=0, **kwargs)
        self.dim = dim
        self.heads = heads
        self.add_self_loops = add_self_loops
        self.dropout = dropout

        assert heads > 0 and isinstance(heads, int) and dim % heads == 0, \
            "The number of heads must be a positive integer divisor of the output dimension."
        head_dim = dim // heads
        if project:
            self.project = nn.Linear(dim, dim, bias=bias)
        else:
            self.project = None

        self.att_src = nn.Parameter(torch.Tensor(1, heads, head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, head_dim))

        if edge_dim is not None:
            self.linear_edge = nn.Linear(edge_dim, heads * head_dim, bias=False)
            self.att_edge = nn.Parameter(torch.Tensor(1, heads, head_dim))
        else:
            self.linear_edge = None
            self.register_parameter('att_edge', None)

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
        if exists(self.linear_edge):
            nn.init.uniform_(self.linear_edge.weight, -stdv, stdv)
            nn.init.uniform_(self.att_edge, -stdv, stdv)
        nn.init.uniform_(self.att_src, -stdv, stdv)
        nn.init.uniform_(self.att_dst, -stdv, stdv)
        if exists(self.bias):
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, edge_index, edge_type=None, edge_weight=None, size=None):

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, torch.Tensor):
            batch_size = x.size(0)
            if exists(self.project):
                x = self.project(x)
            x_src = x_dst = x.view(batch_size, self.heads, -1)
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            batch_size = x_src.size(0)
            if exists(self.project):
                x_src = self.project(x_src).view(batch_size, self.heads, -1)
                x_dst = self.project(x_dst).view(batch_size, self.heads, -1)

        if self.add_self_loops:
            if isinstance(edge_index, torch.Tensor):
                num_nodes = maybe_num_nodes(edge_index)
                edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 1, num_nodes)
            elif isinstance(edge_index, torch_sparse.SparseTensor):
                edge_index = fill_diag(edge_index, 1)

        # edge_updater_type: (alpha: OptPairTensor, edge_feat: OptTensor)
        alpha = self.edge_updater(edge_index, x=(x_src, x_dst), edge_feat=edge_weight)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=(x_src, x_dst), alpha=alpha, edge_weight=edge_weight, size=size)
        out = out.view(batch_size, self.dim)

        if exists(self.bias):
            out += self.bias

        return out

    def edge_update(self, x_j, x_i, edge_feat, index, ptr, size_i):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha_j = (x_j * self.att_src).sum(dim=-1)
        alpha_i = (x_i * self.att_dst).sum(dim=-1)
        alpha = alpha_j + alpha_i

        # we use dot product to compute attention coefficients
        # alpha = ((x_j * self.att_src) * (x_i * self.att_dst) * math.sqrt(self.dim // self.heads)).sum(dim=-1)
        # alpha = ((x_j * self.att_src) * (x_i * self.att_dst)).sum(dim=-1)

        if exists(edge_feat):
            if exists(self.linear_edge):
                edge_feat = self.linear_edge(edge_feat)
            edge_feat = edge_feat.view(edge_feat.size(0), self.heads, -1)
            alpha_edge = (edge_feat * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, 0.2)
        alpha = sparse_softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j, alpha, edge_weight=None):
        return alpha.unsqueeze(-1) * x_j if edge_weight is None else edge_weight.view(-1, 1) * alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(add_self_loops={self.add_self_loops}, aggr={self.aggr}, project={exists(self.project)}, bias={exists(self.bias)})'


class RGCNConv(gconv.MessagePassing):
    """_summary_: implementation of RGCN layer
    """

    def __init__(
        self,
        dim,
        num_relations,
        num_bases=None,
        num_blocks=None,
        add_self_loops=True,
        aggr="mean",
        project=True,
        bias=True,
        **kwargs
    ):
        super(RGCNConv, self).__init__(aggr=aggr, node_dim=0, **kwargs)

        if num_bases is not None and num_blocks is not None:
            raise ValueError(
                'Can not apply both basis-decomposition and '
                'block-diagonal-decomposition at the same time.'
            )

        self.dim = dim
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks

        if project:
            self.project = nn.Linear(dim, dim, bias=bias)
        else:
            self.project = None

        if num_bases is not None:
            assert num_blocks > 0 and isinstance(num_blocks, int), \
                'num_blocks must be a positive integer.'
            self.weight = nn.Parameter(torch.Tensor(num_bases, dim, dim))
            self.comp = nn.Parameter(torch.Tensor(num_relations, num_bases))

        elif num_blocks is not None:
            assert num_blocks > 0 and isinstance(num_blocks, int) and dim % num_blocks == 0, \
                'Number of blocks must divide the dimension of the input tensor.'
            block_dim = dim // num_blocks
            self.weight = nn.Parameter(torch.Tensor(num_relations, num_blocks, block_dim, block_dim))
            self.register_parameter('comp', None)

        else:
            self.weight = nn.Parameter(torch.Tensor(num_relations, dim, dim))
            self.register_parameter('comp', None)

        if add_self_loops:
            self.linear_self = nn.Linear(dim, dim, bias=False)
        else:
            self.linear_self = None

        if bias:
            self.bias = nn.Parameter(torch.Tensor(dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim)
        if exists(self.project):
            nn.init.uniform_(self.project.weight, -stdv, stdv)
            if exists(self.project.bias):
                nn.init.constant_(self.project.bias, 0.0)
        nn.init.uniform_(self.weight, -stdv, stdv)
        if exists(self.comp):
            nn.init.uniform_(self.comp, -stdv, stdv)
        if exists(self.linear_self):
            nn.init.uniform_(self.linear_self.weight, -stdv, stdv)
        if exists(self.bias):
            nn.init.constant_(self.bias, 0)

    def forward(self, x, edge_index, edge_type=None, edge_weight=None, size=None):
        self.fuse = False
        assert self.aggr in ['add', 'sum', 'mean']

        if isinstance(x, torch.Tensor):
            if exists(self.project):
                x_src = self.project(x)
            else:
                x_src = x
            x_dst = x
        else:
            x_src, x_dst = x
            if exists(self.project):
                x_src = self.project(x_src)
        size = (x_src.size(0), x_dst.size(0))

        # propagate_type: (x: Tensor, edge_type: OptTensor)
        out = self.propagate(edge_index, x=x_src, edge_type=edge_type, size=size)

        if exists(self.linear_self):
            out += self.linear_self(x_dst)

        if exists(self.bias):
            out += self.bias

        return out

    def message(self, x_j, edge_type, edge_index_j):
        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(self.num_relations, self.dim, self.dim)

        if self.num_blocks is not None:  # Block-diagonal-decomposition =======
            if x_j.dtype == torch.long:
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            weight = weight[edge_type].view(-1, weight.size(2), weight.size(3))
            x_j = x_j.view(-1, 1, weight.size(1))
            return torch.bmm(x_j, weight).view(-1, self.dim)

        else:  # No regularization/Basis-decomposition ========================
            if x_j.dtype == torch.long:
                weight_index = edge_type * weight.size(1) + edge_index_j
                return weight.view(-1, self.dim)[weight_index]

            return torch.bmm(x_j.unsqueeze(-2), weight[edge_type]).squeeze(-2)

    def aggregate(self, input, edge_type, index, dim_size=None):

        # Compute normalization in separation for each `edge_type`.
        if self.aggr == 'mean':
            norm = F.one_hot(edge_type, self.num_relations).to(torch.float)
            norm = scatter_add(norm, index, dim=0, dim_size=dim_size)[index]
            norm = torch.gather(norm, 1, edge_type.view(-1, 1))
            norm = 1. / norm.clamp_(1.)
            input = norm * input

        return scatter_add(input, index, dim=self.node_dim, dim_size=dim_size)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={self.num_relations}, num_bases={self.num_bases}, num_blocks={self.num_blocks}, add_self_loops={exists(self.linear_self)}, aggr={self.aggr}, project={exists(self.project)}, bias={exists(self.bias)})'


class RGINConv(gconv.MessagePassing):
    """_summary_: implementation of RGIN layer
    """

    def __init__(
        self,
        dim,
        num_relations,
        num_bases=None,
        num_blocks=None,
        init_eps=0.0,
        train_eps=True,
        mlp_layers=0,
        actication="relu",
        batch_norm=True,
        aggr="add",
        project=True,
        bias=True,
        **kwargs
    ):
        super(RGINConv, self).__init__(aggr=aggr, node_dim=0, **kwargs)

        if num_bases is not None and num_blocks is not None:
            raise ValueError(
                'Can not apply both basis-decomposition and '
                'block-diagonal-decomposition at the same time.'
            )

        self.dim = dim
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks

        if project:
            self.project = nn.Linear(dim, dim, bias=bias)
        else:
            self.project = None

        if num_bases is not None:
            assert num_blocks > 0 and isinstance(num_blocks, int), \
                'num_blocks must be a positive integer.'
            self.weight = nn.Parameter(torch.Tensor(num_bases, dim, dim))
            self.comp = nn.Parameter(torch.Tensor(num_relations, num_bases))

        elif num_blocks is not None:
            assert num_blocks > 0 and isinstance(num_blocks, int) and dim % num_blocks == 0, \
                'Number of blocks must divide the dimension of the input tensor.'
            block_dim = dim // num_blocks
            self.weight = nn.Parameter(torch.Tensor(num_relations, num_blocks, block_dim, block_dim))
            self.register_parameter('comp', None)

        else:
            self.weight = nn.Parameter(torch.Tensor(num_relations, dim, dim))
            self.register_parameter('comp', None)

        if mlp_layers > 0:
            mlp = []
            for i in range(mlp_layers):
                mlp.append(nn.Linear(dim, dim, bias=bias))
                mlp.append(get_activation(actication))
                if batch_norm:
                    mlp.append(nn.BatchNorm1d(dim))
            self.mlp = nn.Sequential(*mlp)
        else:
            self.mlp = None

        if bias:
            self.bias = nn.Parameter(torch.Tensor(dim))
        else:
            self.register_parameter('bias', None)

        self.init_eps = init_eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([init_eps]))
        else:
            self.register_buffer('eps', torch.Tensor([init_eps]))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim)
        if exists(self.project):
            nn.init.uniform_(self.project.weight, -stdv, stdv)
            if exists(self.project.bias):
                nn.init.constant_(self.project.bias, 0.0)
        nn.init.uniform_(self.weight, -stdv, stdv)
        if exists(self.comp):
            nn.init.uniform_(self.comp, -stdv, stdv)
        if exists(self.mlp):
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -stdv, stdv)
                    nn.init.constant_(m.bias, 0.0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)
        if exists(self.bias):
            nn.init.constant_(self.bias, 0)

    def forward(self, x, edge_index, edge_type=None, edge_weight=None, size=None):
        self.fuse = False
        assert self.aggr in ['add', 'sum', 'mean']

        if isinstance(x, torch.Tensor):
            if exists(self.project):
                x_src = self.project(x)
            else:
                x_src = x
            x_dst = x
        else:
            x_src, x_dst = x
            if exists(self.project):
                x_src = self.project(x_src)
        size = (x_src.size(0), x_dst.size(0))

        # propagate_type: (x: Tensor, edge_type: OptTensor)
        out = self.propagate(edge_index, x=x_src, edge_type=edge_type, size=size)

        out += (1 + self.eps) * x_dst

        if exists(self.mlp):
            out = self.mlp(out)

        if exists(self.bias):
            out += self.bias

        return out

    def message(self, x_j, edge_type, edge_index_j):
        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(self.num_relations, self.dim, self.dim)

        if self.num_blocks is not None:  # Block-diagonal-decomposition =======
            if x_j.dtype == torch.long:
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            weight = weight[edge_type].view(-1, weight.size(2), weight.size(3))
            x_j = x_j.view(-1, 1, weight.size(1))
            return torch.bmm(x_j, weight).view(-1, self.dim)

        else:  # No regularization/Basis-decomposition ========================
            if x_j.dtype == torch.long:
                weight_index = edge_type * weight.size(1) + edge_index_j
                return weight.view(-1, self.dim)[weight_index]

            return torch.bmm(x_j.unsqueeze(-2), weight[edge_type]).squeeze(-2)

    def aggregate(self, input, edge_type, index, dim_size=None):

        # Compute normalization in separation for each `edge_type`.
        if self.aggr == 'mean':
            norm = F.one_hot(edge_type, self.num_relations).to(torch.float)
            norm = scatter_add(norm, index, dim=0, dim_size=dim_size)[index]
            norm = torch.gather(norm, 1, edge_type.view(-1, 1))
            norm = 1. / norm.clamp_(1.)
            input = norm * input

        return scatter_add(input, index, dim=self.node_dim, dim_size=dim_size)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={self.num_relations}, num_bases={self.num_bases}, num_blocks={self.num_blocks}, eps={self.eps.detach().item()}, mlp={self.mlp}, aggr={self.aggr}, project={exists(self.project)}, bias={exists(self.bias)})'


class RSAGEConv(gconv.MessagePassing):
    """_summary_: implementation of RSAGE layer
    """

    def __init__(
        self,
        dim,
        num_relations,
        num_bases=None,
        num_blocks=None,
        normalize=False,
        add_self_loops=True,
        aggr="add",
        project=True,
        bias=True,
        **kwargs
    ):
        super(RSAGEConv, self).__init__(aggr=aggr, node_dim=0, **kwargs)

        if num_bases is not None and num_blocks is not None:
            raise ValueError(
                'Can not apply both basis-decomposition and '
                'block-diagonal-decomposition at the same time.'
            )

        self.dim = dim
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        if project:
            self.project = nn.Linear(dim, dim, bias=bias)
        else:
            self.project = None

        if self.aggr is None:
            self.fuse = False  # No "fused" message_and_aggregate.
            self.lstm = nn.LSTM(dim, dim, batch_first=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(dim)
        else:
            aggr_out_channels = dim

        self.linear_src = nn.Linear(aggr_out_channels, dim, bias=bias)

        if num_bases is not None:
            assert num_blocks > 0 and isinstance(num_blocks, int), \
                'num_blocks must be a positive integer.'
            self.weight = nn.Parameter(torch.Tensor(num_bases, dim, dim))
            self.comp = nn.Parameter(torch.Tensor(num_relations, num_bases))

        elif num_blocks is not None:
            assert num_blocks > 0 and isinstance(num_blocks, int) and dim % num_blocks == 0, \
                'Number of blocks must divide the dimension of the input tensor.'
            block_dim = dim // num_blocks
            self.weight = nn.Parameter(torch.Tensor(num_relations, num_blocks, block_dim, block_dim))
            self.register_parameter('comp', None)

        else:
            self.weight = nn.Parameter(torch.Tensor(num_relations, dim, dim))
            self.register_parameter('comp', None)

        if add_self_loops:
            self.linear_self = nn.Linear(dim, dim, bias=False)
        else:
            self.linear_self = None

        if bias:
            self.bias = nn.Parameter(torch.Tensor(dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.dim)
        if exists(self.project):
            nn.init.uniform_(self.project.weight, -stdv, stdv)
            if exists(self.project.bias):
                nn.init.constant_(self.project.bias, 0.0)
        nn.init.uniform_(self.weight, -stdv, stdv)
        if exists(self.comp):
            nn.init.uniform_(self.comp, -stdv, stdv)
        nn.init.uniform_(self.linear_src.weight, -stdv, stdv)
        if exists(self.linear_src.bias):
            nn.init.constant_(self.linear_src.bias, 0.0)
        if exists(self.linear_self):
            nn.init.uniform_(self.linear_self.weight, -stdv, stdv)
        if exists(self.bias):
            nn.init.constant_(self.bias, 0)

    def forward(self, x, edge_index, edge_type=None, edge_weight=None, size=None):
        self.fuse = False
        if isinstance(x, torch.Tensor):
            if exists(self.project):
                x_src = self.project(x)
            else:
                x_src = x
            x_dst = x
        else:
            x_src, x_dst = x
            if exists(self.project):
                x_src = self.project(x_src)
        size = (x_src.size(0), x_dst.size(0))

        # propagate_type: (x: Tensor, edge_type: OptTensor)
        out = self.propagate(edge_index, x=(x_src, x_dst), edge_type=edge_type, edge_weight=edge_weight, size=size)
        out = self.linear_src(out)

        if exists(self.linear_self):
            out += self.linear_self(x_dst)

        if exists(self.bias):
            out += self.bias

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j, edge_type, edge_index_j, edge_weight=None):
        weight = self.weight
        if self.num_bases is not None:  # Basis-decomposition =================
            weight = (self.comp @ weight.view(self.num_bases, -1)).view(self.num_relations, self.dim, self.dim)

        if self.num_blocks is not None:  # Block-diagonal-decomposition =======
            if x_j.dtype == torch.long:
                raise ValueError('Block-diagonal decomposition not supported '
                                 'for non-continuous input features.')

            weight = weight[edge_type].view(-1, weight.size(2), weight.size(3))
            x_j = x_j.view(-1, 1, weight.size(1))
            msg = torch.bmm(x_j, weight).view(-1, self.dim)

        else:  # No regularization/Basis-decomposition ========================
            if x_j.dtype == torch.long:
                weight_index = edge_type * weight.size(1) + edge_index_j
                msg = weight.view(-1, self.dim)[weight_index]
            else:
                msg = torch.bmm(x_j.unsqueeze(-2), weight[edge_type]).squeeze(-2)

        if exists(edge_weight):
            msg = msg * edge_weight.view(-1, 1)

        return msg

    # TODO
    def aggregate(self, input, edge_type, index, dim_size=None):

        # Compute normalization in separation for each `edge_type`.
        if self.aggr == 'mean':
            norm = F.one_hot(edge_type, self.num_relations).to(torch.float)
            norm = scatter_add(norm, index, dim=0, dim_size=dim_size)[index]
            norm = torch.gather(norm, 1, edge_type.view(-1, 1))
            norm = 1. / norm.clamp_(1.)
            input = norm * input

        return scatter_add(input, index, dim=self.node_dim, dim_size=dim_size)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={self.num_relations}, num_bases={self.num_bases}, num_blocks={self.num_blocks}, add_self_loops={exists(self.linear_self)}, aggr={self.aggr}, project={exists(self.project)}, bias={exists(self.bias)})'


class RGATConv(gconv.MessagePassing):
    """_summary_: implementation of RGAT layer
    """

    def __init__(
        self,
        dim,
        num_relations,
        heads=1,
        edge_dim=None,
        add_self_loops=True,
        aggr="add",
        project=True,
        bias=True,
        dropout=0.0,
        **kwargs
    ):
        super(RGATConv, self).__init__(aggr=aggr, node_dim=0, **kwargs)
        self.dim = dim
        self.num_relations = num_relations
        self.heads = heads
        self.dropout = dropout

        assert heads > 0 and isinstance(heads, int) and dim % heads == 0, \
            "The number of heads must be a positive integer divisor of the output dimension."
        head_dim = dim // heads
        if project:
            self.project = nn.Linear(dim, dim, bias=bias)
        else:
            self.project = None

        self.att_src = nn.Parameter(torch.Tensor(num_relations, heads, head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(num_relations, heads, head_dim))

        if edge_dim is not None:
            self.linear_edge = nn.Linear(edge_dim, heads * head_dim, bias=False)
            self.att_edge = nn.Parameter(torch.Tensor(1, heads, head_dim))
        else:
            self.linear_edge = None
            self.register_parameter('att_edge', None)

        if add_self_loops:
            self.linear_self = nn.Linear(dim, dim, bias=False)
        else:
            self.linear_self = None

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
        if exists(self.linear_edge):
            nn.init.uniform_(self.linear_edge.weight, -stdv, stdv)
            nn.init.uniform_(self.att_edge, -stdv, stdv)
        if exists(self.linear_self):
            nn.init.uniform_(self.linear_self.weight, -stdv, stdv)
        nn.init.uniform_(self.att_src, -stdv, stdv)
        nn.init.uniform_(self.att_dst, -stdv, stdv)
        if exists(self.bias):
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, edge_index, edge_type=None, edge_weight=None, size=None):

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, torch.Tensor):
            batch_size = x.size(0)
            if exists(self.project):
                x_src = x_dst = self.project(x).view(batch_size, self.heads, -1)
            else:
                x_src = x_dst = x.view(batch_size, self.heads, -1)

        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            batch_size = x_src.size(0)
            if exists(self.project):
                x_src = self.project(x_src).view(batch_size, self.heads, -1)
                x_dst = self.project(x_dst).view(batch_size, self.heads, -1)

        # edge_updater_type: (alpha: OptPairTensor, edge_feat: OptTensor)
        alpha = self.edge_updater(edge_index, x=(x_src, x_dst), edge_type=edge_type, edge_feat=edge_weight)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        out = self.propagate(edge_index, x=(x_src, x_dst), alpha=alpha, edge_weight=edge_weight, size=size)
        out = out.view(batch_size, self.dim)

        if exists(self.linear_self):
            if isinstance(x, torch.Tensor):
                out += self.linear_self(x)
            else:
                out += self.linear_self(x[1])

        if exists(self.bias):
            out += self.bias

        return out

    def edge_update(self, x_j, x_i, edge_type, edge_feat, index, ptr, size_i):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha_j = (x_j * self.att_src[edge_type]).sum(dim=-1)
        alpha_i = (x_i * self.att_dst[edge_type]).sum(dim=-1)
        alpha = alpha_j + alpha_i

        # we use dot product to compute attention coefficients
        # alpha = ((x_j * self.att_src[edge_type]) * (x_i * self.att_dst[edge_type]) * math.sqrt(self.dim // self.heads)).sum(dim=-1)
        # alpha = ((x_j * self.att_src[edge_type]) * (x_i * self.att_dst[edge_type])).sum(dim=-1)

        if exists(edge_feat):
            if exists(self.linear_edge):
                edge_feat = self.linear_edge(edge_feat)
            edge_feat = edge_feat.view(edge_feat.size(0), self.heads, -1)
            alpha_edge = (edge_feat * self.att_edge).sum(dim=-1)
            alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, 0.2)
        alpha = sparse_softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j, alpha, edge_weight=None):
        return alpha.unsqueeze(-1) * x_j if edge_weight is None else edge_weight.view(-1, 1) * alpha.unsqueeze(-1) * x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(add_self_loops={exists(self.linear_self)}, aggr={self.aggr}, project={exists(self.project)}, bias={exists(self.bias)})'


class GNN(nn.Module):
    """_summary_: implementation of GNN as baseline
    """

    def __init__(self, opt, *args, **kwargs):
        super(GNN, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.max_len = opt.max_len
        self.dim = opt.hidden_dim
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.dropout_gnn = opt.dropout_gnn
        self.gnn_type = opt.gnn_type
        self.n_iter = opt.n_iter
        self.sample_num = opt.n_sample_model
        self.clip = opt.clip
        self.mtl = opt.mtl

        self.activate = get_activation(opt.activate)

        self.n_nodes = kwargs["n_nodes"]
        if isinstance(self.n_nodes, int):
            self.n_nodes = {SEQUENCE: self.n_nodes}

        # Aggregator
        if opt.gnn_type == "gcn":
            agg_wrapper = partial(GCNConv, self.dim, project=False, add_self_loops=False)
        elif opt.gnn_type == "rgcn":
            agg_wrapper = partial(RGCNConv, self.dim, num_relations=5, project=False, add_self_loops=False)
        elif opt.gnn_type == "sage":
            agg_wrapper = partial(SAGEConv, self.dim, project=False, add_self_loops=False)
        elif opt.gnn_type == "rsage":
            agg_wrapper = partial(RSAGEConv, self.dim, num_relations=5, project=False, add_self_loops=False)
        elif opt.gnn_type == "gin":
            agg_wrapper = partial(GINConv, self.dim, init_eps=-1.0, train_eps=True, mlp_layers=0, project=False)
        elif opt.gnn_type == "rgin":
            agg_wrapper = partial(
                RGINConv, self.dim, num_relations=5, init_eps=-1.0, train_eps=True, mlp_layers=0, project=False
            )
        elif opt.gnn_type == "gat":
            agg_wrapper = partial(GATConv, self.dim, heads=self.opt.n_head, project=False, add_self_loops=False)
        elif opt.gnn_type == "rgat":
            agg_wrapper = partial(
                RGATConv, self.dim, num_relations=5, heads=self.opt.n_head, project=False, add_self_loops=False
            )
        else:
            raise ValueError("Unknown aggregator type: {}".format(opt.gnn_type))
        self.local_aggs = nn.ModuleList()
        for i in range(self.n_iter):
            # local_aggs = nn.ModuleDict()
            # for key in self.n_nodes:
            #     local_aggs[key] = agg_wrapper()
            # self.local_aggs.append(local_aggs)
            self.local_aggs.append(agg_wrapper())
        self.act = get_activation(opt.activate)

        # Item representation
        self.embeddings = nn.ModuleDict()
        for key in self.n_nodes:
            self.embeddings[key] = nn.Embedding(self.n_nodes[key], self.dim, padding_idx=0)

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
            if name.startswith("local_aggs"):
                continue
            if name.endswith(".bias"):
                nn.init.constant_(param, 0)
            elif name.endswith(".weight"):
                nn.init.uniform_(param, -stdv, stdv)

    def forward(self, input, mask=None, items=None, adj=None, alias=None, shortcut=None, heteitems=None, heteadj=None):
        """_summary_: forward propagation of GNN.
        It uses various GNN to extract local context information.

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

        h_graphs = dict()
        for key in sorted_keys:
            h_graphs[key] = self.embeddings[key](items[key])
        h_graph = torch.stack([h_graphs[key] for key in sorted_keys]).mean(0)

        # local
        h = h_graph
        for i, local_agg in enumerate(self.local_aggs):
            h = local_agg(h, *adj[SEQUENCE])
            h = self.act(h)
            h = F.dropout(h, self.dropout_gnn, training=self.training)
        h_local = h.view(batch_size, seq_len, -1)
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)

        if not exists(alias):
            alias = dict()

        if self.mtl:
            output = {key: None for key in sorted_keys}
        else:
            output = {SEQUENCE: None}
        for key in output:
            if key in alias:
                output[key] = h_local.view((batch_size * seq_len), -1)[(alias[key] + batch_flat_offset).view(-1)].view(batch_size, seq_len, -1)
            else:
                output[key] = h_local

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
            select = self.predictors[key](hidden[key], mask=mask)
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