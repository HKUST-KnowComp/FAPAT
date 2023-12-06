import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import *
from utils import *


class FAPAT(nn.Module):
    """_summary_: implementation of FAPAT proposed by ourselves
    details: local GNN + Transformer with memory + gate for attributes
    """

    def __init__(self, opt, *args, **kwargs):
        super().__init__()

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
        patterns = kwargs.get("patterns", dict())

        self.p_adjs = dict()
        self.p_items = dict()
        for key in self.n_nodes:
            p_adj, p_items = patterns.get(key, (None, None))
            if isinstance(p_adj, np.ndarray):
                p_adj = torch.from_numpy(p_adj)
                p_items = torch.from_numpy(p_items)
            elif isinstance(p_adj, torch.Tensor):
                p_adj = p_adj.clone()
                p_items = p_items.clone()
            elif isinstance(p_adj, (tuple, list)) and len(p_adj) == 2 and isinstance(p_adj[0], torch.Tensor):
                p_adj = to_dense_adj(p_adj[0], None, p_adj[1])[0]
                p_items = p_items.clone().view(p_adj.size(0), -1)
            else:
                p_adj = None
                p_items = None
            # pad
            if p_adj is not None:
                p_adj = torch.cat([torch.zeros((1, ) + tuple(p_adj.shape[1:]), dtype=torch.long), p_adj])
                p_items = torch.cat([torch.zeros((1, ) + tuple(p_items.shape[1:]), dtype=torch.long), p_items])
            else:
                p_adj = torch.zeros((1, 1, 1), dtype=torch.long)
                p_items = torch.zeros((1, 1), dtype=torch.long)
            self.p_adjs[key] = trans_to_cuda(p_adj)
            self.p_items[key] = trans_to_cuda(p_items)

        dim_head = opt.hidden_dim // opt.n_head
        # Memorzing Transformer
        self.mem_transformers = nn.ModuleList()
        for i in range(self.n_iter):
            mem_transformers = nn.ModuleDict()
            for key in self.n_nodes:
                if self.mem_type == KNN_FLAG:
                    mem_transformer = MemorizingTransformer(
                        dim=opt.hidden_dim,
                        depth=2,
                        dim_head=dim_head,
                        heads=opt.n_head,
                        buckets=opt.n_bucket,
                        bidirectional=False,
                        attn_dropout=opt.dropout_attn,
                        ff_mult=4,
                        ff_dropout=opt.dropout_ff,
                        knn_memory_layers=(0, ),
                        knn_max_memories=self.mem_size,
                        knn_num_retrieved_memories=opt.n_bucket,
                        knn_memories_directory=f'./.tmp/FAPAT_{key}_{i}.knn.memories',
                        xl_memory_layers=tuple(),
                        xl_max_memories=0,
                    )
                elif self.mem_type == XL_FLAG:
                    mem_transformer = MemorizingTransformer(
                        dim=opt.hidden_dim,
                        depth=2,
                        dim_head=dim_head,
                        heads=opt.n_head,
                        buckets=opt.n_bucket,
                        bidirectional=False,
                        attn_dropout=opt.dropout_attn,
                        ff_mult=4,
                        ff_dropout=opt.dropout_ff,
                        knn_memory_layers=tuple(),
                        knn_max_memories=0,
                        knn_num_retrieved_memories=opt.n_bucket,
                        knn_memories_directory=f'./.tmp/FAPAT_{key}_{i}.knn.memories',
                        xl_memory_layers=(0, ),
                        xl_max_memories=self.mem_size,
                    )
                else:
                    # default: no memory
                    mem_transformer = MemorizingTransformer(
                        dim=opt.hidden_dim,
                        depth=2,
                        dim_head=dim_head,
                        heads=opt.n_head,
                        buckets=opt.n_bucket,
                        bidirectional=False,
                        attn_dropout=opt.dropout_attn,
                        ff_mult=4,
                        ff_dropout=opt.dropout_ff,
                        knn_memory_layers=tuple(),
                        knn_max_memories=0,
                        knn_num_retrieved_memories=opt.n_bucket,
                        knn_memories_directory=f'./.tmp/FAPAT_{key}_{i}.knn.memories',
                        xl_memory_layers=tuple(),
                        xl_max_memories=0,
                    )
                mem_transformers[key] = mem_transformer
            self.mem_transformers.append(mem_transformers)

        # Aggregator
        self.pattern_aggs = nn.ModuleList()
        self.local_aggs = nn.ModuleList()
        for i in range(self.n_iter):
            pattern_aggs = nn.ModuleDict()
            local_aggs = nn.ModuleDict()
            for key in self.n_nodes:
                pattern_aggs[key] = LocalAggregator(self.dim, dropout=self.dropout_local)
                local_aggs[key] = LocalAggregator(self.dim, dropout=self.dropout_local)
            self.pattern_aggs.append(pattern_aggs)
            self.local_aggs.append(local_aggs)

        # Pattern
        self.pattern_tokvs = nn.ModuleList()
        for i in range(self.n_iter):
            pattern_tokv = nn.ModuleDict()
            for key in self.n_nodes:
                pattern_tokv[key] = nn.Linear(self.dim, dim_head * 2, bias=False)
            self.pattern_tokvs.append(pattern_tokv)

        # Gate
        self.gates = nn.ModuleList()
        self.noise = nn.ModuleList()
        for i in range(2 * self.n_iter):
            gate = nn.Linear(self.dim, len(self.n_nodes), bias=False)
            self.gates.append(gate)
            noise = nn.Linear(self.dim, len(self.n_nodes), bias=False)
            self.noise.append(noise)

        # Item representation & Position representation
        self.pos_embedding = nn.Embedding(self.max_len + 2, self.dim)
        self.embeddings = nn.ModuleDict()
        for key in self.n_nodes:
            self.embeddings[key] = nn.Embedding(self.n_nodes[key], self.dim, padding_idx=0)

        # Parameters
        self.layer_norm = nn.LayerNorm(self.dim)
        self.cls = nn.Parameter(torch.Tensor(1, self.dim))
        self.mask = nn.Parameter(torch.Tensor(1, self.dim))
        self.local_to_qs = nn.ModuleDict()
        self.local_to_ks = nn.ModuleDict()
        self.global_to_qs = nn.ModuleDict()
        self.global_to_ks = nn.ModuleDict()
        self.predictors = nn.ModuleDict()
        for key in self.n_nodes:
            self.local_to_qs[key] = nn.Linear(opt.hidden_dim, opt.hidden_dim, bias=False)
            self.global_to_qs[key] = nn.Linear(opt.hidden_dim, opt.hidden_dim, bias=False)
            self.local_to_ks[key] = nn.Linear(opt.hidden_dim, opt.hidden_dim // opt.n_head, bias=False)
            self.global_to_ks[key] = nn.Linear(opt.hidden_dim, opt.hidden_dim // opt.n_head, bias=False)
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
            if name.startswith("mem_transformer"):
                continue
            if name.endswith("bias"):
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -stdv, stdv)
        for modules in self.mem_transformers:
            for k, v in modules.items():
                v.reset_parameters()
        for gate in self.gates:
            nn.init.constant_(gate.weight, 0.0)
        for noise in self.noise:
            nn.init.constant_(noise.weight, 0.0)
        self.layer_norm.reset_parameters()

    def forward(self, input, mask=None, items=None, adj=None, alias=None, shortcut=None, heteitems=None, heteadj=None):
        """_summary_: forward propagation of MixMemSBRT.
        It first learns session graph representations, 
        distribute graph representations to sequences, 
        and then learns global context representations via Transformer and attribute memory.

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
        h_seq = torch.stack([h_seqs[key] for key in sorted_keys]).mean(0)

        patterns = dict()
        p_reps = dict()
        p_adjs = dict()
        p_items = dict()
        p_mask = dict()
        for key in sorted_keys:
            patterns[key] = select_patterns(
                items[key], self.p_items[key], input_threshold=1, max_patterns=self.mem_size, share=self.mem_share
            )
            unique_patterns, inverse_indices = patterns[key]
            if inverse_indices is None:
                p_num = unique_patterns.size(0)
            else:
                p_num = inverse_indices.size(1)
            p_adjs[key] = self.p_adjs[key][unique_patterns]
            p_items[key] = self.p_items[key][unique_patterns]
            p_mask[key] = (p_items[key] != 0).float()
            p_reps[key] = self.embeddings[key](p_items[key])

        p_mems = {key: [] for key in sorted_keys}
        for i in range(self.n_iter):
            for key in sorted_keys:
                p_rep = p_reps.get(key, None)
                if p_rep is not None:
                    p_reps[key] = self.pattern_aggs[i][key](p_rep, p_adjs[key], mask=p_mask[key])
                    unique_patterns, inverse_indices = patterns[key]
                    if inverse_indices is None:
                        p_num = unique_patterns.size(0)
                    else:
                        p_num = inverse_indices.size(1)
                    p_kv = self.pattern_tokvs[i][key](aggregate_graph(p_rep, p_mask[key], op="mean"))
                    if inverse_indices is None:
                        xl_mem = p_kv.view(1, p_num, 2, -1).repeat(batch_size, 1, 1, 1)  # batch_size x n_patterns x 2 x dim_head
                    else:
                        xl_mem = p_kv[inverse_indices].view(
                            batch_size, p_num, 2, -1
                        )  # batch_size x n_patterns x 2 x dim_head
                else:
                    xl_mem = None
                p_mems[key].append(xl_mem)

        gather_masks = dict()
        for key in sorted_keys:
            gather_masks[key] = (input[key][:, :, None] == items[key][:, None, :]).unsqueeze(-1)

        h_locals = h_graphs.copy()
        h_globals = h_seqs.copy()
        if not exists(alias):
            alias = dict()

        mask_local = None
        mask_global = torch.cat(
            [
                torch.ones(batch_size, 1, device=mask.device, dtype=torch.bool),
                mask,
                torch.ones(batch_size, 1, device=mask.device, dtype=torch.bool),
            ],
            dim=1
        )
        pos_idx = (torch.cumsum(mask_global.long(), dim=1) - 1).clamp_min(0)
        pos_emb = self.pos_embedding(pos_idx)
        gate_bias = torch.zeros((batch_size, 1, len(sorted_keys))).to(pos_emb)
        gate_bias[:, :, 0] = 1.0
        gate_bias = gate_bias

        for i in range(self.n_iter):
            if i == 0:
                cls_rep = self.cls.unsqueeze(0).expand(batch_size, -1, -1)
                mask_rep = self.mask.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                cls_rep = h_global[:, 0:1, :]
                mask_rep = h_global[:, -1:, :]

                for key in sorted_keys:
                    h_locals[key] = h_global[:, 1:-1].unsqueeze(2).repeat(1, 1, seq_len, 1).masked_fill(~gather_masks[key], 0.0)
                    h_locals[key] = h_locals[key].sum(1) / gather_masks[key].sum(1).clamp(min=1.0)

            h_local = torch.stack([h_locals[key] for key in sorted_keys]).mean(0)

            gate = self.gates[2 * i](h_local)  # + gate_bias
            noise = self.noise[2 * i](h_local)
            if len(sorted_keys) > self.topk:
                if self.training:
                    noise_stddev = (F.sigmoid(noise) + 1e-2)
                    gate = gate + (torch.randn_like(gate) * noise_stddev)
                gate_top_k_val, gate_top_k_idx = torch.topk(gate, k=self.topk, dim=-1, largest=True, sorted=False)
                gate_top_k_val = stable_softmax(gate_top_k_val, dim=-1)
                gate = torch.zeros_like(gate).scatter_(-1, gate_top_k_idx, gate_top_k_val)
            else:
                gate = stable_softmax(gate, dim=-1)

            for key in sorted_keys:
                h_locals[key] = self.local_aggs[i][key](h_locals[key], adj[key], mask=mask_local)
                h_locals[key] = F.dropout(h_locals[key], self.dropout_local, training=self.training)
            h_local = (
                torch.stack(
                    [
                        h_locals[key].view((batch_size * seq_len),
                                           -1)[(alias[key] + batch_flat_offset).view(-1)].view(batch_size, seq_len, -1)
                        for key in sorted_keys
                    ]
                ) * gate.permute(2, 0, 1).unsqueeze(-1)
            ).sum(0)

            h_global = torch.cat([cls_rep, h_local, mask_rep], dim=1)
            gate = self.gates[2 * i](h_global)  # + gate_bias
            noise = self.noise[2 * i](h_global)
            if len(sorted_keys) > self.topk:
                if self.training:
                    noise_stddev = (F.sigmoid(noise) + 1e-2)
                    gate = gate + (torch.randn_like(gate) * noise_stddev)
                gate_top_k_val, gate_top_k_idx = torch.topk(gate, k=self.topk, dim=-1, largest=True, sorted=False)
                gate_top_k_val = stable_softmax(gate_top_k_val, dim=-1)
                gate = torch.zeros_like(gate).scatter_(-1, gate_top_k_idx, gate_top_k_val)
            else:
                gate = stable_softmax(gate, dim=-1)

            h_global = h_global + pos_emb
            for key in sorted_keys:
                xl_mem = p_mems[key][i]
                h_globals[key] = self.mem_transformers[i][key](h_global, mask=mask_global, xl_memories=[xl_mem])[0]
                h_globals[key] = F.dropout(h_globals[key], self.dropout_global, training=self.training)
            h_global = (torch.stack([h_globals[key] for key in sorted_keys]) * gate.permute(2, 0, 1).unsqueeze(-1)).sum(0)

        if self.n_iter == 0:
            h_local = torch.stack([h_locals[key] for key in sorted_keys]).mean(0)
            h_global = torch.stack([h_globals[key] for key in sorted_keys]).mean(0)

        if self.mtl:
            output = {key: None for key in sorted_keys}
        else:
            output = {SEQUENCE: None}
        for key in output:
            # # local aggregation
            q = self.local_to_qs[key](h_local)
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.opt.n_head)
            k = self.local_to_ks[key](h_graphs[key])
            v = h_graphs[key]
            sim = einsum('b h i d, b j d -> b h i j', q, k)
            sim = sim.masked_fill(~mask[:, None, None, :], -torch.finfo(q.dtype).max)
            attn = stable_softmax(sim)
            o = einsum('b h i j, b j d -> b i d', attn, v)
            l = F.dropout(o, self.dropout_local, training=self.training)

            # global aggregation
            q = self.global_to_qs[key](h_global)
            q = rearrange(q, 'b n (h d) -> b h n d', h=self.opt.n_head)
            k = self.global_to_ks[key](h_seqs[key])
            v = h_seqs[key]
            sim = einsum('b h i d, b j d -> b h i j', q, k)
            sim = sim.masked_fill(~mask[:, None, None, :], -torch.finfo(q.dtype).max)
            attn = stable_softmax(sim)
            o = einsum('b h i j, b j d -> b i d', attn, v)
            g = F.dropout(o, self.dropout_global, training=self.training)

            output[key] = (l, g[:, 1:-1], g[:, -1])

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