import os
import math
import numpy as np
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from joblib import Parallel, delayed, cpu_count
from pathlib import Path
from config import *
from utils import *

# constants
FAISS_INDEX_GPU_ID = int(os.getenv("FAISS_INDEX_GPU_ID", 0))


class MLP(nn.Module):
    """_summary_: multi-layer perceptron
    """

    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hid_dim, out_dim)
        )

        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("relu"))


class Predictor(nn.Module):
    """_summary_: implementation of Predictor that is used in the paper
    """

    def __init__(self, dim, n_channel=1, max_len=100, dropout=0.0):
        super(Predictor, self).__init__()
        self.dim = dim
        self.n_channel = n_channel
        self.max_len = max_len
        self.dropout = dropout

        self.pos_embedding = nn.Embedding(self.max_len, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor((1 + self.n_channel) * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, channel_rep, global_vec=None, mask=None):
        if isinstance(channel_rep, torch.Tensor):
            if self.n_channel > 1:
                hidden = channel_rep  # cat
                channel_rep = channel_rep.chunk(self.n_channel, dim=-1)
            else:
                hidden = channel_rep
                channel_rep = (channel_rep, )
        elif isinstance(channel_rep, (tuple, list)):
            hidden = torch.cat(channel_rep, dim=-1)
        else:
            raise TypeError("channel_rep must be a torch.Tensor or a list of torch.Tensor")

        batch_size, seq_len = hidden.shape[:2]

        # reverse the pos_emb
        if exists(mask):
            pos_idx = (torch.cumsum(mask.long(), dim=1) - 1).clamp_min(0)
            pos_idx = pos_idx[:, -1].unsqueeze(-1) - pos_idx
            pos_emb = self.pos_embedding(pos_idx)
        else:
            pos_emb = self.pos_embedding.weight[-seq_len:].fllp(0)
            mask = torch.ones(batch_size, seq_len, device=hidden.device, dtype=torch.bool)

        mask = mask.float().unsqueeze(-1)
        if global_vec is None:
            global_vec = torch.sum(hidden * mask, -2) / torch.sum(mask.float(), 1)
            global_vec = global_vec.unsqueeze(-2).repeat(1, seq_len, 1)
            if self.n_channel > 1 and self.n_channel * self.dim == global_vec.size(-1):
                global_vec = global_vec.view(batch_size, seq_len, self.dim, self.n_channel).sum(-1)
        else:
            if global_vec.dim() == 2:
                global_vec = global_vec.unsqueeze(-2).repeat(1, seq_len, 1)
            if self.n_channel > 1 and self.n_channel * self.dim == global_vec.size(-1):
                global_vec = global_vec.view(batch_size, seq_len, self.dim, self.n_channel).sum(-1)

        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)

        nh = torch.sigmoid(self.glu1(nh) + self.glu2(global_vec))
        nh = F.dropout(nh, p=self.dropout, training=self.training)

        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask

        if self.n_channel == 1:
            select = torch.sum(beta * hidden, 1)
        else:
            select = torch.sum(beta * hidden, 1).view(batch_size, self.n_channel, self.dim).sum(1)
        return select


class PatternAggregator(nn.Module):
    """_summary_: relational graph attention network with sigmoid activation
    """

    def __init__(self, dim, dropout=0.):
        super(PatternAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.a_0, -stdv, stdv)
        nn.init.uniform_(self.a_1, -stdv, stdv)
        nn.init.uniform_(self.a_2, -stdv, stdv)
        nn.init.uniform_(self.a_3, -stdv, stdv)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, hidden, adj, mask=None):
        h = hidden
        batch_size, seq_len = h.shape[:2]

        a_input = (h.repeat(1, 1, seq_len).view(batch_size, seq_len * seq_len, self.dim) *
                   h.repeat(1, seq_len, 1)).view(batch_size, seq_len, seq_len, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = F.leaky_relu(e_0, 0.2).squeeze(-1).view(batch_size, seq_len, seq_len)
        e_1 = F.leaky_relu(e_1, 0.2).squeeze(-1).view(batch_size, seq_len, seq_len)
        e_2 = F.leaky_relu(e_2, 0.2).squeeze(-1).view(batch_size, seq_len, seq_len)
        e_3 = F.leaky_relu(e_3, 0.2).squeeze(-1).view(batch_size, seq_len, seq_len)

        if isinstance(adj, tuple):
            adj = adj[0]
        if adj.dim() == 2:
            adj = to_dense_adj(
                adj,
                torch.arange(batch_size, dtype=torch.long, device=adj.device).repeat_interleave(seq_len)
            )

        _inf = -(torch.finfo(e_0.dtype).max)
        alpha = e_0.masked_fill(adj != 1, _inf)
        alpha = torch.where(adj == 2, e_1, alpha)
        alpha = torch.where(adj == 3, e_2, alpha)
        alpha = torch.where(adj == 4, e_3, alpha)
        alpha = F.sigmoid(alpha)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        output = torch.matmul(alpha, h) + self.bias
        return output

    def extra_repr(self):
        return f"dim={self.dim}, dropout={self.dropout}"


class LocalAggregator(nn.Module):
    """_summary_: relational graph attention network with softmax activation
    """

    def __init__(self, dim, dropout=0.):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.a_0, -stdv, stdv)
        nn.init.uniform_(self.a_1, -stdv, stdv)
        nn.init.uniform_(self.a_2, -stdv, stdv)
        nn.init.uniform_(self.a_3, -stdv, stdv)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, hidden, adj, mask=None):
        if isinstance(hidden, (tuple, list)):
            src, dst = hidden
        elif isinstance(hidden, torch.Tensor):
            src, dst = hidden, hidden
        batch_size, seq_len = src.shape[:2]

        # a_input = (src.repeat(1, 1, seq_len).view(batch_size, seq_len * seq_len, self.dim) *
        #            dst.repeat(1, seq_len, 1)).view(batch_size, seq_len, seq_len, self.dim)
        a_input = src[:, :, None, :] * dst[:, None, :, :]

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = F.leaky_relu(e_0, 0.2).view(batch_size, seq_len, seq_len)
        e_1 = F.leaky_relu(e_1, 0.2).view(batch_size, seq_len, seq_len)
        e_2 = F.leaky_relu(e_2, 0.2).view(batch_size, seq_len, seq_len)
        e_3 = F.leaky_relu(e_3, 0.2).view(batch_size, seq_len, seq_len)

        if isinstance(adj, tuple):
            adj = adj[0]
        if adj.dim() == 2:
            adj = to_dense_adj(
                adj,
                torch.arange(batch_size, dtype=torch.long, device=adj.device).repeat_interleave(seq_len)
            )

        _inf = -(torch.finfo(e_0.dtype).max)
        alpha = e_0.masked_fill(adj != 1, _inf)
        alpha = torch.where(adj == 2, e_1, alpha)
        alpha = torch.where(adj == 3, e_2, alpha)
        alpha = torch.where(adj == 4, e_3, alpha)
        alpha = stable_softmax(alpha, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        output = torch.matmul(alpha, src) + self.bias
        return output

    def extra_repr(self):
        return f"dim={self.dim}, dropout={self.dropout}"


class GlobalAggregator(nn.Module):
    """_summary_: relational graph attention network with sampling and softmax activation
    details: local sampling and neighbor attention
    """

    def __init__(self, dim, dropout=0.0, act=torch.relu):
        super(GlobalAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.dim = dim

        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.w_1, -stdv, stdv)
        nn.init.uniform_(self.w_2, -stdv, stdv)
        nn.init.uniform_(self.w_3, -stdv, stdv)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, self_vectors, neighbor_vector, batch_size, mask, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            alpha = torch.matmul(
                torch.cat(
                    [
                        extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1) * neighbor_vector,
                        neighbor_weight.unsqueeze(-1)
                    ], -1
                ), self.w_1
            ).squeeze(-1)
            alpha = F.leaky_relu(alpha, 0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
            alpha = stable_softmax(alpha, -1).unsqueeze(-1)
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        # self_vectors = F.dropout(self_vectors, 0.5, training=self.training)
        output = torch.cat([self_vectors, neighbor_vector], -1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.w_3)
        output = output.view(batch_size, -1, self.dim)
        output = self.act(output)
        return output

    def extra_repr(self):
        return f"dim={self.dim}, act={self.act}, dropout={self.dropout}"


class KNN():
    """_summary_: implementation of KNN
    details: a wrapper around faiss IndexIVFFlat that takes care of expiring old keys automagically
    """

    def __init__(self, dim, max_num_entries, cap_num_entries=False, M=15, keep_stats=False):
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.index = index
        self.max_num_entries = max_num_entries
        self.cap_num_entries = cap_num_entries
        self.is_trained = False
        self.keep_stats = keep_stats

        self.reset()

    def __del__(self):
        if hasattr(self, "index"):
            del self.index

    def reset(self):
        self.ids = np.empty((0, ), dtype=np.int32)

        if self.keep_stats:
            self.hits = np.empty((0, ), dtype=np.int32)
            self.age_num_iterations = np.empty((0, ), dtype=np.int32)
            self.ages_since_last_hit = np.empty((0, ), dtype=np.int32)

        self.index.reset()
        self.is_trained = False

    def train(self, x):
        self.index.train(x)
        self.is_trained = True

    def add(self, x, ids):
        if not self.is_trained:
            self.train(x)

        self.ids = np.concatenate((ids, self.ids))

        if self.keep_stats:
            self.hits = np.concatenate((np.zeros_like(ids), self.hits))
            self.age_num_iterations = np.concatenate((np.zeros_like(ids), self.age_num_iterations))
            self.ages_since_last_hit = np.concatenate((np.zeros_like(ids), self.ages_since_last_hit))

        if self.cap_num_entries and len(self.ids) > self.max_num_entries:
            self.reset()

        return self.index.add(x)

    def search(self, x, topk, nprobe=8, return_distances=False, increment_hits=False, increment_age=True):
        if not self.is_trained:
            return np.full((x.shape[0], topk), -1)

        distances, indices = self.index.search(x, k=topk)

        if increment_hits and self.keep_stats:
            hits = count_intersect(self.ids, rearrange(indices, "... -> (...)"))
            self.hits += hits

            self.ages_since_last_hit += 1
            self.ages_since_last_hit *= (hits == 0)

        if increment_age and self.keep_stats:
            self.age_num_iterations += 1

        if return_distances:
            return indices, distances

        return indices


class KNNMemory():
    """_summary_: memory that store key / value memories for KNN
    details: automatically taking care of a collection of faiss indices (across batch dimension)
    """
    
    def __init__(self, dim, max_memories=16000, num_indices=1, memmap_filename=None, multiprocessing=True):
        self.dim = dim
        self.num_indices = num_indices
        self.scoped_indices = list(range(num_indices))

        self.max_memories = max_memories
        self.shape = (num_indices, max_memories, 2, dim)
        self.db_offsets = np.zeros(num_indices, dtype=np.int32)

        if memmap_filename is not None:
            self.db = np.memmap(memmap_filename, mode="w+", dtype=np.float32, shape=self.shape)
        else:
            self.db = np.zeros(shape=self.shape, dtype=np.float32)
        self.knns = [KNN(dim=dim, max_num_entries=max_memories, cap_num_entries=True) for _ in range(num_indices)]

        self.n_jobs = cpu_count() if multiprocessing else 1

        self.reset()

    def set_scoped_indices(self, indices):
        indices = list(indices)
        assert all_el_unique(indices), f"all scoped batch indices must be unique, received: {indices}"
        assert all(
            [0 <= i < self.num_indices for i in indices]
        ), f"each batch index must be between 0 and less than {self.num_indices}: received {indices}"
        self.scoped_indices = indices

    @contextmanager
    def at_batch_indices(self, indices):
        prev_indices = self.scoped_indices
        self.set_scoped_indices(indices)
        yield self
        self.set_scoped_indices(prev_indices)

    def clear(self, batch_indices=None):
        if not exists(batch_indices):
            batch_indices = list(range(self.num_indices))

        batch_indices = cast_list(batch_indices)

        for index in batch_indices:
            knn = self.knns[index]
            knn.reset()

        self.db_offsets[batch_indices] = 0

    def add(self, memories):
        check_shape(memories, "b n kv d", d=self.dim, kv=2, b=len(self.scoped_indices))

        memories = memories.detach().cpu().numpy()
        memories = memories[:, -self.max_memories:]
        num_memories = memories.shape[1]

        knn_insert_ids = np.arange(num_memories)

        keys = np.ascontiguousarray(memories[..., 0, :])
        knns = [self.knns[i] for i in self.scoped_indices]
        db_offsets = [self.db_offsets[i] for i in self.scoped_indices]

        # use joblib to insert new key / value memories into faiss index

        @delayed
        def knn_add(knn, key, db_offset):
            knn.add(key, ids=knn_insert_ids + db_offset)

        Parallel(n_jobs=self.n_jobs)(knn_add(*args) for args in zip(knns, keys, db_offsets))

        # add the new memories to the memmap "database"

        add_indices = (
            rearrange(np.arange(num_memories), "j -> 1 j") +
            rearrange(self.db_offsets[list(self.scoped_indices)], "i -> i 1")
        ) % self.max_memories
        self.db[rearrange(np.array(self.scoped_indices), "i -> i 1"), add_indices] = memories
        if isinstance(self.db, np.memmap):
            self.db.flush()
        self.db_offsets += num_memories

    def search(self, queries, topk, nprobe=8, increment_hits=True, increment_age=True):
        _, *prec_dims, _ = queries.shape
        check_shape(queries, "b ... d", d=self.dim, b=len(self.scoped_indices))
        queries = rearrange(queries, "b ... d -> b (...) d")

        device = queries.device
        queries = queries.detach().cpu().numpy()

        all_masks = []
        all_key_values = []

        knns = [self.knns[i] for i in self.scoped_indices]

        # parallelize faiss search

        @delayed
        def knn_search(knn, query):
            return knn.search(query, topk, nprobe, increment_hits=increment_hits, increment_age=increment_age)

        fetched_indices = Parallel(n_jobs=self.n_jobs)(knn_search(*args) for args in zip(knns, queries))

        # get all the memory key / values from memmap "database"
        # todo - remove for loop below

        for batch_index, indices in zip(self.scoped_indices, fetched_indices):
            mask = indices != -1
            db_indices = np.where(mask, indices, 0)

            all_masks.append(torch.from_numpy(mask))

            key_values = self.db[batch_index, db_indices % self.max_memories]
            all_key_values.append(torch.from_numpy(key_values))

        all_masks = torch.stack(all_masks)
        all_key_values = torch.stack(all_key_values)
        all_key_values = all_key_values.masked_fill(~rearrange(all_masks, "... -> ... 1 1"), 0.0)

        all_key_values = rearrange_with_anon_dims(all_key_values, "b (...p) ... -> b ...p ...", p=prec_dims)
        all_masks = rearrange_with_anon_dims(all_masks, "b (...p) ... -> b ...p ...", p=prec_dims)

        return all_key_values.to(device), all_masks.to(device)

    def __del__(self):
        if hasattr(self, "knns"):
            for knn in self.knns:
                del knn
        del self.db

    def reset(self):
        for knn in self.knns:
            knn.reset()


class KNNMemoryList(list):
    """_summary_: list with some extra methods for collections of KNN memories
    """

    def cleanup(self):
        for memory in self:
            del memory

    @classmethod
    def create_memories(self, *, batch_size, num_memory_layers, memories_directory=None):
        if memories_directory is not None:
            memories_path = Path(memories_directory)
            memories_path.mkdir(exist_ok=True, parents=True)

        def inner(*args, **kwargs):
            return self(
                [
                    KNNMemory(
                        *args,
                        num_indices=batch_size,
                        memmap_filename=str(memories_path / f"knn.memory.layer.{ind + 1}.memmap")
                        if memories_directory is not None else None,
                        **kwargs
                    ) for ind in range(num_memory_layers)
                ]
            )

        return inner

    @contextmanager
    def at_batch_indices(self, indices):
        knn_batch_indices_contexts = [memory.at_batch_indices(indices) for memory in self]
        with multi_context(*knn_batch_indices_contexts):
            yield

    def clear_memory(self, batch_indices=None, memory_indices=None):
        memory_indices = default(memory_indices, tuple(range(len(self))))

        for memory_index in memory_indices:
            memory = self[memory_index]
            memory.clear(batch_indices)


class PreNormResidual(nn.Module):
    """_summary_: pre-normalization residual block
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

        self.reset_parameters()

    def forward(self, x, **kwargs):
        out = self.fn(self.norm(x), **kwargs)

        if not isinstance(out, tuple):
            return out + x

        head, *tail = out
        return (head + x, *tail)

    def reset_parameters(self):
        self.norm.reset_parameters()


class T5RelativePositionBias(nn.Module):
    """_summary_: T5 relative positional bias (bucket bias)
    """

    def __init__(self, scale, bidirectional=True, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

        self.reset_parameters()

    def reset_parameters(self):
        # init_pos_emb(self.relative_attention_bias.weight)
        nn.init.normal_(self.relative_attention_bias.weight, mean=0.0, std=1.0)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = n.clamp_min(0)
            # n = torch.maximum(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        # val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        val_if_large = torch.clamp_max(val_if_large, num_buckets - 1)
        ret += torch.where(is_small, n, val_if_large)

        return ret

    def forward(self, i, j, *, device):
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos, bidirectional=self.bidirectional, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, "i j h -> () h i j")
        return bias * self.scale


class FeedForward(nn.Module):
    """_summary_: feed forward block
    """

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * mult, dim))

        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("relu"))


class Attention(nn.Module):
    """_summary_: attention block
    """

    def __init__(
        self,
        *,
        dim,
        heads=8,
        dim_head=64,
        bidirectional=True,
        dropout=0.,
        xl_max_memories=0.,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = heads * dim_head
        self.bidirectional = bidirectional
        self.xl_max_memories = xl_max_memories

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("relu"))

    def forward(self, x, *, xl_memory=None, mask=None, rel_pos_bias=None):
        b, n, h, device = *x.shape[:2], self.heads, x.device
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))
        _inf = -torch.finfo(q.dtype).max

        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        q = q * self.scale

        if exists(xl_memory):
            # xl_memory: b, m, 2, d
            k_xl_mem, v_xl_mem = xl_memory.unbind(dim=-2)
            k = torch.cat((k_xl_mem, k), dim=-2)
            v = torch.cat((v_xl_mem, v), dim=-2)

        # rope
        # rope_rotary_dim = q.shape[-1] // 2
        # q_sincos = fixed_pos_embedding(max_len=q.shape[-2], rotary_dims=rope_rotary_dim, device=device)
        # q_rot = apply_rotary_pos_emb(q[:, :, :, :rope_rotary_dim], q_sincos)
        # q_pass = q[:, :, :, rope_rotary_dim:]
        # q = torch.cat((q_rot, q_pass), dim=-1)
        # k_sincos = fixed_pos_embedding(max_len=k.shape[-2], rotary_dims=rope_rotary_dim, device=device)
        # k_rot = apply_rotary_pos_emb(k[:, :, :rope_rotary_dim], k_sincos)
        # k_pass = k[:, :, rope_rotary_dim:]
        # k = torch.cat((k_rot, k_pass), dim=-1)

        sim = einsum("b h i d, b j d -> b h i j", q, k)
        i, j = sim.shape[-2:]

        if exists(rel_pos_bias):
            sim = rel_pos_bias[..., -i:, -j:] + sim

        if not self.bidirectional:
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, _inf)

        if exists(mask):
            if mask.dim() == 2:
                mask = mask[:, None, None, :]  # b, 1, 1, j
                mask = torch.cat([torch.ones((b, 1, 1, j - i), device=device, dtype=torch.bool), mask], dim=-1)
                sim = sim.masked_fill(~mask, _inf)
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]  # b, 1, i, j
                mask = torch.cat([torch.ones((b, 1, i, j - i), device=device, dtype=torch.bool), mask], dim=-1)
                sim = sim.masked_fill(~mask, _inf)
            else:
                raise ValueError(f"invalid mask shape: {mask.shape}")

        attn = stable_softmax(sim)
        attn = self.dropout(attn)

        out = einsum("b h i j, b j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        # new xl memories

        new_kv_memories = torch.stack((k, v), dim=-2).detach()

        if self.xl_max_memories > 0:
            new_xl_kv_memories = new_kv_memories[:, -self.xl_max_memories:]
        else:
            new_xl_kv_memories = None

        return self.to_out(out), new_xl_kv_memories


class KNNAttention(nn.Module):
    """_summary_: approximate nearest neighbor attention
    """

    def __init__(
        self,
        *,
        dim=512,
        heads=8,
        dim_head=64,
        bidirectional=True,
        dropout=0.,
        num_retrieved_memories=32,
        xl_max_memories=0.,
        attn_scale_init=20,
    ):
        super().__init__()
        self.heads = heads
        self.scale = nn.Parameter(torch.ones(heads, 1, 1) * math.log(attn_scale_init))

        inner_dim = heads * dim_head
        self.bidirectional = bidirectional
        self.xl_max_memories = xl_max_memories

        self.num_retrieved_memories = num_retrieved_memories

        self.dropout = nn.Dropout(dropout)
        self.knn_mem_dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("relu"))

    def forward(self, x, *, knn_memory=None, xl_memory=None, mask=None, add_knn_memory=True, rel_pos_bias=None):
        b, n, h, device = *x.shape[:2], self.heads, x.device
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))
        _inf = -torch.finfo(q.dtype).max

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # in paper, they showed normalizing of keys led to more stable training
        # we"ll just go with full cosine sim attention https://arxiv.org/abs/2010.04245

        q, k = map(l2norm, (q, k))

        # handle xl memory

        if exists(xl_memory):
            k_xl_mem, v_xl_mem = xl_memory.unbind(dim=-2)
            k = torch.cat((k_xl_mem, k), dim=-2)
            v = torch.cat((v_xl_mem, v), dim=-2)

        # calculate local attention

        scale = self.scale.exp()

        sim = einsum("b h i d, b j d -> b h i j", q, k) * scale
        i, j = sim.shape[-2:]

        if exists(rel_pos_bias):
            sim = rel_pos_bias[..., -i:, -j:] + sim

        if not self.bidirectional:
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, _inf)

        if exists(mask):
            if mask.dim() == 2:
                mask = mask[:, None, None, :]  # b, 1, 1, j
                mask = torch.cat([torch.ones((b, 1, 1, j - i), device=device, dtype=torch.bool), mask], dim=-1)
                sim = sim.masked_fill(~mask, _inf)
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]  # b, 1, i, j
                mask = torch.cat([torch.ones((b, 1, i, j - i), device=device, dtype=torch.bool), mask], dim=-1)
                sim = sim.masked_fill(~mask, _inf)
            else:
                raise ValueError(f"invalid mask shape: {mask.shape}")

        # calculate knn attention over memory, if index is passed in
        mem_kv, mem_mask = knn_memory.search(q, self.num_retrieved_memories)
        mem_k, mem_v = mem_kv.unbind(dim=-2)

        sim_mem = einsum("b h i d, b h i j d -> b h i j", q, mem_k) * scale
        sim_mem = sim_mem.masked_fill(~mem_mask, _inf)

        # calculate new XL memories, as well as memories to be discarded

        new_kv_memories = torch.stack((k, v), dim=-2).detach()

        if self.xl_max_memories > 0:
            new_kv_memories_discarded, new_xl_kv_memories = new_kv_memories[:, :-self.xl_max_memories
                                                                           ], new_kv_memories[:, -self.xl_max_memories:]
        else:
            new_kv_memories_discarded, new_xl_kv_memories = new_kv_memories, None

        # add memories to be discarded into KNN memory

        if add_knn_memory and new_kv_memories_discarded.numel() > 0:
            knn_memory.add(new_kv_memories_discarded)

        # combining local and distant
        sim = torch.cat((sim_mem, sim), dim=-1)
        i, j = sim.shape[-2:]

        # attention
        attn = stable_softmax(sim)
        attn = self.dropout(attn)

        local_attn, mem_attn = attn[..., self.num_retrieved_memories:], attn[..., :self.num_retrieved_memories]
        local_out = einsum("b h i j, b j d -> b h i d", local_attn, v)
        mem_out = einsum("b h i j, b h i j d -> b h i d", mem_attn, mem_v)

        out = local_out + mem_out

        # combine heads and project out

        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out), new_xl_kv_memories


class Transformer(nn.Module):
    """_summary_: Transformer without memory
    """

    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        buckets=32,
        bidirectional=False,
        attn_dropout=0.,
        ff_mult=4,
        ff_dropout=0.,
        pad_id=0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.dim_head = dim_head

        block_wrapper = partial(PreNormResidual, dim)

        # relative positional bias

        self.rel_pos_bias = T5RelativePositionBias(
            scale=dim_head**0.5,
            bidirectional=bidirectional,
            num_buckets=buckets,
            max_distance=buckets * int(math.log(buckets) / math.log(4) + 0.5),
            heads=heads
        )

        # layers

        self.layers = nn.ModuleList()
        for idx in range(depth):
            layer_num = idx + 1
            attn = Attention(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                bidirectional=bidirectional,
                dropout=attn_dropout,
            )
            ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

            self.layers.append(nn.ModuleList([
                block_wrapper(attn),
                block_wrapper(ff),
            ]))

    def reset_parameters(self):
        pass

    def forward(self, x, mask=None):
        batch_size, seq_len, *_, device = *x.shape, x.device

        # positional bias
        rel_pos_bias = self.rel_pos_bias(seq_len, seq_len, device=device)

        # go through all layers
        for ind, (attn, ff) in enumerate(self.layers):

            # attention
            x, xl_mem = attn(x, rel_pos_bias=rel_pos_bias, mask=mask)

            # feedforward
            x = ff(x)

        return x


class MemorizingTransformer(nn.Module):
    """_summary_: Transformer with memory
    """

    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        buckets=32,
        bidirectional=False,
        knn_attn_heads=None,
        attn_dropout=0.,
        ff_mult=4,
        ff_dropout=0.,
        knn_memory_layers=None,
        knn_max_memories=250000,
        knn_num_retrieved_memories=32,
        knn_memory_multiprocessing=False,
        clear_memories_on_sos_token_id=None,
        clear_memories_on_eos_token_id=None,
        knn_memories_directory=None,
        shift_knn_memories_down=0.,
        xl_memory_layers=None,
        xl_max_memories=0,
        shift_xl_memories_down=0.,
        pad_id=0,
    ):
        super().__init__()
        self.pad_id = pad_id

        block_wrapper = partial(PreNormResidual, dim)
        # valid_layers = set(range(1, depth + 1))

        # default KNN attention layer to midpoint of transformer
        knn_memory_layers = default(knn_memory_layers, (depth // 2, ))
        knn_memory_layers = cast_tuple(knn_memory_layers)
        # knn_memory_layers = tuple(filter(lambda i: i in valid_layers, knn_memory_layers))

        self.dim_head = dim_head

        knn_attn_heads = default(knn_attn_heads, heads)

        # xl memory hyperparameter
        if xl_max_memories > 0:
            xl_memory_layers = default(xl_memory_layers, tuple(range(1, depth + 1)))
            # xl_memory_layers = tuple(filter(lambda i: i in valid_layers, xl_memory_layers))
            self.xl_memory_layers = unique(xl_memory_layers)
            self.num_xl_memory_layers = len(self.xl_memory_layers)
        else:
            self.xl_memory_layers = tuple()
            self.num_xl_memory_layers = 0

        # knn memory hyperparameters

        self.knn_max_memories = knn_max_memories
        self.knn_memories_directory = knn_memories_directory
        self.knn_memory_layers = unique(knn_memory_layers)
        self.num_memory_layers = len(knn_memory_layers)

        self.clear_memories_on_sos_token_id = clear_memories_on_sos_token_id
        self.clear_memories_on_eos_token_id = clear_memories_on_eos_token_id

        # relative positional bias

        self.rel_pos_bias = T5RelativePositionBias(
            scale=dim_head**0.5,
            bidirectional=bidirectional,
            num_buckets=buckets,
            max_distance=buckets * int(math.log(buckets) / math.log(4) + 0.5),
            heads=heads
        )
        self.knn_rel_pos_bias = T5RelativePositionBias(
            scale=dim_head**0.5,
            bidirectional=bidirectional,
            num_buckets=buckets,
            max_distance=buckets * int(math.log(buckets) / math.log(4) + 0.5),
            heads=heads
        )

        # layers

        self.layers = nn.ModuleList()
        for idx in range(depth):
            layer_num = idx + 1

            use_xl_memories = layer_num in self.xl_memory_layers
            use_knn_attention = layer_num in knn_memory_layers
            xl_max_memories_layer = 0 if not use_xl_memories else xl_max_memories

            if use_knn_attention:
                attn = KNNAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=knn_attn_heads,
                    bidirectional=bidirectional,
                    dropout=attn_dropout,
                    num_retrieved_memories=knn_num_retrieved_memories,
                    xl_max_memories=xl_max_memories_layer
                )
            else:
                attn = Attention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    bidirectional=bidirectional,
                    dropout=attn_dropout,
                    xl_max_memories=xl_max_memories_layer
                )
            ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

            self.layers.append(nn.ModuleList([
                block_wrapper(attn),
                block_wrapper(ff),
            ]))

        # memory layer shifting
        # from a little known paper https://arxiv.org/abs/2012.15688

        self.shift_knn_memories_down = shift_knn_memories_down
        self.shift_xl_memories_down = shift_xl_memories_down

        # knn memories init

        self.knn_mem_kwargs = dict(
            dim=self.dim_head, max_memories=self.knn_max_memories, multiprocessing=knn_memory_multiprocessing
        )

    def reset_parameters(self):
        pass

    def create_knn_memories(self, batch_size=1):
        return KNNMemoryList.create_memories(
            batch_size=batch_size,
            num_memory_layers=self.num_memory_layers,
            memories_directory=self.knn_memories_directory,
        )(**self.knn_mem_kwargs)

    @contextmanager
    def knn_memories_context(self, **kwargs):
        if self.knn_memories_directory is not None:
            knn_dir = Path(self.knn_memories_directory)
            knn_dir.mkdir(exist_ok=True, parents=True)
            lock = FileLock(str(knn_dir / "mutex"))

            with lock:
                knn_memories = self.create_knn_memories(**kwargs)
                yield knn_memories
                knn_memories.cleanup()
        else:
            knn_memories = self.create_knn_memories(**kwargs)
            yield knn_memories
            knn_memories.cleanup()

    def clear_memory(self, x, token_id, knn_memories):
        """ clears the KNN memories based on if the batch row contains the specified token id """
        """ for auto-clearing KNN memories based on start and end of strings """

        clear_memory = (x == token_id).any(dim=-1)
        batch_indices, _ = clear_memory.nonzero(as_tuple=True)
        batch_indices_to_clear = batch_indices.tolist()

        if len(batch_indices_to_clear) == 0:
            return

        knn_memories.clear_memory(batch_indices_to_clear)

    def create_xl_memories(self, batch_size=1):
        xl_memories = [None] * self.num_xl_memory_layers
        for ind in range(self.num_xl_memory_layers):
            xl_memories[ind] = torch.zeros((batch_size, 0, 2, self.dim_head), dtype=torch.float32)
        return xl_memories

    def forward(self, x, knn_memories=None, xl_memories=None, mask=None, labels=None, add_knn_memory=True):
        batch_size, seq_len, *_, device = *x.shape, x.device

        # validate KNN memories to have enough indices for batch size

        knn_memories = default(knn_memories, (None, ) * len(self.knn_memory_layers))
        assert all(
            [memory is None or memory.num_indices == batch_size or memory.num_indices == 1 for memory in knn_memories]
        ), f"you passed in an input with batch size {batch_size} but your memories were not instantiated with that number of KNN indices"

        # if KNN memories are passed in, and researcher wants memories auto-cleared on <sos> token detection
        # do the appropriate logic

        if exists(self.clear_memories_on_sos_token_id):
            self.clear_memory(x, self.clear_memories_on_sos_token_id, knn_memories)

        # handle XL memories

        xl_memories = default(xl_memories, (None, ) * self.num_xl_memory_layers)
        assert len(xl_memories) == self.num_xl_memory_layers
        has_xl_memories = len(xl_memories) > 0

        # shifting memories a number of layers down, little known technique shown to enhance memories from Ernie-Doc paper

        if len(knn_memories) > 0 and self.shift_knn_memories_down > 0:
            knn_memories = [*knn_memories[self.shift_knn_memories_down:], *knn_memories[:self.shift_knn_memories_down]]

        if len(xl_memories) > 0 and self.shift_xl_memories_down > 0:
            xl_memories = [*xl_memories[self.shift_xl_memories_down:], *xl_memories[:self.shift_xl_memories_down]]

        # iterate through the memories in order of the ascending layers that contain KNNAttention

        xl_memories_iter = iter(xl_memories)
        knn_memories_iter = iter(knn_memories)

        # positional bias

        max_context_len = max([seq_len, *map(lambda t: (t.shape[-3] if exists(t) else 0) + seq_len, xl_memories)])

        rel_pos_bias = self.rel_pos_bias(seq_len, max_context_len, device=device)
        knn_rel_pos_bias = self.knn_rel_pos_bias(seq_len, max_context_len, device=device)

        # keep track of new xl memories

        new_xl_memories = [] if has_xl_memories else None

        # go through all layers

        for ind, (attn, ff) in enumerate(self.layers):
            layer_num = ind + 1

            is_memorizing_layer = layer_num in self.knn_memory_layers
            is_xl_memory_layer = layer_num in self.xl_memory_layers

            attn_kwargs = dict(rel_pos_bias=rel_pos_bias if not is_memorizing_layer else knn_rel_pos_bias, mask=mask)

            if is_memorizing_layer:
                attn_kwargs = {**attn_kwargs, "knn_memory": next(knn_memories_iter), "add_knn_memory": add_knn_memory}

            if is_xl_memory_layer:
                attn_kwargs = {**attn_kwargs, "xl_memory": next(xl_memories_iter)}

            # attention

            x, xl_mem = attn(x, **attn_kwargs)

            # add new XL memories if needed

            if exists(xl_mem):
                new_xl_memories.append(xl_mem)

            # feedforward

            x = ff(x)

        # auto-clear KNN memories on end of string token
        if exists(self.clear_memories_on_eos_token_id):
            self.clear_memory(x, self.clear_memories_on_eos_token_id, knn_memories)

        return x, new_xl_memories