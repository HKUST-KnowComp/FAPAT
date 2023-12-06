import os
import numpy as np
import math
import torch
import logging
import sys
import socket
import signal
import subprocess
import torch.nn as nn
import torch.nn.functional as F
import datetime
from contextlib import ExitStack, contextmanager
from torch import einsum
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from einops_exts import repeat_many, rearrange_with_anon_dims, check_shape
from einops.layers.torch import Rearrange


def sig_handler(signum, frame):
    """_summary_: Signal handler for SIGUSR1 and SIGTERM for DDP

    :param signum: signal number
    :type signum: int
    :param frame: stack frame
    :type frame: frame
    """

    logger.warning("Signal handler called with signal " + str(signum))
    prod_id = int(os.environ["SLURM_PROCID"])
    logger.warning("Host: %s - Global rank: %i" % (socket.gethostname(), prod_id))
    if prod_id == 0:
        logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
        os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
    else:
        logger.warning("Not the main process, no need to requeue.")
    sys.exit(-1)


def term_handler(signum, frame):
    """_summary_: Signal handler for SIGTERM for DDP

    :param signum: signal number
    :type signum: int
    :param frame: stack frame
    :type frame: frame
    """

    logger.warning("Signal handler called with signal " + str(signum))
    logger.warning("Bypassing SIGTERM.")


def init_signal_handler():
    """_summary_: Initialize signal handler for SIGUSR1 and SIGTERM for DDP by SLURM for time limit / pre-emption
    """

    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)
    #logger.warning("Signal handler installed.")


def init_distributed_mode(params):
    """_summary_: Initialize distributed mode for single and multi-GPU / multi-node / SLURM jobs.
    """

    params.is_slurm_job = "SLURM_JOB_ID" in os.environ
    has_local_rank = hasattr(params, "local_rank")

    print("---" * 10)
    print(params.local_rank)
    # SLURM job
    if params.is_slurm_job and has_local_rank:

        assert params.local_rank == -1  # on the cluster, this is handled by SLURM

        SLURM_VARIABLES = [
            "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_JOB_NUM_NODES", "SLURM_NTASKS", "SLURM_TASKS_PER_NODE",
            "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_NODEID", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_TASK_PID"
        ]

        PREFIX = "%i - " % int(os.environ["SLURM_PROCID"])
        for name in SLURM_VARIABLES:
            value = os.environ.get(name, None)
            #print(PREFIX + "%s: %s" % (name, str(value)))

        # # job ID
        # params.job_id = os.environ["SLURM_JOB_ID"]

        # number of nodes / node ID
        params.n_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        params.node_id = int(os.environ["SLURM_NODEID"])

        # local rank on the current node / global rank
        params.local_rank = int(os.environ["SLURM_LOCALID"])
        params.global_rank = int(os.environ["SLURM_PROCID"])

        # number of processes / GPUs per node
        params.world_size = int(os.environ["SLURM_NTASKS"])
        params.n_gpu_per_node = params.world_size // params.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]])
        params.main_addr = hostnames.split()[0].decode("utf-8")
        assert 10001 <= params.main_port <= 20000 or params.world_size == 1
        #print(PREFIX + "Master address: %s" % params.master_addr)
        #print(PREFIX + "Master port   : %i" % params.master_port)

        # set environment variables for "env://"
        os.environ["MASTER_ADDR"] = params.main_addr
        os.environ["MASTER_PORT"] = str(params.main_port)
        os.environ["WORLD_SIZE"] = str(params.world_size)
        os.environ["RANK"] = str(params.global_rank)
        params.is_distributed = True

    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    elif has_local_rank and params.local_rank != -1:

        assert params.main_port == -1

        # read environment variables
        params.global_rank = int(os.environ["RANK"])
        params.world_size = int(os.environ["WORLD_SIZE"])
        # params.n_gpu_per_node = int(os.environ["NGPU"])
        params.n_gpu_per_node = int(torch.cuda.device_count())

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node
        params.is_distributed = True

    else:
        n_gpu = torch.cuda.device_count()
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = n_gpu
        params.n_gpu_per_node = n_gpu
        params.is_distributed = False

    # define whether this is the master process / if we are in distributed mode
    params.is_main = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1

    # summary
    PREFIX = "%i - " % params.global_rank

    # set GPU device
    if params.is_distributed:
        torch.cuda.set_device(params.local_rank)
        device = torch.device("cuda", params.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.device = device

    # initialize multi-GPU
    if params.is_distributed:

        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # "env://" will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        #print("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            init_method="env://",
            backend="nccl",
            timeout=datetime.timedelta(seconds=1000)
        )


# initialization


def init_seed(seed=None):
    """_summary_: Initialize random seed

    :param seed: random seed, if None, use random seed
    :type seed: int
    """

    if seed is None:
        seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


logger = logging.getLogger(__name__)


def init_logger(is_main=True, is_distributed=False, filename=None):
    """_summary_: Initialize logger

    :param is_main: whether this is the main process, default: True
    :type is_main: bool, optional
    :param is_distributed: whether we are in distributed mode, default: False
    :type is_distributed: bool, optional
    :param filename: log file name, default: None
    :type filename: str, optional
    :return: logger
    :rtype: logging.Logger
    """

    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    return logger


# optimization


class CosineWarmupRestartScheduler(LambdaLR):
    """_summary_: Cosine warmup restart scheduler
    """

    def __init__(
        self,
        optimizer,
        num_warmup_steps=3000,
        num_schedule_steps=1000000,
        num_cycles=3,
        min_percent=0.01,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_schedule_steps = num_schedule_steps
        self.num_cycles = num_cycles
        self.min_percent = min_percent
        super(CosineWarmupRestartScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        progress = float(current_step - self.num_warmup_steps) / \
            float(max(1, self.num_schedule_steps - self.num_warmup_steps))
        if progress >= 1.0:
            return self.min_percent
        return max(self.min_percent, 0.5 * (1.0 + math.cos(math.pi * ((float(self.num_cycles) * progress) % 1.0))))


class ConstantWarmupScheduler(LambdaLR):
    """_summary_: Constant warmup scheduler
    """

    def __init__(
        self,
        optimizer,
        num_warmup_steps=3000,
    ):
        self.num_warmup_steps = num_warmup_steps
        super(ConstantWarmupScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / max(1.0, float(self.num_warmup_steps))
        return 1.0


class LinearWarmupScheduler(LambdaLR):
    """_summary_: Linear warmup scheduler
    """

    def __init__(self, optimizer, num_warmup_steps=3000, num_schedule_steps=100000, min_percent=0.01):
        self.num_warmup_steps = num_warmup_steps
        self.num_schedule_steps = num_schedule_steps
        self.min_percent = min_percent
        super(LinearWarmupScheduler, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, current_step):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            self.min_percent,
            float(self.num_schedule_steps - current_step) / \
                float(max(1, self.num_schedule_steps - self.num_warmup_steps))
        )


# helper functions


class DummyActication(nn.Module):
    """_summary_: dummy activation function
    """

    def forward(self, x):
        return x


def get_activation(activation):
    """_summary_: map activation function name to activation function
    """

    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "tanh":
        return nn.Tanh()
    else:
        raise DummyActication()


def index(array, item):
    """_summary_: get index of item in array

    :param array: data array
    :type array: [np.ndarray, torch.Tensor]
    :param item: item to be searched
    :type item: any
    :return: index of item in array
    :rtype: int
    """

    tmp = (array == item).nonzero()
    return tmp[0][0] if len(tmp) > 0 and len(tmp[0]) > 0 else -1


def sample_neighbors(adj, num, target, n_sample):
    """_summary_: sample neighbors of target node

    :param adj: adjacency matrix
    :type adj: [np.ndarray, torch.Tensor]
    :param num: number of nodes
    :type num: int
    :param target: target node
    :type target: int
    :param n_sample: number of neighbors to be sampled
    :type n_sample: int
    :return: sampled neighbors
    :rtype: [np.ndarray, torch.Tensor]
    """

    item_sample, weight_sample = adj[target], num[target]
    if item_sample.shape[1] > n_sample:
        index = np.arange(item_sample.shape[1])
        np.random.shuffle(index)
        index = index[:n_sample]
        item_sample, weight_sample = item_sample[:, index], weight_sample[:, index]

    return item_sample, weight_sample


def trans_to_cuda(variable):
    """_summary_: transfer variable to cuda
    """

    if variable is None:
        return variable
    elif isinstance(variable, str):
        return variable
    if torch.cuda.is_available():
        if isinstance(variable, torch.Tensor):
            return variable.cuda()
        elif isinstance(variable, np.ndarray):
            return torch.from_numpy(variable).cuda()
        elif isinstance(variable, list):
            return [trans_to_cuda(v) for v in variable]
        elif isinstance(variable, tuple):
            return tuple(trans_to_cuda(v) for v in variable)
        elif isinstance(variable, set):
            return set(trans_to_cuda(v) for v in variable)
        elif isinstance(variable, dict):
            return {k: trans_to_cuda(v) for k, v in variable.items()}
        elif isinstance(variable, nn.Module):
            return variable.cuda()
        else:
            raise ValueError("Unknown variable type: {}".format(type(variable)))
    else:
        return variable


def trans_to_cpu(variable):
    if variable is None:
        return variable
    elif isinstance(variable, str):
        return variable
    if torch.cuda.is_available():
        if isinstance(variable, torch.Tensor):
            return variable.cpu()
        elif isinstance(variable, np.ndarray):
            return torch.from_numpy(variable)
        elif isinstance(variable, list):
            return [trans_to_cpu(v) for v in variable]
        elif isinstance(variable, tuple):
            return (trans_to_cpu(v) for v in variable)
        elif isinstance(variable, set):
            return set(trans_to_cpu(v) for v in variable)
        elif isinstance(variable, dict):
            return {k: trans_to_cpu(v) for k, v in variable.items()}
        elif isinstance(variable, nn.Module):
            return variable.cpu()
        else:
            raise ValueError("Unknown variable type: {}".format(type(variable)))
    else:
        return variable


def identity(t):
    return t


def exists(val):
    return val is not None


def unique(arr):
    return list({el: True for el in arr}.keys())


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, length=1):
    return val if isinstance(val, tuple) else ((val, ) * length)


def l2norm(t):
    return F.normalize(t, dim=-1)


def stable_softmax(t, dim=-1):
    t = t - t.amax(dim=dim, keepdim=True).detach()
    return F.softmax(t, dim=dim)


def topk_softmax(t, topk=2, dim=-1):
    if len(t.size(dim)) > topk:
        t_top_k_val, t_top_k_idx = torch.topk(t, k=topk, dim=dim, largest=True, sorted=False)
        t_top_k_val = stable_softmax(t_top_k_val, dim=dim)
        t = torch.zeros_like(t).scatter_(dim, t_top_k_idx, t_top_k_val)
    else:
        t = stable_softmax(t, dim=dim)
    return t


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_list(val):
    return val if isinstance(val, list) else [val]


def all_el_unique(arr):
    return len(set(arr)) == len(arr)


@contextmanager
def multi_context(*cms):
    with ExitStack() as stack:
        yield [stack.enter_context(cls) for cls in cms]


def count_intersect(x, y):
    # returns an array that shows how many times an element in x is contained in tensor y
    return np.sum(rearrange(x, "i -> i 1") == rearrange(y, "j -> 1 j"), axis=-1)


def fixed_pos_embedding(max_len, rotary_dims, device):
    inv_freq = 1.0 / (10000**(torch.arange(0, rotary_dims, 2, device=device) / rotary_dims))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(max_len, device=device), inv_freq)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), -1)
    return rearrange(x, "... d j -> ... (d j)")


def apply_rotary_pos_emb(x, sincos, offset=0):
    # TODO(jphang): doesn"t work for local
    sin, cos = sincos
    if x.dim() == 2:
        if sin.dim() == 2:
            sin = sin[offset:x.shape[-2], :].repeat(1, 2)
            cos = cos[offset:x.shape[-2], :].repeat(1, 2)
        else:
            raise ValueError("sin and cos must have 2 dimensions")
    elif x.dim() == 3:
        if sin.dim() == 2:
            sin = sin[None, offset:x.shape[-2], :].repeat(1, 1, 2)
            cos = cos[None, offset:x.shape[-2], :].repeat(1, 1, 2)
        elif sin.dim() == 3:
            sin = sin[:, offset:x.shape[-2]].repeat(1, 1, 2)
            cos = cos[:, offset:x.shape[-2]].repeat(1, 1, 2)
        else:
            raise ValueError("sin and cos must have 2 or 3 dimensions")
    elif x.dim() == 4:
        if sin.dim() == 2:
            sin = sin[None, None, offset:x.shape[-2], :].repeat(1, 1, 1, 2)
            cos = cos[None, None, offset:x.shape[-2], :].repeat(1, 1, 1, 2)
        elif sin.dim() == 3:
            sin = sin[:, None, offset:x.shape[-2]].repeat(1, 1, 1, 2)
            cos = cos[:, None, offset:x.shape[-2]].repeat(1, 1, 1, 2)
        else:
            raise ValueError("sin and cos must have 2 or 3 dimensions")
    else:
        raise ValueError("x.dim() must be 2, 3 or 4")
    return (x * cos) + (rotate_every_two(x) * sin)


def init_pos_emb(param):
    with torch.no_grad():
        l, d = param.shape
        device = param.device
        e = nn.init.normal_(param, 0.0, 1.0)
        e[0].fill_(0)
        sincos = fixed_pos_embedding(max_len=l, rotary_dims=d // 2, device=device)
        e_rot = apply_rotary_pos_emb(e[:, :d // 2], sincos)
        e_pass = e[:, d // 2:]
        e = torch.cat((e_rot, e_pass), dim=-1)
        param.copy_(e)
    return param


def select_patterns(input, patterns, input_threshold=1, max_patterns=1000, share=True):
    """_summary_: selects patterns based on Jaccard similarity over node sets.

    :param input: input tensor of shape (batch_size, num_nodes)
    :type input: torch.Tensor
    :param patterns: patterns tensor of shape (batch_size, num_patterns, num_nodes)
    :type patterns: torch.Tensor
    :param input_threshold: threshold for input tensor, defaults to 1
    :type input_threshold: int, optional
    :param max_patterns: maximum number of patterns to select, defaults to 1000
    :type max_patterns: int, optional
    :param share: whether to share patterns across batch, defaults to True
    :type share: bool, optional
    :return: selected patterns tensor of shape (num_patterns, num_nodes), indices of selected patterns for each example
    :rtype: torch.Tensor, torch.Tensor
    """

    if max_patterns == 0:
        return (torch.zeros((1, ), dtype=torch.long, device=input.device), None)
    with torch.no_grad():
        # input = input.to(patterns)
        batch_size = input.shape[0]
        input_len = (input != 0).float().sum(dim=-1)  # batch_size
        pattern_len = (patterns != 0).float().sum(dim=-1)  # n_patterns

        node_match = (input[:, :, None, None] == patterns[None, None, :, :])
        node_match = node_match.masked_fill(patterns[None, None, :, :] == 0, 0)
        node_match = node_match.max(3)[0].float()  # batch_size x seq_len x n_patterns

        session_match = node_match.sum(1)  # batch_size x n_patterns
        session_match = torch.where(session_match >= input_threshold, session_match, torch.zeros_like(session_match))
        jaccard = session_match / (input_len.unsqueeze(1) + pattern_len.unsqueeze(0) -
                                   session_match).clamp(min=1)  # batch_size x n_patterns

        sum_jaccard = jaccard.sum(0)  # n_patterns

        if share:
            # final_patterns = sum_jaccard.argsort(descending=True)[:max_patterns]
            if max_patterns > sum_jaccard.shape[0]:
                return torch.arange(sum_jaccard.shape[0], device=sum_jaccard.device), None
            else:
                # final_patterns = sum_jaccard.argsort(descending=True)[:max_patterns]
                # final_patterns = final_patterns.masked_fill(sum_jaccard[final_patterns] == 0, 0)
                values, final_patterns = torch.topk(sum_jaccard, k=max_patterns, dim=0, largest=True, sorted=False)
                final_patterns = final_patterns.masked_fill(values == 0, 0)

            return final_patterns, None
        else:
            if max_patterns > sum_jaccard.shape[0]:
                return torch.arange(sum_jaccard.shape[0], device=sum_jaccard.device), None
            else:
                # reg_jaccard = torch.where(jaccard > 0, jaccard, (sum_jaccard / batch_size).unsqueeze(0).expand_as(jaccard))
                reg_jaccard = jaccard
                # final_patterns = reg_jaccard.argsort(descending=True, dim=-1)[:, :max_patterns] # batch_size x max_patterns
                # final_patterns = final_patterns.masked_fill(reg_jaccard[torch.arange(batch_size).to(final_patterns).unsqueeze(-1), final_patterns] == 0, 0)
                values, final_patterns = torch.topk(reg_jaccard, k=max_patterns, dim=-1, largest=True, sorted=False)
                final_patterns = final_patterns.masked_fill(values == 0, 0)
                patterns = torch.unique(final_patterns, sorted=False, return_inverse=True)
                return patterns


def aggregate_graph(graph_rep, graph_mask, op="max"):
    """_summary_: aggregates graph representations

    :param graph_rep: graph representations of shape (batch_size, num_nodes, num_features)
    :type graph_rep: torch.Tensor
    :param graph_mask: graph mask of shape (batch_size, num_nodes)
    :type graph_mask: torch.Tensor
    :param op: aggregation operation, defaults to "max"
    :type op: str, optional
    :raises ValueError: if op is not in ["max", "mean"]
    :return: aggregated graph representations of shape (batch_size, num_features)
    :rtype: torch.Tensor
    """

    if op == "mean":
        if graph_mask is None:
            return graph_rep.mean(-2)
        else:
            return (graph_rep * graph_mask.unsqueeze(-1)).sum(-2) / graph_mask.sum(-1).unsqueeze(-1).clamp(min=1e-9)
    elif op == "max":
        if graph_mask is None:
            return graph_rep.max(-2)[0]
        else:
            return (graph_rep * graph_mask.unsqueeze(-1)).max(-2)[0]
    elif op == "sum":
        if graph_mask is None:
            return graph_rep.sum(-2)
        else:
            return (graph_rep * graph_mask.unsqueeze(-1)).sum(-2)
    else:
        raise ValueError("Unknown aggregation operation")


def to_dense_adj(edge_index, batch=None, edge_attr=None, max_num_nodes=None):
    """_summary_: sparse edge index to dense adjacency matrix
    :param edge_index: edge indices
    :type graph_rep: torch.LongTensor
    :param batch: batch indicator, default None
    :type batch: torch.LongTensor
    :param edge_attr: edge weights or multi-dimensional edge, default None
    :type edge_attr: torch.Tensor
    :param max_num_nodes: the size of the output node dimension, default None
    :type max_num_nodes: int
    :return: dense batched adjacency matrix
    :rtype: torch.Tensor
    """
    from torch_scatter import scatter_sum
    if batch is None:
        batch = edge_index.new_zeros(edge_index.max().item() + 1)

    batch_size = batch.max().item() + 1
    one = batch.new_ones(batch.size(0))
    num_nodes = scatter_sum(one, batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    idx0 = batch[edge_index[0]]
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = num_nodes.max().item()

    elif idx1.max() >= max_num_nodes or idx2.max() >= max_num_nodes:
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = torch.ones(idx0.numel(), device=edge_index.device)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size += list(edge_attr.size())[1:]
    adj = torch.zeros(size, dtype=edge_attr.dtype, device=edge_index.device)

    flattened_size = batch_size * max_num_nodes * max_num_nodes
    adj = adj.view([flattened_size] + list(adj.size())[3:])
    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    scatter_sum(edge_attr, idx, dim=0, out=adj)
    adj = adj.view(size)

    return adj
