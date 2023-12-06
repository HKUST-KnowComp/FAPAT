import torch
import os
import numpy as np
import scipy.sparse as sp
import math
import igraph as ig
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from itertools import chain
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from config import *
from utils import default, exists


# data
def split_validation(train_set, valid_portion, seed=2022):
    """_summary_: split the training set into training and validation set

    :param train_set: training set
    :type train_set: dict
    :param valid_portion: portion of validation set
    :type valid_portion: float
    :param seed: random seed, defaults to 2022
    :type seed: int, optional
    :return: training set and validation set
    :rtype: tuple
    """

    indices = np.arange(len(train_set["ids"]))
    n_samples = len(indices)
    np.random.seed(seed)
    np.random.shuffle(indices)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    train_indices = indices[:n_train]
    valid_indices = indices[n_train:]

    train_part = dict()
    for k, v in train_set.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            train_part[k] = v[train_indices]
        elif isinstance(v, (list, tuple)):
            train_part[k] = [v[idx] for idx in train_indices]
        elif isinstance(v, dict):
            _tmp = dict()
            for kk, vv in v.items():
                if isinstance(vv, (np.ndarray, torch.Tensor)):
                    _tmp[kk] = vv[train_indices]
                elif isinstance(vv, (list, tuple)):
                    _tmp[kk] = [vv[idx] for idx in train_indices]
                else:
                    print("Unknown type: {}".format(type(vv)))
                    _tmp[kk] = vv
            train_part[k] = _tmp
        else:
            print("Unknown type: {}".format(type(v)))
            train_part[k] = v

    valid_part = dict()
    for k, v in train_set.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            valid_part[k] = v[valid_indices]
        elif isinstance(v, (list, tuple)):
            valid_part[k] = [v[idx] for idx in valid_indices]
        elif isinstance(v, dict):
            _tmp = dict()
            for kk, vv in v.items():
                if isinstance(vv, (np.ndarray, torch.Tensor)):
                    _tmp[kk] = vv[valid_indices]
                elif isinstance(vv, (list, tuple)):
                    _tmp[kk] = [vv[idx] for idx in valid_indices]
                else:
                    print("Unknown type: {}".format(type(vv)))
                    _tmp[kk] = vv
            valid_part[k] = _tmp
        else:
            print("Unknown type: {}".format(type(v)))
            valid_part[k] = v

    return train_part, valid_part


def handle_adj(adj_dict, num_dict, n_entity, sample_num=None):
    """_summary_: handle the adjacency list and number list and convert them into numpy array

    :param adj_dict: adjacency list
    :type adj_dict: [list, tuple, np.ndarray, torch.Tensor, dict]
    :param num_dict: weight list
    :type num_dict: [list, tuple, np.ndarray, torch.Tensor, dict]
    :param n_entity: number of entities
    :type n_entity: int
    :param sample_num: number of sampled neighbors, defaults to None
    :type sample_num: int, optional
    :raises ValueError: if the length of adjacency list and weight list are in unknown type
    :return: adjacency matrix and weight matrix
    :rtype: tuple
    """

    if sample_num is None:
        if isinstance(adj_dict, (list, tuple, np.ndarray, torch.Tensor)):
            sample_num = max([len(x) for x in adj_dict])
        elif isinstance(adj_dict, dict):
            sample_num = max([len(x) for x in adj_dict.values()])
        else:
            raise ValueError("Unknown type: {}".format(type(adj_dict)))
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    num_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    if isinstance(adj_dict, (list, tuple, np.ndarray, torch.Tensor)):
        for entity in range(1, n_entity):
            if entity == len(adj_dict):
                break
            neighbor = list(adj_dict[entity])
            neighbor_weight = list(num_dict[entity])
            n_neighbor = len(neighbor)
            if n_neighbor == 0:
                continue
            if n_neighbor >= sample_num:
                sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
            else:
                sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
            adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
            num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])
    elif isinstance(adj_dict, dict):
        for entity in range(1, n_entity):
            if entity not in adj_dict:
                continue
            neighbor = list(adj_dict[entity])
            neighbor_weight = list(num_dict[entity])
            n_neighbor = len(neighbor)
            if n_neighbor == 0:
                continue
            if n_neighbor >= sample_num:
                sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
            else:
                sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
            adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
            num_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])
    else:
        raise ValueError("Unknown type: {}".format(type(adj_dict)))

    return adj_entity, num_entity


def merge_adjs(adj_dicts, num_dicts, n_entity):
    """_summary_: merge the adjacency list and weight list from different datasets

    :param adj_dicts: list of adjacencies
    :type adj_dicts: list
    :param num_dicts: list of weights
    :type num_dicts: list
    :param n_entity: number of entities
    :type n_entity: int
    :return: merged adjacency list and weight list
    :rtype: tuple
    """

    assert len(adj_dicts) > 0 and len(num_dicts) > 0 and len(adj_dicts) == len(num_dicts)
    weights = [dict() for _ in range(n_entity)]
    for adj_dict, num_dict in zip(adj_dicts, num_dicts):
        for idx in range(len(adj_dict)):
            for nei, wei in zip(adj_dict[idx], num_dict[idx]):
                if nei in weights[idx]:
                    weights[idx][nei] += wei
                else:
                    weights[idx][nei] = wei
    out_adj = []
    out_num = []
    for idx in range(len(weights)):
        weight = [v for v in sorted(weights[idx].items(), reverse=True, key=lambda x: x[1])]
        out_adj.append([x[0] for x in weight])
        out_num.append([x[1] for x in weight])

    return out_adj, out_num


def merge_data(data_list):
    """_summary_: merge the data from different datasets

    :param data_list: list of data
    :type data_list: list
    :return: merged data
    :rtype: dict
    """

    assert len(data_list) > 0
    data = dict()
    for key in list(data_list[0].keys()):
        if isinstance(data_list[0][key], np.ndarray):
            data[key] = np.concatenate([x[key] for x in data_list], axis=0)
        elif isinstance(data_list[0][key], torch.Tensor):
            data[key] = torch.cat([x[key] for x in data_list], dim=0)
        elif isinstance(data_list[0][key], (list, tuple)):
            data[key] = data_list[0][key].__class__(chain.from_iterable([x[key] for x in data_list]))
        elif isinstance(data_list[0][key], dict):
            data[key] = merge_data([x[key] for x in data_list])
        else:
            print("Unknown type: {}".format(type(data_list[0][key])))
            data[key] = data_list[0][key]
    return data


def read_patterns(filename):
    """_summary_: read patterns from file

    :param filename: file path
    :type filename: str
    :return: patterns
    :rtype: list
    """

    patterns = list()
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            vid2label = dict()
            edges = list()
            cnt = 0
            for line in f:
                if line.startswith("t #"):
                    if len(vid2label) > 0:
                        # remove duplicate edges
                        edges = sorted(set(edges), key=lambda x: (x[0], x[2], x[1]))
                        pattern = (tuple([vid2label[v] for v in range(len(vid2label))]), tuple(edges), cnt)
                        patterns.append(pattern)
                    vid2label = dict()
                    edges = list()
                    cnt = int(line.strip().split(" ")[-1])
                elif line.startswith("v"):
                    vid, vlabel = line[2:-1].split(" ")
                    vid2label[int(vid)] = vlabel
                elif line.startswith("e"):
                    u, v, elabel = line[2:-1].split(" ")
                    edges.append((int(u), elabel, int(v)))  # list of (u, elabel, v)
            if len(vid2label) > 0:
                # remove duplicate edges
                edges = sorted(set(edges), key=lambda x: (x[0], x[2], x[1]))
                pattern = (tuple([vid2label[v] for v in range(len(vid2label))]), tuple(edges), cnt)
                patterns.append(pattern)
    return patterns


def construct_igraph(vlabels, edges):
    """_summary_: construct igraph from pattern

    :param vlabels: vertex labels
    :type vlabels: list
    :param edges: edges
    :type edges: list
    :return: igraph
    :rtype: igraph.Graph
    """

    graph = ig.Graph(directed=True)
    graph.add_vertices(len(vlabels))
    graph.vs["id"] = list(range(len(vlabels)))
    graph.vs["label"] = list(vlabels)
    graph.add_edges([(e[0], e[2]) for e in edges])
    graph.es["id"] = list(range(len(edges)))
    graph.es["label"] = [e[1] for e in edges]
    return graph


def check_connect(pattern):
    """_summary_: check whether the pattern is connected

    :param pattern: pattern
    :type pattern: [tuple, list, ig.Graph]
    :return: whether the pattern is connected
    :rtype: bool
    """

    if isinstance(pattern, ig.Graph):
        if pattern.ecount() < pattern.vcount() - 1:
            return False
        tmp = pattern.copy()
        tmp.to_undirected()
        return tmp.is_connected()
    else:
        return check_connect(construct_igraph(pattern[0], pattern[1]))


def handle_patterns(patterns, pattern_weights=None, pattern_num=None, prepad=False):
    """_summary_: handle patterns and convert them into numpy array

    :param patterns: patterns
    :type patterns: list
    :param pattern_weights: pattern weights, default None
    :type pattern_weights: list, optional
    :param pattern_num: pattern number, default None
    :type pattern_num: list, optional
    :param prepad: whether to prepad, default False
    :type prepad: bool, optional
    :return: adj, items
    :rtype: tuple
    """

    print(len(patterns))
    if len(patterns) == 0:
        adj = np.zeros((len(patterns), 1, 1), dtype=np.int64)
        items = np.zeros((len(patterns), 1), dtype=np.int64)
        return adj, items

    def process_label(label):
        try:
            label = int(label)
        except ValueError:
            label = int(label.split("_")[-1])
        return label

    # tf_idf
    tfidf = TfidfVectorizer(analyzer=lambda x: x)
    tfidf.fit([p[0] for p in patterns])
    if pattern_weights is not None:
        pattern_weights = np.asarray(pattern_weights) * tfidf.transform([p[0] for p in patterns]).sum(1).A1
    else:
        pattern_weights = tfidf.transform([p[0] for p in patterns]).sum(1).A1

    indices = np.argsort(pattern_weights)[::-1]  # reversed order
    if pattern_num is not None:
        indices = indices[:pattern_num]
    patterns = [patterns[i] for i in indices]

    max_n = max([len(p[0]) for p in patterns])

    adj = np.zeros((len(patterns), max_n, max_n), dtype=np.int64)
    items = np.zeros((len(patterns), max_n), dtype=np.int64)

    for i, pattern in enumerate(patterns):
        vlabels, edges = pattern[:2]

        nodes = []
        for vlabel in vlabels:
            vlabel = process_label(vlabel)
            nodes.append(vlabel)
        nodes = np.array(nodes)
        n = len(nodes)

        # dense graph
        p_adj = np.zeros((n, n), dtype=np.int64)
        for src, elabel, dst in edges:
            p_adj[src, src] = 1
            p_adj[dst, dst] = 1

            elabel = process_label(elabel)

            p_adj[src, dst] = elabel
            if elabel == 2:
                p_adj[dst, src] = 3
            elif elabel == 4:
                p_adj[dst, src] = 4

        if prepad:
            adj[i, -n:, -n:] = p_adj
            items[i, -n:] = nodes
        else:
            adj[i, :n, :n] = p_adj
            items[i, :n] = nodes

    return adj, items


def create_node_index(seq):
    seq_ = seq[::-1]
    nodes = seq_[np.sort(np.unique(seq_, return_index=True)[1])]
    nodes = np.copy(nodes[::-1])
    node_index = dict(zip(nodes, range(len(nodes))))

    return nodes, node_index


def create_dense_adj(seq, node_index=None):
    """_summary_: create dense adjacency matrix from sequence

    :param seq: sequence
    :type seq: (list, np.ndarray, torch.Tensor)
    :param node_index: node index, default None
    :type node_index: dict, optional
    :return: dense adjacency matrix
    :rtype: np.ndarray
    """

    if node_index is not None:
        adj = np.zeros((len(node_index), len(node_index)), dtype=np.int64)
        for i in np.arange(len(seq) - 1):
            u = node_index[seq[i]]
            v = node_index[seq[i + 1]]
            adj[u, u] = 1  # loop
            if u == v or adj[u, v] == 4:
                continue
            adj[v, v] = 1

            if adj[v, u] == 2:
                adj[u, v] = 4  # bi-directional
                adj[v, u] = 4  # bi-directional
            else:
                adj[u, v] = 2  # (u, v)
                adj[v, u] = 3  # reverse
    else:
        adj = np.zeros((len(seq), len(seq)), dtype=np.int64)
        adj[0, 0] = 1  # loop
        for i in np.arange(len(seq) - 1):
            u = seq[i]
            v = seq[i + 1]
            # adj[i, i] = 1  # loop
            adj[i + 1, i + 1] = 1  # loop
            # print(u, v)
            if u == v:
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1
                continue

            if adj[i + 1, i] == 2:
                adj[i, i + 1] = 4  # bi-directional
                adj[i + 1, i] = 4  # bi-directional
            elif adj[i + 1, i] == 0:
                adj[i, i + 1] = 2  # (u, v)
                adj[i + 1, i] = 3  # reverse
    return adj


def create_sparse_adj(seq, node_index=None):
    """_summary_: create sparse adjacency matrix from sequence

    :param seq: sequence
    :type seq: (list, np.ndarray, torch.Tensor)
    :param node_index: node index, default None
    :type node_index: dict, optional
    :return: sparse adjacency matrix
    :rtype: scipy.sparse.csr_matrix
    """

    if node_index is not None:
        adj = sp.lil_matrix((len(node_index), len(node_index)), dtype=np.int64)
        for i in np.arange(len(seq) - 1):
            u = node_index[seq[i]]
            v = node_index[seq[i + 1]]
            adj[u, u] = 1  # loop
            if u == v or adj[u, v] == 4:
                continue
            adj[v, v] = 1

            if adj[v, u] == 2:
                adj[u, v] = 4  # bi-directional
                adj[v, u] = 4  # bi-directional
            else:
                adj[u, v] = 2  # (u, v)
                adj[v, u] = 3  # reverse
    else:
        adj = sp.lil_matrix((len(seq), len(seq)), dtype=np.int64)
        adj[0, 0] = 1  # loop
        for i in np.arange(len(seq) - 1):
            u = seq[i]
            v = seq[i + 1]
            adj[i + 1, i + 1] = 1  # loop
            if u == v:
                adj[i, i + 1] = 1
                adj[i + 1, i] = 1
                continue

            if adj[i + 1, i] == 2:
                adj[i, i + 1] = 4  # bi-directional
                adj[i + 1, i] = 4  # bi-directional
            elif adj[i + 1, i] == 0:
                adj[i, i + 1] = 2  # (u, v)
                adj[i + 1, i] = 3  # reverse
    return adj


def create_dense_shortcut(seq, node_index=None):
    """_summary_: create dense shortcut matrix from sequence

    :param seq: sequence
    :type seq: (list, np.ndarray, torch.Tensor)
    :param node_index: node index, default None
    :type node_index: dict, optional
    :return: dense shortcut matrix
    :rtype: np.ndarray
    """

    if node_index is not None:
        sc = np.zeros((len(node_index), len(node_index)), dtype=np.int64)
        for i in np.arange(len(seq)):
            u = node_index[seq[i]]
            sc[u, u] = 1  # loop
            for j in np.arange(i + 1, len(seq)):
                v = node_index[seq[j]]
                if u == v or sc[u, v] == 4:
                    continue

                if sc[v, u] == 2:
                    sc[u, v] = 4  # bi-directional
                    sc[v, u] = 4  # bi-directional
                else:
                    sc[u, v] = 2  # (u, v)
                    sc[v, u] = 3  # reverse
    else:
        sc = np.zeros((len(seq), len(seq)), dtype=np.int64)
        for i in np.arange(len(seq)):
            u = seq[i]
            sc[i, i] = 1 # loop
            for j in np.arange(i + 1, len(seq)):
                v = seq[j]
                # print(u, v)
                if u == v:
                    sc[i, j] = 1
                    sc[j, i] = 1
                    continue

                if sc[j, i] == 2:
                    sc[i, j] = 4  # bi-directional
                    sc[j, i] = 4  # bi-directional
                elif sc[j, i] == 0:
                    sc[i, j] = 2  # (u, v)
                    sc[j, i] = 3  # reverse
    return sc


def create_sparse_shortcut(seq, node_index=None):
    """_summary_: create sparse shortcut matrix from sequence

    :param seq: sequence
    :type seq: (list, np.ndarray, torch.Tensor)
    :param node_index: node index, default None
    :type node_index: dict, optional
    :return: sparse shortcut matrix
    :rtype: scipy.sparse.csr_matrix
    """

    if node_index is not None:
        sc = sp.lil_matrix((len(node_index), len(node_index)), dtype=np.int64)
        for i in np.arange(len(seq) - 1):
            u = node_index[seq[i]]
            sc[u, u] = 1  # loop
            for j in np.arange(i + 1, len(seq)):
                v = node_index[seq[j]]
                if u == v or sc[u, v] == 4:
                    continue
                sc[v, v] = 1

                if sc[v, u] == 2:
                    sc[u, v] = 4  # bi-directional
                    sc[v, u] = 4  # bi-directional
                else:
                    sc[u, v] = 2  # (u, v)
                    sc[v, u] = 3  # reverse
    else:
        sc = sp.lil_matrix((len(seq), len(seq)), dtype=np.int64)
        for i in np.arange(len(seq)):
            u = seq[i]
            sc[i, i] = 1  # loop
            for j in np.arange(i + 1, len(seq)):
                v = seq[j]
                if u == v:
                    sc[i, j] = 1
                    sc[j, i] = 1
                    continue

                if sc[j, i] == 2:
                    sc[i, j] = 4  # bi-directional
                    sc[j, i] = 4  # bi-directional
                elif sc[j, i] == 0:
                    sc[i, j] = 2  # (u, v)
                    sc[j, i] = 3  # reverse
    return sc


def create_dense_heteadj(seq, node_index=None, order=2):
    """_summary_: create dense heterogeneous adjacency matrix from sequence

    :param seq: sequence
    :type seq: (list, np.ndarray, torch.Tensor)
    :param node_index: node index, default None
    :type node_index: dict, optional
    :param order: adjacency order, default 2
    :type order: int, optional
    :return: (heterogeneous tuples, heterogeneous adjacency)
    :rtype: (np.ndarray, np.ndarray)
    """

    if order == 1:
        items = np.arange(len(node_index), dtype=np.int64).reshape(len(node_index), 1)
        return (items, create_dense_adj(seq, node_index))
    else:
        if node_index is not None:
            seq_order = []
            items = dict()
            for i in np.arange(len(seq) - order + 1):
                x = tuple([node_index[seq[j]] for j in np.arange(i, i + order)])
                if x not in items:
                    items[x] = len(items)
                seq_order.append(x)
        else:
            seq_order = []
            items = dict()
            for i in np.arange(len(seq) - order + 1):
                x = tuple([seq[j] for j in np.arange(i, i + order)])
                if x not in items:
                    items[x] = len(items)
                seq_order.append(x)

        heteadj = create_dense_adj(seq_order, items)
        return (np.array(sorted(items.keys(), key=lambda k: items[k]), dtype=np.int64), heteadj)

def create_sparse_heteadj(seq, node_index=None, order=2):
    """_summary_: create sparse heterogeneous adjacency matrix from sequence

    :param seq: sequence
    :type seq: (list, np.ndarray, torch.Tensor)
    :param node_index: node index, default None
    :type node_index: dict, optional
    :param order: adjacency order, default 2
    :type order: int, optional
    :return: heterogeneous adjacency
    :rtype: (np.ndarray, np.ndarray)
    """

    if order == 1:
        items = np.arange(len(node_index), dtype=np.int64).reshape(len(node_index), 1)
        return (items, create_sparse_adj(seq, node_index))
    else:
        if node_index is not None:
            seq_order = []
            items = dict()
            for i in np.arange(len(seq) - order + 1):
                x = tuple([node_index[seq[j]] for j in np.arange(i, i + order)])
                if x not in items:
                    items[x] = len(items)
                seq_order.append(x)
        else:
            seq_order = []
            items = dict()
            for i in np.arange(len(seq) - order + 1):
                x = tuple([seq[j] for j in np.arange(i, i + order)])
                if x not in items:
                    items[x] = len(items)
                seq_order.append(x)

        heteadj = create_sparse_adj(seq_order, items)
        return (np.asarray(sorted(items.keys(), key=lambda k: items[k])), heteadj)


class Data(Dataset):
    """_summary_: dataset class for session-based recommendation
    """

    def __init__(self, data, attributes=None, max_len=None):
        """_summary_: initialize dataset perserving ids, input, target, and attributes

        :param data: the input data organized as a dictionary
        where the data look like {"ids": List, "input": {"sequence": List, "attribute1": List, ...}, "target": {"sequence:" int, "attribute1": int, ...}}
        :type data: dict
        :param attributes: attributes, default None
        :type attributes: list, optional
        :param max_len: max length, default None
        :type max_len: int, optional
        """

        if max_len is None:
            self.lens = np.asarray([len(x) for x in data["input"][SEQUENCE]], dtype=np.int64)
            self.max_len = int(self.lens.max())
            self.attributes = tuple(default(attributes, tuple()))
            self.ids = data["ids"]
            # {"ids": List, "input": {"sequence": List, "attribute1": List, ...}
            self.input = {
                attr: value
                for attr, value in data["input"].items() if attr == SEQUENCE or attr in self.attributes
            }
            #  {"sequence:" int, "attribute1": int, ...}
            self.target = {
                attr: value
                for attr, value in data["target"].items() if attr == SEQUENCE or attr in self.attributes
            }
        else:
            self.max_len = max_len
            self.lens = np.asarray([len(x) for x in data["input"][SEQUENCE]], dtype=np.int64)
            self.lens = np.clip(self.lens, 0, self.max_len)
            self.attributes = tuple(default(attributes, tuple()))
            self.ids = data["ids"]
            # {"ids": List, "input": {"sequence": List, "attribute1": List, ...}
            self.input = {
                attr: [v[-self.lens[i]:] for i, v in enumerate(value)]
                for attr, value in data["input"].items() if attr == SEQUENCE or attr in self.attributes
            }
            #  {"sequence:" int, "attribute1": int, ...}
            self.target = {
                attr: value
                for attr, value in data["target"].items() if attr == SEQUENCE or attr in self.attributes
            }

    def __getitem__(self, idx):
        """_summary_: get a session data

        :param idx: index
        :type idx: int
        :return: id, input, target, length
        where the input and target look like {"sequence": torch.Tensor, "attribute1", torch.Tensor, ...}
        :rtype: int, dict, dict, int
        """

        id = self.ids[idx]
        input = {attr: torch.tensor(value[idx]) for attr, value in self.input.items()}
        target = {
            attr: torch.tensor(value[idx]) if value is not None else torch.tensor(0)
            for attr, value in self.target.items()
        }
        l = torch.tensor(self.lens[idx])
        return id, input, target, l

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def batchify(batch, prepad=False, return_adj=DENSE_FLAG, return_alias=False, return_shortcut=False, return_hete=False):
        """_summary_: batchify data for training and evaluation

        :param batch: batch of data
        :type batch: list
        :param prepad: prepad, default False
        :type prepad: bool, optional
        :param return_adj: how to return adjacency matrix, default "dense"
        :type return_adj: str, optional, choices=["dense", "sparse", "none"]
        :param return_alias: whether to return alias, default False
        :type return_alias: bool, optional
        :param return_shortcut: whether to return shortcut, default False
        :type return_shortcut: bool, optional
        :param return_hete: whether to return heterogeneous adjacency matrix, default False
        :type return_hete: bool, optional
        :return: batch of data in torch.Tensor
        :rtype: dict
        """

        assert return_adj in [DENSE_FLAG, SPARSE_FLAG, NONE_FLAG]
        heteorder = 2

        id = [x[0] for x in batch]
        l = torch.tensor([x[3] for x in batch])
        max_len = l.max().item()
        mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        items_dict = dict()
        adj_dict = dict()
        alias_dict = dict()
        shortcut_dict = dict()
        heteitems_dict = dict()
        heteadj_dict = dict()

        target_dict = {attr: torch.zeros(len(batch), dtype=torch.long) for attr in batch[0][2].keys()}
        for i, x in enumerate(batch):
            for attr, value in x[2].items():
                if value is None:
                    continue
                target_dict[attr][i].copy_(value)

        if prepad:
            for i in range(len(batch)):
                mask[i, -l[i]:].fill_(1)
            input_dict = {attr: torch.zeros((len(batch), max_len), dtype=torch.long) for attr in batch[0][1].keys()}
            for i, x in enumerate(batch):
                for attr, value in x[1].items():
                    if value is None:
                        continue
                    input_dict[attr][i, -l[i]:].copy_(value)
        else:
            for i in range(len(batch)):
                mask[i, :l[i]].fill_(1)
            input_dict = {attr: torch.zeros((len(batch), max_len), dtype=torch.long) for attr in batch[0][1].keys()}
            for i, x in enumerate(batch):
                for attr, value in x[1].items():
                    if value is None:
                        continue
                    input_dict[attr][i, :l[i]].copy_(value)

        if return_adj == DENSE_FLAG:
            for key in batch[0][1].keys():
                items = torch.zeros((len(batch), max_len), dtype=torch.long)
                adj = torch.zeros((len(batch), max_len, max_len), dtype=torch.long)
                alias = torch.zeros((len(batch), max_len), dtype=torch.long)
                shortcut = torch.zeros((len(batch), max_len, max_len), dtype=torch.long)
                heteitems = torch.zeros((len(batch), max_len, heteorder), dtype=torch.long)
                heteadj = torch.zeros((len(batch), max_len, max_len), dtype=torch.long)
                for i, x in enumerate(batch):
                    seq = x[1][key].numpy()
                    nodes, node_index = create_node_index(seq)
                    n = nodes.shape[0]
                    if prepad:
                        items[i, -n:].copy_(torch.from_numpy(nodes))
                        adj[i, -n:, -n:].copy_(torch.from_numpy(create_dense_adj(seq, node_index)))
                        alias[i, -l[i]:].copy_(max_len - n + torch.tensor([node_index[i] for i in seq]))
                        if return_shortcut:
                            shortcut[i, -n:, -n:].copy_(torch.from_numpy(create_dense_shortcut(seq, node_index)))
                        if return_hete:
                            hete = create_dense_heteadj(seq, node_index, order=heteorder)
                            m = hete[0].shape[0]
                            if m > 0:
                                heteitems[i, -m:].copy_(torch.from_numpy(nodes[hete[0].reshape(-1)].reshape(m, -1)))
                                heteadj[i, -m:, -m:].copy_(torch.from_numpy(hete[1]))
                    else:
                        items[i, :n].copy_(torch.from_numpy(nodes))
                        adj[i, :n, :n].copy_(torch.from_numpy(create_dense_adj(seq, node_index)))
                        alias[i, :l[i]].copy_(torch.tensor([node_index[i] for i in seq]))
                        if return_shortcut:
                            shortcut[i, :n, :n].copy_(torch.from_numpy(create_dense_shortcut(seq, node_index)))
                        if return_hete:
                            hete = create_dense_heteadj(seq, node_index, order=heteorder)
                            m = hete[0].shape[0]
                            if m > 0:
                                heteitems[i, :m].copy_(torch.from_numpy(nodes[hete[0].reshape(-1)].reshape(m, -1)))
                                heteadj[i, :m, :m].copy_(torch.from_numpy(hete[1]))
                items_dict[key] = items
                adj_dict[key] = adj
                alias_dict[key] = alias
                if return_shortcut:
                    shortcut_dict[key] = shortcut
                if return_hete:
                    heteitems_dict[key] = heteitems
                    heteadj_dict[key] = heteadj
            return {
                "id": id,
                "input": input_dict,
                "target": target_dict,
                "len": l,
                "mask": mask,
                "items": items_dict,
                "adj": adj_dict,
                "alias": alias_dict,
                "shortcut": shortcut_dict if return_shortcut else None,
                "heteitems": heteitems_dict if return_hete else None,
                "heteadj": heteadj_dict if return_hete else None,
            }
        elif return_adj == SPARSE_FLAG:
            for key in batch[0][1].keys():
                items = torch.zeros((len(batch), max_len), dtype=torch.long)
                adj_edge_row = []
                adj_edge_col = []
                adj_edge_attr = []
                alias = torch.zeros((len(batch), max_len), dtype=torch.long)
                shortcut_edge_row = []
                shortcut_edge_col = []
                shortcut_edge_attr = []
                heteitems = torch.zeros((len(batch), max_len, 2), dtype=torch.long)
                heteadj_edge_row = []
                heteadj_edge_col = []
                heteadj_edge_attr = []
                for i, x in enumerate(batch):
                    seq = x[1][key].numpy()
                    nodes, node_index = create_node_index(seq)
                    n = nodes.shape[0]
                    if prepad:
                        offset = (i + 1) * max_len - n
                        items[i, -n:].copy_(torch.from_numpy(nodes))
                        adj = create_sparse_adj(seq, node_index).tocoo()
                        adj_edge_row.append(torch.from_numpy(adj.row + offset))
                        adj_edge_col.append(torch.from_numpy(adj.col + offset))
                        adj_edge_attr.append(torch.from_numpy(adj.data))
                        alias[i, -l[i]:].copy_(max_len - n + torch.tensor([node_index[i] for i in seq]))
                        if return_shortcut:
                            shortcut = create_sparse_shortcut(seq, node_index).tocoo()
                            shortcut_edge_row.append(torch.from_numpy(shortcut.row + offset))
                            shortcut_edge_col.append(torch.from_numpy(shortcut.col + offset))
                            shortcut_edge_attr.append(torch.from_numpy(shortcut.data))
                        if return_hete:
                            hete = create_sparse_heteadj(seq, node_index, order=heteorder)
                            m = hete[0].shape[0]
                            if m > 0:
                                offset = (i + 1) * max_len - m
                                heteitems[i, -m:].copy_(torch.from_numpy(nodes[hete[0].reshape(-1)].reshape(m, -1)))
                                heteadj = hete[1].tocoo()
                                heteadj_edge_row.append(torch.from_numpy(heteadj.row + offset))
                                heteadj_edge_col.append(torch.from_numpy(heteadj.col + offset))
                                heteadj_edge_attr.append(torch.from_numpy(heteadj.data))

                    else:
                        offset = i * max_len
                        items[i, :n].copy_(torch.from_numpy(nodes))
                        adj = create_sparse_adj(seq, node_index).tocoo()
                        adj_edge_row.append(torch.from_numpy(adj.row + offset))
                        adj_edge_col.append(torch.from_numpy(adj.col + offset))
                        adj_edge_attr.append(torch.from_numpy(adj.data))
                        alias[i, :l[i]].copy_(torch.tensor([node_index[i] for i in seq]))
                        if return_shortcut:
                            shortcut = create_sparse_shortcut(seq, node_index).tocoo()
                            shortcut_edge_row.append(torch.from_numpy(shortcut.row + offset))
                            shortcut_edge_col.append(torch.from_numpy(shortcut.col + offset))
                            shortcut_edge_attr.append(torch.from_numpy(shortcut.data))
                        if return_hete:
                            hete = create_sparse_heteadj(seq, node_index, order=heteorder)
                            m = hete[0].shape[0]
                            if m > 0:
                                offset = i * max_len
                                heteitems[i, -m:].copy_(torch.from_numpy(nodes[hete[0].reshape(-1)].reshape(m, -1)))
                                heteadj = hete[1].tocoo()
                                heteadj_edge_row.append(torch.from_numpy(heteadj.row + offset))
                                heteadj_edge_col.append(torch.from_numpy(heteadj.col + offset))
                                heteadj_edge_attr.append(torch.from_numpy(heteadj.data))
                items_dict[key] = items
                adj_dict[key] = (
                    torch.stack([torch.cat(adj_edge_row), torch.cat(adj_edge_col)],
                                dim=0).long(), torch.cat(adj_edge_attr)
                )
                alias_dict[key] = alias
                if return_shortcut:
                    shortcut_dict[key] = (
                        torch.stack([torch.cat(shortcut_edge_row), torch.cat(shortcut_edge_col)],
                                    dim=0).long(), torch.cat(shortcut_edge_attr)
                    )
                if return_hete:
                    heteitems_dict[key] = heteitems
                    heteadj_dict[key] = (
                        torch.stack([torch.cat(heteadj_edge_row), torch.cat(heteadj_edge_col)],
                                    dim=0).long(), torch.cat(heteadj_edge_attr)
                    )

            return {
                "id": id,
                "input": input_dict,
                "target": target_dict,
                "len": l,
                "mask": mask,
                "items": items_dict,
                "adj": adj_dict,
                "alias": alias_dict,
                "shortcut": shortcut_dict if return_shortcut else None,
                "heteitems": heteitems_dict if return_hete else None,
                "heteadj": heteadj_dict if return_hete else None,
            }
        elif return_alias:
            for key in batch[0][1].keys():
                items = torch.zeros((len(batch), max_len), dtype=torch.long)
                alias = torch.zeros((len(batch), max_len), dtype=torch.long)
                for i, x in enumerate(batch):
                    seq = x[1][key].numpy()
                    nodes, node_index = create_node_index(seq)
                    n = nodes.shape[0]
                    if prepad:
                        items[i, -n:].copy_(torch.from_numpy(nodes))
                        alias[i, -l[i]:].copy_(torch.tensor([node_index[i] for i in seq]))
                    else:
                        items[i, :n].copy_(torch.from_numpy(nodes))
                        alias[i, :l[i]].copy_(torch.tensor([node_index[i] for i in seq]))
                items_dict[key] = items
                alias_dict[key] = alias
            return {
                "id": id,
                "input": input_dict,
                "target": target_dict,
                "len": l,
                "mask": mask,
                "items": items_dict,
                "adj": None,
                "alias": alias_dict,
                "shortcut": None,
                "heteitems": None,
                "heteadj": None,
            }
        else:
            return {
                "id": id,
                "input": input_dict,
                "target": target_dict,
                "len": l,
                "mask": mask,
                "items": None,
                "adj": None,
                "alias": None,
                "shortcut": None,
                "heteitems": None,
                "heteadj": None,
            }


class BucketSampler(Sampler):
    """_summary_: Bucketing sampler to group sequences of similar length together
    """

    def __init__(self, dataset, batch_size, shuffle=False, seed=0, drop_last=False, pad_last=False):
        """_summary_: Bucketing sampler to group sequences of similar length together

        :param dataset: dataset to sample from
        :type dataset: torch.utils.data.Dataset
        :param batch_size: batch size
        :type batch_size: int
        :param shuffle: whether to shuffle the dataset, defaults to False
        :type shuffle: bool, optional
        :param seed: random seed, defaults to 0
        :type seed: int, optional
        :param drop_last: whether to drop the last batch if it is smaller than batch_size, defaults to False
        :type drop_last: bool, optional
        :param pad_last: whether to pad the last batch if it is smaller than batch_size, defaults to False
        :type pad_last: bool, optional
        """

        super(BucketSampler, self).__init__(dataset)
        self.dataset = dataset
        self.lens = dataset.lens.copy()
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

        if drop_last:
            self.total_size = math.ceil((len(self.dataset) - self.batch_size) / self.batch_size) * self.batch_size
        elif pad_last:
            self.total_size = math.ceil(len(self.dataset) / self.batch_size) * self.batch_size
        else:
            self.total_size = len(self.dataset)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)

        if self.drop_last:
            data_indices = np.arange(self.total_size, dtype=np.int64)
            lens = self.lens[:self.total_size].astype(np.float32)
        else:
            data_indices = np.arange(len(self.dataset), dtype=np.int64)
            padding_size = self.total_size - data_indices.shape[0]
            if padding_size > len(self.dataset):
                data_indices = np.repeat(data_indices, padding_size // len(self.dataset) + 1)
                padding_size = self.total_size - data_indices.shape[0]
            if padding_size > 0:
                data_indices = np.hstack([data_indices, rng.randint(0, len(self.dataset), (padding_size, ))])
            lens = self.lens[data_indices].astype(np.float32)
        assert len(lens) == self.total_size

        rand = rng.rand(self.total_size)
        array = lens + rand
        indices = np.argsort(array, axis=0)
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

        if self.shuffle:
            batch_indices = rng.permutation(len(batches))
            batches = [batches[i] for i in batch_indices]

        batch_idx = 0
        while batch_idx < len(batches):
            yield data_indices[batches[batch_idx]]
            batch_idx += 1

    def __len__(self):
        return math.ceil(self.total_size / self.batch_size)

    def set_epoch(self, epoch=-1):
        if epoch == -1:
            self.epoch += 1
        else:
            self.epoch = epoch
