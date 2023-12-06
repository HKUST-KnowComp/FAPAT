import os
import sys

f_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f_path)

import argparse
import numpy as np
import pickle5 as pickle
import os
from collections import Counter


def process(seq, directed=False):
    """_summary_: process the sequence for gSpan

    :param seq: list of sequences
    :type seq: list
    :param directed: whether the graph is directed, defaults to False
    :type directed: bool, optional
    :return: node list, edge labels, edge weights, node index
    :rtype: list, dict, Counter, dict
    """

    nodes = np.unique(seq).tolist()
    node_index = dict(zip(nodes, range(len(nodes))))

    edge_labels = dict()
    edge_weights = Counter()

    # edge_labels[(0, 0)] = 1 # self loop
    # edge_weights[(0, 0)] += 1 # self loop
    for i in range(len(seq) - 1):
        # ignore unknown nodes
        if seq[i] == 0 or seq[i + 1] == 0:
            continue
        u = node_index[seq[i]]
        v = node_index[seq[i + 1]]

        # edge_labels[(v, v)] = 1 # self loop
        # edge_weights[(v, v)] += 1 # self loop

        if directed:
            if (u == v):
                edge_labels[(u, v)] = 4
            elif edge_labels.get((v, u), -1) == 2:
                edge_labels[(u, v)] = 4  # bi-directional
                edge_labels[(v, u)] = 4  # bi-directional
            else:
                edge_labels[(u, v)] = 2  # (u, v)
                edge_labels[(v, u)] = 3  # reverse

            edge_weights[(u, v)] += 1
            edge_weights[(v, u)] += 1
        else:
            if u < v:
                edge_labels[(u, v)] = 4
                edge_weights[(u, v)] += 1
            else:
                edge_labels[(v, u)] = 4
                edge_weights[(v, u)] += 1

    return nodes, edge_labels, edge_weights, node_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        type=str, default="../datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str, default="diginetica",
        help="diginetica/Tmall/Nowplaying/Amazon"
    )
    parser.add_argument(
        "--n_sample",
        type=int, default=12
    )
    parser.add_argument(
        "--category",
        type=str, default=""
    )
    parser.add_argument(
        "--attributes", nargs="+", default=[])
    parser.add_argument(
        "--date",
        type=str, default=""
    )
    parser.add_argument(
        "--multi_edge",
        action="store_true"
    )
    parser.add_argument(
        "--directed",
        action="store_true"
    )
    opt = parser.parse_args()

    datadir = opt.datadir
    dataset = opt.dataset
    n_sample = opt.n_sample
    category = opt.category + "-" if opt.category else ""
    date = opt.date + "-" if opt.date else ""

    prefix = category + date

    seqs = pickle.load(open(f"{datadir}/{dataset}/{prefix}all_train_sequence.pkl", "rb"))
    print(f"{datadir}/{dataset}/{prefix}sequence.lg")
    with open(f"{datadir}/{dataset}/{prefix}sequence.lg", "w") as ff:
        for t, seq in enumerate(seqs):
            nodes, edge_labels, edge_weights, node_index = process(seq, directed=opt.directed)
            ff.write(f"t # {t}\n")
            for vid, vlabel in enumerate(nodes):
                ff.write(f"v {vid} item_{vlabel}\n")
            for (u, v), elabel in edge_labels.items():
                line = f"e {u} {v} item_{elabel}\n"
                if opt.multi_edge:
                    for i in range(edge_weights[(u, v)]):
                        ff.write(line)
                else:
                    ff.write(line)

    # separate attribute
    attr_seqs = dict()
    for attr in opt.attributes:
        attr_seqs[attr] = pickle.load(open(f"{datadir}/{dataset}/{prefix}all_train_{attr}.pkl", "rb"))
        print(f"{datadir}/{dataset}/{prefix}{attr}.lg")
        with open(f"{datadir}/{dataset}/{prefix}{attr}.lg", "w") as ff:
            for t, seq in enumerate(attr_seqs[attr]):
                nodes, edge_labels, edge_weights, node_index = process(seq)
                ff.write(f"t # {t}\n")
                for vid, vlabel in enumerate(nodes):
                    ff.write(f"v {vid} {attr}_{vlabel}\n")
                for (u, v), elabel in edge_labels.items():
                    line = f"e {u} {v} {attr}_{elabel}\n"
                    if opt.multi_edge:
                        for i in range(edge_weights[(u, v)]):
                            ff.write(line)
                    else:
                        ff.write(line)

    # combination
    print(f"{datadir}/{dataset}/{prefix}attributes.lg")
    with open(f"{datadir}/{dataset}/{prefix}attributes.lg", "w") as ff:
        for t, seq in enumerate(seqs):
            node_offset = 0
            combined_nodes = []
            combined_edge_labels = dict()
            combined_edge_weights = Counter()
            nodes, edge_labels, edge_weights, node_index = process(seq)
            for vid, vlabel in enumerate(nodes):
                combined_nodes.append("item_0")  # mask item information
            for (u, v), elabel in edge_labels.items():
                combined_edge_labels[(u, v)] = f"item_{elabel}"
                combined_edge_weights[(u, v)] = edge_weights[(u, v)]
            node_offset += len(nodes)

            for attr in sorted(opt.attributes):
                attr_seq = attr_seqs[attr][t]
                attr_nodes, attr_edge_labels, attr_edge_weights, attr_node_index = process(attr_seq)
                for vid, vlabel in enumerate(attr_nodes):
                    combined_nodes.append(f"{attr}_{vlabel}")
                for (u, v), elabel in attr_edge_labels.items():
                    combined_edge_labels[(u + node_offset, v + node_offset)] = f"{attr}_{elabel}"
                    combined_edge_weights[(u + node_offset, v + node_offset)] = attr_edge_weights[(u, v)]
                # attr -> item
                for idx in range(len(seq)):
                    src = attr_node_index[attr_seq[idx]] + node_offset
                    dst = node_index[seq[idx]]
                    combined_edge_labels[(src, dst)] = f"{attr}-item"
                    combined_edge_weights[(src, dst)] += 1
                node_offset += len(attr_nodes)

            ff.write(f"t # {t}\n")
            for vid, vlabel in enumerate(combined_nodes):
                ff.write(f"v {vid} {vlabel}\n")
            for (u, v), elabel in combined_edge_labels.items():
                line = f"e {u} {v} {elabel}\n"
                if opt.multi_edge:
                    for i in range(combined_edge_weights[(u, v)]):
                        ff.write(line)
                else:
                    ff.write(line)