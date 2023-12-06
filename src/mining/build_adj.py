import os
import sys

f_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f_path)

import pickle5 as pickle
import argparse
import os


def process(seqs):
    """_summary_: process the sequence to adjacency linked list 

    :param seqs: list of sequences
    :type seqs: list
    :return: adjacency linked list , neighbor weight linked list 
    :rtype: list, list
    """

    num_items = max([max(s) for s in seqs]) + 1

    relation = []

    adj1 = [dict() for _ in range(num_items)]
    adj = [[] for _ in range(num_items)]

    for seq in seqs:
        for k in range(1, 4):
            for j in range(len(seq) - k):
                relation.append([seq[j], seq[j + k]])
                relation.append([seq[j + k], seq[j]])

    for tup in relation:
        if tup[1] in adj1[tup[0]].keys():
            adj1[tup[0]][tup[1]] += 1
        else:
            adj1[tup[0]][tup[1]] = 1

    weight = [[] for _ in range(num_items)]

    for t in range(num_items):
        x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
        adj[t] = [v[0] for v in x][:n_sample]
        weight[t] = [v[1] for v in x][:n_sample]

    return adj, weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        type=str, default="../datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str, default="lastfm",
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
        "--attributes", nargs="+", default=["track"]
    )
    parser.add_argument(
        "--date",
        type=str, default=""
    )
    opt = parser.parse_args()

    datadir = opt.datadir
    dataset = opt.dataset
    n_sample = opt.n_sample
    category = opt.category + "-" if opt.category else ""
    date = opt.date + "-" if opt.date else ""

    prefix = category + date

    seqs = pickle.load(open(f"{datadir}/{dataset}/{prefix}all_train_sequence.pkl", "rb"))
    adj, weight = process(seqs)

    print(f"{datadir}/{dataset}/{prefix}sequence_adj_{n_sample}.pkl")
    pickle.dump(adj, open(f"{datadir}/{dataset}/{prefix}sequence_adj_{n_sample}.pkl", "wb"))
    print(f"{datadir}/{dataset}/{prefix}sequence_num_{n_sample}.pkl")
    pickle.dump(weight, open(f"{datadir}/{dataset}/{prefix}sequence_num_{n_sample}.pkl", "wb"))

    for attr in opt.attributes:
        seqs = pickle.load(open(f"{datadir}/{dataset}/{prefix}all_train_{attr}.pkl", "rb"))
        adj, weight = process(seqs)
        print(f"{datadir}/{dataset}/{prefix}{attr}_adj_{n_sample}.pkl")
        pickle.dump(adj, open(f"{datadir}/{dataset}/{prefix}{attr}_adj_{n_sample}.pkl", "wb"))
        print(f"{datadir}/{dataset}/{prefix}{attr}_num_{n_sample}.pkl")
        pickle.dump(weight, open(f"{datadir}/{dataset}/{prefix}{attr}_num_{n_sample}.pkl", "wb"))
