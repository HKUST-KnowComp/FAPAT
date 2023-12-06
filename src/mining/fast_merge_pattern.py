import os
import sys

f_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f_path)

import argparse
import igraph as ig
import numpy as np
import datetime
import os
from tqdm import tqdm
from collections import Counter, defaultdict
from gspan import *
from merge_pattern import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        type=str, default="../datasets"
    )
    parser.add_argument(
        "--dataset",
        help="diginetica/Tmall/Nowplaying/Amazon"
    )
    parser.add_argument(
        "--attribute",
        type=str, default="browse"
    )
    parser.add_argument(
        "--category",
        type=str, default=""
    )
    parser.add_argument(
        "--dates",
        type=str, default=""
    )
    parser.add_argument(
        "--min_node",
        type=int, default=3
    )
    parser.add_argument(
        "--max_node",
        type=int, default=4
    )
    parser.add_argument(
        "--min_edge",
        type=int, default=3
    )
    parser.add_argument(
        "--max_edge",
        type=int, default=6
    )
    parser.add_argument(
        "--min_freq",
        type=int, default=5
    )
    parser.add_argument(
        "--max_freq",
        type=int, default=65536
    )
    parser.add_argument(
        "--max_candicate",
        type=int, default=1000000
    )

    opt = parser.parse_args()

    # load patterns
    if len(opt.dates) > 0:
        start_date = opt.dates.split("-")[0]
        start_date = datetime.datetime.strptime(start_date, "%Y%m%d").date()
        end_date = opt.dates.split("-")[-1]
        end_date = datetime.datetime.strptime(end_date, "%Y%m%d").date()
        delta = datetime.timedelta(days=1)
        dates = []
        while start_date <= end_date:
            dates.append(start_date.strftime("%Y%m%d"))
            start_date += delta

        category = opt.category
        if category:
            category += "-"
        patterns = []
        for date in dates:
            prefix = ""
            prefix += category + date + "-"
            for vlabels, edges, cnt in read_patterns(
                os.path.join(opt.datadir, opt.dataset, prefix + opt.attribute + f".gspan")
            ):
                # we do not use cnt here because the gspan output does not return the correct counts
                pattern_g = construct_igraph(vlabels, edges)
                if check_connect(pattern_g):
                    patterns.append((vlabels, edges, pattern_g))
    else:
        category = opt.category
        if category:
            category += "-"
        patterns = []
        for vlabels, edges, cnt in read_patterns(os.path.join(opt.datadir, opt.dataset, opt.attribute + ".gspan")):
            # we do not use cnt here because the gspan output does not return the correct counts
            pattern_g = construct_igraph(vlabels, edges)
            if check_connect(pattern_g):
                patterns.append((vlabels, edges, pattern_g))

    num_patterns = len(patterns)
    print(f"{num_patterns} patterns before removing redundancies")

    # further remove low-frequency patterns
    for num_node in range(opt.min_node, opt.max_node + 1):
        input_file = os.path.join(opt.datadir, opt.dataset, category + opt.attribute + f".{num_node}.lg")
        output_file = os.path.join(opt.datadir, opt.dataset, category + opt.attribute + f"_freq.{num_node}.gspan")
        write_patterns([(p[:2], i) for i, p in enumerate(patterns) if len(p[0]) == num_node], input_file)
        try:
            mine_frequent_patterns(
                parsemis_path, input_file, output_file, num_node, num_node, opt.min_edge, opt.max_edge, opt.min_freq,
                opt.max_freq
            )
        except:
            pass

    # filter loose patterns
    pattern_retriever = PatternRetriever()
    patterns = []
    for num_node in range(opt.min_node, opt.max_node + 1):
        output_file = os.path.join(opt.datadir, opt.dataset, category + opt.attribute + f"_freq.{num_node}.gspan")
        for vlabels, edges, cnt in read_patterns(output_file):
            pattern_g = construct_igraph(
                [int(vl.split("_")[-1]) for vl in vlabels], [(e[0], int(e[1].split("_")[-1]), e[2]) for e in edges]
            )
            patterns.append((vlabels, edges, pattern_g))
    lens = np.array([(len(pattern[0]), len(pattern[1])) for pattern in patterns])
    if len(lens) > 0:
        indices = np.lexsort(lens.T)
        patterns = [patterns[i] for i in indices]

    N = len(patterns)
    pattern_cnt = [0] * N
    pattern_vlabels = [set(pattern[2].vs["label"]) for pattern in patterns]
    pattern_elabels = [set(pattern[2].es["label"]) for pattern in patterns]
    i = 0
    j = N - 1
    duplicate_indices = set()
    for i in tqdm(range(N)):
        if i in duplicate_indices:
            continue
        m = len(patterns[i][0])
        n = len(patterns[i][1])
        vlabels = pattern_vlabels[i]
        elabels = pattern_elabels[i]

        pattern = patterns[i]
        pattern_cnt[i] += 1
        for j in range(i + 1, N):
            if j in duplicate_indices:
                continue
            mm = len(patterns[j][0])
            nn = len(patterns[j][1])

            if mm > m + 1:
                break
            if nn < n:
                continue
            if mm == m and nn == n:
                continue

            # manually check labels to speed up
            if len(pattern_vlabels[j] - vlabels) > 0:
                continue
            if len(pattern_elabels[j] - elabels) > 0:
                continue

            cnt_sub_iso = pattern_retriever.count_subisomorphisms(patterns[j][2], pattern[2])
            if cnt_sub_iso > 0:
                duplicate_indices.add(j)
                pattern_cnt[i] += 1

    # save compact patterns
    num_patterns = len(patterns) - len(duplicate_indices)
    print(f"{num_patterns} patterns after removing redundancies")
    write_patterns(
        [(p[:2], pattern_cnt[i]) for i, p in enumerate(patterns) if i not in duplicate_indices],
        os.path.join(opt.datadir, opt.dataset, category + opt.attribute + "_freq.lg")
    )
