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
from collections import Counter
from utils import *


def retrieve_multiple_edges(graph, source=-1, target=-1):
    """_summary_: retrieve multiple edges from a graph

    :param graph: graph
    :type graph: igraph.Graph
    :param source: source node, defaults to -1
    :type source: int, optional
    :param target: target node, defaults to -1
    :type target: int, optional
    :return: edges between source and target
    :rtype: list
    """

    if source != -1:
        e = graph.incident(source, mode=ig.OUT)
        if target != -1:
            e = set(e).intersection(graph.incident(target, mode=ig.IN))
        return ig.EdgeSeq(graph, e)
    else:
        if target != -1:
            e = graph.incident(target, mode=ig.IN)
        else:
            e = list()
        return ig.EdgeSeq(graph, e)


class PatternRetriever(object):
    """_summary_: retrieve frequent patterns from an input graph
    """

    def __init__(self):
        pass

    @classmethod
    def node_compat_fn(cls, g1, g2, v1, v2):
        if g1.indegree(v1) < g2.indegree(v2):
            return False
        vl2 = g2.vs[v2]["label"]
        vl1 = g1.vs[v1]["label"]
        return vl1 == vl2

    @classmethod
    def multi_edge_compat_fn(cls, g1, g2, e1, e2):
        edge1 = g1.es[e1]
        edge2 = g2.es[e2]
        if edge1.is_loop() != edge2.is_loop():
            return False
        # for multiedges
        edges1 = retrieve_multiple_edges(g1, edge1.source, edge1.target)
        edges2 = retrieve_multiple_edges(g2, edge2.source, edge2.target)
        if len(edges1) < len(edges2):
            return False
        edge1_labels = set(edges1["label"])
        for el in edges2["label"]:
            if el not in edge1_labels:
                return False
        return True

    @classmethod
    def edge_compat_fn(cls, g1, g2, e1, e2):
        edge1 = g1.es[e1]
        edge2 = g2.es[e2]
        if edge1.is_loop() != edge2.is_loop():
            return False
        if edge1["label"] != edge2["label"]:
            return False
        return True

    @classmethod
    def get_vertex_color_vectors(cls, g1, g2, seed_v1=-1, seed_v2=-1):
        N1 = g1.vcount()
        N2 = g2.vcount()
        color_vectors = list()

        color1, color2 = None, None
        if seed_v1 != -1:
            color1 = g1.vs[seed_v1]["label"]
        if seed_v2 != -1:
            color2 = g2.vs[seed_v2]["label"]

        if color1 is None and color2 is None:
            color_vectors.append((None, None))
        elif color1 is not None and color2 is not None:
            if color1 == color2:
                color1 = np.zeros((N1, ), dtype=np.int64)
                color2 = np.zeros((N2, ), dtype=np.int64)
                color1[seed_v1] = 1
                color2[seed_v2] = 1
                color_vectors.append((color1, color2))
        elif color1 is not None:
            seed_label = color1
            color1 = np.zeros((N1, ), dtype=np.int64)
            color1[seed_v1] = 1
            for seed_v2, vertex in enumerate(g2.vs):
                if vertex["label"] == seed_label:
                    color2 = np.zeros((N2, ), dtype=np.int64)
                    color2[seed_v2] = 1
                    color_vectors.append((color1, color2))
        else:  # color2 is not None
            seed_label = color2
            color2 = np.zeros((N2, ), dtype=np.int64)
            color2[seed_v2] = 1
            for seed_v1, vertex in enumerate(g1.vs):
                if vertex["label"] == seed_label:
                    color1 = np.zeros((N1, ), dtype=np.int64)
                    color1[seed_v1] = 1
                    color_vectors.append((color1, color2))
        return color_vectors

    @classmethod
    def get_edge_color_vectors(cls, g1, g2, seed_e1=-1, seed_e2=-1):
        E1 = len(g1.es)
        E2 = len(g2.es)
        edge_color_vectors = list()

        color1, color2 = None, None
        if seed_e1 != -1:
            color1 = g1.es[seed_e1]["label"]
        if seed_e2 != -1:
            color2 = g2.es[seed_e2]["label"]

        if color1 is None and color2 is None:
            edge_color_vectors.append((None, None))
        elif color1 is not None and color2 is not None:
            if color1 == color2 and g1.es[seed_e1].is_loop() == g2.es[seed_e2].is_loop():
                edge_color_vectors.append((color1, color2))
        elif color1 is not None:
            seed_label = color1
            is_loop = g1.es[seed_e1].is_loop()
            color1 = [0] * E1
            color1[seed_e1] = 1
            for seed_e2, edge in enumerate(g2.es):
                if edge["label"] == seed_label and is_loop == edge.is_loop():
                    color2 = [0] * E2
                    color2[seed_e2] = 1
                    edge_color_vectors.append((color1, color2))
        else:  # color2 is not None:
            seed_label = color2
            is_loop = g2.es[seed_e2].is_loop()
            color2 = [0] * E2
            color2[seed_e2] = 1
            for seed_e1, edge in enumerate(g1.es):
                if edge["label"] == seed_label and is_loop == edge.is_loop():
                    color1 = [0] * E1
                    color1[seed_e1] = 1
                    edge_color_vectors.append((color1, color2))

        return edge_color_vectors

    def check(self, graph, pattern, **kwargs):
        """_summary_: check if a pattern is a subgraph of a graph

        :param graph: graph
        :type graph: igraph.Graph
        :param pattern: pattern
        :type pattern: igraph.Graph
        :return: True if pattern is a subgraph of graph, False otherwise
        :rtype: bool
        """

        # valid or not
        if graph.vcount() < pattern.vcount():
            return False
        if graph.ecount() < pattern.ecount():
            return False

        graph_vlabels = Counter(graph.vs["label"])
        pattern_vlabels = Counter(pattern.vs["label"])
        if len(graph_vlabels) < len(pattern_vlabels):
            return False
        for vertex_label, pv_cnt in pattern_vlabels.most_common():
            diff = graph_vlabels[vertex_label] - pv_cnt
            if diff < 0:
                return False

        graph_elabels = set(graph.es["label"])
        pattern_elabels = set(pattern.es["label"])
        if len(graph_elabels) < len(pattern_elabels):
            return False

        return True

    def get_subisomorphisms(self, graph, pattern, **kwargs):
        """_summary_: get all subisomorphisms of a pattern in a graph

        :param graph: graph
        :type graph: igraph.Graph
        :param pattern: pattern
        :type pattern: igraph.Graph
        :return: all subisomorphisms of pattern in graph
        :rtype: list
        """

        if not self.check(graph, pattern):
            return list()

        seed_v1 = kwargs.get("seed_v1", -1)
        seed_v2 = kwargs.get("seed_v2", -1)
        seed_e1 = kwargs.get("seed_e1", -1)
        seed_e2 = kwargs.get("seed_e2", -1)

        vertex_color_vectors = PatternRetriever.get_vertex_color_vectors(
            graph, pattern, seed_v1=seed_v1, seed_v2=seed_v2
        )
        edge_color_vectors = PatternRetriever.get_edge_color_vectors(graph, pattern, seed_e1=seed_e1, seed_e2=seed_e2)

        vertices_in_graph = list()
        if seed_v1 != -1:
            vertices_in_graph.append(seed_v1)
        if seed_e1 != -1:
            vertices_in_graph.extend(graph.es[seed_e1].tuple)
        subisomorphisms = list()  # [(component, mapping), ...]
        for vertex_colors in vertex_color_vectors:
            for edge_colors in edge_color_vectors:
                for subisomorphism in graph.get_subisomorphisms_vf2(
                    pattern,
                    color1=vertex_colors[0],
                    color2=vertex_colors[1],
                    edge_color1=edge_colors[0],
                    edge_color2=edge_colors[1],
                    node_compat_fn=PatternRetriever.node_compat_fn,
                    edge_compat_fn=PatternRetriever.edge_compat_fn
                ):
                    if len(vertices_in_graph) == 0 or all([v in subisomorphism for v in vertices_in_graph]):
                        subisomorphisms.append(subisomorphism)
        return subisomorphisms

    def count_subisomorphisms(self, graph, pattern, **kwargs):
        """_summary_: count all subisomorphisms of a pattern in a graph

        :param graph: graph
        :type graph: igraph.Graph
        :param pattern: pattern
        :type pattern: igraph.Graph
        :return: number of subisomorphisms of pattern in graph
        :rtype: int
        """

        if not self.check(graph, pattern):
            return 0

        seed_v1 = kwargs.get("seed_v1", -1)
        seed_v2 = kwargs.get("seed_v2", -1)
        seed_e1 = kwargs.get("seed_e1", -1)
        seed_e2 = kwargs.get("seed_e2", -1)

        vertex_color_vectors = PatternRetriever.get_vertex_color_vectors(
            graph, pattern, seed_v1=seed_v1, seed_v2=seed_v2
        )
        edge_color_vectors = PatternRetriever.get_edge_color_vectors(graph, pattern, seed_e1=seed_e1, seed_e2=seed_e2)

        vertices_in_graph = list()
        if seed_v1 != -1:
            vertices_in_graph.append(seed_v1)
        if seed_e1 != -1:
            vertices_in_graph.extend(graph.es[seed_e1].tuple)
        if len(vertices_in_graph) == 0:
            counts = 0
            for vertex_colors in vertex_color_vectors:
                for edge_colors in edge_color_vectors:
                    counts += graph.count_subisomorphisms_vf2(
                        pattern,
                        color1=vertex_colors[0],
                        color2=vertex_colors[1],
                        edge_color1=edge_colors[0],
                        edge_color2=edge_colors[1],
                        node_compat_fn=PatternRetriever.node_compat_fn,
                        edge_compat_fn=PatternRetriever.edge_compat_fn
                    )
            return counts
        else:
            counts = 0
            for vertex_colors in vertex_color_vectors:
                for edge_colors in edge_color_vectors:
                    for subisomorphism in graph.get_subisomorphisms_vf2(
                        pattern,
                        color1=vertex_colors[0],
                        color2=vertex_colors[1],
                        edge_color1=edge_colors[0],
                        edge_color2=edge_colors[1],
                        node_compat_fn=PatternRetriever.node_compat_fn,
                        edge_compat_fn=PatternRetriever.edge_compat_fn
                    ):
                        if all([v in subisomorphism for v in vertices_in_graph]):
                            counts += 1
            return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        type=str, default="../datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str, help="diginetica/Tmall/Nowplaying/Amazon"
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
                os.path.join(opt.datadir, opt.dataset, prefix + opt.attribute + ".gspan")
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

    # filter loose patterns
    pattern_retriever = PatternRetriever()
    lens = np.array([(len(pattern[0]), len(pattern[1])) for pattern in patterns])
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
