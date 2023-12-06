import os
import igraph as ig
from collections import Counter


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


def write_patterns(patterns, filename):
    """_summary_: write patterns to file

    :param patterns: patterns
    :type patterns: [list, Counter, dict]
    :param filename: file path
    :type filename: str
    """

    if isinstance(patterns, Counter):
        patterns = patterns.most_common()
    elif isinstance(patterns, dict):
        patterns = Counter(patterns).most_common()

    with open(filename, "w", encoding="utf-8") as f:
        for i, pattern in enumerate(patterns):
            if len(pattern) == 2 and len(pattern[0]) == 2:
                if isinstance(pattern[1], int):
                    cnt = pattern[1]
                    pattern = pattern[0]
                else:
                    cnt = i
            elif len(pattern) == 3:
                cnt = pattern[2]
                pattern = pattern[:2]
            f.write("t # %d\n" % (cnt))
            for v, p in enumerate(pattern[0]):
                f.write("v %d %s\n" % (v, p))
            for dep in pattern[1]:
                f.write("e %d %d %s\n" % (dep[0], dep[2], dep[1]))


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
