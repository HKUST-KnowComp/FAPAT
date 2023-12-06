import os
import sys

f_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f_path)

import argparse
import os
import subprocess
import multiprocessing

parsemis_path = f"{f_path}/parsemis/bin"


def iter_files(path):
    """_summary_: iterate all files in a directory

    :param path: directory path
    :type path: str
    :raises RuntimeError: if path is not a directory
    :yield: file path
    :rtype: str
    """

    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, file_names in os.walk(path):
            for f in file_names:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError("Path %s is invalid" % path)


def mine_frequent_patterns(
    class_path, input_file, output_file, min_node, max_node, min_edge, max_edge, min_freq, max_freq
):
    """_summary_: mine frequent patterns using parsemis

    :param class_path: path to parsemis jar file
    :type class_path: str
    :param input_file: input file path
    :type input_file: str
    :param output_file: output file path
    :type output_file: str
    :param min_node: minimum number of nodes
    :type min_node: int
    :param max_node: maximum number of nodes
    :type max_node: int
    :param min_edge: minimum number of edges
    :type min_edge: int
    :param max_edge: maximum number of edges
    :type max_edge: int
    :param min_freq: minimum frequency
    :type min_freq: int
    :param max_freq: maximum frequency
    :type max_freq: int
    :return: parsemis output
    :rtype: bytes
    """

    output = subprocess.check_output(
        f"java -Xmx128g -cp {class_path} de.parsemis.Miner --graphFile={input_file} --outputFile={output_file} --algorithm=gspan --threads=4 --minimumEdgeCount={min_edge} --maximumEdgeCount={max_edge} --minimumNodeCount={min_node} --maximumNodeCount={max_node} --minimumFrequency={min_freq} --maximumFrequency={max_freq}",
        shell=True
    )
    return output


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
        "--attributes",
        nargs="+", default=[]
    )
    parser.add_argument(
        "--reg_file",
        type=str, default=""
    )
    parser.add_argument(
        "--date",
        type=str, default=""
    )
    parser.add_argument(
        "--n_extractors",
        type=int, default=5
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
        type=int, default=2
    )
    parser.add_argument(
        "--max_freq",
        type=int, default=65535
    )
    opt = parser.parse_args()

    datadir = opt.datadir
    dataset = opt.dataset
    n_sample = opt.n_sample
    category = opt.category + "-" if opt.category else ""
    date = opt.date + "-" if opt.date else ""

    prefix = category + date

    process_files = [f"{datadir}/{dataset}/{prefix}sequence.lg"] + \
        [f"{datadir}/{dataset}/{prefix}{attr}.lg" for attr in opt.attributes]

    with multiprocessing.Pool(opt.n_extractors) as pool:
        results = []
        for input_file in process_files:
            num_graphs = 0
            with open(input_file, "r") as f:
                for line in f:
                    if line.startswith("t #"):
                        num_graphs += 1
            # min_freq = int(opt.min_freq * num_graphs)
            # max_freq = int(opt.max_freq * num_graphs)
            output_file = input_file.replace(".lg", ".gspan")
            results.append(
                (
                    input_file, num_graphs,
                    pool.apply_async(
                        mine_frequent_patterns,
                        args=(
                            parsemis_path, input_file, output_file, opt.min_node, opt.max_node, opt.min_edge,
                            opt.max_edge, opt.min_freq, opt.max_freq
                        )
                    )
                )
            )
        pool.close()

        for result in results:
            input_file, num_graphs, process = result
            info = process.get().decode("utf-8")
            print(f"{input_file} with {num_graphs} graphs finished: {info}")
