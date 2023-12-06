import os
import sys

f_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f_path)

import csv
import math
import pickle5 as pickle
import time
import datetime
import torch
from functools import partial
from tensorboardX import SummaryWriter
from tqdm import tqdm
from option import get_option, get_model_config
from config import *
from data import *
from utils import *
from train import test, logger


def main(opt):
    model_name = opt.model
    datadir = opt.datadir
    dataset = opt.dataset
    n_sample = opt.n_sample
    batch_size = opt.batch_size
    hidden_dim = opt.hidden_dim
    n_iter = opt.n_iter
    seed = opt.seed
    prepad = opt.prepad
    mtl = "mtl_" if opt.mtl else ""
    attributes = "+".join(opt.attributes)
    model_class, return_adj, return_alias, return_shortcut, return_hete, load_adj, load_pattern = get_model_config(opt.model)

    n_nodes = dict()
    for key in ["sequence"] + opt.attributes:
        with open(f"{datadir}/{dataset}/{key}.tsv", "r") as f:
            n_nodes[key] = 0
            reader = csv.DictReader(f, delimiter="\t")
            for line in reader:
                n_nodes[key] = max(n_nodes[key], int(line["index"]) + 1)

    adjs = dict()
    nums = dict()
    patterns = dict()
    for key in ["sequence"] + opt.attributes:
        if load_adj:
            adjs[key] = pickle.load(open(f"{datadir}/{dataset}/{key}_adj_{n_sample}.pkl", "rb"))
            nums[key] = pickle.load(open(f"{datadir}/{dataset}/{key}_num_{n_sample}.pkl", "rb"))
            adjs[key], nums[key] = handle_adj(adjs[key], nums[key], n_nodes[key], n_sample)
        if load_pattern:
            patterns[key] = read_patterns(f"{datadir}/{dataset}/{key}_freq.lg")
            patterns[key] = handle_patterns(
                patterns[key], pattern_weights=[p[2] for p in patterns[key]], pattern_num=opt.n_pattern, prepad=prepad
            )

    test_data = pickle.load(open(f"{datadir}/{dataset}/test.pkl", "rb"))
    test_data = Data(test_data, opt.attributes, opt.max_len)

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        num_workers=1,
        batch_sampler=BucketSampler(
            test_data, batch_size=batch_size, shuffle=False, seed=len(test_data), drop_last=False
        ),
        collate_fn=partial(Data.batchify, prepad=prepad, return_adj=return_adj, return_alias=return_alias, return_shortcut=return_shortcut, return_hete=return_hete),
        pin_memory=False
    )
    model = model_class(opt, n_nodes=n_nodes, adjs=adjs, nums=nums, patterns=patterns)

    checkpoint_dir = dataset
    model.load_state_dict(
        torch.load(
            f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_mrr20.pt",
            map_location="cpu"
        ),
        strict=False
    )
    model = trans_to_cuda(model)

    global logger
    logger = init_logger(
        filename=
        f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_eval.log"
    )
    logger.info(
        f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_mrr20.pt"
    )
    result = test(model, test_dataloader)["sequence"]

    logger.info("Evaluate Result:")
    logger.info(
        "\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tNDCG@10:\t%.4f" % (result["hit10"], result["mrr10"], result["ndcg10"])
    )
    logger.info(
        "\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tNDCG@20:\t%.4f" % (result["hit20"], result["mrr20"], result["ndcg20"])
    )


if __name__ == "__main__":
    opt = get_option()
    init_seed(opt.seed)

    main(opt)
