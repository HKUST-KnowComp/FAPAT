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
from train import train_epoch, debug, test, logger, writer


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
            patterns[key] = handle_patterns(patterns[key], pattern_weights=[p[2] for p in patterns[key]], pattern_num=opt.n_pattern, prepad=prepad)

    train_data = pickle.load(open(f"{datadir}/{dataset}/train.pkl", "rb"))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion, seed=opt.seed)
    else:
        valid_data = pickle.load(open(f"{datadir}/{dataset}/valid.pkl", "rb"))
    ind = [i for i in range(len(train["ids"])) if i % opt.world_size == opt.global_rank]
    train = {
        "ids": [train["ids"][i] for i in ind],
        "dates": [train["dates"][i] for i in ind],
        "input": {key: [train["input"][key][i] for i in ind] for key in ["sequence"] + opt.attributes},
        "target": {key: [train["target"][key][i] for i in ind] for key in ["sequence"] + opt.attributes},
    }
    train_data = Data(train_data, opt.attributes, opt.max_len)

    ind = [i for i in range(len(valid_data["ids"])) if i % opt.world_size == opt.global_rank]
    valid_data = {
        "ids": [valid_data["ids"][i] for i in ind],
        "dates": [valid_data["dates"][i] for i in ind],
        "input": {key: [valid_data["input"][key][i] for i in ind] for key in ["sequence"] + opt.attributes},
        "target": {key: [valid_data["target"][key][i] for i in ind] for key in ["sequence"] + opt.attributes},
    }
    valid_data = Data(valid_data, opt.attributes, opt.max_len)

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        num_workers=1,
        batch_sampler=BucketSampler(train_data, batch_size=batch_size, shuffle=True, seed=len(train_data), drop_last=False),
        collate_fn=partial(Data.batchify, prepad=prepad, return_adj=return_adj, return_alias=return_alias, return_shortcut=return_shortcut, return_hete=return_hete),
        pin_memory=False
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data,
        num_workers=1,
        batch_sampler=BucketSampler(valid_data, batch_size=100, shuffle=False, seed=len(valid_data), drop_last=False),
        collate_fn=partial(Data.batchify, prepad=prepad, return_adj=return_adj, return_alias=return_alias, return_shortcut=return_shortcut, return_hete=return_hete),
        pin_memory=False
    )
    opt.lr_dc_step *= len(train_dataloader)
    model = model_class(opt, n_nodes=n_nodes, adjs=adjs, nums=nums, patterns=patterns)
    # model = trans_to_cuda(model)
    model.to(opt.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank,
        find_unused_parameters=True,
    )
    model._set_static_graph()

    if opt.is_main:
        checkpoint_dir = dataset
        os.makedirs(f"checkpoints/{checkpoint_dir}", exist_ok=True)
        global logger
        logger = init_logger(
            filename=f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_train.log"
        )
        logger.info(str(opt))
        logger.info(str(model))
        logger.info("num of trainable parameters: %d" % (sum(param.numel() for param in model.parameters() if param.requires_grad)))
        global writer
        writer = SummaryWriter(f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_train.tensorboard")
    else:
        logger = None
        writer = None


    start = time.time()
    best_result = {k: 0 for k in ["hit10", "hit20", "mrr10", "mrr20", "ndcg10", "ndcg20"]}
    best_epoch = {k: 0 for k in ["hit10", "hit20", "mrr10", "mrr20", "ndcg10", "ndcg20"]}
    bad_counter = 0

    for epoch in range(opt.epoch):
        if opt.is_main:
            logger.info("-" * 80)
            logger.info("Epoch: %d\tTraining LR:\t%.5f" % (epoch, model.optimizer.param_groups[0]["lr"]))
        train_loss = train_epoch(model, train_dataloader)
        if opt.is_main:
            logger.info("\tTraining Loss:\t%.3f" % (train_loss))

        if False and opt.is_main:
            result = debug(model, train_dataloader)["sequence"]
            logger.info("Debug Result:")
            logger.info("\tRecall@10:\t%.4f\t\tMMR@10:\t%.4f\t\tNDCG@10:\t%.4f" % (result["hit10"], result["mrr10"], result["ndcg10"]))
            logger.info("\tRecall@20:\t%.4f\t\tMMR@20:\t%.4f\t\tNDCG@20:\t%.4f" % (result["hit20"], result["mrr20"], result["ndcg20"]))
            for k, v in result.items():
                writer.add_scalar("debug/eval-%s" % (k), v, epoch)

        result = test(model, valid_dataloader)["sequence"]
        if opt.is_main:
            logger.info("Evaluate Result:")
            logger.info(
                "\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tNDCG@10:\t%.4f" % (result["hit10"], result["mrr10"], result["ndcg10"])
            )
            logger.info(
                "\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tNDCG@20:\t%.4f" % (result["hit20"], result["mrr20"], result["ndcg20"])
            )
            for k, v in result.items():
                writer.add_scalar("valid/eval-%s" % (k), v, epoch)

            flag = 0
            for key in result:
                if result[key] - best_result[key] > 1e-6:
                    best_result[key] = result[key]
                    best_epoch[key] = epoch
                    flag = 1
                    torch.save(
                        model.state_dict(),
                        f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_%s.pt" % key
                    )

            logger.info("Best Evaluate Result:")
            logger.info(
                "\tRecall@10:\t%.4f (e%d)\tMMR@10:\t%.4f (e%d)\tNDCG@10:\t%.4f (e%d)" % (
                    best_result["hit10"], best_epoch["hit10"], best_result["mrr10"], best_epoch["mrr10"],
                    best_result["ndcg10"], best_epoch["ndcg10"]
                )
            )
            logger.info(
                "\tRecall@20:\t%.4f (e%d)\tMMR@20:\t%.4f (e%d)\tNDCG@20:\t%.4f (e%d)" % (
                    best_result["hit20"], best_epoch["hit20"], best_result["mrr20"], best_epoch["mrr20"],
                    best_result["ndcg20"], best_epoch["ndcg20"]
                )
            )
            logger.info("-" * 80)

            if flag:
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter >= opt.patience:
                    break

    if opt.is_main:
        logger.info("=" * 80)
        end = time.time()
        logger.info("Run time: %f s" % (end - start))
    
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    opt = get_option()
    init_seed(opt.seed)
    init_distributed_mode(opt)
    init_signal_handler()
    torch.distributed.barrier()

    main(opt)
