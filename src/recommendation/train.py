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
from collections import defaultdict
from functools import partial
from tensorboardX import SummaryWriter
from tqdm import tqdm
from option import get_option, get_model_config
from config import *
from data import *
from utils import *

logger = None
writer = None


def forward(model, data):
    input = trans_to_cuda(data["input"])
    target = trans_to_cuda(data["target"])
    mask = trans_to_cuda(data["mask"])
    items = trans_to_cuda(data["items"])
    adj = trans_to_cuda(data["adj"])
    alias = trans_to_cuda(data["alias"])
    shortcut = trans_to_cuda(data["shortcut"])
    heteitems = trans_to_cuda(data["heteitems"])
    heteadj = trans_to_cuda(data["heteadj"])

    output = model(input, mask, items=items, adj=adj, alias=alias, shortcut=shortcut, heteitems=heteitems, heteadj=heteadj)

    model = model.module if hasattr(model, "module") else model
    scores = model.compute_scores(output, mask)
    loss = model.compute_loss(scores, target)

    return target, scores, loss


def train_epoch(model, train_dataloader):
    # logger.info("start training: %s" % datetime.datetime.now())
    model.train()
    total_loss = 0.0
    cnt = 0

    if not hasattr(model, "module") or model.output_device == 0:
        train_dataloader = tqdm(train_dataloader)

    for data in train_dataloader:
        cnt += 1

        if hasattr(model, "module"):
            model.module.optimizer.zero_grad()
        else:
            model.optimizer.zero_grad()
        targets, scores, loss = forward(model, data)
        loss.backward()
        if hasattr(model, "module"):
            if model.module.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), model.module.clip)
            model.module.optimizer.step()
            model.module.scheduler.step()
        else:
            if model.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), model.clip)
            model.optimizer.step()
            model.scheduler.step()

        total_loss += loss.item()

    return total_loss / cnt if cnt > 0 else 0.0


def pretrain_epoch(model, train_dataloader):
    # logger.info("start training: %s" % datetime.datetime.now())
    model.train()
    total_loss = 0.0
    cnt = 0

    if not hasattr(model, "module") or model.output_device == 0:
        train_dataloader = tqdm(train_dataloader)

    for data in train_dataloader:
        cnt += 1
        input = trans_to_cuda(data["input"])
        mask = trans_to_cuda(data["mask"])

        if hasattr(model, "module"):
            model.module.optimizer.zero_grad()
            pretrain_data = model.module.reconstruct_pretrain_data(input, mask)
            loss = model.module.pretrain(*pretrain_data)
            loss.backward()
            if model.module.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), model.module.clip)
            model.module.optimizer.step()
            model.module.scheduler.step()
        else:
            model.optimizer.zero_grad()
            pretrain_data = model.reconstruct_pretrain_data(input, mask)
            loss = model.pretrain(*pretrain_data)
            loss.backward()
            if model.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), model.clip)
            model.optimizer.step()
            model.scheduler.step()

        total_loss += loss.item()

    return total_loss / cnt if cnt > 0 else 0.0


def debug(model, train_dataloader):
    model.eval()

    results = dict()
    cnt = 0
    with torch.no_grad():
        for idx, data in enumerate(train_dataloader):
            if idx % 10 != 0:
                continue
            targets, scores, loss = forward(model, data)
            cnt += targets["sequence"].shape[0]

            for key in scores:
                if key not in results:
                    results[key] = {"hit10": 0, "hit20": 0, "mrr10": 0.0, "mrr20": 0.0, "ndcg10": 0.0, "ndcg20": 0.0}

                target = targets[key].detach()
                score = scores[key].detach()
                # argsort = torch.argsort(score, dim=1, descending=True)
                # argsort = trans_to_cpu(argsort[:, :20]).numpy()

                value, argsort = torch.topk(score, min(20, score.shape[1]), dim=1, largest=True, sorted=True)
                argsort = trans_to_cpu(argsort).numpy()

                target = trans_to_cpu(target).numpy()
                score = trans_to_cpu(score).numpy()
                for i in range(argsort.shape[0]):
                    rank = 1 + index(argsort[i], target[i])
                    if rank == 0:
                        rank = 21
                    results[key]["hit10"] += 1 if rank <= 10 else 0
                    results[key]["hit20"] += 1 if rank <= 20 else 0
                    results[key]["mrr10"] += 1.0 / rank if rank <= 10 else 0
                    results[key]["mrr20"] += 1.0 / rank if rank <= 20 else 0
                    results[key]["ndcg10"] += 1.0 / math.log(rank + 1, 2) if rank <= 10 else 0
                    results[key]["ndcg20"] += 1.0 / math.log(rank + 1, 2) if rank <= 20 else 0
    for key in results:
        for metric in results[key]:
            results[key][metric] /= cnt

    if len(results) == 0:
        results["sequence"] = {"hit10": 0, "hit20": 0, "mrr10": 0.0, "mrr20": 0.0, "ndcg10": 0.0, "ndcg20": 0.0}

    return results


def test(model, test_dataloader):
    model.eval()

    results = dict()
    cnt = 0
    with torch.no_grad():
        for data in test_dataloader:
            targets, scores, loss = forward(model, data)
            cnt += targets["sequence"].shape[0]

            for key in scores:
                if key not in results:
                    results[key] = {
                        "hit10": 0,
                        "hit20": 0,
                        "mrr10": 0.0,
                        "mrr20": 0.0,
                        "ndcg10": 0.0,
                        "ndcg20": 0.0,
                    }

                target = targets[key].detach()
                score = scores[key].detach()
                # argsort = torch.argsort(score, dim=1, descending=True)
                # argsort = trans_to_cpu(argsort[:, :20]).numpy()

                value, argsort = torch.topk(score, min(20, score.shape[1]), dim=1, largest=True, sorted=True)
                argsort = trans_to_cpu(argsort).numpy()

                target = trans_to_cpu(target).numpy()
                score = trans_to_cpu(score).numpy()
                for i in range(argsort.shape[0]):
                    rank = 1 + index(argsort[i], target[i])
                    if rank == 0:
                        rank = 21
                    results[key]["hit10"] += 1 if rank <= 10 else 0
                    results[key]["hit20"] += 1 if rank <= 20 else 0
                    results[key]["mrr10"] += 1.0 / rank if rank <= 10 else 0
                    results[key]["mrr20"] += 1.0 / rank if rank <= 20 else 0
                    results[key]["ndcg10"] += 1.0 / math.log(rank + 1, 2) if rank <= 10 else 0
                    results[key]["ndcg20"] += 1.0 / math.log(rank + 1, 2) if rank <= 20 else 0
    for key in results:
        for metric in results[key]:
            results[key][metric] /= cnt

    if len(results) == 0:
        results["sequence"] = {"hit10": 0, "hit20": 0, "mrr10": 0.0, "mrr20": 0.0, "ndcg10": 0.0, "ndcg20": 0.0}

    return results


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

    train_data = pickle.load(open(f"{datadir}/{dataset}/train.pkl", "rb"))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion, seed=opt.seed)
    else:
        valid_data = pickle.load(open(f"{datadir}/{dataset}/valid.pkl", "rb"))
    train_data = Data(train_data, opt.attributes, opt.max_len)
    valid_data = Data(valid_data, opt.attributes, opt.max_len)

    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        num_workers=1,
        batch_sampler=BucketSampler(
            train_data, batch_size=batch_size, shuffle=True, seed=len(train_data), drop_last=False
        ),
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
    model = trans_to_cuda(model)

    checkpoint_dir = dataset
    os.makedirs(f"checkpoints/{checkpoint_dir}", exist_ok=True)
    global logger
    logger = init_logger(
        filename=
        f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_train.log"
    )
    logger.info(str(opt))
    logger.info(str(model))
    logger.info(
        "num of trainable parameters: %d" % (sum(param.numel() for param in model.parameters() if param.requires_grad))
    )
    global writer
    writer = SummaryWriter(
        f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_train.tensorboard"
    )

    start = time.time()
    best_result = {k: 0 for k in ["hit10", "hit20", "mrr10", "mrr20", "ndcg10", "ndcg20"]}
    best_epoch = {k: 0 for k in ["hit10", "hit20", "mrr10", "mrr20", "ndcg10", "ndcg20"]}
    bad_counter = 0

    if (hasattr(model, "pretrain")) or (hasattr(model, "module") and hasattr(model.module, "pretrain")):
        logger.info("Pretraining...")
        for epoch in range(min(opt.epoch, 10)):
            logger.info("-" * 80)
            logger.info("Epoch: %d\tPretraining LR:\t%.5f" % (epoch, model.optimizer.param_groups[0]["lr"]))
            train_loss = pretrain_epoch(model, train_dataloader)
            logger.info("\tPretraining Loss:\t%.3f" % (train_loss))
        logger.info("Pretraining finished.")

        # reset optimizer
        if hasattr(model, "module"):
            model.module.optimizer.__setstate__({'state': defaultdict(dict)})
            for g in model.module.optimizer.param_groups:
                g['lr'] = opt.lr
            # model.module.scheduler._initial_step()
            model.module.scheduler.optimizer._step_count = 0
            model.module.scheduler._step_count = 0
            model.module.scheduler.step(0)
        else:
            model.optimizer.__setstate__({'state': defaultdict(dict)})
            for g in model.optimizer.param_groups:
                g['lr'] = opt.lr
            # model.scheduler._initial_step()
            model.scheduler.optimizer._step_count = 0
            model.scheduler._step_count = 0
            model.scheduler.step(0)

    for epoch in range(opt.epoch):
        logger.info("-" * 80)
        logger.info("Epoch: %d\tTraining LR:\t%.5f" % (epoch, model.optimizer.param_groups[0]["lr"]))
        train_loss = train_epoch(model, train_dataloader)
        logger.info("\tTraining Loss:\t%.3f" % (train_loss))

        result = debug(model, train_dataloader)["sequence"]
        logger.info("Debug Result:")
        logger.info(
            "\tRecall@10:\t%.4f\t\tMMR@10:\t%.4f\t\tNDCG@10:\t%.4f" %
            (result["hit10"], result["mrr10"], result["ndcg10"])
        )
        logger.info(
            "\tRecall@20:\t%.4f\t\tMMR@20:\t%.4f\t\tNDCG@20:\t%.4f" %
            (result["hit20"], result["mrr20"], result["ndcg20"])
        )
        for k, v in result.items():
            writer.add_scalar("debug/eval-%s" % (k), v, epoch)

        result = test(model, valid_dataloader)["sequence"]
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
                    f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_%s.pt"
                    % key
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

    logger.info("=" * 80)
    end = time.time()
    logger.info("Run time: %f s" % (end - start))


if __name__ == "__main__":
    opt = get_option()
    init_seed(opt.seed)

    main(opt)
