import os
import sys

f_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f_path)

import pandas as pd
from collections import Counter
from train import forward
from eval_amazon import *

# IDCG table
idcg_10 = {0: 0.0}
idcg_10.update(dict(zip(range(1, 11), (1 / np.log2(np.arange(10) + 2)).cumsum().tolist())))
idcg_20 = {0: 0.0}
idcg_20.update(dict(zip(range(1, 21), (1 / np.log2(np.arange(20) + 2)).cumsum().tolist())))


def attr_test(model, test_dataloader, node2attrs):
    model.eval()

    results = dict()
    for key in node2attrs:
        results[key] = {
            "hit10": 0,
            "hit20": 0,
            "mrr10": 0.0,
            "mrr20": 0.0,
            "ndcg10": 0.0,
            "ndcg20": 0.0,
        }
    cnt = 0

    attr_cnts = dict()
    for key in node2attrs:
        attr_cnts[key] = Counter(node2attrs[key].values())

    with torch.no_grad():
        for data in test_dataloader:
            targets, scores, loss = forward(model, data)
            cnt += targets["sequence"].shape[0]

            seq_score = scores["sequence"].detach()

            # seq_argsort = torch.argsort(seq_score, dim=1, descending=True)
            # seq_argsort = trans_to_cpu(seq_argsort[:, :20]).numpy()

            value, seq_argsort = torch.topk(seq_score, min(20, seq_score.shape[1]), dim=1, largest=True, sorted=True)
            seq_argsort = trans_to_cpu(seq_argsort).numpy()
            seq_score = trans_to_cpu(seq_score).numpy()

            for key in node2attrs:
                target = targets[key].detach()
                target = trans_to_cpu(target).numpy()
                if key in scores:
                    attr_score = scores[key].detach()
                    value, attr_argsort = torch.topk(
                        attr_score, min(20, attr_score.shape[1]), dim=1, largest=True, sorted=True
                    )
                    attr_argsort = trans_to_cpu(attr_argsort).numpy()
                    attr_score = trans_to_cpu(attr_score).numpy()
                for i in range(seq_argsort.shape[0]):
                    if key == "sequence":
                        argsort_i = seq_argsort[i]
                    else:
                        if key in scores:
                            argsort_i = attr_argsort[i]
                        else:
                            argsort_i = np.asarray([node2attrs[key][j] for j in seq_argsort[i]])
                    rank = 1 + index(argsort_i, target[i])
                    if rank == 0:
                        rank = 21
                    results[key]["hit10"] += 1 if rank <= 10 else 0
                    results[key]["hit20"] += 1 if rank <= 20 else 0
                    results[key]["mrr10"] += 1.0 / rank if rank <= 10 else 0
                    results[key]["mrr20"] += 1.0 / rank if rank <= 20 else 0
                    results[key]["ndcg10"] += 1.0 / math.log(rank + 1, 2) if rank <= 10 else 0
                    results[key]["ndcg20"] += 1.0 / math.log(rank + 1, 2) if rank <= 20 else 0
                    # dcg = 0
                    # for j in range(min(10, len(argsort_i))):
                    #     if argsort_i[j] == target[i]:
                    #         dcg += 1 / math.log(j + 2, 2)
                    # attr_cnt = attr_cnts[key][int(target[i])]
                    # results[key]["ndcg10"] = dcg / idcg_10.get(attr_cnt, idcg_10[10]) if idcg_10.get(attr_cnt, idcg_10[10]) != 0 else 0

                    # for j in range(min(20, len(argsort_i))):
                    #     if argsort_i[j] == target[i]:
                    #         dcg += 1 / math.log(j + 2, 2)
                    # results[key]["ndcg20"] = dcg / idcg_20.get(attr_cnt, idcg_20[20]) if idcg_20.get(attr_cnt, idcg_20[20]) != 0 else 0

    for key in results:
        for metric in results[key]:
            results[key][metric] /= cnt

    return results


def main_amazon(opt):
    model_name = opt.model
    datadir = opt.datadir
    dataset = opt.dataset
    category = opt.category
    dates = opt.dates
    n_sample = opt.n_sample
    batch_size = opt.batch_size
    hidden_dim = opt.hidden_dim
    n_iter = opt.n_iter
    seed = opt.seed
    prepad = opt.prepad
    mtl = "mtl_" if opt.mtl else ""
    attributes = "+".join(opt.attributes)
    # if model_name.startswith("fapat") or opt.mtl:
    #     attributes = "+".join(opt.attributes)
    # else:
    #     attributes = ""
    model_class, return_adj, return_alias, return_shortcut, return_hete, load_adj, load_pattern = get_model_config(opt.model)

    prefix = ""
    if category:
        prefix = category + "-"

    n_nodes = dict()
    for key in ["sequence"] + attributes.split("+"):
        if len(key.strip()) == 0:
            continue
        with open(f"{datadir}/Amazon/{prefix}{key}.tsv", "r") as f:
            n_nodes[key] = 0
            reader = csv.DictReader(f, delimiter="\t")
            for line in reader:
                n_nodes[key] = max(n_nodes[key], int(line["index"]) + 1)

    node2attrs = {key: {0: 99999} for key in ["sequence"] + opt.attributes}
    df = pd.read_csv(f"{datadir}/Amazon/{prefix}meta.tsv", sep="\t")
    for idx, row in tqdm(df.iterrows()):
        for attr in opt.attributes:
            node2attrs[attr][row[0]] = row[attr]
        node2attrs["sequence"][row[0]] = row[0]

    start_date = opt.dates.split("-")[0]
    start_date = datetime.datetime.strptime(start_date, "%Y%m%d").date()
    end_date = opt.dates.split("-")[-1]
    end_date = datetime.datetime.strptime(end_date, "%Y%m%d").date()
    delta = datetime.timedelta(days=1)
    dates = []
    while start_date <= end_date:
        dates.append(start_date.strftime("%Y%m%d"))
        start_date += delta

    # load adj, num, and patterns
    prefixes = []
    adjs = {key: [] for key in n_nodes}
    nums = {key: [] for key in n_nodes}
    patterns = {key: [] for key in n_nodes}
    tests = []
    for date in dates:
        prefix = ""
        if category:
            prefix += category + "-"
        prefix += date + "-"
        prefixes.append(prefix)
        for key in n_nodes:
            if load_adj:
                adjs[key].append(pickle.load(open(f"{datadir}/Amazon/{prefix}{key}_adj_{n_sample}.pkl", "rb")))
                nums[key].append(pickle.load(open(f"{datadir}/Amazon/{prefix}{key}_num_{n_sample}.pkl", "rb")))
        tests.append(pickle.load(open(f"{datadir}/{dataset}/{prefix}test.pkl", "rb")))

    prefix = ""
    if category:
        prefix += category + "-"
    for key in n_nodes:
        if load_adj:
            adjs[key], nums[key] = merge_adjs(adjs[key], nums[key], n_nodes[key])
            adjs[key], nums[key] = handle_adj(adjs[key], nums[key], n_nodes[key], n_sample)
        if load_pattern:
            patterns[key] = read_patterns(f"{datadir}/Amazon/{prefix}{key}_freq.lg")
            patterns[key] = handle_patterns(
                patterns[key], pattern_weights=[p[2] for p in patterns[key]], pattern_num=opt.n_pattern, prepad=prepad
            )

    test_data = merge_data(tests)
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

    opt.lr_dc_step = opt.lr_dc_step * len(test_dataloader) * 10 * len(dates)
    # model = model_class(opt, n_nodes=n_nodes, adjs=adjs, nums=nums, patterns=patterns)
    attributes = "" # test models without attribute prediction
    model = model_class(opt, n_nodes={"sequence": n_nodes["sequence"]}, adjs=adjs, nums=nums, patterns=patterns)

    checkpoint_dir = dataset + "-"
    if category:
        checkpoint_dir += category + "-"
    checkpoint_dir += opt.dates
    model.load_state_dict(
        torch.load(
            f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_mrr20.pt",
            map_location="cpu"
        ),
        strict=True
    )
    model = trans_to_cuda(model)

    global logger
    logger = init_logger(
        filename=
        f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_attr_eval.log"
    )
    logger.info(
        f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_mrr20.pt"
    )
    result = attr_test(model, test_dataloader, node2attrs)
    for key in ["sequence"] + opt.attributes:
        logger.info(f"Evaluate Result (attribute: {key}):")
        logger.info(
            "\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tNDCG@10:\t%.4f" %
            (result[key]["hit10"], result[key]["mrr10"], result[key]["ndcg10"])
        )
        logger.info(
            "\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tNDCG@20:\t%.4f" %
            (result[key]["hit20"], result[key]["mrr20"], result[key]["ndcg20"])
        )


def main_public(opt):
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
    # if model_name.startswith("fapat") or opt.mtl:
    #     attributes = "+".join(opt.attributes)
    # else:
    #     attributes = ""
    model_class, return_adj, return_alias, return_shortcut, return_hete, load_adj, load_pattern = get_model_config(opt.model)

    n_nodes = dict()
    for key in ["sequence"] + attributes.split("+"):
        if len(key.strip()) == 0:
            continue
        with open(f"{datadir}/{dataset}/{key}.tsv", "r") as f:
            n_nodes[key] = 0
            reader = csv.DictReader(f, delimiter="\t")
            for line in reader:
                n_nodes[key] = max(n_nodes[key], int(line["index"]) + 1)

    node2attrs = {key: {0: 99999} for key in ["sequence"] + opt.attributes}
    df = pd.read_csv(f"{datadir}/{dataset}/meta.tsv", sep="\t")
    for idx, row in tqdm(df.iterrows()):
        for attr in opt.attributes:
            node2attrs[attr][row[0]] = row[attr]
        node2attrs["sequence"][row[0]] = row[0]

    adjs = dict()
    nums = dict()
    patterns = dict()
    for key in n_nodes:
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
    # model = model_class(opt, n_nodes=n_nodes, adjs=adjs, nums=nums, patterns=patterns)
    attributes = "" # test models without attribute prediction
    model = model_class(opt, n_nodes={"sequence": n_nodes["sequence"]}, adjs=adjs, nums=nums, patterns=patterns)

    checkpoint_dir = dataset
    model.load_state_dict(
        torch.load(
            f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_mrr20.pt",
            map_location="cpu"
        ),
        strict=True
    )
    model = trans_to_cuda(model)

    global logger
    logger = init_logger(
        filename=
        f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_attr_eval.log"
    )
    logger.info(
        f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_mrr20.pt"
    )
    result = attr_test(model, test_dataloader, node2attrs)
    for key in ["sequence"] + opt.attributes:
        logger.info(f"Evaluate Result (attribute: {key}):")
        logger.info(
            "\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tNDCG@10:\t%.4f" %
            (result[key]["hit10"], result[key]["mrr10"], result[key]["ndcg10"])
        )
        logger.info(
            "\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tNDCG@20:\t%.4f" %
            (result[key]["hit20"], result[key]["mrr20"], result[key]["ndcg20"])
        )


if __name__ == "__main__":
    opt = get_option()
    init_seed(opt.seed)

    if opt.dataset.startswith("Amazon"):
        main_amazon(opt)
    else:
        main_public(opt)