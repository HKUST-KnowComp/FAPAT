import os
import sys

f_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f_path)

import pandas as pd
from eval_amazon import *


def period_batchify(batch, prepad=False, return_adj="dense", return_alias=False, return_shortcut=False, return_hete=False):
    assert return_adj in ["dense", "sparse", "none", False]
    heteorder = 2

    id = [x[0] for x in batch]
    l = torch.tensor([x[3] for x in batch])
    max_len = l.max().item()
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    items_dict = dict()
    adj_dict = dict()
    alias_dict = dict()
    shortcut_dict = dict()
    heteitems_dict = dict()
    heteadj_dict = dict()

    target_len = dict()
    for i, x in enumerate(batch):
        for attr, value in x[2].items():
            if value is None:
                continue
            target_len[attr] = max(target_len.get(attr, 0), value.shape[0])

    target_dict = {attr: torch.zeros((len(batch), target_len[attr]), dtype=torch.long) for attr in batch[0][2].keys()}
    for i, x in enumerate(batch):
        for attr, value in x[2].items():
            if value is None:
                continue
            target_dict[attr][i, :value.shape[0]].copy_(value)

    if prepad:
        for i in range(len(batch)):
            mask[i, -l[i]:].fill_(1)
        input_dict = {attr: torch.zeros((len(batch), max_len), dtype=torch.long) for attr in batch[0][1].keys()}
        for i, x in enumerate(batch):
            for attr, value in x[1].items():
                if value is None:
                    continue
                input_dict[attr][i, -l[i]:].copy_(value)
    else:
        for i in range(len(batch)):
            mask[i, :l[i]].fill_(1)
        input_dict = {attr: torch.zeros((len(batch), max_len), dtype=torch.long) for attr in batch[0][1].keys()}
        for i, x in enumerate(batch):
            for attr, value in x[1].items():
                if value is None:
                    continue
                input_dict[attr][i, :l[i]].copy_(value)

    if return_adj == "dense":
        for key in batch[0][1].keys():
            items = torch.zeros((len(batch), max_len), dtype=torch.long)
            adj = torch.zeros((len(batch), max_len, max_len), dtype=torch.long)
            alias = torch.zeros((len(batch), max_len), dtype=torch.long)
            shortcut = torch.zeros((len(batch), max_len, max_len), dtype=torch.long)
            heteitems = torch.zeros((len(batch), max_len, heteorder), dtype=torch.long)
            heteadj = torch.zeros((len(batch), max_len, max_len), dtype=torch.long)
            for i, x in enumerate(batch):
                seq = x[1][key].numpy()
                nodes, node_index = create_node_index(seq)
                n = nodes.shape[0]
                if prepad:
                    items[i, -n:].copy_(torch.from_numpy(nodes))
                    adj[i, -n:, -n:].copy_(torch.from_numpy(create_dense_adj(seq, node_index)))
                    alias[i, -l[i]:].copy_(max_len - n + torch.tensor([node_index[i] for i in seq]))
                    if return_shortcut:
                        shortcut[i, -n:, -n:].copy_(torch.from_numpy(create_dense_shortcut(seq, node_index)))
                    if return_hete:
                        hete = create_dense_heteadj(seq, node_index, order=heteorder)
                        m = hete[0].shape[0]
                        if m > 0:
                            heteitems[i, -m:].copy_(torch.from_numpy(nodes[hete[0].reshape(-1)].reshape(m, -1)))
                            heteadj[i, -m:, -m:].copy_(torch.from_numpy(hete[1]))
                else:
                    items[i, :n].copy_(torch.from_numpy(nodes))
                    adj[i, :n, :n].copy_(torch.from_numpy(create_dense_adj(seq, node_index)))
                    alias[i, :l[i]].copy_(torch.tensor([node_index[i] for i in seq]))
                    if return_shortcut:
                        shortcut[i, :n, :n].copy_(torch.from_numpy(create_dense_shortcut(seq, node_index)))
                    if return_hete:
                        hete = create_dense_heteadj(seq, node_index, order=heteorder)
                        m = hete[0].shape[0]
                        if m > 0:
                            heteitems[i, :m].copy_(torch.from_numpy(nodes[hete[0].reshape(-1)].reshape(m, -1)))
                            heteadj[i, :m, :m].copy_(torch.from_numpy(hete[1]))
            items_dict[key] = items
            adj_dict[key] = adj
            alias_dict[key] = alias
            if return_shortcut:
                shortcut_dict[key] = shortcut
            if return_hete:
                heteitems_dict[key] = heteitems
                heteadj_dict[key] = heteadj
        return {
            "id": id,
            "input": input_dict,
            "target": target_dict,
            "len": l,
            "mask": mask,
            "adj": adj_dict,
            "items": items_dict,
            "alias": alias_dict,
            "shortcut": shortcut_dict if return_shortcut else None,
            "heteitems": heteitems_dict if return_hete else None,
            "heteadj": heteadj_dict if return_hete else None,
        }
    elif return_adj == "sparse":
        for key in batch[0][1].keys():
            items = torch.zeros((len(batch), max_len), dtype=torch.long)
            adj_edge_row = []
            adj_edge_col = []
            adj_edge_attr = []
            alias = torch.zeros((len(batch), max_len), dtype=torch.long)
            shortcut_edge_row = []
            shortcut_edge_col = []
            shortcut_edge_attr = []
            heteitems = torch.zeros((len(batch), max_len, 2), dtype=torch.long)
            heteadj_edge_row = []
            heteadj_edge_col = []
            heteadj_edge_attr = []
            for i, x in enumerate(batch):
                seq = x[1][key].numpy()
                nodes, node_index = create_node_index(seq)
                n = nodes.shape[0]
                if prepad:
                    offset = (i + 1) * max_len - n
                    items[i, -n:].copy_(torch.from_numpy(nodes))
                    adj = create_sparse_adj(seq, node_index).tocoo()
                    adj_edge_row.append(torch.from_numpy(adj.row + offset))
                    adj_edge_col.append(torch.from_numpy(adj.col + offset))
                    adj_edge_attr.append(torch.from_numpy(adj.data))
                    alias[i, -l[i]:].copy_(max_len - n + torch.tensor([node_index[i] for i in seq]))
                    if return_shortcut:
                        shortcut = create_sparse_shortcut(seq, node_index).tocoo()
                        shortcut_edge_row.append(torch.from_numpy(shortcut.row + offset))
                        shortcut_edge_col.append(torch.from_numpy(shortcut.col + offset))
                        shortcut_edge_attr.append(torch.from_numpy(shortcut.data))
                    if return_hete:
                        hete = create_sparse_heteadj(seq, node_index, order=heteorder)
                        m = hete[0].shape[0]
                        if m > 0:
                            offset = (i + 1) * max_len - m
                            heteitems[i, -m:].copy_(torch.from_numpy(nodes[hete[0].reshape(-1)].reshape(m, -1)))
                            heteadj = hete[1].tocoo()
                            heteadj_edge_row.append(torch.from_numpy(heteadj.row + offset))
                            heteadj_edge_col.append(torch.from_numpy(heteadj.col + offset))
                            heteadj_edge_attr.append(torch.from_numpy(heteadj.data))
                else:
                    offset = i * max_len
                    items[i, :n].copy_(torch.from_numpy(nodes))
                    adj = create_sparse_adj(seq, node_index).tocoo()
                    adj_edge_row.append(torch.from_numpy(adj.row + offset))
                    adj_edge_col.append(torch.from_numpy(adj.col + offset))
                    adj_edge_attr.append(torch.from_numpy(adj.data))
                    alias[i, :l[i]].copy_(torch.tensor([node_index[i] for i in seq]))
                    if return_shortcut:
                        shortcut = create_sparse_shortcut(seq, node_index).tocoo()
                        shortcut_edge_row.append(torch.from_numpy(shortcut.row + offset))
                        shortcut_edge_col.append(torch.from_numpy(shortcut.col + offset))
                        shortcut_edge_attr.append(torch.from_numpy(shortcut.data))
                    if return_hete:
                        hete = create_sparse_heteadj(seq, node_index, order=heteorder)
                        m = hete[0].shape[0]
                        if m > 0:
                            offset = i * max_len
                            heteitems[i, -m:].copy_(torch.from_numpy(nodes[hete[0].reshape(-1)].reshape(m, -1)))
                            heteadj = hete[1].tocoo()
                            heteadj_edge_row.append(torch.from_numpy(heteadj.row + offset))
                            heteadj_edge_col.append(torch.from_numpy(heteadj.col + offset))
                            heteadj_edge_attr.append(torch.from_numpy(heteadj.data))
            items_dict[key] = items
            adj_dict[key] = (
                torch.stack([torch.cat(adj_edge_row), torch.cat(adj_edge_col)],
                            dim=0).long(), torch.cat(adj_edge_attr)
            )
            alias_dict[key] = alias
            if return_shortcut:
                shortcut_dict[key] = (
                    torch.stack([torch.cat(shortcut_edge_row), torch.cat(shortcut_edge_col)],
                                dim=0).long(), torch.cat(shortcut_edge_attr)
                )
            if return_hete:
                heteitems_dict[key] = heteitems
                heteadj_dict[key] = (
                    torch.stack([torch.cat(heteadj_edge_row), torch.cat(heteadj_edge_col)],
                                dim=0).long(), torch.cat(heteadj_edge_attr)
                )
        return {
            "id": id,
            "input": input_dict,
            "target": target_dict,
            "len": l,
            "mask": mask,
            "items": items_dict,
            "adj": adj_dict,
            "alias": alias_dict,
            "shortcut": shortcut_dict if return_shortcut else None,
            "heteitems": heteitems_dict if return_hete else None,
            "heteadj": heteadj_dict if return_hete else None,
        }
    elif return_alias:
        for key in batch[0][1].keys():
            items = torch.zeros((len(batch), max_len), dtype=torch.long)
            alias = torch.zeros((len(batch), max_len), dtype=torch.long)
            for i, x in enumerate(batch):
                seq = x[1][key].numpy()
                nodes, node_index = create_node_index(seq)
                n = nodes.shape[0]
                if prepad:
                    items[i, -n:].copy_(torch.from_numpy(nodes))
                    alias[i, -l[i]:].copy_(torch.tensor([node_index[i] for i in seq]))
                else:
                    items[i, :n].copy_(torch.from_numpy(nodes))
                    alias[i, :l[i]].copy_(torch.tensor([node_index[i] for i in seq]))
            items_dict[key] = items
            alias_dict[key] = alias
        return {
            "id": id,
            "input": input_dict,
            "target": target_dict,
            "len": l,
            "mask": mask,
            "items": items_dict,
            "adj": None,
            "alias": alias_dict,
            "shortcut": None,
            "heteitems": None,
            "heteadj": None,
        }
    else:
        return {
            "id": id,
            "input": input_dict,
            "target": target_dict,
            "len": l,
            "mask": mask,
            "items": None,
            "adj": None,
            "alias": None,
            "shortcut": None,
            "heteitems": None,
            "heteadj": None,
        }


def period_forward(model, data):
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

    return target, scores, trans_to_cuda(torch.zeros(1, dtype=torch.float, requires_grad=True))


# IDCG table
_idcg = {0: 0.0}
_idcg.update(dict(zip(range(1, 21), (1 / np.log2(np.arange(20) + 2)).cumsum().tolist())))


def period_test(model, test_dataloader):
    model.eval()

    results = dict()
    cnt = 0
    with torch.no_grad():
        for data in tqdm(test_dataloader, total=len(test_dataloader)):
            targets, scores, loss = period_forward(model, data)
            cnt += targets["sequence"].shape[0]

            for key in scores:
                if key not in results:
                    results[key] = {
                        "recall10": 0.0,
                        "recall20": 0.0,
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
                    recall10 = 0
                    recall20 = 0
                    dcg10 = 0.0
                    dcg20 = 0.0
                    idcg = 0.0
                    j = 0
                    for t in target[i]:
                        if t == 0:
                            continue
                        idcg += _idcg[j + 1]

                        rank = 1 + index(argsort[i], t)
                        if rank == 0:
                            rank = 21

                        if rank <= 10:
                            recall10 += 1
                            dcg10 += 1 / math.log(rank + 1, 2)
                        if rank <= 20:
                            recall20 += 1
                            dcg20 += 1 / math.log(rank + 1, 2)

                        j += 1

                    results[key]["recall10"] += recall10 / target.shape[1]
                    results[key]["recall20"] += recall20 / target.shape[1]
                    results[key]["ndcg10"] += dcg10 / idcg
                    results[key]["ndcg20"] += dcg20 / idcg
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
    # if opt.mtl:  # if model_name.startswith("fapat") or opt.mtl:
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
            adjs[key].append(pickle.load(open(f"{datadir}/Amazon/{prefix}{key}_adj_{n_sample}.pkl", "rb")))
            nums[key].append(pickle.load(open(f"{datadir}/Amazon/{prefix}{key}_num_{n_sample}.pkl", "rb")))
            # patterns[key].append(read_patterns(f"{datadir}/Amazon/{prefix}{key}.gspan") # [(vlabels, elabels, cnt)])
        if os.path.exists(f"{datadir}/{dataset}/{prefix}test.pkl"):
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
        collate_fn=partial(period_batchify, prepad=prepad, return_adj=return_adj, return_alias=return_alias, return_shortcut=return_shortcut, return_hete=return_hete),
        pin_memory=False
    )

    opt.lr_dc_step = opt.lr_dc_step * len(test_dataloader) * 10 * len(dates)
    model = model_class(opt, n_nodes=n_nodes, adjs=adjs, nums=nums, patterns=patterns)

    # checkpoint_dir = dataset + "-"
    checkpoint_dir = "Amazon_1-"
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

    # checkpoint_dir = dataset + "-"
    # if category:
    #     checkpoint_dir += category + "-"
    # checkpoint_dir += opt.dates
    # if not os.path.exists(f"checkpoints/{checkpoint_dir}"):
    #     os.makedirs(f"checkpoints/{checkpoint_dir}")
    global logger
    logger = init_logger(
        filename=
        f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_period_eval.log"
    )
    logger.info(
        f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_mrr20.pt"
    )
    result = period_test(model, test_dataloader)["sequence"]
    logger.info(f"Evaluate Result:")
    logger.info("\tRecall@10:\t%.4f\tNDCG@10:\t%.4f" % (result["recall10"], result["ndcg10"]))
    logger.info("\tRecall@20:\t%.4f\tNDCG@20:\t%.4f" % (result["recall20"], result["ndcg20"]))


def main_public(opt):
    model_name = opt.model
    datadir = opt.datadir
    dataset = opt.dataset.split("_period_")[0]
    n_sample = opt.n_sample
    batch_size = opt.batch_size
    hidden_dim = opt.hidden_dim
    n_iter = opt.n_iter
    seed = opt.seed
    prepad = opt.prepad
    mtl = "mtl_" if opt.mtl else ""
    attributes = "+".join(opt.attributes)
    # if opt.mtl:  # if model_name.startswith("fapat") or opt.mtl:
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

    dataset = opt.dataset
    test_data = pickle.load(open(f"{datadir}/{dataset}/test.pkl", "rb"))
    test_data = Data(test_data, opt.attributes, opt.max_len)

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        num_workers=1,
        batch_sampler=BucketSampler(
            test_data, batch_size=batch_size, shuffle=False, seed=len(test_data), drop_last=False
        ),
        collate_fn=partial(period_batchify, prepad=prepad, return_adj=return_adj, return_alias=return_alias, return_shortcut=return_shortcut, return_hete=return_hete),
        pin_memory=False
    )
    model = model_class(opt, n_nodes=n_nodes, adjs=adjs, nums=nums, patterns=patterns)

    checkpoint_dir = dataset.split("_period_")[0]
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
        f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_period_eval.log"
    )
    logger.info(
        f"checkpoints/{checkpoint_dir}/{mtl}{model_name}_attributes{attributes}batch{batch_size}_hidden{hidden_dim}_niter{n_iter}_seed{seed}_mrr20.pt"
    )
    result = period_test(model, test_dataloader)["sequence"]
    logger.info(f"Evaluate Result:")
    logger.info("\tRecall@10:\t%.4f\tNDCG@10:\t%.4f" % (result["recall10"], result["ndcg10"]))
    logger.info("\tRecall@20:\t%.4f\tNDCG@20:\t%.4f" % (result["recall20"], result["ndcg20"]))


if __name__ == "__main__":
    opt = get_option()
    init_seed(opt.seed)

    if opt.dataset.startswith("Amazon"):
        main_amazon(opt)
    else:
        main_public(opt)
