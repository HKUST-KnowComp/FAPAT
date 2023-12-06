import os
import sys

f_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f_path)

import argparse
import time
import csv
import pickle5 as pickle
import operator
import datetime
import os
import datetime
import random
import numpy as np
from tqdm import tqdm
from collections import Counter
from itertools import chain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str, default="diginetica/train-item-views.csv"
    )
    parser.add_argument(
        "--categories",
        type=str, default="diginetica/product-categories.csv"
    )
    parser.add_argument(
        "--seed",
        type=int, default=2022
    )
    parser.add_argument(
        "--steps",
        nargs="+", default=["3", "5", "10"]
    )
    opt = parser.parse_args()
    print(opt)

    if not os.path.exists("diginetica"):
        os.makedirs("diginetica")

    print("-- Starting @ %ss" % datetime.datetime.now())

    sess_user = dict()
    sess_date = dict()
    sess_sequences = dict()
    n_clicks = 0
    with open(opt.dataset, "r") as f:
        reader = csv.DictReader(f, delimiter=";")
        cur_id = -1
        cur_user = None
        cur_date = []
        cur_time = []
        cur_seq = []
        for line in reader:
            sess_id = line["sessionId"]
            if cur_date and cur_id != sess_id:
                indices = np.argsort(np.asarray(cur_time, dtype=np.int64))
                sess_user[cur_id] = cur_user
                sess_date[cur_id] = datetime.datetime.fromisoformat(cur_date[indices[-1]]).timestamp()
                sess_sequences[cur_id] = _cur_seq = []
                for idx in indices:
                    item = cur_seq[idx]
                    _cur_seq.append(item)
                cur_date = []
                cur_time = []
                cur_seq = []
            cur_id = sess_id
            item = line["itemId"]
            cur_user = line["userId"]
            cur_time.append(int(line["timeframe"]))
            cur_date.append(line["eventdate"])
            cur_seq.append(item)
            n_clicks += 1

        # for i in list(sess_sequences):
        #     sorted_sequence = sorted(sess_sequences[i], key=operator.itemgetter(1))
        #     sess_sequences[i] = [c[0] for c in sorted_sequence]
        indices = np.argsort(np.asarray(cur_time, dtype=np.int64))
        sess_user[cur_id] = cur_user
        sess_date[cur_id] = datetime.datetime.fromisoformat(cur_date[indices[-1]]).timestamp()
        sess_sequences[cur_id] = _cur_seq = []
        for idx in indices:
            item = cur_seq[idx]
            _cur_seq.append(item)
    print("-- Reading data @ %ss" % datetime.datetime.now())
    print("n_clicks:", n_clicks)

    # Filter out length 1 sessions
    for sess_id in list(sess_sequences):
        if len(sess_sequences[sess_id]) == 1:
            del sess_user[sess_id]
            del sess_date[sess_id]
            del sess_sequences[sess_id]

    # Count number of times each item appears
    item_counts = Counter(chain.from_iterable(sess_sequences.values()))
    sorted_counts = sorted(item_counts.items(), key=operator.itemgetter(1))

    length = len(sess_sequences)
    for sess_id in list(sess_sequences):
        cur_seq = sess_sequences[sess_id]
        seq = []
        for idx, item in enumerate(cur_seq):
            if item_counts[item] >= 5:
                seq.append(item)
        if len(seq) < 2:
            del sess_user[sess_id]
            del sess_date[sess_id]
            del sess_sequences[sess_id]
        else:
            sess_sequences[sess_id] = seq

    # Split out test set based on dates
    # the last 7 days for test
    # the last 14-7 days for validation
    sess = list(sess_date.items())
    maxdate = max(sess_date.values())
    valid_splitdate = maxdate - 86400 * 14
    test_splitdate = maxdate - 86400 * 7
    train_sessions = filter(lambda x: x[1] < valid_splitdate, sess)
    valid_sessions = filter(lambda x: valid_splitdate <= x[1] < test_splitdate, sess)
    test_sessions = filter(lambda x: x[1] >= test_splitdate, sess)
    train_sess_ids = list(map(operator.itemgetter(0), sorted(train_sessions, key=operator.itemgetter(1))))
    valid_sess_ids = list(map(operator.itemgetter(0), sorted(valid_sessions, key=operator.itemgetter(1))))
    test_sess_ids = list(map(operator.itemgetter(0), sorted(test_sessions, key=operator.itemgetter(1))))

    # Split out test set based on disjoint user sets
    # users = sorted(set(sess_user.values()))
    # random.seed(opt.seed)
    # random.shuffle(users)
    # user_sep1_idx = int(len(users) * 0.8)
    # user_sep2_idx = int(len(users) * 0.9)
    # train_users = set(users[:user_sep1_idx])
    # valid_users = set(users[user_sep1_idx:user_sep2_idx])
    # test_users = set(users[user_sep2_idx:])
    # if "NA" in train_users:
    #     train_users.remove("NA")
    # if "NA" in valid_users:
    #     valid_users.remove("NA")
    # if "NA" in test_users:
    #     test_users.add("NA")
    # print(
    #     "train users - valid users - test users :",
    #     len(train_users) + 1, "-",
    #     len(valid_users) + 1, "-",
    #     len(test_users) + 1
    # )

    # train_sess_ids = [x[0] for x in sess_user.items() if x[1] in train_users]
    # valid_sess_ids = [x[0] for x in sess_user.items() if x[1] in valid_users]
    # test_sess_ids = [x[0] for x in sess_user.items() if x[1] in test_users]
    # nan_sess_ids = [x[0] for x in sess_user.items() if x[1] == "NA"]
    # random.shuffle(nan_sess_ids)
    # nan_sess_sep1_idx = int(len(nan_sess_ids) * 0.8)
    # nan_sess_sep2_idx = int(len(nan_sess_ids) * 0.9)
    # train_sess_ids = train_sess_ids + nan_sess_ids[:nan_sess_sep1_idx]
    # valid_sess_ids = valid_sess_ids + nan_sess_ids[nan_sess_sep1_idx:nan_sess_sep2_idx]
    # test_sess_ids = test_sess_ids + nan_sess_ids[nan_sess_sep2_idx:]
    print(
        "train sessions - valid sessions - test sessions :", len(train_sess_ids), "-", len(valid_sess_ids), "-",
        len(test_sess_ids)
    )

    # Sort sessions by date
    train_sess_ids = sorted(train_sess_ids, key=lambda sess_id: sess_date[sess_id])
    valid_sess_ids = sorted(valid_sess_ids, key=lambda sess_id: sess_date[sess_id])
    test_sess_ids = sorted(test_sess_ids, key=lambda sess_id: sess_date[sess_id])
    print("-- Splitting train set, valid set, and test set @ %ss" % datetime.datetime.now())

    # Choosing item count >=5 gives approximately the same number of items as reported in paper
    item2idx = dict()
    item2category = dict()
    category2idx = dict()


    def obtain_mappings(train_sess_ids):
        for sess_id in train_sess_ids:
            seq = sess_sequences[sess_id]
            for item in seq:
                if item not in item2idx:
                    item2idx[item] = len(item2idx) + 1

        with open(opt.categories, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            # itemId;categoryId
            for line in tqdm(reader):
                item, cat = line["itemId"], line["categoryId"]
                if item in item2idx:
                    if item not in item2category:
                        item2category[item] = cat
                    if cat not in category2idx:
                        category2idx[cat] = len(category2idx) + 1


    obtain_mappings(train_sess_ids)
    print("item_ctr:", len(item2idx))
    print("category_ctr:", len(category2idx))


    # Convert test sessions to sequences, ignoring items that do not appear in training set
    def obtian_data(sess_ids):
        out_ids = []
        out_dates = []
        out_sequences = []
        out_categories = []
        for sess_id in sess_ids:
            date = sess_date[sess_id]
            sequence = sess_sequences[sess_id]
            out_seq = []
            out_cat = []
            for item in sequence:
                if item in item2idx:
                    out_seq.append(item2idx[item])
                    if item in item2category:
                        cat = category2idx[item2category[item]]
                    else:
                        cat = 0
                    out_cat.append(cat)
            if len(out_seq) < 2:
                continue
            out_ids.append(sess_id)
            out_dates.append(date)
            out_sequences.append(out_seq)
            out_categories.append(out_cat)
        return out_ids, out_dates, out_sequences, {"category": out_categories}


    train_ids, train_dates, train_sequences, train_attributes = obtian_data(train_sess_ids)
    valid_ids, valid_dates, valid_sequences, valid_attributes = obtian_data(valid_sess_ids)
    test_ids, test_dates, test_sequences, test_attributes = obtian_data(test_sess_ids)

    l = 0
    for seq in train_sequences:
        l += len(seq)
    for seq in valid_sequences:
        l += len(seq)
    for seq in test_sequences:
        l += len(seq)
    print("avg length:", l / (len(train_sequences) + len(valid_sequences) + len(test_sequences)))
    print("all:", l)

    with open("diginetica/meta.tsv", "w") as ff:
        csv_writer = csv.writer(ff, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["artist", "category"])
        for art, idx in item2idx.items():
            if art in item2category:
                cat = category2idx[item2category[art]]
            else:
                cat = 0
            csv_writer.writerow([str(idx), str(cat)])

    with open("diginetica/sequence.tsv", "w") as ff:
        csv_writer = csv.writer(ff, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["artist", "index"])
        for art, idx in item2idx.items():
            csv_writer.writerow([art, str(idx)])
    pickle.dump(train_sequences, open("diginetica/all_train_sequence.pkl", "wb"))

    with open("diginetica/category.tsv", "w") as ff:
        csv_writer = csv.writer(ff, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["category", "index"])
        for cat, idx in category2idx.items():
            csv_writer.writerow([cat, str(idx)])
    pickle.dump(train_attributes["category"], open("diginetica/all_train_category.pkl", "wb"))


    def process_sequences(ids, dates, sequences, attributes, step=1):
        out_ids = []
        out_dates = []
        out_sequences = []
        out_attributes = {k: [] for k in attributes}
        out_seq_labels = []
        out_attr_labels = {k: [] for k in attributes}
        for idx, _id in enumerate(ids):
            date = dates[idx]
            seq = sequences[idx]
            if len(seq) <= step:
                continue
            for i in range(1, len(seq) - step + 1):
                out_ids.append(str(_id) + "-" + str(i))
                out_dates.append(date)
                # out_sequences.append(seq[:-i])
                # out_seq_labels.append(seq[-i])
                out_sequences.append(seq[:i])
                out_seq_labels.append(seq[i:i + step])
                for k, v in attributes.items():
                    out_attributes[k].append(v[idx][:i])
                    out_attr_labels[k].append(v[idx][i:i + step])

        input = {"sequence": out_sequences}
        target = {"sequence": out_seq_labels}
        for k in attributes:
            input[k] = out_attributes[k]
            target[k] = out_attr_labels[k]
        return out_ids, out_dates, input, target

    for step in opt.steps:
        step = int(step)
        if not os.path.exists("diginetica_period_{}".format(step)):
            os.makedirs("diginetica_period_{}".format(step))
        train = process_sequences(train_ids, train_dates, train_sequences, train_attributes, step)
        valid = process_sequences(valid_ids, valid_dates, valid_sequences, valid_attributes, step)
        test = process_sequences(test_ids, test_dates, test_sequences, test_attributes, step)

        print("train - valid - test:", len(train[0]), "-", len(valid[0]), "-", len(test[0]))

        keys = ["ids", "dates", "input", "target"]
        pickle.dump(dict(zip(keys, train)), open("diginetica_period_{}/train.pkl".format(step), "wb"))
        pickle.dump(dict(zip(keys, valid)), open("diginetica_period_{}/valid.pkl".format(step), "wb"))
        pickle.dump(dict(zip(keys, test)), open("diginetica_period_{}/test.pkl".format(step), "wb"))

# Namespace(categories='diginetica/product-categories.csv', dataset='diginetica/train-item-views.csv', seed=2022)
# -- Starting @ 2022-10-05 21:15:40.823531s
# -- Reading data @ 2022-10-05 21:15:43.924243s
# n_clicks: 1235380
# train users - valid users - test users : 46098 - 5763 - 5764
# train sessions - valid sessions - test sessions : 163794 - 20490 - 20505
# -- Splitting train set, valid set, and test set @ 2022-10-05 21:15:44.832378s
# 184047it [00:00, 923055.02it/s]
# item_ctr: 43074
# category_ctr: 995
# avg length: 4.850399738228845
# all: 993163
# train - valid - test: 630789 - 78708 - 78907