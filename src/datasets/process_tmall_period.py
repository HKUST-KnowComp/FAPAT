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
import random
from collections import Counter
from itertools import chain
from tqdm import tqdm


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str, default="Tmall/dataset15.csv"
    )
    parser.add_argument(
        "--items",
        type=str, default="Tmall/user_log_format1.csv"
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

    if not os.path.exists("Tmall"):
        os.makedirs("Tmall")

    if not os.path.exists("Tmall/tmall_data.csv"):
        with open("Tmall/tmall_data.csv", "w") as tmall_data:
            with open(opt.dataset, "r") as tmall_file:
                header = tmall_file.readline()
                tmall_data.write(header)
                for line in tmall_file:
                    data = line[:-1].split("\t")
                    if int(data[2]) > 120000:
                        break
                    tmall_data.write(line)

    sess_user = dict()
    sess_date = dict()
    sess_sequences = dict()
    n_clicks = 0
    print("-- Starting @ %ss" % datetime.datetime.now())
    with open("Tmall/tmall_data.csv", "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cur_id = -1
        cur_user = None
        cur_date = None
        for line in reader:
            sess_id = int(line["SessionId"])
            if cur_date and not cur_id == sess_id:
                sess_user[cur_id] = cur_user
                sess_date[cur_id] = cur_date
            cur_id = sess_id

            item = line["ItemId"]
            cur_user = line["UserId"]
            cur_date = float(line["Time"])

            if sess_id in sess_sequences:
                sess_sequences[sess_id].append(item)
            else:
                sess_sequences[sess_id] = [item]
            n_clicks += 1
        sess_user[cur_id] = cur_user
        sess_date[cur_id] = cur_date
    print("-- Reading data @ %ss" % datetime.datetime.now())

    # Filter out length 1 sessions
    for sess_id in list(sess_sequences):
        if len(sess_sequences[sess_id]) == 1:
            del sess_sequences[sess_id]
            del sess_date[sess_id]

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
        if len(seq) < 2 or len(seq) > 40:
            del sess_user[sess_id]
            del sess_date[sess_id]
            del sess_sequences[sess_id]
        else:
            sess_sequences[sess_id] = seq

    # Split out test set based on dates
    # the last of 100 seconds for test
    sess = list(sess_date.items())
    maxdate = max(sess_date.values())
    valid_splitdate = maxdate - 200
    test_splitdate = maxdate - 100
    train_sess = filter(lambda x: x[1] < valid_splitdate, sess)
    valid_sess = filter(lambda x: valid_splitdate <= x[1] < test_splitdate, sess)
    test_sess = filter(lambda x: x[1] >= test_splitdate, sess)

    train_sess_ids = list(map(operator.itemgetter(0), sorted(train_sess, key=operator.itemgetter(1))))
    valid_sess_ids = list(map(operator.itemgetter(0), sorted(valid_sess, key=operator.itemgetter(1))))
    test_sess_ids = list(map(operator.itemgetter(0), sorted(test_sess, key=operator.itemgetter(1))))

    # Split out test set based on disjoint user sets
    # users = sorted(set(sess_user.values()))
    # random.seed(opt.seed)
    # random.shuffle(users)
    # user_sep1_idx = int(len(users) * 0.8)
    # user_sep2_idx = int(len(users) * 0.9)
    # train_users = set(users[:user_sep1_idx])
    # valid_users = set(users[user_sep1_idx:user_sep2_idx])
    # test_users = set(users[user_sep2_idx:])
    # print("train users - valid users - test users :", len(train_users), "-", len(valid_users), "-", len(test_users))

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
    item2brand = dict()
    brand2idx = dict()


    def obtian_mappings(train_sess_ids):
        for sess_id in train_sess_ids:
            sequence = sess_sequences[sess_id]
            for item in sequence:
                if item not in item2idx:
                    item2idx[item] = len(item2idx) + 1

        with open(opt.items, "r") as f:
            reader = csv.DictReader(f, delimiter=",")
            # item_id,cat_id,seller_id,brand_id
            for line in tqdm(reader):
                item, cat, brand = line["item_id"], line["cat_id"], line["brand_id"]
                if item in item2idx:
                    if item not in item2category:
                        item2category[item] = cat
                    if cat not in category2idx:
                        category2idx[cat] = len(category2idx) + 1
                    if item not in item2brand:
                        item2brand[item] = brand
                    if brand not in brand2idx:
                        brand2idx[brand] = len(brand2idx) + 1


    obtian_mappings(train_sess_ids)
    print("item_ctr:", len(item2idx))
    print("category_ctr:", len(item2category), len(category2idx))
    print("brand_ctr:", len(item2brand), len(brand2idx))


    # Convert test sessions to sequences, ignoring items that do not appear in training set
    def obtian_data(sess_ids):
        out_ids = []
        out_sequences = []
        out_dates = []
        out_categories = []
        out_brands = []
        for sess_id in sess_ids:
            date = sess_date[sess_id]
            sequence = sess_sequences[sess_id]
            out_seq = []
            out_cat = []
            out_brand = []
            for item in sequence:
                if item in item2idx:
                    out_seq.append(item2idx[item])
                    if item in item2category:
                        cat = category2idx[item2category[item]]
                    else:
                        cat = 0
                    out_cat.append(cat)
                    if item in item2brand:
                        brand = brand2idx[item2brand[item]]
                    else:
                        brand = 0
                    out_brand.append(brand)
            if len(out_seq) < 2:
                continue
            out_ids.append(sess_id)
            out_dates.append(date)
            out_sequences.append(out_seq)
            out_categories.append(out_cat)
            out_brands.append(out_brand)
        return out_ids, out_dates, out_sequences, {"category": out_categories, "brand": out_brands}


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

    with open("Tmall/meta.tsv", "w") as ff:
        csv_writer = csv.writer(ff, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["item", "category", "brand"])
        for item, idx in item2idx.items():
            if item in item2category:
                cat = category2idx[item2category[item]]
            else:
                cat = 0
            if item in item2brand:
                brand = brand2idx[item2brand[item]]
            else:
                brand = 0
            csv_writer.writerow([str(idx), str(cat), str(brand)])

    with open("Tmall/sequence.tsv", "w") as ff:
        csv_writer = csv.writer(ff, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["item", "index"])
        for item, idx in item2idx.items():
            csv_writer.writerow([item, str(idx)])
    pickle.dump(train_sequences, open("Tmall/all_train_sequence.pkl", "wb"))

    with open("Tmall/category.tsv", "w") as ff:
        csv_writer = csv.writer(ff, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["category", "index"])
        for cat, idx in category2idx.items():
            csv_writer.writerow([cat, str(idx)])
    pickle.dump(train_attributes["category"], open("Tmall/all_train_category.pkl", "wb"))

    with open("Tmall/brand.tsv", "w") as ff:
        csv_writer = csv.writer(ff, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(["brand", "index"])
        for cat, idx in brand2idx.items():
            csv_writer.writerow([cat, str(idx)])
    pickle.dump(train_attributes["brand"], open("Tmall/all_train_brand.pkl", "wb"))


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
            for i in range(1, len(seq)- step + 1):
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
        if not os.path.exists("Tmall_period_{}".format(step)):
            os.makedirs("Tmall_period_{}".format(step))
        train = process_sequences(train_ids, train_dates, train_sequences, train_attributes, step)
        valid = process_sequences(valid_ids, valid_dates, valid_sequences, valid_attributes, step)
        test = process_sequences(test_ids, test_dates, test_sequences, test_attributes, step)

        print("train - valid - test:", len(train[0]), "-", len(valid[0]), "-", len(test[0]))

        keys = ["ids", "dates", "input", "target"]
        pickle.dump(dict(zip(keys, train)), open("Tmall_period_{}/train.pkl".format(step), "wb"))
        pickle.dump(dict(zip(keys, valid)), open("Tmall_period_{}/valid.pkl".format(step), "wb"))
        pickle.dump(dict(zip(keys, test)), open("Tmall_period_{}/test.pkl".format(step), "wb"))

# -- Starting @ 2022-10-05 21:49:47.991262s
# -- Reading data @ 2022-10-05 21:49:49.204988s
# train users - valid users - test users : 6060 - 758 - 758
# train sessions - valid sessions - test sessions : 53003 - 6308 - 7064
# -- Splitting train set, valid set, and test set @ 2022-10-05 21:49:49.535733s
# 54925330it [01:21, 672818.50it/s]
# item_ctr: 39768
# category_ctr: 39768 821
# brand_ctr: 39768 4304
# avg length: 6.649397736581814
# all: 438315
# train - valid - test: 303181 - 33735 - 35481