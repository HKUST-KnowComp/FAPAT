import os
import sys
f_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f_path)

import re
import os
import numpy as np


user_rex1 = re.compile(r'train users . valid users . test users\s*:\s*(\d+) . (\d+) . (\d+)')
user_rex2 = re.compile(r'train users . test users\s*:\s*(\d+) . (\d+)')
session_rex1 = re.compile(r'train sessions . valid sessions . test sessions\s*:\s*(\d+) . (\d+) . (\d+)')
session_rex2 = re.compile(r'train sessions . test sessions\s*:\s*(\d+) . (\d+) . (\d+)')
length_rex = re.compile(r'avg length: ([\d\.]+)')
all_rex = re.compile(r'all: (\d+)')
size_rex1 = re.compile(r'.*?train . valid . test\s*:\s*(\d+) . (\d+) . (\d+)')
size_rex2 = re.compile(r'.*?train . test\s*:\s*(\d+) . (\d+)')

def iter_files(path):
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            yield os.path.join(dirpath, filename)

def read_log(path):
    users = []
    sessions = []
    sizes = []
    length = 0.0
    all = 0.0
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            m = user_rex1.match(line)
            if m:
                users.append([int(x) for x in m.groups()])
                continue
            m = user_rex2.match(line)
            if m:
                users.append([int(x) for x in m.groups()])
                continue
            m = session_rex1.match(line)
            if m:
                sessions.append([int(x) for x in m.groups()])
                continue
            m = session_rex2.match(line)
            if m:
                sessions.append([int(x) for x in m.groups()])
                continue
            m = length_rex.match(line)
            if m:
                length = float(m.group(1))
                continue
            m = all_rex.match(line)
            if m:
                all = int(m.group(1))
                continue
            m = size_rex1.match(line)
            if m:
                sizes.append([int(x) for x in m.groups()])
                continue
            m = size_rex2.match(line)
            if m:
                sizes.append([int(x) for x in m.groups()])
                continue
    return users, sessions, sizes, length, all

results = []
files = sorted([file_name for file_name in iter_files("Amazon") if file_name.endswith(".log") and "electronics" in file_name])
for file_name in files:
    users, sessions, sizes, length, all = read_log(file_name)
    results.append((all, np.array(sessions).sum(), length, *(np.array(users).reshape(3).tolist()), *(np.array(sizes).reshape(-1, 3)[0].tolist())))
results = np.array(results)
print("clicks\tsessions\tavg length\ttrain users:sessions\tvalid users:sessions\ttest users:sessions")
print(results[:, 0].sum().astype(np.int64), end="\t")
print(results[:, 1].sum().astype(np.int64), end="\t")
print(((results[:, 2] * results[:, 1]).sum() / (results[:, 1].sum())).round(3), end="\t")
print("\t".join(map(lambda x: str(x[0]) + ":" + str(x[1]), list(zip(results[:, [3,4,5]].sum(0).astype(np.int64), results[:, [6,7,8]].sum(0).astype(np.int64))))))
