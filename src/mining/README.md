### Data Mining

We provide scripts to sample neighbors and construct attribute patterns. You can also run `build.sh` to reprocude experiments.

#### Neighbor Sampling

We follow the setting of GCE-GNN that one node (i.e. item) only perserves at most 12 neighbors, where 12 is an empirical hyperparameter.
For example, this sampling for ```Tmall``` can be done by:
```bash
python build_adj.py --dataset Tmall --n_sample 12
```

We also extend the sampling to attribute graphs:
```bash
python build_adj.py --dataset Tmall --n_sample 12 --attributes brand category
```

#### Pattern Mining

We utilize the gSpan algorithm to mine attribute patterns. Before that, we need to process sequences to a specific data format by `build_lg.py`. And then we unzip ```parsemis.zip``` and run `gspan.py` to extract patterns:
```bash
python build_lg.py --dataset Tmall --n_sample 12 --attributes brand category
python gspan.py --dataset --min_freq 2 Tmall --attributes brand category
```