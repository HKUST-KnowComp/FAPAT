unzip parsemis.zip 

# sample neighbors for graph neural networks
python build_adj.py --dataset Tmall --n_sample 12 --attributes brand category
# write the sampled neighbors to the file in the format "t # graph_id\nv node_id node_label\n ... \ne source_id target_id edge_label\n ..."
python build_lg.py --dataset Tmall --n_sample 12 --attributes brand category
# use gspan to mine frequent subgraphs
python gspan.py --dataset Tmall --min_freq 2 --attributes brand category

# similar steps for diginetica
python build_adj.py --dataset diginetica --n_sample 12 --attributes category
python build_lg.py --dataset diginetica --n_sample 12 --attributes category
python gspan.py --dataset diginetica --min_freq 2 --attributes category

# filter loose patterns via vf2 algorithm
python merge_pattern.py --dataset Tmall --attribute brand
python merge_pattern.py --dataset Tmall --attribute category
python merge_pattern.py --dataset diginetica --attribute category
