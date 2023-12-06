import argparse
from config import *

def get_option():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        type=str, default="../datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str, default="Tmall",
        help="diginetica/Nowplaying/Tmall"
    )
    parser.add_argument(
        "--attributes",
        nargs="+", default=[]
    )
    parser.add_argument(
        "--category",
        type=str, default="books"
    )
    parser.add_argument(
        "--dates",
        type=str, default="20220501-20220630"
    )
    parser.add_argument(
        "--seed",
        type=int, default=2022
    )
    parser.add_argument(
        "--hidden_dim",
        type=int, default=128
    )
    parser.add_argument(
        "--epoch",
        type=int, default=20
    )
    parser.add_argument(
        "--activate",
        type=str, default="relu"
    )
    parser.add_argument(
        "--n_sample_model",
        type=int, default=12
    )
    parser.add_argument(
        "--n_sample",
        type=int, default=12
    )
    parser.add_argument(
        "--n_pattern",
        type=int, default=30000
    )
    parser.add_argument(
        "--max_len",
        type=int, default=100
    )
    parser.add_argument(
        "--batch_size",
        type=int, default=100
    )
    parser.add_argument(
        "--lr",
        type=float, default=0.001,
        help="learning rate."
    )
    parser.add_argument(
        "--lr_dc",
        type=float, default=0.1,
        help="learning rate decay."
    )
    parser.add_argument(
        "--lr_dc_step",
        type=int, default=3,
        help="the number of steps after which the learning rate decay"
    )
    parser.add_argument(
        "--l2",
        type=float, default=1e-5,
        help="l2 penalty"
    )
    parser.add_argument(
        "--clip",
        type=float, default=-1.0,
        help="gradient clipping"
    )
    parser.add_argument(
        "--model",
        type=str, default="gce_gnn",
        help="gnn/sbrt/gce_gnn/mix_sbrt/hier_sbrt"
    )
    parser.add_argument(
        "--rnn_type",
        type=str, default="lstm",
        help="lstm/gru/rnn"
    )
    parser.add_argument(
        "--gnn_type",
        type=str, default="gat",
        help="gat/gcn/gin/rgat/rgcn/rgin"
    )
    parser.add_argument(
        "--mem_type",
        type=str, default="xl",
        help="xl/knn/none"
    )
    parser.add_argument(
        "--mem_size",
        type=int, default=100)
    parser.add_argument(
        "--mem_share", action="store_true",
        help="memory share in batch"
    )
    parser.add_argument(
        "--n_iter",
        type=int, default=1)
    parser.add_argument(
        "--n_bucket",
        type=int, default=8)
    parser.add_argument(
        "--n_head",
        type=int, default=4)
    parser.add_argument(
        "--topk",
        type=int, default=2)
    parser.add_argument(
        "--dropout_gnn",
        type=float, default=0.2,
        help="Dropout rate for graph neural networks"
    )  # [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    parser.add_argument(
        "--dropout_attn",
        type=float, default=0.2,
        help="Dropout rate for attention"
    )  # [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    parser.add_argument(
        "--dropout_ff",
        type=float, default=0.2,
        help="Dropout rate for feed-forward layers"
    )  # [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    parser.add_argument(
        "--dropout_local",
        type=float, default=0.2,
        help="Dropout rate for local representations"
    )  # [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    parser.add_argument(
        "--dropout_global",
        type=float, default=0.2,
        help="Dropout rate for global representations"
    )  # [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    parser.add_argument(
        "--dropout_pred",
        type=float, default=0.2,
        help="Dropout rate for predictors"
    )  # [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    parser.add_argument(
        "--prepad", action="store_true",
        help="pre padding"
    )
    parser.add_argument(
        "--mtl", action="store_true",
        help="multi-task learning"
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="subsampling for validation"
    )
    parser.add_argument(
        "--valid_portion",
        type=float, default=0.1,
        help="split the portion"
    )
    parser.add_argument(
        "--patience",
        type=int, default=5
    )
    parser.add_argument(
        "--lazy",
        action="store_true"
    )
    parser.add_argument(
        "--main_port",
        type=int, default=-1
    )
    parser.add_argument(
        "--local_rank",
        type=int, default=0
    )

    opt = parser.parse_args()

    return opt


def get_model_config(model_name):
    if model_name == FPMC_MODEL:
        from fpmc import FPMC
        model_class = FPMC
        return_adj, return_alias = NONE_FLAG, False
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = False, False
    elif model_name == RNN_MODEL:
        from rnn import RNN
        model_class = RNN
        return_adj, return_alias = NONE_FLAG, False
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = False, False
    elif model_name == GRU4REC_MODEL:
        from gru4rec import GRU4Rec
        model_class = GRU4Rec
        return_adj, return_alias = NONE_FLAG, False
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = False, False
    elif model_name == NARM_MODEL:
        from narm import NARM
        model_class = NARM
        return_adj, return_alias = NONE_FLAG, False
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = False, False
    elif model_name == STAMP_MODEL:
        from stamp import STAMP
        model_class = STAMP
        return_adj, return_alias = NONE_FLAG, False
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = False, False
    elif model_name == REPEATNET_MODEL:
        from repeatnet import RepeatNet
        model_class = RepeatNet
        return_adj, return_alias = NONE_FLAG, False
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = False, False
    elif model_name == S3REC_MODEL:
        from s3rec import S3Rec
        model_class = S3Rec
        return_adj, return_alias = NONE_FLAG, False
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = False, False
    elif model_name == CSRM_MODEL:
        from csrm import CSRM
        model_class = CSRM
        return_adj, return_alias = NONE_FLAG, False
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = False, False
    elif model_name == M2TREC_MODEL:
        from m2trec import M2TREC
        model_class = M2TREC
        return_adj, return_alias = NONE_FLAG, False
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = False, False
    elif model_name == SBRT_MODEL:
        from sbrt import SBRT
        model_class = SBRT
        return_adj, return_alias = NONE_FLAG, False
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = False, False
    elif model_name == GRAPHFORMER_MODEL:
        from graph_former import GRAPHFORMER
        model_class = GRAPHFORMER
        return_adj, return_alias = DENSE_FLAG, True
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = False, False
    elif model_name == GNN_MODEL:
        from gnn import GNN        
        model_class = GNN
        return_adj, return_alias = SPARSE_FLAG, True
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = False, False
    elif model_name == SRGNN_MODEL:
        from srgnn import SRGNN
        model_class = SRGNN
        return_adj, return_alias = DENSE_FLAG, True
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = False, False
    elif model_name == GCSAN_MODEL:
        from gcsan import GCSAN
        model_class = GCSAN
        return_adj, return_alias = DENSE_FLAG, True
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = False, False
    elif model_name == GCEGNN_MODEL:
        from gce_gnn import GCEGNN
        model_class = GCEGNN
        return_adj, return_alias = DENSE_FLAG, True
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = True, False
    elif model_name == DHCN_MODEL:
        from dhcn import DHCN
        model_class = DHCN
        return_adj, return_alias = DENSE_FLAG, True
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = True, False
    elif model_name == NCL_MODEL:
        from ncl import NCL
        model_class = NCL
        return_adj, return_alias = DENSE_FLAG, True
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = True, False
    elif model_name == LESSR_MODEL:
        from lessr import LESSR
        model_class = LESSR
        return_adj, return_alias = DENSE_FLAG, True
        return_shortcut, return_hete = True, False
        load_adj, load_pattern = False, False
    elif model_name == MSGIFSR_MODEL:
        from msgifsr import MSGIFSR
        model_class = MSGIFSR
        return_adj, return_alias = DENSE_FLAG, True
        return_shortcut, return_hete = True, True
        load_adj, load_pattern = False, False
    elif model_name == FAPAT_MODEL:
        from fapat import FAPAT
        model_class = FAPAT
        return_adj, return_alias = DENSE_FLAG, True
        return_shortcut, return_hete = False, False
        load_adj, load_pattern = False, True
    else:
        raise ValueError("Unknown model: {}".format(model_name))
    
    return model_class, return_adj, return_alias, return_shortcut, return_hete, load_adj, load_pattern
