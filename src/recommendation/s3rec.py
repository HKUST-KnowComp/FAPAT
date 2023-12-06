import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import *
from utils import *


class S3Rec(nn.Module):
    """_summary_: implementation of S3Rec
    reference:  Kun Zhou, Hui Wang, Wayne Xin Zhao, Yutao Zhu, Sirui Wang, Fuzheng Zhang, Zhongyuan Wang, and Ji-Rong Wen. 2020. S^3-Rec: Self-Supervised Learning for Sequential Recommendation with Mutual Information Maximizatino. In CIKM. 1893-1902.
    code reference: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/s3rec.py
    """

    def __init__(self, opt, **kwargs):
        super(S3Rec, self).__init__()

        self.opt = opt

        self.batch_size = opt.batch_size
        self.max_len = opt.max_len
        self.dim = opt.hidden_dim
        self.n_iter = opt.n_iter
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.clip = opt.clip
        self.mtl = opt.mtl

        self.n_nodes = kwargs["n_nodes"]
        if isinstance(self.n_nodes, int):
            self.n_nodes = {SEQUENCE: self.n_nodes}

        # Transformer
        self.transformer = Transformer(
            dim=opt.hidden_dim,
            depth=opt.n_iter,
            dim_head=opt.hidden_dim // opt.n_head,
            heads=opt.n_head,
            buckets=opt.n_bucket,
            bidirectional=False,
            attn_dropout=opt.dropout_attn,
            ff_mult=4,
            ff_dropout=opt.dropout_ff,
        )

        # Item representation & Position representation & Feature representation
        self.pos_embedding = nn.Embedding(self.max_len + 2, self.dim)
        self.embeddings = nn.ModuleDict()
        for key in self.n_nodes:
            if key == SEQUENCE:
                self.embeddings[key] = nn.Embedding(self.n_nodes[key] + 1, self.dim, padding_idx=0) # -1 for mask
            else:
                self.embeddings[key] = nn.Embedding(self.n_nodes[key], self.dim, padding_idx=0)

        # Parameters
        self.layer_norm = nn.LayerNorm(self.dim)
        self.aap_norms = nn.ModuleDict()
        for key in self.n_nodes:
            if key != SEQUENCE:
                self.aap_norms[key] = nn.Linear(self.dim, self.dim)
        self.mip_norm = nn.Linear(self.dim, self.dim)
        self.map_norms = nn.ModuleDict()
        for key in self.n_nodes:
            if key != SEQUENCE:
                self.map_norms[key] = nn.Linear(self.dim, self.dim)
        self.sp_norm = nn.Linear(self.dim, self.dim)

        # Optimization
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = LinearWarmupScheduler(
            self.optimizer,
            num_warmup_steps=opt.lr_dc_step,
            num_schedule_steps=opt.lr_dc_step * opt.epoch,
            min_percent=opt.lr_dc**2
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for name, param in self.named_parameters():
            if name.startswith("transformer"):
                continue
            if name.endswith("bias"):
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -stdv, stdv)
        nn.init.ones_(self.layer_norm.weight)
        nn.init.constant_(self.layer_norm.bias, 0.0)
    
    def _associated_attribute_prediction(self, sequence_output, feature_embeddings):
        scores = dict()
        for key in feature_embeddings:
            select = self.aap_norms[key](sequence_output)  # [B L H]
            select = select.view([-1, select.size(-1), 1])  # [B*L H 1]
            # [feature_num H] [B*L H 1] -> [B*L feature_num 1]
            scores[key] = torch.matmul(feature_embeddings[key], select).squeeze(-1)
        return scores
    
    def _masked_item_prediction(self, sequence_output, target_item_emb):
        sequence_output = self.mip_norm(
            sequence_output.view([-1, sequence_output.size(-1)])
        )  # [B*L H]
        target_item_emb = target_item_emb.view(
            [-1, sequence_output.size(-1)]
        )  # [B*L H]
        score = torch.mul(sequence_output, target_item_emb)  # [B*L H]
        return torch.sigmoid(torch.sum(score, -1))  # [B*L]
    
    def _masked_attribute_prediction(self, sequence_output, feature_embeddings):
        scores = dict()
        for key in feature_embeddings:
            select = sequence_output = self.map_norms[key](sequence_output)  # [B L H]
            select = select.view([-1, select.size(-1), 1])  # [B*L H 1]
            # [feature_num H] [B*L H 1] -> [B*L feature_num 1]
            scores[key] = torch.matmul(feature_embeddings[key], select).squeeze(-1)
        return scores

    def _segment_prediction(self, context, segment_emb):
        context = self.sp_norm(context)
        score = torch.mul(context, segment_emb)  # [B H]
        return torch.sigmoid(torch.sum(score, dim=-1))  # [B]
    
    def reconstruct_pretrain_data(self, input, mask, mask_ratio=0.2):

        """_summary_: generate pre-training data for the pre-training stage.
        It uses Transformer to extract local context and global information.

        :param input: input data, including sequence and attributes
        :type input: dict
        :param mask: mask for valid data
        :type mask: torch.Tensor
        :return: hidden states for sequence and attributes
        :rtype: dict
        """
        device = input[SEQUENCE].device
        batch_size, seq_len = input[SEQUENCE].shape[:2]
        item_seq = input[SEQUENCE].cpu().numpy()
        seq_real_len = torch.sum(mask.long(), dim=1).cpu().numpy()
        start_indices = seq_len - seq_real_len

        # associated attribute prediction
        associated_features = {key: input[key] for key in input if key != SEQUENCE}

        # masked item prediction
        mask_item = self.n_nodes[SEQUENCE] - 1
        pos_items = item_seq.copy()
        mask_prob = np.random.rand(batch_size, seq_len)
        mask_bool = mask_prob < mask_ratio
        mask_bool = (mask_bool & (~mask.cpu().numpy()))
        neg_items = np.random.random_integers(0, self.n_nodes[SEQUENCE] - 2, size=pos_items.shape)
        neg_items = neg_items + (neg_items >= item_seq)
        neg_items = np.where(mask_bool, neg_items, pos_items)
        masked_item_sequence = np.where(mask_bool, mask_item, pos_items)

        masked_item_sequence = torch.from_numpy(masked_item_sequence).to(device)
        pos_items = torch.from_numpy(pos_items).to(device)
        neg_items = torch.from_numpy(neg_items).to(device)

        # segment prediction
        sample_lengths = np.random.random_integers(0, 1000000007, (batch_size,)) % (seq_real_len // 2) + 1
        sample_start_ids = np.random.random_integers(0, 1000000007, (batch_size,)) % (seq_real_len - sample_lengths)
        long_sequence = item_seq[mask.cpu().numpy()].reshape(-1)
        neg_start_ids = torch.randint(0, 1000000007, (batch_size,)) % (long_sequence.shape[0] - sample_lengths)

        masked_segment_list = []
        pos_segment_list = []
        neg_segment_list = []
        for i, start_index in enumerate(start_indices):
            if seq_real_len[i] - start_index < 2:
                masked_segment = item_seq[i].copy()
                pos_segment = item_seq[i].copy()
                neg_segment = item_seq[i].copy()
            else:
                sample_length = sample_lengths[i]
                sample_start_id = sample_start_ids[i] + start_index
                neg_start_id = neg_start_ids[i]

                masked_segment = item_seq[i].copy()
                pos_segment = np.zeros(seq_len, dtype=np.int64)
                neg_segment = np.zeros(seq_len, dtype=np.int64)

                masked_segment[sample_start_id:sample_start_id+sample_length] = mask_item
                pos_segment[:sample_start_id] = mask_item
                pos_segment[sample_start_id:sample_start_id+sample_length] = item_seq[i, sample_start_id:sample_start_id+sample_length]
                pos_segment[sample_start_id+sample_length:] = mask_item
                neg_segment[:sample_start_id] = mask_item
                neg_segment[sample_start_id:sample_start_id+sample_length] = long_sequence[neg_start_id:neg_start_id+sample_length]
                neg_segment[sample_start_id+sample_length:] = mask_item

            masked_segment_list.append(masked_segment)
            pos_segment_list.append(pos_segment)
            neg_segment_list.append(neg_segment)
        masked_segment_list = torch.from_numpy(np.stack(masked_segment_list)).to(device)
        pos_segment_list = torch.from_numpy(np.stack(pos_segment_list)).to(device)
        neg_segment_list = torch.from_numpy(np.stack(neg_segment_list)).to(device)

        return (
            associated_features,
            masked_item_sequence,
            pos_items,
            neg_items,
            masked_segment_list,
            pos_segment_list,
            neg_segment_list,
        )
    
    def pretrain(
        self,
        features,
        masked_item_seq,
        pos_items,
        neg_items,
        masked_seg_seq,
        pos_segs,
        neg_segs,
        app_weight=0.2,
        mip_weight=1.0,
        map_weight=1.0,
        sp_weight=0.5
    ):
        """_summary_: pretrain S3Rec.

        :param features: associated features
        :type features: dict
        :param masked_item_seq: masked item sequence
        :type masked_item_seq: torch.Tensor
        :param pos_items: positive items
        :type pos_items: torch.Tensor
        :param neg_items: negative items
        :type neg_items: torch.Tensor
        :param masked_seg_seq: masked segment sequence
        :type masked_seg_seq: torch.Tensor
        :param pos_segs: positive segments
        :type pos_segs: torch.Tensor
        :param neg_segs: negative segments
        :type neg_segs: torch.Tensor
        :param app_weight: associated attribute prediction weight, default 0.2
        :type app_weight: float
        :param mip_weight: masked item prediction weight, default 1.0
        :type mip_weight: float
        :param map_weight: masked attribute prediction weight, default 1.0
        :type map_weight: float
        :param sp_weight: segment prediction weight, default 0.5
        :type sp_weight: float
        :return: loss
        :rtype: torch.Tensor
        """
        mask = masked_item_seq != 0
        seq_hidden = self._forward(masked_item_seq, mask)
        
        mask_item = self.n_nodes[SEQUENCE] - 1

        # AAP
        aap_scores = self._associated_attribute_prediction(seq_hidden, {key: self.embeddings[key].weight for key in features})
        aap_mask = ((masked_item_seq != mask_item) & mask).flatten()
        aap_loss = 0
        if len(features) > 0:
            for key in features:
                aap_loss += F.cross_entropy(aap_scores[key], features[key].flatten(), reduction='none').masked_fill(aap_mask, 0)
            aap_loss = aap_loss.sum() * app_weight

        # MIP
        pos_item_embs = self.embeddings[SEQUENCE](pos_items)
        neg_item_embs = self.embeddings[SEQUENCE](neg_items)
        pos_score = self._masked_item_prediction(seq_hidden, pos_item_embs)
        neg_score = self._masked_item_prediction(seq_hidden, neg_item_embs)
        mip_mask = (masked_item_seq == mask_item).flatten()
        mip_distance = pos_score - neg_score
        mip_loss = F.binary_cross_entropy_with_logits(mip_distance, torch.ones_like(mip_distance), reduction="none").masked_fill(mip_mask, 0)
        mip_loss = mip_loss.sum() * mip_weight

        # MAP
        map_scores = self._masked_attribute_prediction(seq_hidden, {key: self.embeddings[key].weight for key in features})
        map_mask = (masked_item_seq == mask_item).flatten()
        map_loss = 0
        if len(features) > 0:
            for key in features:
                map_loss += F.cross_entropy(map_scores[key], features[key].flatten(), reduction="none").masked_fill(map_mask, 0)
            map_loss = map_loss.sum() * map_weight
        
        # SP
        segment_context = self._forward(masked_seg_seq)[:, -1, :]
        pos_seg_emb = self._forward(pos_segs)[:, -1, :]
        neg_seg_emb = self._forward(neg_segs)[:, -1, :]
        pos_seg_score = self._segment_prediction(segment_context, pos_seg_emb)
        neg_seg_score = self._segment_prediction(segment_context, neg_seg_emb)
        sp_distance = pos_seg_score - neg_seg_score
        sp_loss = F.binary_cross_entropy_with_logits(sp_distance, torch.ones_like(sp_distance, dtype=torch.float32), reduction="none")
        sp_loss = sp_loss.sum() * sp_weight

        return aap_loss + mip_loss + map_loss + sp_loss


    def _forward(self, item_seq, mask=None):
        """_summary_: forward propagation of S3Rec.

        :param item_seq: item sequence
        :type item_seq: torch.Tensor
        :param mask: mask for valid data, default None
        :type mask: torch.Tensor
        :return: hidden states for sequence and attributes
        :rtype: dict
        """

        batch_size, seq_len = item_seq.shape[:2]
        if mask is None:
            mask = item_seq > 0
        h_seq = self.embeddings[SEQUENCE](item_seq)
        pos_idx = (torch.cumsum(mask.long(), dim=1) - 1).clamp_min(0)
        pos_emb = self.pos_embedding(pos_idx)
        h_trans = self.layer_norm(h_seq + pos_emb)
        h_trans = self.transformer(h_trans, mask=mask)

        return h_trans


    def forward(self, input, mask=None, items=None, adj=None, alias=None, shortcut=None, heteitems=None, heteadj=None):
        """_summary_: forward propagation of S3Rec.
        It uses Transformer to extract local context and global information.

        :param input: input data, including sequence and attributes
        :type input: dict
        :param mask: mask for valid data
        :type mask: torch.Tensor
        :param items: sequence of row numbers aligned with adjacency matrix
        :type items: torch.Tensor
        :param adj: adjacency matrix
        :type adj: torch.Tensor
        :param alias: real item ids that can map "items" to real item ids
        :type alias: torch.Tensor
        :return: hidden states for sequence and attributes
        :rtype: dict
        """

        sorted_keys = sorted(self.embeddings.keys())
        hidden = self._forward(input[SEQUENCE], mask=mask)[:, -1, :]

        if self.mtl:
            output = {key: hidden for key in sorted_keys}
        else:
            output = {SEQUENCE: hidden}

        return output

    def compute_scores(self, hidden, mask=None):
        """_summary_: compute inner-product scores for ranking

        :param hidden: hidden states for seqeuence and attributes
        :type hidden: dict
        :param mask: mask for valid data, defaults to None
        :type mask: torch.Tensor, optional
        :return: scores for sequence and attributes
        :rtype: dict
        """

        scores = dict()
        for key in hidden.keys():
            select = hidden[key]
            b = self.embeddings[key].weight[:-1]  # n_nodes x latent_size
            scores[key] = torch.matmul(select, b.transpose(1, 0))
        return scores

    def compute_loss(self, scores, target, coefs=None):
        if self.mtl:
            if coefs is None:
                coefs = {key: 1.0 for key in scores.keys()}
            loss = torch.stack([coef * self.loss_function(scores[key], target[key])
                                for key, coef in coefs.items()]).sum()
        else:
            loss = self.loss_function(scores[SEQUENCE], target[SEQUENCE])
        return loss