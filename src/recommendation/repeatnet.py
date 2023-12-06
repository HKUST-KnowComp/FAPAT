import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from layers import *
from utils import *


class RepeatExploreMechanism(nn.Module):
    """_summary_: implementation of RepeatExploreMechanism that explores the next item based on the previous memory
    details: attentive soft attention
    """

    def __init__(self, hidden_size, seq_len, dropout_prob):
        super(RepeatExploreMechanism, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.Wre = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ure = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Vre = nn.Linear(hidden_size, 1, bias=False)
        self.Wcre = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, all_memory, last_memory):
        """_summary_: forward propagation of RepeatExploreMechanism
        It calculates the probability of Repeat and explore

        :param all_memory: the memory of all previous sessions
        :type all_memory: torch.Tensor
        :param last_memory: the last memory
        :type last_memory: torch.Tensor
        :return: the probability of repeat and explore
        :rtype: torch.Tensor
        """

        all_memory_values = all_memory

        all_memory = self.dropout(self.Ure(all_memory))

        last_memory = self.dropout(self.Wre(last_memory))
        last_memory = last_memory.unsqueeze(1)

        output_ere = F.tanh(all_memory + last_memory)

        output_ere = self.Vre(output_ere)
        alpha_are = stable_softmax(output_ere, dim=1)
        output_cre = alpha_are * all_memory_values
        output_cre = output_cre.sum(dim=1)

        output_cre = self.Wcre(output_cre)

        repeat_explore_mechanism = nn.Softmax(dim=-1)(output_cre)

        return repeat_explore_mechanism


class RepeatRecommendationDecoder(nn.Module):
    """_summary_: implementation of RepeatRecommendationDecoder that finds out the repeat consume in sequential recommendation
    details: attentive soft attention
    """

    def __init__(self, hidden_size, seq_len, num_item, dropout_prob):
        super(RepeatRecommendationDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_item = num_item
        self.Wr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Vr = nn.Linear(hidden_size, 1)

    def forward(self, all_memory, last_memory, item_seq, mask=None):
        """_summary_: forward propagation of RepeatRecommendationDecoder
        It calculates the the force of repeat

        :param all_memory: the memory of all previous sessions
        :type all_memory: torch.Tensor
        :param last_memory: the last memory
        :type last_memory: torch.Tensor
        :param item_seq: the current sequence
        :type item_seq: torch.Tensor
        :param mask: the mask of the current sequence, defaults to None
        :type mask: torch.Tensor, optional
        :return: the force of repeat
        :rtype: torch.Tensor
        """

        all_memory = self.dropout(self.Ur(all_memory))

        last_memory = self.dropout(self.Wr(last_memory))
        last_memory = last_memory.unsqueeze(1)

        output_er = F.tanh(last_memory + all_memory)
        output_er = self.Vr(output_er).squeeze(2)

        if mask is not None:
            output_er.masked_fill_(~mask, -1e9)

        output_er = stable_softmax(output_er, dim=-1)  # (batch_size, seq_len)
        output_er = torch.zeros((output_er.shape[0], self.num_item), device=output_er.device,
                                dtype=output_er.dtype).scatter_add_(1, item_seq, output_er)
        return output_er


class ExploreRecommendationDecoder(nn.Module):
    """_summary_: implementation of ExploreRecommendationDecoder that explores new items for recommendation
    details: attentive soft attention
    """

    def __init__(self, hidden_size, seq_len, num_item, dropout_prob):
        super(ExploreRecommendationDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_item = num_item
        self.We = nn.Linear(hidden_size, hidden_size)
        self.Ue = nn.Linear(hidden_size, hidden_size)
        self.Ve = nn.Linear(hidden_size, 1)
        self.matrix_for_explore = nn.Linear(2 * self.hidden_size, self.num_item, bias=False)

    def forward(self, all_memory, last_memory, item_seq, mask=None):
        """_summary_: forward propagation of ExploreRecommendationDecoder
        It calculates the force of explore

        :param all_memory: the memory of all previous sessions
        :type all_memory: torch.Tensor
        :param last_memory: the last memory
        :type last_memory: torch.Tensor
        :param item_seq: the current sequence
        :type item_seq: torch.Tensor
        :param mask: the mask of the current sequence, defaults to None
        :type mask: torch.Tensor, optional
        :return: the force of explore
        :rtype: torch.Tensor
        """
      
        all_memory_values, last_memory_values = all_memory, last_memory

        all_memory = self.dropout(self.Ue(all_memory))

        last_memory = self.dropout(self.We(last_memory))
        last_memory = last_memory.unsqueeze(1)

        output_ee = F.tanh(all_memory + last_memory)
        output_ee = self.Ve(output_ee).squeeze(-1)

        if mask is not None:
            output_ee.masked_fill_(~mask, -1e9)

        output_ee = output_ee.unsqueeze(-1)

        alpha_e = stable_softmax(output_ee, dim=1)
        output_e = (alpha_e * all_memory_values).sum(dim=1)
        output_e = torch.cat([output_e, last_memory_values], dim=1)
        output_e = self.dropout(self.matrix_for_explore(output_e))  # (batch_size, num_item)

        output_e = output_e.scatter_add_(
            1, item_seq,
            torch.full((output_e.shape[0], item_seq.shape[1]), -1e9, dtype=output_e.dtype, device=output_e.device)
        )  # (batch_size, num_item)

        return stable_softmax(output_e, dim=1)


class RepeatNet(nn.Module):
    """_summary_: implementation of NARM
    reference: Pengjie Ren, Jing Li, Zhumin Chen, Zhaochun Ren, Jun Ma, and Maarten de Rijke. 2019. RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation. In AAAI. 4806-4813.
    code reference: https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/sequential_recommender/repeatnet.py
    """

    def __init__(self, opt, **kwargs):
        super(RepeatNet, self).__init__()

        self.opt = opt

        self.batch_size = opt.batch_size
        self.max_len = opt.max_len
        self.dim = opt.hidden_dim
        self.n_iter = opt.n_iter
        self.rnn_type = opt.rnn_type
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.clip = opt.clip
        self.mtl = opt.mtl

        self.n_nodes = kwargs["n_nodes"]
        if isinstance(self.n_nodes, int):
            self.n_nodes = {SEQUENCE: self.n_nodes}

        # Item representation & Position representation
        self.embeddings = nn.ModuleDict()
        for key in self.n_nodes:
            self.embeddings[key] = nn.Embedding(self.n_nodes[key], self.dim, padding_idx=0)

        # define the layers and loss function
        self.gru = nn.GRU(self.dim, self.dim, batch_first=True)
        self.repeat_explore_mechanism = RepeatExploreMechanism(self.dim, self.max_len, self.dropout_local)
        self.repeat_recommendation_decoder = RepeatRecommendationDecoder(
            self.dim, self.max_len, self.n_nodes[SEQUENCE], self.dropout_local
        )
        self.explore_recommendation_decoder = ExploreRecommendationDecoder(
            self.dim, self.max_len, self.n_nodes[SEQUENCE], self.dropout_local
        )

        # Optimization
        self.loss_function = nn.NLLLoss(ignore_index=0)
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
            if name.endswith("bias"):
                nn.init.constant_(param, 0.0)
            else:
                nn.init.uniform_(param, -stdv, stdv)

    def forward(self, input, mask=None, items=None, adj=None, alias=None, shortcut=None, heteitems=None, heteadj=None):
        """_summary_: forward propagation of RepeatNet.
        It emphasizes repeat consumption with neural networks and 
        explores new items with attention mechanism.

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

        batch_size, seq_len = input[SEQUENCE].shape[:2]
        sorted_keys = sorted(self.embeddings.keys())
        assert len(set(sorted_keys) - set(input.keys())) == 0
        seq_idx = sorted_keys.index(SEQUENCE)
        sorted_keys[0], sorted_keys[seq_idx] = sorted_keys[seq_idx], sorted_keys[0]

        h_seqs = dict()
        for key in sorted_keys:
            h_seqs[key] = self.embeddings[key](input[key])
        h_seq = torch.stack([h_seqs[key] for key in sorted_keys]).mean(0)

        all_memory, _ = self.gru(h_seq)
        last_memory = all_memory[:, -1, :]

        repeat_explore = self.repeat_explore_mechanism.forward(all_memory=all_memory, last_memory=last_memory)
        # batch_size * 2
        repeat_rec = self.repeat_recommendation_decoder.forward(
            all_memory=all_memory,
            last_memory=last_memory,
            item_seq=input[SEQUENCE],
            mask=mask,
        )
        # batch_size * num_item
        explore_rec = self.explore_recommendation_decoder.forward(
            all_memory=all_memory,
            last_memory=last_memory,
            item_seq=input[SEQUENCE],
            mask=mask,
        )
        # batch_size * num_item

        if self.mtl:
            output = {key: (repeat_explore, repeat_rec, explore_rec) for key in sorted_keys}
        else:
            output = {SEQUENCE: (repeat_explore, repeat_rec, explore_rec)}

        return output

    def compute_scores(self, hidden, mask=None):
        """_summary_: compute repeat and explore scores for ranking

        :param hidden: hidden states for seqeuence and attributes
        :type hidden: dict
        :param mask: mask for valid data, defaults to None
        :type mask: torch.Tensor, optional
        :return: scores for sequence and attributes
        :rtype: dict
        """

        scores = dict()
        for key in hidden.keys():
            repeat_explore, repeat_rec, explore_rec = hidden[key]
            scores[key] = repeat_rec * repeat_explore[:, 0].unsqueeze(1) + \
                explore_rec * repeat_explore[:, 1].unsqueeze(1)
        return scores

    def compute_loss(self, scores, target, coefs=None):
        if self.mtl:
            if coefs is None:
                coefs = {key: 1.0 for key in scores.keys()}
            loss = torch.stack(
                [coef * self.loss_function((scores[key] + 1e-8).log(), target[key]) for key, coef in coefs.items()]
            ).sum()
        else:
            loss = self.loss_function((scores[SEQUENCE] + 1e-8).log(), target[SEQUENCE])
        return loss
