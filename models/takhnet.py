# coding: utf-8

r"""
TaKHNet: Time-aware KAN-based Heterogeneous Multimodal Recommendation.

This file contains the model only. It is written for an MMRec-style framework,
so the base class it inherits from is expected to expose ``inter_matrix``,
``v_feat``, ``t_feat``, ``n_users`` and ``n_items``, and the two graph helpers
at the bottom are expected to exist in ``utils``. The training and evaluation
pipeline is not part of this release, so the file is not meant to run on its
own.

Graphs (all sparse, O(E) memory):
    interact (u <-> i) : LightGCN-normalised rating matrix, the collaborative base
    cooc     (i <-> i) : item co-occurrence with event-level time-decayed weights
    img-sim  (i <-> i) : visual kNN graph, built on decorrelated visual features
    txt-sim  (i <-> i) : textual kNN graph

Two channels:
    collaborative channel : interaction edges only, propagated with LightGCN and
                            never contaminated by content;
    fused channel         : all edge types, with type-aware messages aggregated
                            inside propagation rather than after it.

Ingredients:
    1. fusion inside propagation, so the views meet at every layer;
    2. KAN cross-view fusion: a Kolmogorov-Arnold layer models the nonlinear
       interaction among cooc / visual / textual instead of a linear attention
       subtraction;
    3. time-aware relations: co-occurrence edges are weighted by exp(-|dt| / tau)
       and averaged over the co-purchase events of an item pair;
    4. visual decorrelation: ZCA-shrinkage whitening of the frozen visual
       features, applied through a learned gate;
    5. text-guided removal of redundancy: the visual gate is multiplied by the
       image-text semantic consistency;
    6. dual-channel InfoNCE, aligning the collaborative and fused views on both
       the user and the item side.

The switches in ``TaKHNet.__init__`` reproduce the ablation variants of the
paper; their default values are the configuration reported there.
"""

import os
import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, build_knn_normalized_graph


class KANFusionLayer(nn.Module):
    r"""Kolmogorov-Arnold view-fusion layer.

    Input: concatenated multi-view messages (I, k*d); output (I, out_dim).
    One grid is shared per input dimension (FastKAN style), and the basis is
    selectable:
      'rbf'     : Gaussian exp(-u^2/2) with u = (x - c) / h, fixed centres and width;
      'wav'     : Mexican hat (1 - u^2) exp(-u^2/2) on several scales, Wav-KAN style;
      'fourier' : cos(k pi x) and sin(k pi x), k = 1..G, as a baseline;
      'cheby'   : T_k(tanh(x)), k = 0..G-1, a global polynomial basis.

    Optional refinements:
      - h_mult scales the Gaussian width; the useful range is 1/(G-1) to 2/(G-1),
        because a much wider kernel makes the design matrix ill-conditioned;
      - grid_learnable lets the centres and the width be trained, in the spirit of
        free-knot splines;
      - hidden > 0 inserts one hidden layer (in -> hidden -> out) with layer
        normalisation, which adds capacity without stacking many KAN layers;
      - init_zero starts from an all-zero read-out, so the branch begins as a pure
        residual and grows only where it helps;
      - dropout is applied to the output.

    Everything is computed per node and never through a sparse matmul, so the
    layer is safe on the full item set of the largest benchmark.
    """

    def __init__(self, in_dim, out_dim, basis='rbf', grid_size=8, scales=(1.0, 0.5, 0.25),
                 h_mult=2.0, grid_learnable=False, hidden=0, init_zero=False, dropout=0.0,
                 grid_range=(-3.0, 3.0)):
        super(KANFusionLayer, self).__init__()
        assert basis in ('rbf', 'wav', 'fourier', 'cheby')
        if hidden > 0:
            assert basis in ('rbf', 'cheby'), 'the two-layer variant supports only rbf and cheby'
        self.in_dim, self.out_dim = in_dim, out_dim
        self.basis = basis
        self.grid_size = grid_size
        self.scales = scales
        self.hidden = hidden
        grid_init = torch.linspace(grid_range[0], grid_range[1], grid_size)
        if basis == 'fourier':
            std = 1.0 / (in_dim * grid_size)
            self.W = nn.Parameter(torch.randn(in_dim, out_dim, 2, grid_size) * std)   # (in, out, 2, G)
        elif basis == 'wav':
            self.register_buffer('grid', grid_init.clone())  # fixed wavelet centres
            self.register_buffer('scales_b', torch.tensor(scales, dtype=torch.float32))  # (S,)
            std = 1.0 / (in_dim * grid_size * len(scales))
            self.W = nn.Parameter(torch.randn(in_dim, out_dim, len(scales), grid_size) * std)  # (in, out, S, G)
        else:  # rbf / cheby
            if grid_learnable:
                self.grid = nn.Parameter(grid_init.clone())
            else:
                self.register_buffer('grid', grid_init)
            h = h_mult * (6.0 / (grid_size - 1))
            if grid_learnable:
                self.h = nn.Parameter(torch.tensor(h, dtype=torch.float32))
            else:
                self.register_buffer('h', torch.tensor(h, dtype=torch.float32))
            std = 1.0 / (in_dim * grid_size)
            if hidden > 0:
                self.W1 = nn.Parameter(torch.randn(in_dim, hidden, grid_size) * std)      # (in, hid, G)
                self.bias1 = nn.Parameter(torch.zeros(1, hidden))
                std2 = 1.0 / (hidden * grid_size)
                self.W2 = nn.Parameter(torch.randn(hidden, out_dim, grid_size) * std2)    # (hid, out, G)
                self.bias2 = nn.Parameter(torch.zeros(1, out_dim))
            else:
                self.W = nn.Parameter(torch.randn(in_dim, out_dim, grid_size) * std)      # (in, out, G)
        if hidden == 0:
            self.bias = nn.Parameter(torch.zeros(1, out_dim))
        if init_zero:
            for p in self.parameters():
                if p.requires_grad and p.dim() >= 2:
                    nn.init.zeros_(p)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def _expand(self, x):
        r"""Expand into basis activations: (N, in) -> (N, in, [S,] G)."""
        if self.basis == 'cheby':
            t = torch.tanh(x)
            feats = [torch.ones_like(t), t]
            T_prev, T_curr = feats[0], feats[1]
            for _ in range(2, self.grid_size):
                T_next = 2.0 * t * T_curr - T_prev
                feats.append(T_next)
                T_prev, T_curr = T_curr, T_next
            return torch.stack(feats, dim=-1)                      # (N, in, G)
        if self.basis == 'rbf':
            u = (x.unsqueeze(-1) - self.grid) / (self.h.abs() + 1e-8)   # (N, in, G)
            return torch.exp(-0.5 * u * u)
        if self.basis == 'wav':
            g = self.grid.view(1, 1, -1)                           # (1,1,G)
            s = self.scales_b.view(1, -1, 1)                       # (1,S,1)
            u = (x.unsqueeze(-1).unsqueeze(-1) - g) / s            # (N, in, S, G)
            return (1.0 - u * u) * torch.exp(-0.5 * u * u)  # Mexican hat
        # fourier
        k = torch.arange(1, self.grid_size + 1, device=x.device).float()
        phase = x.unsqueeze(-1) * (k * math.pi)                    # (N, in, G)
        return torch.stack([torch.cos(phase), torch.sin(phase)], dim=-2)   # (N, in, 2, G)

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_dim,)
        x = x.reshape(-1, self.in_dim)                            # (N, in)
        feat = self._expand(x)                                    # (N, in, [S|2,] G)
        if self.hidden > 0:
            h1 = torch.einsum('nig,iog->no', feat, self.W1) + self.bias1
            h1 = F.layer_norm(h1, (self.hidden,))
            feat2 = self._expand(h1)
            out = torch.einsum('nig,iog->no', feat2, self.W2) + self.bias2
        elif self.basis == 'wav':
            out = torch.einsum('nidg,iodg->no', feat, self.W)
        elif self.basis == 'fourier':
            out = torch.einsum('nidg,iodg->no', feat, self.W)
        else:
            out = torch.einsum('nig,idg->nd', feat, self.W)
        if self.hidden == 0:
            out = out + self.bias
        if self.dropout is not None:
            out = self.dropout(out)
        return out.reshape(out_shape)


class TaKHNet(GeneralRecommender):
    def __init__(self, config, dataset):
        super(TaKHNet, self).__init__(config, dataset)
        self._gates_logged = False
        self.logger = getLogger()

        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.knn_k = config['knn_k']
        self.is_sparse = config['is_sparse']
        self.fusion_coeff = config['fusion_coeff']
        self.tau = config['tau']
        self.cl_weight = config['cl_weight']
        self.reg_weight = config['reg_weight']

        # edge-type switches (used by the ablation)
        self.use_cooc = config['use_cooc'] if config['use_cooc'] is not None else True
        self.use_img = config['use_img'] if config['use_img'] is not None else True
        self.use_txt = config['use_txt'] if config['use_txt'] is not None else True

        # time-weighted co-occurrence: event-level mean of exp(-dt / tau)
        self.time_weighted_cooc = config['time_weighted_cooc'] if config['time_weighted_cooc'] is not None else False
        self.time_decay_tau = config['time_decay_tau'] if config['time_decay_tau'] is not None else 7776000

        # visual decorrelation: gated ZCA-shrinkage whitening
        self.image_decorr = config['image_decorr'] if config['image_decorr'] is not None else False
        self.image_decorr_shrinkage = config['image_decorr_shrinkage'] if config['image_decorr_shrinkage'] is not None else 0.03

        # type-aware fusion through scalar gates
        self.type_gate = config['type_gate'] if config['type_gate'] is not None else True
        self.type_gate_init = config['type_gate_init'] if config['type_gate_init'] is not None else -0.69

        # alignment between the collaborative and fused channels
        self.use_align = config['use_align'] if config['use_align'] is not None else True

        # text-guided removal of redundant visual signal
        self.use_txt_guided_decorr = config['use_txt_guided_decorr'] if config['use_txt_guided_decorr'] is not None else False
        self.txt_guided_beta = config['txt_guided_beta'] if config['txt_guided_beta'] is not None else 5.0

        # KAN cross-view fusion
        self.use_kan_fusion = config['use_kan_fusion'] if config['use_kan_fusion'] is not None else False
        self.kan_fusion_basis = config['kan_fusion_basis'] if config['kan_fusion_basis'] is not None else 'rbf'
        self.kan_fusion_grid = config['kan_fusion_grid'] if config['kan_fusion_grid'] is not None else 8
        self.kan_fusion_coeff = config['kan_fusion_coeff'] if config['kan_fusion_coeff'] is not None else 0.01
        self.kan_fusion_norm = config['kan_fusion_norm'] if config['kan_fusion_norm'] is not None else True
        # fusion coefficient, kernel width, grid, depth and initialisation
        self.kan_fusion_scale_mode = config['kan_fusion_scale_mode'] if config['kan_fusion_scale_mode'] is not None else 'match'
        self.kan_fusion_coeff_learnable = config['kan_fusion_coeff_learnable'] if config['kan_fusion_coeff_learnable'] is not None else False
        self.kan_fusion_coeff_max = config['kan_fusion_coeff_max'] if config['kan_fusion_coeff_max'] is not None else 0.5
        self.kan_fusion_h_mult = config['kan_fusion_h_mult'] if config['kan_fusion_h_mult'] is not None else 2.0
        self.kan_fusion_grid_learnable = config['kan_fusion_grid_learnable'] if config['kan_fusion_grid_learnable'] is not None else False
        self.kan_fusion_hidden = config['kan_fusion_hidden'] if config['kan_fusion_hidden'] is not None else 0
        self.kan_fusion_init_zero = config['kan_fusion_init_zero'] if config['kan_fusion_init_zero'] is not None else False
        self.kan_fusion_dropout = config['kan_fusion_dropout'] if config['kan_fusion_dropout'] is not None else 0.0
        self.kan_gate_mode = config['kan_gate_mode'] if config['kan_gate_mode'] is not None else 'scalar'   # scalar|kan
        self.kan_gate_hidden = config['kan_gate_hidden'] if config['kan_gate_hidden'] is not None else 0
        # the semantic gate is either a fixed cosine curve or a learned KAN curve
        self.sem_gate_mode = config['sem_gate_mode'] if config['sem_gate_mode'] is not None else 'cos'      # cos|kan
        self.sem_kan_input = config['sem_kan_input'] if config['sem_kan_input'] is not None else 'cos'      # cos|cos_gv
        self.sem_kan_grid = config['sem_kan_grid'] if config['sem_kan_grid'] is not None else 8
        # interactions used to build the co-occurrence graph: 'train' (default) or 'all'
        self.cooc_build_on = config['cooc_build_on'] if config['cooc_build_on'] is not None else 'train'

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        # ID embeddings
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        # graphs (sparse throughout)
        self.norm_R, self.norm_adj = self.get_norm_adj_mat()
        self.norm_R_t = self.norm_R.transpose(0, 1)

        self.A_cooc = self._build_cooc_adj(config) if self.use_cooc else None
        self.A_img = self._build_modal_adj('image', config) if (self.use_img and self.v_feat is not None) else None
        self.A_txt = self._build_modal_adj('text', config) if (self.use_txt and self.t_feat is not None) else None

        # modality projection, then whitening, then a learned gate
        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)
        self.gate_v = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())

        # type-aware scalar gates: sigmoid, no softmax competition, so no saturation
        self.type_gate_p = nn.ParameterDict({
            'cooc': nn.Parameter(torch.tensor(self.type_gate_init, dtype=torch.float32)),
            'img': nn.Parameter(torch.tensor(self.type_gate_init, dtype=torch.float32)),
            'txt': nn.Parameter(torch.tensor(self.type_gate_init, dtype=torch.float32)),
        }) if self.type_gate else None

        # KAN view-fusion layer
        n_views = int(self.use_cooc) + int(self.use_img and self.v_feat is not None) \
            + int(self.use_txt and self.t_feat is not None)
        self._fusion_in_dim = self.embedding_dim * n_views
        self.kan_fusion = KANFusionLayer(self._fusion_in_dim, self.embedding_dim,
                                         basis=self.kan_fusion_basis,
                                         grid_size=self.kan_fusion_grid,
                                         h_mult=self.kan_fusion_h_mult,
                                         grid_learnable=self.kan_fusion_grid_learnable,
                                         hidden=self.kan_fusion_hidden,
                                         init_zero=self.kan_fusion_init_zero,
                                         dropout=self.kan_fusion_dropout) if self.use_kan_fusion else None
        # optional per-item gate scored by a KAN instead of one global scalar
        self.kan_gate = KANFusionLayer(self._fusion_in_dim, n_views,
                                       basis=self.kan_fusion_basis,
                                       grid_size=self.kan_fusion_grid,
                                       h_mult=self.kan_fusion_h_mult,
                                       hidden=self.kan_gate_hidden,
                                       init_zero=True) \
            if (self.type_gate and self.kan_gate_mode == 'kan' and n_views > 1) else None
        if self.kan_gate is not None:
            # start from the global scalar gate, sigmoid(-0.69) ~ 1/3; with a hidden layer the bias sits in the second one
            gate_bias = self.kan_gate.bias2 if self.kan_gate_hidden > 0 else self.kan_gate.bias
            nn.init.constant_(gate_bias, self.type_gate_init)
        # semantic gate implemented as a KAN over [cos(img, txt)] or [cos, mean(gate_v)]
        self.sem_kan = None
        if (self.sem_gate_mode == 'kan' and self.use_txt_guided_decorr
                and self.use_img and self.use_txt
                and self.v_feat is not None and self.t_feat is not None):
            sem_in = 2 if self.sem_kan_input == 'cos_gv' else 1
            self.sem_kan = KANFusionLayer(sem_in, 1, basis='rbf',
                                          grid_size=self.sem_kan_grid,
                                          h_mult=0.5, init_zero=True,
                                          grid_range=(-1.3, 1.5))
            nn.init.constant_(self.sem_kan.bias, 0.0)  # start at w = sigmoid(0) = 0.5
        # fusion coefficient: frozen, or learned through a bounded sigmoid
        if self.use_kan_fusion:
            if self.kan_fusion_coeff_learnable:
                target = min(max(self.kan_fusion_coeff / self.kan_fusion_coeff_max, 1e-4), 0.9999)
                theta = math.log(target / (1 - target))
                self._kan_fusion_theta = nn.Parameter(torch.tensor(theta, dtype=torch.float32))
            else:
                self._kan_fusion_coeff_p = nn.Parameter(torch.tensor(self.kan_fusion_coeff, dtype=torch.float32))
                self._kan_fusion_coeff_p.requires_grad_(False)

        self.restore_user_e, self.restore_item_e = None, None
        self._last_ii_msgs = []

    # ------------------------------------------------------------------ #
    # graph construction
    # ------------------------------------------------------------------ #
    def get_norm_adj_mat(self):
        R = self.interaction_matrix.tocsr()
        user_degrees = np.array(R.sum(axis=1)).flatten()
        item_degrees = np.array(R.sum(axis=0)).flatten()
        d_user = np.power(user_degrees, -0.5)
        d_user[np.isinf(d_user)] = 0.
        d_item = np.power(item_degrees, -0.5)
        d_item[np.isinf(d_item)] = 0.
        D_user = sp.diags(d_user, shape=(self.n_users, self.n_users))
        D_item = sp.diags(d_item, shape=(self.n_items, self.n_items))
        norm_R = D_user.dot(R).dot(D_item).tocoo()
        top_mat = sp.hstack([sp.csr_matrix((self.n_users, self.n_users)), norm_R])
        bottom_mat = sp.hstack([norm_R.T, sp.csr_matrix((self.n_items, self.n_items))])
        norm_adj_mat = sp.vstack([top_mat, bottom_mat]).tocoo()
        norm_R = torch.sparse.FloatTensor(
            torch.LongTensor(np.array([norm_R.row, norm_R.col])),
            torch.FloatTensor(norm_R.data),
            torch.Size(norm_R.shape)).float().to(self.device)
        norm_adj_mat = torch.sparse.FloatTensor(
            torch.LongTensor(np.array([norm_adj_mat.row, norm_adj_mat.col])),
            torch.FloatTensor(norm_adj_mat.data),
            torch.Size(norm_adj_mat.shape)).float().to(self.device)
        return norm_R, norm_adj_mat

    def _build_cooc_adj(self, config):
        r"""Co-occurrence graph: topology from counts above one, per-row top-k and
        self-loops; the edge weight is one or the time-decayed mean."""
        R = self.interaction_matrix.tocsr()
        if self.time_weighted_cooc:
            cand_row, cand_col, cand_count, cand_weight = self._cooc_time_weighted(config)
        else:
            C = (R.T @ R).tocoo()
            mask = (C.row != C.col) & (C.data > 1)
            cand_row = C.row[mask].astype(np.int64)
            cand_col = C.col[mask].astype(np.int64)
            cand_count = C.data[mask].astype(np.float32)
            cand_weight = np.ones_like(cand_count)

        row_list, col_list, weight_list = self._cooc_topk_edges(cand_row, cand_col, cand_count, cand_weight)
        indices = torch.stack([torch.tensor(row_list, dtype=torch.long),
                               torch.tensor(col_list, dtype=torch.long)], dim=0).to(self.device)
        values = torch.tensor(weight_list, dtype=torch.float32).to(self.device)
        adj = self.compute_normalized_laplacian(indices, (self.n_items, self.n_items), values)
        self.logger.info('[TaKHNet] cooc graph built: nnz=%d (time_weighted=%s, tau=%ds).'
                         % (indices.shape[1], self.time_weighted_cooc, self.time_decay_tau))
        return adj

    def _cooc_topk_edges(self, cand_row, cand_col, cand_count, cand_weight):
        r"""Per-row top-k edge selection through the csr indptr, so that no dense
        block is ever formed; self-loops are added afterwards."""
        n = self.n_items
        count_csr = sp.csr_matrix((cand_count, (cand_row, cand_col)), shape=(n, n))
        w_map = {(int(r), int(c)): float(w) for r, c, w in zip(cand_row, cand_col, cand_weight)}
        row_list, col_list, weight_list = [], [], []
        for i in range(n):
            row_list.append(i)
            col_list.append(i)
            weight_list.append(1.0)
            s, e = count_csr.indptr[i], count_csr.indptr[i + 1]
            if s == e:
                continue
            idx = count_csr.indices[s:e]
            data = count_csr.data[s:e]
            kk = min(self.knn_k, data.size)
            # deterministic top-k: descending value, ascending index, matching the tie-break
            order = np.lexsort((idx, -data))
            for t in order[:kk]:
                j = int(idx[t])
                row_list.append(i)
                col_list.append(j)
                weight_list.append(w_map.get((i, j), 1.0))
        return row_list, col_list, weight_list

    def _cooc_time_weighted(self, config):
        r"""Event-level time weighting: W[i, j] is the mean over the co-purchase
        events of exp(-|dt| / tau). Returns the candidate edges as
        (row, col, count, weight), where count is the raw co-occurrence
        count. The result is cached per (dataset, tau), since it depends
        only on the input data."""
        cache_file = os.path.join(self.dataset_path,
                                  'cooc_tw_cache_tau%d_%s.npz' % (int(self.time_decay_tau), self.cooc_build_on))
        if os.path.exists(cache_file):
            z = np.load(cache_file)
            return z['r'], z['c'], z['n'], z['w']
        user_hist = self._load_train_histories(config)
        acc = {}
        for pairs in user_hist.values():
            m = len(pairs)
            if m < 2:
                continue
            for a in range(m):
                ia, ta = pairs[a]
                for b in range(a + 1, m):
                    ib, tb = pairs[b]
                    if ib == ia:
                        continue
                    w = math.exp(-abs(ta - tb) / self.time_decay_tau)
                    key = (ia, ib) if ia < ib else (ib, ia)
                    if key in acc:
                        acc[key][0] += 1.0
                        acc[key][1] += w
                    else:
                        acc[key] = [1.0, w]
        if not acc:
            raise RuntimeError('time-weighted co-occurrence: no co-occurring pair.')
        # keep the upper triangle and expand it to both (i, j) and (j, i), so that the
            # per-row top-k sees every neighbour
        rows, cols, counts, decays = [], [], [], []
        for (i, j), (c, w) in acc.items():
            if c <= 1:
                continue
            rows.extend([i, j])
            cols.extend([j, i])
            counts.extend([c, c])
            decays.extend([w, w])
        cand_row = np.array(rows, dtype=np.int64)
        cand_col = np.array(cols, dtype=np.int64)
        cand_count = np.array(counts, dtype=np.float32)
        decay_sum = np.array(decays, dtype=np.float32)
        cand_weight = decay_sum / cand_count
        np.savez(cache_file, r=cand_row, c=cand_col, n=cand_count, w=cand_weight)
        return cand_row, cand_col, cand_count, cand_weight

    def _build_modal_adj(self, modality, config):
        r"""Modality kNN graph, built with the shared similarity and kNN
        normalisation helpers and cached next to the dataset."""
        feat = self.v_feat if modality == 'image' else self.t_feat
        emb = nn.Embedding.from_pretrained(feat, freeze=False)
        if modality == 'image':
            self.image_embedding = emb
        else:
            self.text_embedding = emb
        adj_file = os.path.join(self.dataset_path, '{}_adj_{}_{}.pt'.format(modality, self.knn_k, self.is_sparse))
        if os.path.exists(adj_file):
            adj = torch.load(adj_file, map_location=self.device)
        else:
            adj = build_sim(emb.weight.detach())
            adj = build_knn_normalized_graph(adj, topk=self.knn_k, is_sparse=self.is_sparse, norm_type='sym')
            torch.save(adj, adj_file)
        return adj.to(self.device)

    def compute_normalized_laplacian(self, indices, adj_size, values=None):
        r"""Symmetric normalised Laplacian D^{-1/2} A D^{-1/2}, kept sparse."""
        if values is None:
            values = torch.ones_like(indices[0], dtype=torch.float32)
        else:
            values = values.to(torch.float32)
        adj = torch.sparse.FloatTensor(indices, values, adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt * values
        return torch.sparse.FloatTensor(indices, values, adj_size)

    # ------------------------------------------------------------------ #
    # helpers: interaction timestamps and ZCA whitening
    # ------------------------------------------------------------------ #
    def _load_train_histories(self, config):
        r"""Read the raw .inter file and return one time-ordered interaction list
        per user. With cooc_build_on = 'train' (the default) only the
        training split, x_label = 0, is used, so validation and test
        interactions never enter the graph."""
        import pandas as pd
        from collections import defaultdict
        inter_file = os.path.join(self.dataset_path, config['inter_file_name'])
        df = pd.read_csv(inter_file, sep=config['field_separator'],
                         usecols=[config['USER_ID_FIELD'], config['ITEM_ID_FIELD'], config['TIME_FIELD'],
                                  config['inter_splitting_label']])
        if self.cooc_build_on != 'all':
            df = df[df[config['inter_splitting_label']] == 0]
        user_hist = defaultdict(list)
        for u, t in df.groupby(config['USER_ID_FIELD']):
            pairs = sorted(zip(t[config['ITEM_ID_FIELD']].astype(int).tolist(),
                               t[config['TIME_FIELD']].astype(float).tolist()), key=lambda x: x[1])
            user_hist[int(u)] = pairs
        return user_hist

    def _zca_shrink(self, z):
        r"""ZCA-shrinkage whitening, with statistics over all items and fully
        differentiable. If the covariance yields non-finite values the
        shrinkage is increased and the step retried; if it still fails the
        input is returned unchanged, so training never breaks on a badly
        conditioned dataset."""
        d = z.shape[1]
        mu = z.mean(dim=0, keepdim=True)
        zc = z - mu
        cov = zc.t() @ zc / (z.shape[0] - 1)
        if not torch.isfinite(cov).all():
            return z
        eps = self.image_decorr_shrinkage * (torch.diag(cov).mean() + 1e-8)
        cov_j = cov + eps * torch.eye(d, device=z.device)
        try:
            evals, evecs = torch.linalg.eigh(cov_j)
        except RuntimeError:
            try:
                evals, evecs = torch.linalg.eigh(cov + 1e-3 * torch.eye(d, device=z.device))
            except RuntimeError:
                self.logger.info('[TaKHNet] ZCA numerically unstable, falling back to no whitening.')
                return z
        evals = evals.clamp(min=1e-8)
        w = evecs @ torch.diag(evals.pow(-0.5)) @ evecs.t()
        return zc @ w

    # ------------------------------------------------------------------ #
    # forward pass
    # ------------------------------------------------------------------ #
    def _item_sources(self):
        r"""Independent source of the three item-item message types (single hop,
        they do not evolve with propagation):
        - cooc: the bare item embedding;
        - visual: item_id * gate_v(z_img), with z_img optionally whitened;
        - textual: item_id * gate_t(z_txt)."""
        cooc_src = self.item_id_embedding.weight
        img_src = txt_src = None
        if self.use_img and self.v_feat is not None:
            z_img = self.image_trs(self.image_embedding.weight)
            if self.image_decorr:
                z_img = self._zca_shrink(z_img)
            # text-guided visual gating: the visual gate follows the image-text consistency
            if self.use_txt_guided_decorr and self.use_txt and self.t_feat is not None:
                z_txt_detach = self.text_trs(self.text_embedding.weight).detach()
                cos_img_txt = F.cosine_similarity(F.normalize(z_img, dim=1), F.normalize(z_txt_detach, dim=1), dim=1)
                if self.sem_kan is not None:
                    # a KAN learns the shape of the gate curve instead of fixing it
                    if self.sem_kan_input == 'cos_gv':
                        gv_mean = self.gate_v(z_img).mean(dim=1)  # (I,) mean visual gate as context
                        feats = torch.stack([cos_img_txt, gv_mean], dim=1)  # (I, 2)
                    else:
                        feats = cos_img_txt.unsqueeze(1)                  # (I, 1)
                    w_txt_guided = torch.sigmoid(self.sem_kan(feats)).squeeze(1)
                else:
                    w_txt_guided = torch.sigmoid(self.txt_guided_beta * (cos_img_txt - cos_img_txt.mean()))
                img_src = self.item_id_embedding.weight * self.gate_v(z_img) * w_txt_guided[:, None]
            else:
                img_src = self.item_id_embedding.weight * self.gate_v(z_img)
        if self.use_txt and self.t_feat is not None:
            z_txt = self.text_trs(self.text_embedding.weight)
            txt_src = self.item_id_embedding.weight * self.gate_t(z_txt)
        return cooc_src, img_src, txt_src

    def _aggregate_ii(self, ii_msgs):
        r"""Type-aware fusion written as a subtraction of redundancy:
        ii_lin = total - redundant = sum_r (1 - g_r) * m_r.
        The gate is either one global scalar per type, g_r = sigmoid(p_r),
        or, with kan_gate_mode = 'kan', a per-item score
        g_r(i) = sigmoid(KAN(LN([m_r]_i))) initialised at the global value.
        An optional KAN branch adds coeff * KAN(LN([m_r])); with
        scale_mode = 'match' it is rescaled to ||delta|| = coeff * ||ii_lin||
        per item, with 'free' its magnitude is learned."""
        if not ii_msgs:
            return 0.0
        total = sum(m for _, m in ii_msgs)
        n_views = len(ii_msgs)
        M = None
        if n_views > 1 and ((self.use_kan_fusion and self.kan_fusion is not None)
                            or self.kan_gate is not None):
            views = [F.layer_norm(m, (m.shape[1],)) if self.kan_fusion_norm else m for _, m in ii_msgs]
            M = torch.cat(views, dim=1)                        # (I, k·d)
        if self.type_gate and n_views > 1:
            if self.kan_gate is not None:
                gates = torch.sigmoid(self.kan_gate(M))  # (I, k) per-item gates
                redundant = sum(gates[:, r].unsqueeze(1) * m for r, (_, m) in enumerate(ii_msgs))
            else:
                redundant = sum(torch.sigmoid(self.type_gate_p[name]) * m for name, m in ii_msgs)
            ii_lin = total - redundant
        else:
            ii_lin = total
        if self.use_kan_fusion and self.kan_fusion is not None and n_views > 1:
            h = self.kan_fusion(M)                             # (I, d)
            if self.kan_fusion_coeff_learnable:
                coeff = self.kan_fusion_coeff_max * torch.sigmoid(self._kan_fusion_theta)
            else:
                coeff = self._kan_fusion_coeff_p
            if self.kan_fusion_scale_mode == 'match':
                # per-item scale matching, ||delta|| = coeff * ||ii_lin||, so that a growing W cannot
                # take over the fused representation; skipped while h is close to zero, to avoid a spike
                hnorm = h.norm(dim=-1, keepdim=True)
                scale = torch.where(hnorm > 1e-8,
                                    ii_lin.norm(dim=-1, keepdim=True) / (hnorm + 1e-8),
                                    torch.zeros_like(hnorm))
                ii_lin = ii_lin + coeff * scale * h
            else:  # free: the magnitude of the KAN output is left to W
                ii_lin = ii_lin + coeff * h
        return ii_lin

    def forward(self, test=True):
        cooc_src, img_src, txt_src = self._item_sources()

        # collaborative channel: interaction edges only
        ego = torch.cat([self.user_embedding.weight, self.item_id_embedding.weight], dim=0)
        collab_layers = []
        for _ in range(self.n_layers):
            ego = torch.sparse.mm(self.norm_adj, ego)
            collab_layers.append(ego)
        collab = torch.mean(torch.stack(collab_layers, dim=1), dim=1)

        # single-hop item-item messages, fused as messages and then lifted to users
        ii_msgs = []
        if self.use_cooc and self.A_cooc is not None:
            ii_msgs.append(('cooc', torch.sparse.mm(self.A_cooc, cooc_src)))
        if self.use_img and self.A_img is not None and img_src is not None:
            ii_msgs.append(('img', torch.sparse.mm(self.A_img, img_src)))
        if self.use_txt and self.A_txt is not None and txt_src is not None:
            ii_msgs.append(('txt', torch.sparse.mm(self.A_txt, txt_src)))
        self._last_ii_msgs = ii_msgs
        ii_enh = self._aggregate_ii(ii_msgs)

        # fused channel: fusion inside propagation, item = collaborative + item-item per layer
        if isinstance(ii_enh, float):
            fused = torch.zeros_like(collab)
        else:
            m_enh_u = torch.sparse.mm(self.norm_R, ii_enh)  # lift the item-item enhancement to users (global semantics)
            g_u = self.user_embedding.weight
            g_i = self.item_id_embedding.weight
            fused_u_layers, fused_i_layers = [], []
            for _ in range(self.n_layers):
                g_u = torch.sparse.mm(self.norm_R, g_i)  # collaborative message, user <- item
                m_cf = torch.sparse.mm(self.norm_R_t, g_u)  # collaborative message, item <- user
                g_i = m_cf + ii_enh  # fusion inside propagation: item = collaborative + enhancement
                fused_u_layers.append(g_u + m_enh_u)  # user = collaborative + global semantics
                fused_i_layers.append(g_i)
            fused_u = torch.mean(torch.stack(fused_u_layers, dim=1), dim=1)
            fused_i = torch.mean(torch.stack(fused_i_layers, dim=1), dim=1)
            fused = torch.cat([fused_u, fused_i], dim=0)

        final = collab + self.fusion_coeff * fused
        final_u, final_i = torch.split(final, [self.n_users, self.n_items])

        if test:
            return final_u, final_i
        return final_u, final_i, collab, fused

    # ------------------------------------------------------------------ #
    # losses
    # ------------------------------------------------------------------ #
    def cal_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        return -torch.mean(F.logsigmoid(pos_scores - neg_scores))

    def cal_cl_loss(self, embed_1, embed_2, tau):
        embed_1, embed_2 = F.normalize(embed_1, p=2, dim=1), F.normalize(embed_2, p=2, dim=1)
        pos = torch.sum(embed_1 * embed_2, dim=-1)
        tot = torch.matmul(embed_1, torch.transpose(embed_2, 0, 1))
        gcl_logits = tot - pos[:, None]
        clogits = torch.logsumexp(gcl_logits / tau, dim=1)
        return torch.mean(clogits)

    def cal_reg_loss(self):
        reg_loss = 0
        for param in self.parameters():
            reg_loss += param.norm(2).square()
        return reg_loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        final_u, final_i, collab, fused = self.forward(test=False)

        bpr = self.cal_bpr_loss(final_u[users], final_i[pos_items], final_i[neg_items])
        reg = self.cal_reg_loss()
        loss = bpr + self.reg_weight * reg

        if self.use_align:
            collab_u, collab_i = torch.split(collab, [self.n_users, self.n_items])
            align_u = 0.0
            align_i = 0.0
            for _, m in self._last_ii_msgs:
                view_u = torch.sparse.mm(self.norm_R, m)
                align_u = align_u + self.cal_cl_loss(collab_u[users], view_u[users], self.tau)
                align_i = align_i + self.cal_cl_loss(collab_i[pos_items], m[pos_items], self.tau)
            loss = loss + self.cl_weight * (align_u + align_i)

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        self.restore_user_e, self.restore_item_e = self.forward()
        # print the diagnostics once, in the first round only
        if not self._gates_logged:
            self._gates_logged = True
            if self.type_gate:
                gates = {k: round(torch.sigmoid(v).item(), 4) for k, v in self.type_gate_p.items()}
                self.logger.info('TYPE_GATE (cooc/img/txt) = %s' % gates)
            if self.use_kan_fusion:
                if self.kan_fusion_coeff_learnable:
                    coeff = (self.kan_fusion_coeff_max * torch.sigmoid(self._kan_fusion_theta)).item()
                else:
                    coeff = self._kan_fusion_coeff_p.item()
                self.logger.info('KAN_FUSION basis=%s coeff=%.4f scale_mode=%s gate_mode=%s'
                                 % (self.kan_fusion_basis, coeff, self.kan_fusion_scale_mode,
                                    self.kan_gate_mode))
        scores = torch.matmul(self.restore_user_e[user], self.restore_item_e.transpose(0, 1))
        return scores


takhnet = TaKHNet
