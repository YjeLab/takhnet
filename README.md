# TaKHNet

Model code for **TaKHNet: Time-aware KAN-based Heterogeneous Multimodal
Recommendation** (Machine Intelligence Research, 2026).

## What this repository contains

The model definition only: the network, the graph construction inside it, and the
Kolmogorov–Arnold view-fusion layer. It is the code that implements the method, not the
project that produced the numbers.

What is **not** here, on purpose:

- the training loop, the evaluation protocol and the hyper-parameter search,
- the data pipeline that loads images, text features and interaction splits,
- the baselines used in the comparison,
- the datasets.

Because of that, the files are **not runnable as they stand**. They are written for an
MMRec-style framework and expect the host project to provide the base class, the
similarity helpers and the data. Nothing here depends on private data or on a particular
machine.

## Files

```
models/takhnet.py     TaKHNet and KANFusionLayer
config/takhnet.yaml   the configuration reported in the paper (Baby), with the
                      per-dataset settings of Sports and Electronics as comments
LICENSE               MIT
```

## Method in one screen

The item graph has three content views plus the user-item bipartite graph, and every view
is sparse, so memory grows with the number of edges rather than with items × items.

| View | Edges | Note |
| --- | --- | --- |
| interaction | user ↔ item | LightGCN-normalised rating matrix |
| temporal co-occurrence | item ↔ item | weight = mean over co-purchase events of exp(-Δt / τ), τ = 90 days |
| visual | item ↔ item | kNN on ZCA-whitened visual features, behind a learned gate |
| textual | item ↔ item | kNN on text features |

The model keeps two channels. A **collaborative channel** propagates only interaction
edges, so it stays free of content noise. A **fused channel** propagates all edges and
aggregates type-aware messages inside propagation, layer by layer, rather than at the
output.

Three ingredients act on the content side:

1. **Visual decorrelation.** Frozen visual features are projected, whitened with
   ZCA-shrinkage, and passed through a learned gate; the whitening removes the low-rank
   redundancy that makes cosine similarity between images nearly uniform.
2. **Text-guided redundancy removal.** The visual gate is multiplied by the image–text
   semantic consistency, so a visual dimension that carries no relation to the item text
   is down-weighted.
3. **KAN cross-view fusion.** A Kolmogorov–Arnold layer over the concatenated view
   messages adds a nonlinear residual to the linear type-aware sum,
   `ii_enh = Σ (1 - g_r) m_r + coeff · KAN(LN([m_cooc; m_img; m_txt]))`,
   with the residual rescaled per item so that it cannot dominate the linear term.

The two channels are aligned with an InfoNCE term on both the user and the item side.

## What the host project must provide

`TaKHNet` inherits from a `GeneralRecommender` base class and calls two graph helpers, so
the surrounding project has to supply:

| Name | Expected interface |
| --- | --- |
| `common.abstract_recommender.GeneralRecommender` | sets `n_users`, `n_items`, `v_feat`, `t_feat`, `inter_matrix` |
| `utils.utils.build_sim` | cosine similarity between an item feature matrix and itself |
| `utils.utils.build_knn_normalized_graph` | kNN sparsification and symmetric normalisation of a similarity matrix |

The model also reads the raw `.inter` file named by `data_path` + `dataset` to obtain the
timestamps used by the co-occurrence weights; the column holding them is `TIME_FIELD`.

Every hyper-parameter is read from the config dictionary, so a YAML file such as
`config/takhnet.yaml` is enough to instantiate the model. The switches starting with
`use_` and the `_mode` keys reproduce the ablation variants of the paper.

## Citation

```bibtex
@article{takhnet2026,
  title   = {TaKHNet: Time-aware {KAN}-based Heterogeneous Multimodal Recommendation},
  author  = {Yin, Jie and Li, Donghuan and Lu, Keke and Liu, Chunjiang},
  journal = {Machine Intelligence Research},
  year    = {2026}
}
```

## License

MIT. Use it, modify it, ship it in a product if you like — just keep the copyright
notice and do not expect a warranty. If this code helps your work, a citation is
appreciated.

## Contact

Correspondence: Chunjiang Liu, Chengdu Library and Information Center, Chinese Academy of
Sciences (liucj@clas.ac.cn).
