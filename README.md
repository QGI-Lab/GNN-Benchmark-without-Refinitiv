# RareGraph: Graph Neural Networks on Rare Classes

RareGraph is a benchmark for Graph Neural Networks (GNNs) when some classes are rare. It trains GCN, GraphSAGE, GAT and GATv2 models under controlled class imbalance, with optional minority oversampling (upsampling or SMOTE), and reports metrics suited to imbalanced data. It supports node classification and link prediction on several datasets.

Developed at the Quantum Geometric Intelligence Lab (QGI Lab), NYU Tandon School of Engineering, led by Prof. Amine M. Aboussalah.

## Requirements

The code requires Python >= 3.8. Install the dependencies from the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Run all commands from the repository root: the data loaders read files from `./data`.

## Usage

You can customize the execution of the benchmark through various command-line arguments:

- **`--task`**: Type of task to perform.

  - `node`: For node classification tasks.
  - `edge`: For link prediction tasks.
- **`--dataset`**: Datasets available. Default: `Cora`

  - `Cora`, `Citeseer`, `Pubmed`: Traditional citation network datasets (included).
  - `NCI1`, `Proteins`: TUDataset graph collections (included).
  - `Reddit`: Larger dataset, downloaded by PyTorch Geometric on first use.
  - `Wiki`: Company knowledge graph built from Wikidata (included).
  - `Refinitiv`: M&A deal graph. **Requires licensed data that is not included**; see [data/Refinitiv/README.md](data/Refinitiv/README.md).
- **`--model`**: GNN models supported.

  - `gcn`: Graph Convolutional Network.
  - `sage`: GraphSAGE, uses sampled neighborhood aggregation.
  - `gat`: Graph Attention Network, applies attention mechanisms in message passing.
  - `gatv2`: Graph Attention Network v2, fixes the static attention problem of the standard [GAT](https://paperswithcode.com/method/gat) layer.
- **`--device`**: Execution device.

  - `cuda`: For GPU acceleration (if available).
  - `cpu`: For CPU processing.
- **`--num_layers`**: Controls the number of GNN layers. Default: `2`
- **`--hidden_dim`**: Sets the dimension of hidden layers in GNN models. Default: `128`
- **`--gat_heads`**: Specifies the number of attention heads in GAT models. Default: `8`
- **`--gatv2_heads`**: Specifies the number of attention heads in GATv2 models. Default: `8`
- **`--seed`**: Seed for random number generation. Default: `42`
- **`--epochs`**: Determines the number of training epochs. Default: `100`
- **`--lr`** (Learning Rate): Configures the learning rate for the optimizer. Default: `0.01`
- **`--weight_decay`**: Applies weight decay (L2 penalty) to prevent overfitting. Default: `0.0005`
- **`--batch_size`**: Sets the batch size for training sessions. Default: `256`
- **`--loss`**: Chooses the loss function, with options of `ce` (Cross-Entropy Loss) or `focal` (Focal Loss, Lin et al. 2017, with `alpha = 1`, `gamma = 2`). Default: `ce`

### Imbalance settings

The training split is made imbalanced on purpose, following the protocol of GraphSMOTE (Zhao et al., WSDM 2021):

- **`--class_sample_num`**: Number of training nodes per majority class. Default: `20`
- **`--im_class_num`**: Number of minority classes; these are the classes with the largest label ids. Default: `1`
- **`--imb_ratio`**: Imbalance ratio. Each minority class gets `int(class_sample_num * imb_ratio)` training nodes. Default: `1`
- Every class also gets 25 validation nodes and 55 test nodes.
- **`--imb_setting`**: How the minority classes are oversampled before training. Default: `upsample`

  - `upsample`: duplicate the minority training nodes together with their edges.
  - `smote`: make synthetic minority nodes by interpolating each minority training node with its nearest same-class training node in feature space; a new node takes the edges of the node it was made from.
- **`--up_scale`**: Number of synthetic nodes per minority training node. With `0`, the amount is set from the mean number of training nodes per class. Default: `1`

Note: the split and oversampling step in `main.py` uses the dense adjacency matrix `adj` that `utils.load_data` builds for `Cora`, `Citeseer`, `Pubmed`, `NCI1` and `Proteins`. The `Reddit`, `Wiki` and `Refinitiv` loaders do not build it yet.

### Metrics

For node classification, the script reports test loss, accuracy, balanced accuracy, micro / macro / weighted F1, AUC and macro AUC (one-vs-rest), per-class G-mean, Cohen's kappa and the Matthews correlation coefficient. For link prediction, it reports test loss and AUC.

### Example Command

For running a node classification on the Citeseer dataset with the GraphSAGE model on a GPU:

```bash
python main.py --task node --dataset Citeseer --model sage --device cuda
```

For a GCN on Cora with one minority class at 10% of the majority training size, oversampled with SMOTE, on CPU:

```bash
python main.py --task node --dataset Cora --model gcn --device cpu --im_class_num 1 --imb_ratio 0.1 --imb_setting smote
```

## Data

**Included** (each keeps the terms of its source; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)):

- `data/Cora`, `data/CiteSeer`, `data/PubMed`: Planetoid splits of the citation networks (Yang, Cohen & Salakhutdinov, ICML 2016; MIT License), raw files plus the PyTorch Geometric processed files.
- `data/NCI1`, `data/PROTEINS`: from the TUDataset collection (Morris et al., 2020), raw files with the upstream `README.txt` plus the PyTorch Geometric processed files.
- `data/Wiki`: a company knowledge graph built at the QGI Lab from Wikidata (CC0 1.0). See [data/Wiki/README.md](data/Wiki/README.md).

**Not included:**

- `Refinitiv`: proprietary LSEG (Refinitiv) SDC M&A deal data. It cannot be redistributed. You can use this option only under your own LSEG Workspace or WRDS license; [data/Refinitiv/README.md](data/Refinitiv/README.md) lists the fields and the file format the loader expects.
- `Reddit`: downloaded by `torch_geometric.datasets.Reddit2` into `data/Reddit` on first use.

## Acknowledgments

Parts of this code come from other open research projects, or implement methods published by other authors. We thank them. Reused files carry a short header comment, and license texts are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- **GraphSMOTE** (Zhao, Zhang & Wang, WSDM 2021).
  The imbalanced split (`utils.split_arti`), the upsampling (`utils.src_upsample`), the SMOTE oversampling (`utils.src_smote`) and the split set-up in `main.py` implement the method of GraphSMOTE. This is an independent implementation written from the paper; it contains no code from the GraphSMOTE repository.
- **GraphENS** (Park, Song & Yang, ICLR 2022), https://github.com/JoonHyung-Park/GraphENS, MIT License.
  The GNN backbones in `models/gcn.py`, `models/gat.py` and `models/sage.py` are adapted from its `nets/` folder, and `losses/ce.py` is copied from its `losses/ce.py`. The overall script layout (`args.py`, `main.py`, `utils.load_data`) follows the same pattern.
- **PyTorch Geometric** (Fey & Lenssen, 2019), https://github.com/pyg-team/pytorch_geometric, MIT License.
  The convolution layers `GCNConv`, `GATConv` and `SAGEConv` (through GraphENS) and `GATv2Conv` in `models/gatv2.py` are modified copies of PyG layers.
- **Focal loss** (Lin et al., ICCV 2017): `losses/focal.py` is our own implementation of the published formula.
- **Datasets.** Cora, CiteSeer and PubMed raw files are from Planetoid (Yang, Cohen & Salakhutdinov, ICML 2016), https://github.com/kimiyoung/planetoid, MIT License. NCI1 and PROTEINS are from the TUDataset collection (Morris et al., 2020), https://chrsmrrs.github.io/datasets/. The Wiki graph is built from Wikidata, https://www.wikidata.org (CC0 1.0).

If you use these methods or datasets, please cite the original papers:

```bibtex
@inproceedings{zhao2021graphsmote,
  title     = {GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks},
  author    = {Zhao, Tianxiang and Zhang, Xiang and Wang, Suhang},
  booktitle = {Proceedings of the 14th ACM International Conference on Web Search and Data Mining (WSDM)},
  pages     = {833--841},
  year      = {2021},
  doi       = {10.1145/3437963.3441720}
}

@inproceedings{park2022graphens,
  title     = {Graph{ENS}: Neighbor-Aware Ego Network Synthesis for Class-Imbalanced Node Classification},
  author    = {Park, Joonhyung and Song, Jaeyun and Yang, Eunho},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  url       = {https://openreview.net/forum?id=MXEl7i-iru}
}

@inproceedings{fey2019pyg,
  title     = {Fast Graph Representation Learning with {PyTorch Geometric}},
  author    = {Fey, Matthias and Lenssen, Jan E.},
  booktitle = {ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year      = {2019}
}

@inproceedings{lin2017focal,
  title     = {Focal Loss for Dense Object Detection},
  author    = {Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages     = {2980--2988},
  year      = {2017}
}

@inproceedings{yang2016planetoid,
  title     = {Revisiting Semi-Supervised Learning with Graph Embeddings},
  author    = {Yang, Zhilin and Cohen, William W. and Salakhutdinov, Ruslan},
  booktitle = {Proceedings of the 33rd International Conference on Machine Learning (ICML)},
  pages     = {40--48},
  year      = {2016}
}

@inproceedings{morris2020tudataset,
  title     = {{TUDataset}: A Collection of Benchmark Datasets for Learning with Graphs},
  author    = {Morris, Christopher and Kriege, Nils M. and Bause, Franka and Kersting, Kristian and Mutzel, Petra and Neumann, Marion},
  booktitle = {ICML 2020 Workshop on Graph Representation Learning and Beyond (GRL+)},
  year      = {2020}
}
```

## License

The QGI Lab's own code in this repository is released under the MIT License; see [LICENSE](LICENSE).

- Third-party code keeps its own license and copyright notices (GraphENS and PyTorch Geometric, both MIT). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- Data is **not** covered by the MIT License. Each included dataset keeps the terms of its source (see the Data section and THIRD_PARTY_NOTICES.md). The licensed Refinitiv data is not included.
