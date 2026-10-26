# Third-Party Notices

RareGraph includes code and data from the projects below. For each one we list what was reused, where it lives in this repository, and the upstream license. Where the upstream license requires it, the full copyright and permission notice is reproduced.

The MIT license in [LICENSE](LICENSE) covers the QGI Lab's own code only. The third-party code below keeps its own license and copyright notices. No data in this repository is covered by the MIT license: each dataset keeps the terms of its source (sections 4 to 6). The licensed Refinitiv (LSEG) dataset is not included at all; see `data/Refinitiv/README.md`.

---

## 1. GraphENS

- Source: https://github.com/JoonHyung-Park/GraphENS (compared at commit 67833317)
- Paper: Park, Song & Yang, "GraphENS: Neighbor-Aware Ego Network Synthesis for Class-Imbalanced Node Classification", ICLR 2022.
- Used in:
  - `models/gcn.py` (from `nets/gcn.py`; only the `GCN` wrapper class is changed)
  - `models/gat.py` (from `nets/gat.py`; only the `GAT` wrapper class is changed)
  - `models/sage.py` (from `nets/sage.py`; only the `SAGE` wrapper class is changed)
  - `losses/ce.py` (copied from `losses/ce.py`)
  - `models/gatv2.py` (`GATv2` wrapper class follows the GraphENS `GAT` wrapper)
- License: MIT

```
MIT License

Copyright (c) 2022 JoonHyung-Park

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. PyTorch Geometric (PyG)

- Source: https://github.com/pyg-team/pytorch_geometric
- Used in:
  - `models/gcn.py` (`gcn_norm`, `GCNConv`), `models/gat.py` (`GATConv`), `models/sage.py` (`SAGEConv`): modified copies of PyG layers at commit `97d55577f1d0bf33c1bfbe0ef864923ad5cb844d`, reached through GraphENS.
  - `models/gatv2.py` (`GATv2Conv`): copy of `torch_geometric/nn/conv/gatv2_conv.py` from PyG 2.5.x.
- License: MIT. The license text is the same for both versions; only the copyright line differs.

Copyright line at commit 97d55577:

```
Copyright (c) 2021 Matthias Fey, Jiaxuan You <matthias.fey@tu-dortmund.de, jiaxuan@cs.stanford.edu>
```

License at PyG 2.5.x:

```
Copyright (c) 2023 PyG Team <team@pyg.org>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## 3. Methods implemented from papers (no code reused)

These parts are our own code, written from the published method descriptions. No code from the authors' repositories is included. We credit the papers as the method source.

- **GraphSMOTE**: Zhao, Zhang & Wang, "GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks", WSDM 2021, doi:10.1145/3437963.3441720. Implemented in `utils.py` (`split_arti`, `src_upsample`, `src_smote`) and in the imbalanced split set-up in `main.py`. Independent implementation. The authors' repository (https://github.com/TianxiangZhao/GraphSmote) has no license file, so none of its code is used here.
- **Focal loss**: Lin, Goyal, Girshick, He & Dollar, "Focal Loss for Dense Object Detection", ICCV 2017. Implemented in `losses/focal.py`.

---

## 4. Planetoid datasets (Cora, CiteSeer, PubMed)

- Source: https://github.com/kimiyoung/planetoid (`data/ind.*` files; the `raw/ind.*` files in `data/Cora`, `data/CiteSeer` and `data/PubMed` are byte-identical to upstream). The `processed/*.pt` files are the same data converted by PyTorch Geometric's `Planetoid` loader.
- Paper: Yang, Cohen & Salakhutdinov, "Revisiting Semi-Supervised Learning with Graph Embeddings", ICML 2016. The underlying citation networks come from the LINQS group (Sen et al., "Collective Classification in Network Data", AI Magazine 2008).
- License: MIT

```
The MIT License (MIT)
Copyright (c) 2016 Zhilin Yang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

---

## 5. TUDataset (NCI1, PROTEINS)

- Source: https://chrsmrrs.github.io/datasets/ (the `raw/` files in `data/NCI1` and `data/PROTEINS`, including the upstream `README.txt`; the `processed/*.pt` files are the same data converted by PyTorch Geometric's `TUDataset` loader)
- Paper: Morris et al., "TUDataset: A Collection of Benchmark Datasets for Learning with Graphs", GRL+ Workshop at ICML 2020. Please also cite the original dataset papers listed in each `README.txt`.
- License: no explicit license is given by the collection; the datasets are distributed for research use with citation.

---

## 6. Wikidata (Wiki graph)

- Source: https://www.wikidata.org. The file `data/Wiki/wiki.gpickle` is a company graph built at the QGI Lab from Wikidata items; see `data/Wiki/README.md`.
- License: Wikidata content is released under CC0 1.0 (https://www.wikidata.org/wiki/Wikidata:Licensing). We credit Wikidata as the source.

---

## 7. Data downloaded at run time (not included)

- Reddit (`--dataset Reddit`) is downloaded by `torch_geometric.datasets.Reddit2` on first use (GraphSAINT version: Zeng et al., ICLR 2020; original data: Hamilton, Ying & Leskovec, NeurIPS 2017). It is not part of this repository.
