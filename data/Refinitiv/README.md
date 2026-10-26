# Refinitiv M&A graph (not included)

The `Refinitiv` dataset option uses **proprietary LSEG (formerly Refinitiv / Thomson Reuters) SDC mergers-and-acquisitions deal data**. This data is licensed. It is **not included** in this repository, and it is not covered by the repository's MIT license.

You can use this option only if you have your own license for SDC M&A data, for example through LSEG Workspace or through WRDS where your institution's subscription includes it. Build the file yourself from your own export, keep it on your machine, and follow the terms of your license. Do not commit it or share it: LSEG and WRDS terms do not allow public or bulk redistribution, and SDC records can name private individuals.

`.gitignore` excludes everything in `data/Refinitiv/` except this README. Without the file, `python main.py --dataset Refinitiv` stops with a `FileNotFoundError` that points to this page.

## Fields to export

One row per M&A deal:

| Field | Used for |
|---|---|
| SDC deal number | edge attribute `deal_id` (dropped by the loader, kept for traceability) |
| Target name | source node of the deal edge |
| Target macro industry | node attribute `industry` (the class label) |
| Acquirer name | target node of the deal edge |
| Acquirer macro industry | node attribute `industry` (the class label) |
| Target city and nation | `city` nodes with the node attribute `nation` (used as the node feature) |

The labels are the SDC "macro industry" classes, for example Financials, Industrials, High Technology, Healthcare, Energy and Power, Real Estate, Materials, Retail, Telecommunications, Media and Entertainment, Consumer Products and Services, Consumer Staples, and Government and Agencies. The file used during development had about 175,000 nodes (about 159,000 companies and 16,000 cities), about 98,000 deal edges and 14 label values.

## Expected file format

Path: `data/Refinitiv/refinitiv.gpickle`, relative to the folder you run `main.py` from.

The file is a Python pickle (written with `pickle.dump`) of a `networkx.DiGraph`:

- **Nodes** are keyed by the entity name (a string).
  - `type`: `'company'` or `'city'`.
  - `industry`: the SDC macro industry (a string). This becomes the label `y`.
  - `nation`: the nation (a string). This becomes a one-hot node feature `x`.
  - A node may miss `industry` or `nation`; the loader fills a missing value with the string `'None'`, which then acts as its own class or feature value.
- **Edges** go from the target to the acquirer, one edge per deal, each with the attribute `deal_id` (an integer). Every edge must carry the same attribute keys.

`utils.preprocess_refinitiv` then:

1. sets `y` to the index of `industry` in the sorted list of industry values;
2. sets `x` to the one-hot code of `nation`, then row-normalises it;
3. converts the graph with `torch_geometric.utils.from_networkx` and drops the `industry`, `nation` and `deal_id` attributes;
4. uses all nodes for training (`split_type='full'`), or a 60/20/20 random split (`split_type='random'`).

## Building the file

A minimal sketch, to run on your own licensed export (the column names are examples; use the names of your export):

```python
import pickle

import networkx as nx
import pandas as pd

deals = pd.read_csv("my_sdc_export.csv")  # your own licensed export; never commit it

G = nx.DiGraph()
for row in deals.itertuples(index=False):
    G.add_node(row.target_name, type="company", industry=row.target_macro_industry)
    G.add_node(row.acquirer_name, type="company", industry=row.acquirer_macro_industry)
    G.add_node(row.target_city, type="city", nation=row.target_nation)
    G.add_edge(row.target_name, row.acquirer_name, deal_id=int(row.deal_number))

with open("data/Refinitiv/refinitiv.gpickle", "wb") as f:
    pickle.dump(G, f)
```

Only load pickle files that you built yourself: loading a pickle can run code.

## Citing

If you report results on this dataset, cite the data source as your license requires (for example LSEG Data & Analytics, SDC Platinum / SDC M&A database; or WRDS if you accessed it there) and add a data-availability statement saying the data is proprietary and not redistributed.
