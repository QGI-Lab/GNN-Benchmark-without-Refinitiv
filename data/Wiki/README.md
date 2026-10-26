# Wiki company graph

`wiki.gpickle` is a company knowledge graph built at the QGI Lab from **Wikidata** (https://www.wikidata.org). Wikidata content is released under CC0 1.0 (public domain dedication; see https://www.wikidata.org/wiki/Wikidata:Licensing). We credit Wikidata as the source.

## Contents

The file is a Python pickle of a `networkx.DiGraph` with 17,713 nodes and 29,839 directed edges. Nodes are keyed by integers.

- Node attribute `label`, one of: `Company` (13,235), `Location` (2,066), `Industry` (1,252), `Person` (973), `Group` (160), `Quant_Metric` (27).
- Node attribute `id`: the Wikidata item id (for example `Q4545752`). `Quant_Metric` nodes are value bands instead (for example `3_profit`, `6_market_cap`).
- `Company` nodes also carry `name` and, when known, `start_date` (inception date). `Person` nodes carry only their Wikidata id.
- Edge attribute `label`, one of: `RESIDES_IN`, `HEADQUARTERS`, `INDUSTRY`, `ACQUIRED`, `CEO`, `CHAIRMAN`, `GROUPED_IN`, and the band edges `PROFIT_BAND`, `EQUITY_BAND`, `MARKET_CAP_BAND`, `EMPLOYEE_COUNT_BAND`, `ASSETS_BAND`.

`utils.preprocess_wiki` loads it with `utils.gpickle_to_PyG` (identity node features, edge list from the graph).
