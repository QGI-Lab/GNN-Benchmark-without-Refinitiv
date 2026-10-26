import torch
import numpy as np
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data
import os
from torch_geometric.utils import to_dense_adj, subgraph
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import InMemoryDataset
import pickle
import random

def load_data(name, path, split_type='full'):
    # Load Planetoid Dataset
    if name in ['Cora', 'Citeseer', 'Pubmed', 'NCI1', 'Proteins']:
        if name == 'Cora':
            data_path = './data/Cora/Cora/processed/data.pt'
            loaded_data = torch.load(data_path)
            graph_data = loaded_data[0]  
            x = graph_data['x']
            edge_index = graph_data['edge_index']
            y = graph_data['y']
            # Assuming 'edge_index' contains no edge attributes or they should be ignored
            adjacency_matrix = to_dense_adj(edge_index, edge_attr=None)
            # Convert any non-zero value to 1 to enforce a binary adjacency matrix
            binary_adjacency_matrix = (adjacency_matrix > 0).float()
            # Flatten the tensor if extra dimension exists
            if binary_adjacency_matrix.dim() > 2:
                binary_adjacency_matrix = binary_adjacency_matrix.squeeze(0)
            # Create a PyTorch Geometric Data object
            dataset = Data(x=x, edge_index=edge_index, y=y, adj = binary_adjacency_matrix)

        if name == 'Citeseer':
            data_path = './data/CiteSeer/CiteSeer/processed/data.pt'
            loaded_data = torch.load(data_path)
            graph_data = loaded_data[0]  
            x = graph_data['x']
            edge_index = graph_data['edge_index']
            y = graph_data['y']
            # Assuming 'edge_index' contains no edge attributes or they should be ignored
            adjacency_matrix = to_dense_adj(edge_index, edge_attr=None)
            # Convert any non-zero value to 1 to enforce a binary adjacency matrix
            binary_adjacency_matrix = (adjacency_matrix > 0).float()
            # Flatten the tensor if extra dimension exists
            if binary_adjacency_matrix.dim() > 2:
                binary_adjacency_matrix = binary_adjacency_matrix.squeeze(0)
            # Create a PyTorch Geometric Data object
            dataset = Data(x=x, edge_index=edge_index, y=y, adj = binary_adjacency_matrix)

        if name == 'Pubmed':
            data_path = './data/PubMed/PubMed/processed/data.pt'
            loaded_data = torch.load(data_path)
            graph_data = loaded_data[0]  
            x = graph_data['x']
            edge_index = graph_data['edge_index']
            y = graph_data['y']
            # Assuming 'edge_index' contains no edge attributes or they should be ignored
            adjacency_matrix = to_dense_adj(edge_index, edge_attr=None)
            # Convert any non-zero value to 1 to enforce a binary adjacency matrix
            binary_adjacency_matrix = (adjacency_matrix > 0).float()
            # Flatten the tensor if extra dimension exists
            if binary_adjacency_matrix.dim() > 2:
                binary_adjacency_matrix = binary_adjacency_matrix.squeeze(0)
            # Create a PyTorch Geometric Data object
            dataset = Data(x=x, edge_index=edge_index, y=y, adj = binary_adjacency_matrix)
        
        if name == 'NCI1':
            data_path = './data/NCI1/NCI1/processed/data.pt'
            loaded_data = torch.load(data_path)
            graph_data = loaded_data[0]  
            x = graph_data['x']
            edge_index = graph_data['edge_index']
            y = graph_data['y']
            # Assuming 'edge_index' contains no edge attributes or they should be ignored
            adjacency_matrix = to_dense_adj(edge_index, edge_attr=None)
            # Convert any non-zero value to 1 to enforce a binary adjacency matrix
            binary_adjacency_matrix = (adjacency_matrix > 0).float()
            # Flatten the tensor if extra dimension exists
            if binary_adjacency_matrix.dim() > 2:
                binary_adjacency_matrix = binary_adjacency_matrix.squeeze(0)
            # Create a PyTorch Geometric Data object
            dataset = Data(x=x, edge_index=edge_index, y=y, adj = binary_adjacency_matrix)
        
        if name == 'Proteins':
            data_path = './data/PROTEINS/PROTEINS/processed/data.pt'
            loaded_data = torch.load(data_path)
            graph_data = loaded_data[0]  
            x = graph_data['x']
            edge_index = graph_data['edge_index']
            y = graph_data['y']
            # Assuming 'edge_index' contains no edge attributes or they should be ignored
            adjacency_matrix = to_dense_adj(edge_index, edge_attr=None)
            # Convert any non-zero value to 1 to enforce a binary adjacency matrix
            binary_adjacency_matrix = (adjacency_matrix > 0).float()
            # Flatten the tensor if extra dimension exists
            if binary_adjacency_matrix.dim() > 2:
                binary_adjacency_matrix = binary_adjacency_matrix.squeeze(0)
            # Create a PyTorch Geometric Data object
            dataset = Data(x=x, edge_index=edge_index, y=y, adj = binary_adjacency_matrix)
    # Load Custom Dataset
    elif name in ['Reddit', 'Wiki', 'Refinitiv']:
        if name == 'Reddit':
            dataset = torch_geometric.datasets.Reddit2(path, transform=T.NormalizeFeatures())[0]
        if name == 'Wiki':
            dataset = preprocess_wiki()
        if name == 'Refinitiv':
            dataset = preprocess_refinitiv()
    else:
        raise NotImplementedError('Dataset not supported')
    return dataset

# ---------------------------------------------------------------------------
# Class-imbalance tools: imbalanced split, upsampling and SMOTE oversampling.
# Implements the method of GraphSMOTE (Zhao et al., WSDM 2021); independent implementation.
# Method source: T. Zhao, X. Zhang and S. Wang, "GraphSMOTE: Imbalanced Node Classification
# on Graphs with Graph Neural Networks", WSDM 2021, doi:10.1145/3437963.3441720.
# ---------------------------------------------------------------------------

VAL_NODES_PER_CLASS = 25
TEST_NODES_PER_CLASS = 55


def split_arti(labels, c_train_num):
    """Class-imbalanced train/val/test split of the nodes.

    Labels are class ids 0..K-1. For class k, the nodes of that class are put in a
    random order: the first ``c_train_num[k]`` go to the training set, the next
    VAL_NODES_PER_CLASS to the validation set and the next TEST_NODES_PER_CLASS to
    the test set. The training set is shuffled at the end.

    Returns three LongTensors of node ids (train, val, test) and a (K, 3) integer
    array that holds the requested train/val/test size of each class.
    """
    n_classes = len(set(labels.tolist()))
    sizes = np.zeros((n_classes, 3), dtype=int)
    sizes[:, 1] = VAL_NODES_PER_CLASS
    sizes[:, 2] = TEST_NODES_PER_CLASS

    parts = ([], [], [])  # train, val, test
    for k in range(n_classes):
        members = torch.nonzero(labels == k, as_tuple=True)[0].tolist()
        print(f'Class {k}: {len(members)} nodes')
        random.shuffle(members)
        sizes[k, 0] = c_train_num[k]
        start = 0
        for part, stop in zip(parts, np.cumsum(sizes[k])):
            part.extend(members[start:stop])
            start = stop

    random.shuffle(parts[0])
    train, val, test = (torch.tensor(p, dtype=torch.long) for p in parts)
    return train, val, test, sizes


def _minority_train_nodes(labels, idx_train, im_class_num):
    """Yield the training node ids of each minority class.

    The minority classes are the ``im_class_num`` classes with the largest
    labels; they are visited from the largest label down.
    """
    top = int(labels.max())
    train_labels = labels[idx_train]
    for cls in range(top, top - im_class_num, -1):
        yield idx_train[train_labels == cls]


def _append_nodes(adj, features, labels, idx_train, sources, new_features):
    """Add one new training node per entry of ``sources``.

    New node j gets the feature row ``new_features[j]``, the label of node
    ``sources[j]`` and a copy of that node's edges: its adjacency row and column
    are those of ``sources[j]``, so two new nodes are linked when their source
    nodes are linked. Returns the sparse adjacency, features, labels and
    training ids of the enlarged graph.
    """
    dense = adj.to_dense()
    n_old = dense.shape[0]
    sources = sources.to(dense.device)
    # Row/column order of the enlarged graph: all old nodes, then one copy per source.
    order = torch.cat([torch.arange(n_old, device=dense.device), sources])
    grown = dense.index_select(0, order).index_select(1, order)

    new_ids = torch.arange(n_old, n_old + sources.numel(),
                           dtype=idx_train.dtype, device=idx_train.device)
    return (grown.to_sparse(),
            torch.cat([features, new_features], dim=0),
            torch.cat([labels, labels[sources.to(labels.device)]], dim=0),
            torch.cat([idx_train, new_ids], dim=0))


def src_upsample(adj, features, labels, idx_train, portion=1.0, im_class_num=3):
    """Oversample the minority classes by duplicating their training nodes.

    A copy keeps the features, label and edges of its source node.
    - ``portion > 0``: every training node of a minority class is copied
      ``int(portion)`` times; the fractional part of ``portion`` adds one more
      copy of the first ``int(n * frac)`` training nodes of that class.
    - ``portion == 0``: each minority class is copied
      ``int(mean training nodes per class / training nodes in this class)`` times.
    Returns the sparse adjacency, features, labels and training ids after
    oversampling.
    """
    mean_per_class = int(idx_train.shape[0] / (int(labels.max()) + 1))
    copies = []
    for members in _minority_train_nodes(labels, idx_train, im_class_num):
        n = members.numel()
        if n == 0:
            continue  # no training node of this class to copy
        if portion == 0:
            full, extra = int(mean_per_class / n), 0
        else:
            full = int(portion)
            extra = int(n * (portion - full))
        copies.append(members.repeat(full))
        copies.append(members[:extra])

    sources = torch.cat(copies) if copies else idx_train.new_empty(0)
    return _append_nodes(adj, features, labels, idx_train, sources, features[sources])


def src_downsample(data, portion=0.5):
    num_nodes = data.num_nodes
    num_sampled = int(num_nodes * portion)
    node_indices = torch.randperm(num_nodes)[:num_sampled]
    
    new_edge_indices, _ = subgraph(node_indices, data.edge_index)
    
    # Check if new_edge_indices is empty or None
    if new_edge_indices is None or new_edge_indices.size(1) == 0:
        print("No edges were retained in the downsampled graph.")
        return Data(x=data.x[node_indices], edge_index=torch.empty((2, 0), dtype=torch.long))
    
    new_data = Data(x=data.x[node_indices], edge_index=new_edge_indices)
    
    # Copying over other attributes
    for key, item in data:
        if key in ['x', 'edge_index']:
            continue
        # Node-level attribute
        if item.size(0) == num_nodes:
            new_data[key] = item[node_indices]
        else:
            new_data[key] = item

    return new_data


def _interpolate_to_nearest(x):
    """Make one synthetic row per row of ``x`` (SMOTE step).

    Each row is moved towards its nearest other row (Euclidean distance) by a
    random fraction delta in [0, 1); one delta is drawn per call and shared by
    all rows of the call.
    """
    x64 = x.detach().double()
    dist = torch.cdist(x64, x64, compute_mode='donot_use_mm_for_euclid_dist')
    dist.fill_diagonal_(float('inf'))
    nearest = dist.argmin(dim=1).to(x.device)
    delta = random.random()
    return x + delta * (x[nearest] - x)


def src_smote(adj, features, labels, idx_train, portion=1.0, im_class_num=3):
    """Oversample the minority classes with SMOTE in input-feature space.

    For a minority class with n >= 2 training nodes, one synthetic node is made
    per training node by interpolating its features towards its nearest
    same-class training node. The synthetic node takes the label and the edges
    of the node it was made from. Classes with fewer than 2 training nodes are
    skipped.
    - ``portion > 0``: ``int(portion)`` rounds over all training nodes of the
      class, plus one round over the first ``int(n * frac)`` nodes for the
      fractional part (only if that is at least 2 nodes; neighbours are then
      searched within those nodes).
    - ``portion == 0``: the number of rounds is set from the mean number of
      training nodes per class (at least one round), with the same rule for the
      fractional part.
    Returns the sparse adjacency, features, labels and training ids after
    oversampling.
    """
    mean_per_class = int(idx_train.shape[0] / (int(labels.max()) + 1))
    sources, synthetic = [], []
    for members in _minority_train_nodes(labels, idx_train, im_class_num):
        n = members.numel()
        if n < 2:
            continue  # SMOTE needs a same-class neighbour
        if portion == 0:
            ratio = mean_per_class / n
            rounds = max(int(ratio), 1)
            frac = max(ratio - rounds, 0)
        else:
            rounds = int(portion)
            frac = portion - rounds

        batches = [members] * rounds
        n_extra = int(n * frac)
        if n_extra >= 2:
            batches.append(members[:n_extra])
        for batch in batches:
            sources.append(batch)
            synthetic.append(_interpolate_to_nearest(features[batch]))

    if sources:
        sources, new_features = torch.cat(sources), torch.cat(synthetic)
    else:
        sources, new_features = idx_train.new_empty(0), features[:0]
    return _append_nodes(adj, features, labels, idx_train, sources, new_features)


def imbalance_processing():
    pass

# Preprocessing for Wiki dataset
def preprocess_wiki(split_type='full'):
    current_path = os.getcwd()
    data_path = os.path.join(current_path, 'data', 'Wiki')
    gpickle_path = os.path.join(data_path, 'wiki.gpickle')
    wiki_data = gpickle_to_PyG(gpickle_path)
    wiki_data = NormalizeFeatures()(wiki_data)
    
    # Split the dataset
    if split_type == 'full':
        wiki_data.train_mask = torch.ones(wiki_data.num_nodes, dtype=torch.bool)
        wiki_data.val_mask = torch.zeros(wiki_data.num_nodes, dtype=torch.bool)
        wiki_data.test_mask = torch.zeros(wiki_data.num_nodes, dtype=torch.bool)
    elif split_type == 'random':
        torch.manual_seed(0)
        num_nodes = wiki_data.num_nodes
        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2
        
        perm = torch.randperm(num_nodes)
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)
        
        wiki_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        wiki_data.train_mask[perm[:train_size]] = True
        wiki_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        wiki_data.val_mask[perm[train_size:train_size+val_size]] = True
        wiki_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        wiki_data.test_mask[perm[train_size+val_size:]] = True
    print("After preprocessing: ", wiki_data)
    return wiki_data

# Preprocessing for Refinitiv dataset
def preprocess_refinitiv(split_type='full'):
    current_path = os.getcwd()
    data_path = os.path.join(current_path, 'data', 'Refinitiv')
    gpickle_path = os.path.join(data_path, 'refinitiv.gpickle')
    if not os.path.isfile(gpickle_path):
        raise FileNotFoundError(
            f"Refinitiv dataset not found at {gpickle_path}. This dataset is proprietary "
            "LSEG (Refinitiv) SDC M&A deal data and is NOT distributed with this repository. "
            "You can only use it under your own LSEG Workspace or WRDS license: build the "
            "file yourself as described in data/Refinitiv/README.md.")
    with open(gpickle_path, 'rb') as f:
        refinitiv_graph=pickle.load(f)

    # Identify all unique attributes and set default values for missing attributes
    default_value = 'None'
    unique_attrs = set()
    for _, attrs in refinitiv_graph.nodes(data=True):
        unique_attrs.update(attrs.keys())
    
    for node, attrs in refinitiv_graph.nodes(data=True):
        for attr in unique_attrs:
            refinitiv_graph.nodes[node].setdefault(attr, default_value)

    # Encode the industry attribute into numerical labels
    industry_labels = [attrs['industry'] for _, attrs in refinitiv_graph.nodes(data=True)]
    industries = sorted(set(industry_labels))
    industry_to_idx = {industry: i for i, industry in enumerate(industries)}
    y = torch.tensor([industry_to_idx[attrs['industry']] for _, attrs in refinitiv_graph.nodes(data=True)], dtype=torch.long)

    # Encode the nation attribute into numerical labels for use as features
    nation_labels = [attrs.get('nation', 'None') for _, attrs in refinitiv_graph.nodes(data=True)]
    nations = sorted(set(nation_labels))
    nation_to_idx = {nation: i for i, nation in enumerate(nations)}
    x = torch.tensor([nation_to_idx[nation] for nation in nation_labels], dtype=torch.long)
    x = torch.nn.functional.one_hot(x, num_classes=len(nations)).to(torch.float)

    refinitiv_data = from_networkx(refinitiv_graph)
    refinitiv_data.x = x
    refinitiv_data.y = y
    num_nodes = refinitiv_data.num_nodes
    
    # Remove unwanted attributes
    unwanted_attrs = ['industry', 'nation', 'deal_id']
    for attr in unwanted_attrs:
        if hasattr(refinitiv_data, attr):
            delattr(refinitiv_data, attr)

    refinitiv_data = NormalizeFeatures()(refinitiv_data)

    # Setup data splits
    if split_type == 'full':
        refinitiv_data.train_mask = torch.ones(num_nodes, dtype=torch.bool)
        refinitiv_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        refinitiv_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    elif split_type == 'random':
        torch.manual_seed(0)
        perm = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        refinitiv_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        refinitiv_data.train_mask[perm[:train_size]] = True
        refinitiv_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        refinitiv_data.val_mask[perm[train_size:train_size + val_size]] = True
        refinitiv_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        refinitiv_data.test_mask[perm[train_size + val_size:]] = True

    return refinitiv_data
    

def gpickle_to_PyG(gpickle_name):
    with open(gpickle_name, 'rb') as f:
        graph = pickle.load(f)

    # Convert to PyTorch Geometric Data object
    edge_index = torch.tensor(list(graph.edges())).t().contiguous()
    x = torch.tensor(np.eye(graph.number_of_nodes()), dtype=torch.float)  # Identity feature matrix
    y = torch.tensor([graph.has_edge(e[0], e[1]) for e in graph.edges()], dtype=torch.float)  # Binary edge existence labels
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

class CustomDataset(InMemoryDataset):
    def __init__(self, data, include_edge_weight=False):
        super().__init__('.', transform=None, pre_transform=None)
        self._data = data
        self.include_edge_weight = include_edge_weight

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.include_edge_weight and hasattr(self._data, 'edge_weight'):
            edge_weight = self._data.edge_weight
        else:
            edge_weight = None

        return Data(
            x=self._data.x,
            edge_index=self._data.edge_index,
            edge_weight=edge_weight,
            y=self._data.y if hasattr(self._data, 'y') else None,
            train_mask=self._data.train_mask if hasattr(self._data, 'train_mask') else None,
            val_mask=self._data.val_mask if hasattr(self._data, 'val_mask') else None,
            test_mask=self._data.test_mask if hasattr(self._data, 'test_mask') else None,
            train_pos_edge_index=self._data.train_pos_edge_index if hasattr(self._data, 'train_pos_edge_index') else None,
            train_neg_edge_index=self._data.train_neg_edge_index if hasattr(self._data, 'train_neg_edge_index') else None,
            test_pos_edge_index=self._data.test_pos_edge_index if hasattr(self._data, 'test_pos_edge_index') else None,
            test_neg_edge_index=self._data.test_neg_edge_index if hasattr(self._data, 'test_neg_edge_index') else None
        )
# class CustomDataset(InMemoryDataset):
#     def __init__(self, data):
#         super().__init__('.', transform=None, pre_transform=None)
#         self._data = data

#     def len(self):
#         return 1

#     def get(self, idx):
#         return self._data
