from .gat import GAT
import torch.nn as nn

class GATNodeClassification(GAT):
    def __init__(self, n_layer, input_dim, feat_dim, n_cls, heads, **kwargs):
        super(GATNodeClassification, self).__init__(n_layer, input_dim, feat_dim, n_cls, n_head=heads, **kwargs)