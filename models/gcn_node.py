from .gcn import GCN
import torch

class GCNNodeClassification(GCN):
    def __init__(self, n_layer, input_dim, feat_dim, n_cls, normalize=True, is_add_self_loops=True):
        super(GCNNodeClassification, self).__init__(n_layer, input_dim, feat_dim, n_cls, normalize, is_add_self_loops)