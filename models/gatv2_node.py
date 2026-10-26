from .gatv2 import GATv2
import torch.nn as nn

class GATv2NodeClassification(GATv2):
    def __init__(self, n_layer, input_dim, feat_dim, n_cls, heads, **kwargs):
        super(GATv2NodeClassification, self).__init__(n_layer, input_dim, feat_dim, n_cls, n_head=heads, **kwargs)