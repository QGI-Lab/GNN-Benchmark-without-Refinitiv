from .gatv2 import GATv2

class GATv2LinkPrediction(GATv2):
    def __init__(self, n_layer, input_dim, feat_dim, heads, **kwargs):
        super(GATv2LinkPrediction, self).__init__(n_layer, input_dim, feat_dim, n_cls=None, n_head=heads, **kwargs)

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)