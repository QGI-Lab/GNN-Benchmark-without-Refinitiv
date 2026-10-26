from .gcn import GCN

class GCNLinkPrediction(GCN):
    def __init__(self, n_layer, input_dim, feat_dim, normalize=True, is_add_self_loops=True):
        super(GCNLinkPrediction, self).__init__(n_layer, input_dim, feat_dim, n_cls=None, normalize=normalize, is_add_self_loops=is_add_self_loops)
    
    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)