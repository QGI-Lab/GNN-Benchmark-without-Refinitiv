from .sage import SAGE

class SAGELinkPrediction(SAGE):
    def __init__(self, n_layer, input_dim, feat_dim, **kwargs):
        super(SAGELinkPrediction, self).__init__(n_layer, input_dim, feat_dim, n_cls=None, **kwargs)

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)