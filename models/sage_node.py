from .sage import SAGE

class SAGENodeClassification(SAGE):
    def __init__(self, n_layer, input_dim, feat_dim, n_cls):
        super(SAGENodeClassification, self).__init__(n_layer, input_dim, feat_dim, n_cls)