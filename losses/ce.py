# Copied from GraphENS (Park, Song & Yang, ICLR 2022), https://github.com/JoonHyung-Park/GraphENS
# (losses/ce.py). Copyright (c) 2022 JoonHyung-Park. MIT License; see THIRD_PARTY_NOTICES.md.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, input, target, weight=None, reduction='mean'):
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)