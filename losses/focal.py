# Focal loss of Lin, Goyal, Girshick, He and Dollar, "Focal Loss for Dense Object Detection",
# ICCV 2017 (arXiv:1708.02002). Own implementation.
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss: FL(p_t) = -alpha * (1 - p_t) ** gamma * log(p_t).

    p_t is the predicted probability of the true class. With gamma = 0 and
    alpha = 1 this is the usual cross-entropy.

    - Multi-class: ``input`` has shape (N, C) and holds logits; ``target`` has
      shape (N,) and holds class ids.
    - Binary: ``input`` has shape (N,) and holds logits; ``target`` has shape
      (N,) and holds 0/1 labels.

    Args:
        alpha: constant weight applied to every term.
        gamma: focusing parameter (gamma >= 0).
        reduction: 'mean', 'sum' or 'none'.
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        if input.dim() == 1:
            log_pt = -F.binary_cross_entropy_with_logits(
                input, target.to(input.dtype), reduction='none')
        else:
            log_pt = -F.cross_entropy(input, target, reduction='none')
        loss = -self.alpha * (1.0 - log_pt.exp()).pow(self.gamma) * log_pt
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
