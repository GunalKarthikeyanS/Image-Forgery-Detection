"""Losses. CombinedLoss is from the original code (legacy/assets.py, Hyeonwook Kim)."""
import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """BCE + Dice + IoU. Set dice_weight=iou_weight=0 for plain BCE."""

    def __init__(self, bce_weight=1.0, dice_weight=1.0, iou_weight=1.0, eps=1e-6):
        super().__init__()
        self.bce_weight, self.dice_weight, self.iou_weight = bce_weight, dice_weight, iou_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = eps

    def dice_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        return 1 - ((2 * inter + self.eps) / (union + self.eps)).mean()

    def iou_loss(self, logits, targets):
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
        return 1 - ((inter + self.eps) / (union + self.eps)).mean()

    def forward(self, logits, targets):
        loss = self.bce_weight * self.bce(logits, targets)
        if self.dice_weight:
            loss = loss + self.dice_weight * self.dice_loss(logits, targets)
        if self.iou_weight:
            loss = loss + self.iou_weight * self.iou_loss(logits, targets)
        return loss


class DeepSupervisionLoss(nn.Module):
    """Applies a loss to EVERY output head (coarse to fine). The original training loop
    only used the first, lowest-resolution head."""

    def __init__(self, base_loss, weights=(0.1, 0.2, 0.3, 0.4, 1.0)):
        super().__init__()
        self.base = base_loss
        self.weights = weights

    def forward(self, outputs, targets):
        total = sum(w * self.base(o, targets) for w, o in zip(self.weights, outputs))
        return total / sum(self.weights)
