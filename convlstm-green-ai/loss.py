import torch
import torch.nn as nn


class SequenceMSELoss(nn.Module):
    """Mean squared error between two sequences (B, T, C, H, W)."""

    def __init__(self):
        super(SequenceMSELoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)
