import torch
from torch import nn

class CorrectionModule(nn.Module):
    """Light-weight residual MLP used in DPBL.

    F_out = F_in + MLP(F_in)
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) 