from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class EBOPostprocessor(BasePostprocessor):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = self.temperature * torch.logsumexp(output / self.temperature,
                                                  dim=1)
        return pred, conf

    def set_hyperparam(self, temperature: float):
        self.temperature = temperature

    def get_hyperparam(self):
        return self.temperature
