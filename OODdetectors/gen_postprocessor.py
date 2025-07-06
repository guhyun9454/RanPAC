from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class GENPostprocessor(BasePostprocessor):
    def __init__(self, gamma: float = 0.1, M: int = 100):
        super().__init__()
        self.gamma = gamma
        self.M = M

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = self.generalized_entropy(score, self.gamma, self.M)
        return pred, conf



    def generalized_entropy(self, softmax_id_val, gamma=0.1, M=100):
        probs = softmax_id_val
        probs_sorted = torch.sort(probs, dim=1)[0][:, -M:]
        scores = torch.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma),
                           dim=1)
        return -scores
    def set_hyperparam(self, gamma: float, M: int):
        self.gamma = gamma
        self.M = M

    def get_hyperparam(self):
        return [self.gamma, self.M]