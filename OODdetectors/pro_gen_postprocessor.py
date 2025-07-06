from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor


class PRO_GENPostprocessor(BasePostprocessor):
    def __init__(self, gamma: float = 0.1, M: int = 100, noise_level: float = 1e-4, gd_steps: int = 3):
        super().__init__()
        self.gamma = gamma
        self.M = M
        self.noise_level = noise_level
        self.gd_steps = gd_steps

    def singlepostprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        _, pred = torch.max(score, dim=1)
        conf = self.generalized_entropy(score, self.gamma, self.M)
        return pred, conf

    def postprocess(self, net: nn.Module, data: Any):
        tempInputs=data.clone().detach()
        conf_record = [] 
        for step in range(self.gd_steps):
            tempInputs.requires_grad=True

            pred,conf = self.singlepostprocess(net,tempInputs)
            conf_record.append(conf.detach().clone())
            if step==0:
                unperturbed_pred = pred
            loss = conf.mean()
            loss.backward()
            gradient = tempInputs.grad.data
            tempInputs = torch.add(tempInputs.detach(), gradient.sign(), alpha=-self.noise_level) # decrease msp

        pred,conf = self.singlepostprocess(net,tempInputs)
        if self.gd_steps==0:
            unperturbed_pred = pred #just for debug
        conf_record.append(conf.detach().clone())
        conf_record_tensor = torch.stack(conf_record, dim=0)
        min_conf = conf_record_tensor.min(dim=0).values
        return unperturbed_pred, min_conf

    def set_hyperparam(self, gamma: float, M: int, noise_level: float, gd_steps: int):
        self.gamma = gamma
        self.M = M
        self.noise_level = noise_level
        self.gd_steps = gd_steps

    def get_hyperparam(self):
        return [self.gamma, self.M, self.noise_level, self.gd_steps]

    def generalized_entropy(self, softmax_id_val, gamma=0.1, M=100):
        probs = softmax_id_val
        probs_sorted = torch.sort(probs, dim=1)[0][:, -M:]
        scores = torch.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma),
                           dim=1)
        return -scores
