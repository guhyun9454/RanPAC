"""Adapted from: https://github.com/facebookresearch/odin."""
from typing import Any

import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor

class PROv2_TEMPSCALE_Postprocessor(BasePostprocessor):
    def __init__(self, temperature: float = 1.0, noise_level: float = 0.003, gd_steps: int = 1):
        super().__init__()
        self.temperature = temperature
        self.noise_level = noise_level
        self.gd_steps = gd_steps

    def postprocess(self, net: nn.Module, data: Any):
        #data.requires_grad = True

        tempInputs=data.clone().detach()
        #criterion = nn.CrossEntropyLoss()
        conf_record = [] 
        for step in range(self.gd_steps):
            tempInputs.requires_grad=True
            output = net(tempInputs)
            score = torch.softmax(output / self.temperature, dim=1)
            conf, pred = torch.max(score, dim=1)
            conf_record.append(conf.detach().clone())
            if step==0:
                unperturbed_pred = pred
            loss = conf.mean()
            loss.backward()
            # Normalizing the gradient to binary in {0, 1}
            gradient = tempInputs.grad.data

            # Adding small perturbations to images
            #tempInputs = torch.add(data.detach(), gradient, alpha=-self.noise)# increase msp
            tempInputs = torch.add(tempInputs.detach(), gradient.sign(), alpha=-self.noise_level) # decrease msp

        output = net(tempInputs)
        output = output / self.temperature
        # Calculating the confidence after adding perturbations
        nnOutput = output.detach()
        nnOutput = nnOutput - nnOutput.max(dim=1, keepdims=True).values
        nnOutput = nnOutput.exp() / nnOutput.exp().sum(dim=1, keepdims=True)
        conf, _ = nnOutput.max(dim=1)
        conf_record.append(conf.detach().clone())
        conf_record_tensor = torch.stack(conf_record, dim=0)
        min_conf = conf_record_tensor.min(dim=0).values
        return unperturbed_pred, min_conf

    def set_hyperparam(self, noise_level: float, gd_steps: int, temperature: float):
        self.noise_level = noise_level
        self.gd_steps = gd_steps
        self.temperature = temperature

    def get_hyperparam(self):
        return [self.noise_level, self.gd_steps, self.temperature]