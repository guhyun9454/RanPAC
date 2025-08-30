from typing import Any
import torch
import torch.nn as nn

from .base_postprocessor import BasePostprocessor

class PROv2_ENT_Postprocessor(BasePostprocessor):
    prob = True
    def __init__(self, noise_level: float = 0.0014, gd_steps: int = 2):
        super().__init__()
        self.noise_level = noise_level
        self.gd_steps = gd_steps  # Gradient descent steps

    def postprocess(self, net: nn.Module, data: Any):
        tempInputs = data.clone().detach()

        conf_record = [] 
        for step in range(self.gd_steps):
            tempInputs.requires_grad = True
            output = net(tempInputs)
            if step==0:
                pred = output.argmax(dim=1)
            probabilities=torch.softmax(output,dim=1)
            entropy =-torch.sum(probabilities * torch.log(probabilities+1e-10),dim=1)
            conf_record.append(-entropy.detach().clone())
            # We want to maximize energy (or minimize negative energy)
            loss = -entropy.mean()
            loss.backward()
            gradient = tempInputs.grad.data

            tempInputs = torch.add(tempInputs.detach(), gradient.sign(), alpha=-self.noise_level)

        output = net(tempInputs.detach())
        probabilities=torch.softmax(output,dim=1)
        entropy =-torch.sum(probabilities * torch.log(probabilities+1e-10),dim=1)
        conf_record.append(-entropy.detach().clone())
        conf_record_tensor = torch.stack(conf_record, dim=0)
        min_conf = conf_record_tensor.min(dim=0).values
        return pred, min_conf

    def set_hyperparam(self, noise_level: float, gd_steps: int):
        self.noise_level = noise_level
        self.gd_steps = gd_steps

    def get_hyperparam(self):
        return [self.noise_level, self.gd_steps]
