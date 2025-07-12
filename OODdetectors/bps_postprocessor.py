import torch
import torch.nn.functional as F
from .base_postprocessor import BasePostprocessor

class BPSPostprocessor(BasePostprocessor):
    """Boundary Perturbation Stability (BPS) score for OOD detection.

    BPS(x) = Conf(x) - Conf(x_p)
    where x_p is an adversarial sample toward the 2nd predicted class.
    """

    def __init__(self, epsilon: float = 1.0/255.0, **kwargs):
        super().__init__()
        self.epsilon = epsilon

    def postprocess(self, net, inputs):
        # Enable gradient on inputs for adversarial generation
        x = inputs.clone().detach().requires_grad_(True)

        logits = net(x)
        if isinstance(logits, dict):
            logits = logits["logits"]

        prob = F.softmax(logits, dim=1)
        conf, pred = torch.max(prob, dim=1)

        # Determine target labels: 2nd highest prob class for each sample
        top2 = torch.topk(prob, k=2, dim=1).indices
        target_labels = top2[:, 1]

        loss_adv = F.cross_entropy(logits, target_labels)
        net.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        loss_adv.backward(retain_graph=True)
        x_p = torch.clamp(x + self.epsilon * x.grad.sign(), 0.0, 1.0)

        with torch.no_grad():
            logits_p = net(x_p)
            if isinstance(logits_p, dict):
                logits_p = logits_p["logits"]
            conf_p = F.softmax(logits_p, dim=1).max(dim=1)[0]

        bps = conf - conf_p
        return pred, bps 