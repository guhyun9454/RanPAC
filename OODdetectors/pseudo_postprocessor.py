import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional

from .base_postprocessor import BasePostprocessor

class PseudoOODPostprocessor(BasePostprocessor):
    """ID 샘플을 targeted FGSM으로 perturbation 하여 pseudo-OOD를 생성하고,
    network logit 기반의 MLP 분류기를 학습해 ID / OOD 점수를 반환한다.
    학습은 `fit()` 에서 수행되며, `postprocess()` 는 훈련된 분류기의
    sigmoid 확률을 confidence score 로 사용한다.
    """

    def __init__(self, eps: float = 0.02, max_train_batches: int = 0, 
                 lr: float = 1e-2, epochs: int = 3,
                 hidden_dim: int = 128, layers: int = 2):
        super().__init__()
        self.eps = eps
        self.max_train_batches = max_train_batches  # 0이면 전체 사용
        self.lr = lr
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.trained = False
        self.classifier: Optional[nn.Module] = None

    def _generate_pseudo(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Targeted FGSM 으로 두 번째로 높은 class 로 공격하여 pseudo-OOD 샘플 생성"""
        x_adv = x.clone().detach().requires_grad_(True)
        logits = net(x_adv)
        # 두 번째로 높은 class 선택
        top2 = logits.topk(2, dim=1).indices
        target = top2[:, 1]
        loss = F.cross_entropy(logits, target)
        net.zero_grad(set_to_none=True)
        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv - self.eps * grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv.detach()

    def fit(self, net: nn.Module, id_loader: torch.utils.data.DataLoader, device: torch.device):
        """pseudo-OOD 생성 후 logistic classifier 학습"""
        if self.trained:
            return  # 이미 학습 완료
        feats, labels = [], []
        processed_batches = 0
        net.eval()
        for batch in id_loader:
            # DataLoader 에 따라 (idx, img, label) 또는 (img, label) 형태가 올 수 있음
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                _, inputs, _lbl = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, _lbl = batch
            else:
                # 예상치 못한 포맷일 경우 첫 요소를 이미지로 간주
                inputs = batch[0]

            inputs = inputs.to(device)
            # 원본 logits (ID)
            with torch.no_grad():
                logits_id = net(inputs)
            # pseudo-OOD logits
            inputs_adv = self._generate_pseudo(net, inputs)
            with torch.no_grad():
                logits_ood = net(inputs_adv)
            feats.append(torch.cat([logits_id, logits_ood], dim=0).cpu())
            id_label = torch.zeros(logits_id.size(0))
            ood_label = torch.ones(logits_ood.size(0))
            labels.append(torch.cat([id_label, ood_label], dim=0))

            processed_batches += 1
            if self.max_train_batches > 0 and processed_batches >= self.max_train_batches:
                break
        X = torch.cat(feats, dim=0).to(device)
        y = torch.cat(labels, dim=0).to(device).unsqueeze(1).float()

        # MLP 분류기 초기화
        dims = [X.size(1)]
        if self.layers >= 2:
            dims.append(self.hidden_dim)
        if self.layers == 3:
            dims.append(self.hidden_dim)
        dims.append(1)

        modules = []
        for i in range(len(dims)-2):
            modules.append(nn.Linear(dims[i], dims[i+1]))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(dims[-2], dims[-1]))
        self.classifier = nn.Sequential(*modules).to(device)

        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.lr)
        for ep in range(self.epochs):
            permutation = torch.randperm(X.size(0))
            epoch_loss, cnt = 0.0, 0
            for i in range(0, X.size(0), 256):
                idx = permutation[i:i+256]
                batch_x = X[idx]
                batch_y = y[idx]
                logits_bin = self.classifier(batch_x)
                loss = F.binary_cross_entropy_with_logits(logits_bin, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)
                cnt += batch_x.size(0)
            avg_loss = epoch_loss / max(cnt, 1)
            import logging
            logging.info(f"[PseudoOOD] Epoch {ep+1}/{self.epochs} - Loss: {avg_loss:.4f}")
        self.trained = True

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """logit → 선형 분류기 sigmoid → anomaly score (확률이 높을수록 OOD)"""
        assert self.trained and self.classifier is not None, "PseudoOODPostprocessor: fit() must be called before postprocess()"
        logits = net(data)
        conf = torch.sigmoid(self.classifier(logits)).squeeze(1)
        pred = logits.argmax(dim=1)
        return pred, conf 