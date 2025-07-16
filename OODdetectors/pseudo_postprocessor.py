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

    past_classes (List[int]): 현재 task 이전에 학습된 클래스 인덱스 집합.
       Meaningful-Boundary Generation 시, 이들 클래스 중 logit 최댓값을
       갖는 클래스를 targeted FGSM 의 목표로 사용한다.
    """

    def __init__(self, eps: float = 0.02, max_train_batches: int = 0, 
                 lr: float = 1e-2, epochs: int = 3,
                 hidden_dim: int = 128, layers: int = 0, lambda_: float = 1e-3,
                 past_classes: Optional[list] = None):
        super().__init__()
        self.eps = eps
        self.max_train_batches = max_train_batches  # 0이면 전체 사용
        self.lr = lr
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.lambda_ = lambda_

        # === Meaningful Boundary Generation 관련 ===
        # 과거 task 에서 등장했던 클래스 인덱스 리스트 (중복 제거). 비어있으면 첫 태스크.
        self.past_classes = sorted(set(past_classes)) if past_classes else []

        # random projection matrix (for layers==0). If network has one, we will reuse it.
        self.W_rand: Optional[torch.Tensor] = None
        self.trained = False
        self.classifier: Optional[nn.Module] = None

    def _generate_pseudo(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Meaningful Boundary Generation: 
        과거 task 클래스( label < known_class_boundary ) 중에서 가장 점수가 높은 클래스로
        targeted FGSM 공격을 수행하여 pseudo-OOD 샘플을 생성한다.
        단, 과거 클래스가 없으면 (첫 task) 기존 방식(두 번째로 높은 class)으로 대체한다."""
        x_adv = x.clone().detach().requires_grad_(True)
        out = net(x_adv)
        logits = out["logits"] 
        if self.past_classes:
            # 과거 task 클래스들(logit 열) 중 최고 점수 class 선택
            past_logits = logits[:, self.past_classes]  # (N, |past|)
            rel_idx = past_logits.argmax(dim=1)  # relative idx w.r.t past_classes list
            target = torch.tensor([self.past_classes[i] for i in rel_idx.cpu().numpy()], device=logits.device, dtype=torch.long)
        else:
            # fallback: 두 번째로 높은 class
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
            # 원본 conv features (ID)
            with torch.no_grad():
                feats_id_dict = net.convnet(inputs)
                feats_id = feats_id_dict["features"] if isinstance(feats_id_dict, dict) else feats_id_dict
            # pseudo-OOD logits
            inputs_adv = self._generate_pseudo(net, inputs)
            with torch.no_grad():
                feats_ood_dict = net.convnet(inputs_adv)
                feats_ood = feats_ood_dict["features"] if isinstance(feats_ood_dict, dict) else feats_ood_dict

            feats.append(torch.cat([feats_id, feats_ood], dim=0).cpu())
            # NOTE: ID=1, OOD=0 로 레이블을 뒤집어 다른 postprocessor 와 동일하게 conf 값이 높을수록 ID 임을 의미하도록 수정
            id_label = torch.ones(feats_id.size(0))
            ood_label = torch.zeros(feats_ood.size(0))
            labels.append(torch.cat([id_label, ood_label], dim=0))

            processed_batches += 1
            if self.max_train_batches > 0 and processed_batches >= self.max_train_batches:
                break
        X = torch.cat(feats, dim=0).to(device)
        y = torch.cat(labels, dim=0).to(device).unsqueeze(1).float()

        # Build classifier depending on layers
        if self.layers == 0:
            # Reuse network's random projection if available; else create
            if self.W_rand is None:
                # expect net passed later, here derive from X dim (placeholder)
                self.W_rand = torch.randn(X.size(1), self.hidden_dim, device=device)
            def proj(z):
                return torch.relu(z @ self.W_rand)
            X_proj = proj(X)

            # decorrelation whitening using Gram matrix (ridge)
            G = X_proj.t() @ X_proj  # (M,M)
            lam = self.lambda_ * torch.trace(G) / G.size(0)
            eigvals, eigvecs = torch.linalg.eigh(G + lam * torch.eye(G.size(0), device=device))
            inv_sqrt = eigvecs @ torch.diag(torch.rsqrt(eigvals)) @ eigvecs.t()
            X_whiten = X_proj @ inv_sqrt

            self.decorr_mat = inv_sqrt  # store for inference

            self.classifier = nn.Linear(self.hidden_dim, 1).to(device)
            optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.lr)
            feature_func = lambda z: torch.relu(z @ self.W_rand) @ self.decorr_mat
        else:
            # MLP according to layers
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
            feature_func = lambda z: z  # identity

        for ep in range(self.epochs):
            permutation = torch.randperm(X.size(0))
            epoch_loss, cnt = 0.0, 0
            for i in range(0, X.size(0), 256):
                idx = permutation[i:i+256]
                batch_x = X[idx]
                batch_y = y[idx]
                if self.layers == 0:
                    bx = feature_func(batch_x)
                else:
                    bx = batch_x
                logits_bin = self.classifier(bx)
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
        """네트워크 convnet feature → RP(+ReLU) → decorrelation → sigmoid(score)"""
        assert self.trained and self.classifier is not None, "PseudoOODPostprocessor: fit() must be called before postprocess()"

        # Extract base features from convnet
        with torch.no_grad():
            feats_dict = net.convnet(data)
            feats = feats_dict["features"] if isinstance(feats_dict, dict) else feats_dict

        if self.layers == 0:
            # Use RP matrix from network if present and not yet cached
            if self.W_rand is None and hasattr(net, 'fc') and getattr(net.fc, 'use_RP', False) and hasattr(net.fc, 'W_rand') and net.fc.W_rand is not None:
                self.W_rand = net.fc.W_rand.detach()
            if self.W_rand is None:
                self.W_rand = torch.randn(feats.size(1), self.hidden_dim, device=feats.device)

            feats_rp = torch.relu(feats @ self.W_rand)
            feats_rp = feats_rp @ self.decorr_mat.to(feats_rp.device)
            conf = torch.sigmoid(self.classifier(feats_rp)).squeeze(1)
        else:
            conf = torch.sigmoid(self.classifier(feats)).squeeze(1)
        # prediction used for bookkeeping; use network final logits argmax
        with torch.no_grad():
            pred = net(data)["logits"].argmax(dim=1)
        return pred, conf 