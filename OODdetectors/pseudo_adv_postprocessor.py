from typing import Any, List

import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from .base_postprocessor import BasePostprocessor


class PseudoADVPostprocessor(BasePostprocessor):
    """Pseudo-OOD 샘플(경계 근처 targeted FGSM) + Logistic Regression 보조 분류기
    ID vs. Pseudo-OOD 구분기로부터 score를 산출한다.

    Note
    -----
    • setup() 단계에서 id_loader 로부터 pseudo OOD 데이터를 생성하고 보조 분류기를 학습한다.
    • 이후 postprocess() 는 입력에 대한 모델 logits -> 보조 분류기 확률(ID class)을 반환한다.
    """

    def __init__(self, epsilon: float = 0.003, 
                 attack_type: str = "fgsm_second",
                 classifier_type: str = "logistic",
                 use_feature_combination: bool = False,
                 temperature: float = 1.0,
                 num_steps: int = 1,
                 step_size: float = 0.001,
                 random_start: bool = False,
                 use_confidence_weight: bool = False):
        super().__init__()
        self.epsilon = epsilon
        self.attack_type = attack_type
        self.classifier_type = classifier_type
        self.use_feature_combination = use_feature_combination
        self.temperature = temperature
        self.num_steps = num_steps
        self.step_size = step_size
        self.random_start = random_start
        self.use_confidence_weight = use_confidence_weight
        self.clf = None
        self.num_classes: int | None = None
        self.feature_mean = None
        self.feature_std = None

    # ---------------------------------------------------------------------
    # Helper : 다양한 pseudo OOD 생성 방법들
    # ---------------------------------------------------------------------
    def _generate_pseudo(self, inputs: torch.Tensor, net: nn.Module) -> torch.Tensor:
        """선택된 attack_type에 따라 pseudo-OOD 샘플 생성"""
        if self.attack_type == "fgsm_second":
            return self._fgsm_second_highest(inputs, net)
        elif self.attack_type == "fgsm_random":
            return self._fgsm_random_target(inputs, net)
        elif self.attack_type == "pgd_second":
            return self._pgd_second_highest(inputs, net)
        elif self.attack_type == "fgsm_least_likely":
            return self._fgsm_least_likely(inputs, net)
        elif self.attack_type == "boundary_noise":
            return self._boundary_noise(inputs, net)
        elif self.attack_type == "mixup_boundary":
            return self._mixup_boundary(inputs, net)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")

    def _fgsm_second_highest(self, inputs: torch.Tensor, net: nn.Module) -> torch.Tensor:
        """FGSM 공격으로 두 번째로 높은 logit 클래스를 타겟으로"""
        inputs_adv = inputs.clone().detach().to(inputs.device)
        inputs_adv.requires_grad = True

        with torch.enable_grad():
            logits = net(inputs_adv) / self.temperature
            # 타겟 클래스 : 두 번째로 높은 logit index
            second_target = logits.topk(2, dim=1).indices[:, 1]
            loss = nn.CrossEntropyLoss()(logits, second_target)
            loss.backward()
            grad_sign = inputs_adv.grad.data.sign()
            inputs_adv = inputs_adv + self.epsilon * grad_sign
            inputs_adv = torch.clamp(inputs_adv, 0.0, 1.0)
        return inputs_adv.detach()

    def _fgsm_random_target(self, inputs: torch.Tensor, net: nn.Module) -> torch.Tensor:
        """FGSM 공격으로 랜덤 타겟 클래스로"""
        inputs_adv = inputs.clone().detach().to(inputs.device)
        inputs_adv.requires_grad = True

        with torch.enable_grad():
            logits = net(inputs_adv) / self.temperature
            pred = logits.argmax(dim=1)
            # 현재 예측과 다른 랜덤 클래스 선택
            num_classes = logits.size(1)
            random_targets = torch.randint(0, num_classes, (inputs.size(0),), device=inputs.device)
            # 현재 예측과 같은 경우 다시 선택
            mask = random_targets == pred
            while mask.any():
                random_targets[mask] = torch.randint(0, num_classes, (mask.sum(),), device=inputs.device)
                mask = random_targets == pred
            
            loss = nn.CrossEntropyLoss()(logits, random_targets)
            loss.backward()
            grad_sign = inputs_adv.grad.data.sign()
            inputs_adv = inputs_adv + self.epsilon * grad_sign
            inputs_adv = torch.clamp(inputs_adv, 0.0, 1.0)
        return inputs_adv.detach()

    def _pgd_second_highest(self, inputs: torch.Tensor, net: nn.Module) -> torch.Tensor:
        """PGD 공격으로 두 번째로 높은 logit 클래스를 타겟으로"""
        inputs_adv = inputs.clone().detach()
        
        if self.random_start:
            # 랜덤 시작점
            random_noise = torch.zeros_like(inputs).uniform_(-self.epsilon, self.epsilon)
            inputs_adv = inputs_adv + random_noise
            inputs_adv = torch.clamp(inputs_adv, 0.0, 1.0)
        
        for _ in range(self.num_steps):
            inputs_adv.requires_grad = True
            
            logits = net(inputs_adv) / self.temperature
            second_target = logits.topk(2, dim=1).indices[:, 1]
            loss = nn.CrossEntropyLoss()(logits, second_target)
            loss.backward()
            
            grad_sign = inputs_adv.grad.data.sign()
            inputs_adv = inputs_adv.detach() + self.step_size * grad_sign
            
            # Projection
            delta = torch.clamp(inputs_adv - inputs, min=-self.epsilon, max=self.epsilon)
            inputs_adv = torch.clamp(inputs + delta, 0.0, 1.0)
            
        return inputs_adv.detach()

    def _fgsm_least_likely(self, inputs: torch.Tensor, net: nn.Module) -> torch.Tensor:
        """FGSM 공격으로 가장 낮은 logit 클래스를 타겟으로"""
        inputs_adv = inputs.clone().detach().to(inputs.device)
        inputs_adv.requires_grad = True

        with torch.enable_grad():
            logits = net(inputs_adv) / self.temperature
            # 가장 낮은 logit 클래스
            least_likely = logits.argmin(dim=1)
            loss = nn.CrossEntropyLoss()(logits, least_likely)
            loss.backward()
            grad_sign = inputs_adv.grad.data.sign()
            inputs_adv = inputs_adv + self.epsilon * grad_sign
            inputs_adv = torch.clamp(inputs_adv, 0.0, 1.0)
        return inputs_adv.detach()

    def _boundary_noise(self, inputs: torch.Tensor, net: nn.Module) -> torch.Tensor:
        """Decision boundary 방향으로 노이즈 추가"""
        inputs_adv = inputs.clone().detach().to(inputs.device)
        inputs_adv.requires_grad = True

        with torch.enable_grad():
            logits = net(inputs_adv) / self.temperature
            # Top-2 logits의 차이를 최소화
            top2_logits = logits.topk(2, dim=1).values
            loss = (top2_logits[:, 0] - top2_logits[:, 1]).mean()
            loss.backward()
            grad = inputs_adv.grad.data
            # Gradient 방향으로 이동 (boundary로 접근)
            inputs_adv = inputs_adv + self.epsilon * grad / (grad.norm(dim=(1,2,3), keepdim=True) + 1e-8)
            inputs_adv = torch.clamp(inputs_adv, 0.0, 1.0)
        return inputs_adv.detach()

    def _mixup_boundary(self, inputs: torch.Tensor, net: nn.Module) -> torch.Tensor:
        """다른 클래스 샘플과 mixup하여 boundary 샘플 생성"""
        with torch.no_grad():
            logits = net(inputs)
            pred = logits.argmax(dim=1)
            
            # 각 샘플에 대해 다른 클래스로 예측된 샘플 찾기
            mixed_inputs = inputs.clone()
            for i in range(inputs.size(0)):
                # 다른 예측을 가진 샘플들의 인덱스
                diff_pred_idx = (pred != pred[i]).nonzero(as_tuple=True)[0]
                if len(diff_pred_idx) > 0:
                    # 랜덤하게 하나 선택
                    j = diff_pred_idx[torch.randint(len(diff_pred_idx), (1,))].item()
                    # Mixup (경계 근처로 이동)
                    alpha = 0.5 + 0.3 * torch.rand(1).item()  # 0.5~0.8 사이
                    mixed_inputs[i] = alpha * inputs[i] + (1 - alpha) * inputs[j]
            
        return mixed_inputs.detach()

    def _extract_features(self, logits: torch.Tensor) -> np.ndarray:
        """로짓에서 특징 추출 (선택적으로 조합 특징 포함)"""
        features = [logits]
        
        if self.use_feature_combination:
            # Softmax probabilities
            probs = torch.softmax(logits / self.temperature, dim=1)
            features.append(probs)
            
            # Top-k differences
            top5 = logits.topk(5, dim=1).values
            if top5.size(1) >= 2:
                diff_features = []
                for i in range(1, min(5, top5.size(1))):
                    diff_features.append((top5[:, 0] - top5[:, i]).unsqueeze(1))
                features.append(torch.cat(diff_features, dim=1))
            
            # Entropy
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)
            features.append(entropy)
            
            # Energy score
            energy = self.temperature * torch.logsumexp(logits / self.temperature, dim=1, keepdim=True)
            features.append(energy)
        
        return torch.cat(features, dim=1).cpu().numpy()

    # ------------------------------------------------------------------
    # Public API used by ood_adapter.compute_ood_scores
    # ------------------------------------------------------------------
    def setup(self, net: nn.Module, id_loader, ood_loader=None, device=None):
        """id_loader 로부터 pseudo OOD 를 생성하고 보조 분류기 학습"""
        net.eval()
        feats: List[np.ndarray] = []  # features (logits)
        labels: List[int] = []        # 1=ID, 0=Pseudo
        confidences: List[float] = [] # confidence scores for weighting

        for inputs, _ in id_loader:
            inputs = inputs.to(device)
            if len(labels) >= 4000:  # 최대 4000 샘플(2000 ID + 2000 pseudo)
                break
            
            # 1) ID features
            with torch.no_grad():
                logits_id = net(inputs)
                if self.use_confidence_weight:
                    probs = torch.softmax(logits_id / self.temperature, dim=1)
                    conf_id = probs.max(dim=1).values.cpu().numpy()
                    confidences.extend(conf_id)
                
            feats.append(self._extract_features(logits_id))
            labels.extend([1] * inputs.size(0))

            # 2) Pseudo OOD features
            inputs_adv = self._generate_pseudo(inputs, net)
            with torch.no_grad():
                logits_adv = net(inputs_adv)
                if self.use_confidence_weight:
                    probs_adv = torch.softmax(logits_adv / self.temperature, dim=1)
                    conf_adv = probs_adv.max(dim=1).values.cpu().numpy()
                    # Pseudo OOD는 낮은 confidence를 가중치로
                    confidences.extend(1 - conf_adv)
                    
            feats.append(self._extract_features(logits_adv))
            labels.extend([0] * inputs_adv.size(0))

        X = np.concatenate(feats, axis=0)
        y = np.asarray(labels)
        
        # Feature normalization
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-8
        X = (X - self.feature_mean) / self.feature_std

        # 분류기 학습
        if self.classifier_type == "logistic":
            self.clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced')
        elif self.classifier_type == "svm":
            self.clf = SVC(probability=True, kernel='rbf', gamma='scale', class_weight='balanced')
        elif self.classifier_type == "rf":
            self.clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced')
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
        
        if self.use_confidence_weight and confidences:
            sample_weight = np.asarray(confidences)
            self.clf.fit(X, y, sample_weight=sample_weight)
        else:
            self.clf.fit(X, y)
            
        self.num_classes = logits_id.shape[1]

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        """보조 분류기의 ID 확률을 score 로 사용"""
        logits = net(data)
        if self.clf is None:
            raise RuntimeError("PseudoADVPostprocessor.setup() 이 먼저 호출되어야 합니다.")
        
        features = self._extract_features(logits)
        # Normalize features
        features = (features - self.feature_mean) / self.feature_std
        
        probs = self.clf.predict_proba(features)[:, 1]  # ID class 확률
        conf = torch.tensor(probs, dtype=torch.float)
        _, pred = torch.max(logits, dim=1)
        return pred.to(conf.device), conf 