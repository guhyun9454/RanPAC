import argparse, datetime, logging, os, sys, time, copy
import numpy as np
import torch, pandas as pd
import random
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from torchvision import transforms as T

from continual_datasets.build_incremental_scenario import build_continual_dataloader
from RanPAC import Learner
from continual_datasets.dataset_utils import set_data_config
from continual_datasets.dataset_utils import get_ood_dataset
from continual_datasets.dataset_utils import RandomSampleWrapper, UnknownWrapper

from torch.utils.data import ConcatDataset
from utils.acc_heatmap import save_accuracy_heatmap
from utils import save_anomaly_histogram, save_logits_statistics, update_ood_hyperparams


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class IndexedDataset(torch.utils.data.Dataset):
    """Wrap any (img, label) dataset -> (idx, img, label) & expose .labels"""
    def __init__(self, base_ds):
        self.base = base_ds
        # try common names for label list; fall back to brute-force read
        if hasattr(base_ds, "targets"):
            lab = base_ds.targets
        elif hasattr(base_ds, "labels"):
            lab = base_ds.labels
        else:                       # worst-case: iterate once
            lab = [base_ds[i][1] for i in range(len(base_ds))]
        self.labels = np.asarray(lab)

    def __len__(self):  return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]          # original (<img>, <label>)
        return idx, img, label               # <- 3-tuple expected by Learner


class LimitIterationDataloader:
    """develop 모드에서 dataloader의 iteration 수를 제한하는 래퍼 클래스"""
    def __init__(self, dataloader, max_iterations=10):
        self.dataloader = dataloader
        self.max_iterations = max_iterations
        
    def __iter__(self):
        iterator = iter(self.dataloader)
        for i in range(self.max_iterations):
            try:
                yield next(iterator)
            except StopIteration:
                break
                
    def __len__(self):
        return min(len(self.dataloader), self.max_iterations)
    
    @property
    def dataset(self):
        return self.dataloader.dataset


# ------------------------------------------------------------------ #
# Utility wrapper : (img, label) -> (img, 0)  for ID samples
# ------------------------------------------------------------------ #
class IDLabelWrapper(torch.utils.data.Dataset):
    """ID 데이터의 라벨을 0으로 고정하여 반환합니다 (OOD 분류 학습용)."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        return x, 0

# ------------------------------------------------------------------ #
# Synthetic OOD dataset utilities                                    #
# ------------------------------------------------------------------ #
class GaussianNoiseDataset(torch.utils.data.Dataset):
    """랜덤 가우시안 노이즈 이미지를 생성하는 데이터셋."""
    def __init__(self, num_samples, image_shape, seed=0):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.tensor(self.rng.randn(*self.image_shape), dtype=torch.float32)
        return img, -1  # dummy label


class BlurNoiseWrapper(torch.utils.data.Dataset):
    """기존 ID 이미지를 GaussianBlur & 가우시안 노이즈로 변환하여 OOD 샘플을 생성합니다.

    Args:
        dataset: 원본 ID 데이터셋
        noise_std: 추가할 가우시안 노이즈 표준편차
        blur_kernel: GaussianBlur 커널 크기(홀수)
        blur_sigma:  블러 강도 (sigma). 크면 더 흐려짐
    """
    def __init__(self, dataset, noise_std=0.2, blur_kernel=7, blur_sigma=3.0):
        self.dataset = dataset
        self.noise_std = noise_std
        self.blur = T.GaussianBlur(kernel_size=blur_kernel, sigma=(blur_sigma, blur_sigma*1.5))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)
        img = self.blur(img)
        img = img + torch.randn_like(img) * self.noise_std
        img = torch.clamp(img, 0.0, 1.0)
        return img, -1  # dummy label

# ------------------------------------------------------------------ #
# Task-specific OOD classifier 학습 함수
# ------------------------------------------------------------------ #
def train_te_classifier(learner, id_dataset, ood_dataset, device, args):
    """현재 task에 대한 binary TE(Task Expert) classifier(0:ID, 1:OOD)를 학습합니다."""

    # 1) 데이터셋 크기 맞추기
    id_size, ood_size = len(id_dataset), len(ood_dataset)
    min_size = min(id_size, ood_size)
    if args.develop:
        min_size = min(min_size, 1000)

    id_ds_aligned  = RandomSampleWrapper(id_dataset,  min_size, args.seed) if id_size  > min_size else id_dataset
    ood_ds_aligned = RandomSampleWrapper(ood_dataset, min_size, args.seed) if ood_size > min_size else ood_dataset

    # 라벨 래퍼 적용
    id_wrapped  = IDLabelWrapper(id_ds_aligned)                 # label 0
    ood_wrapped = UnknownWrapper(ood_ds_aligned, unknown_label=1)  # label 1

    combined_dataset = ConcatDataset([id_wrapped, ood_wrapped])
    loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=args.te_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------
    # feature extraction helper
    # ------------------------------------------------------------
    feature_type = args.te_feature.lower()

    def extract_feats(inputs):
        """Return feature tensor according to feature_type."""
        if feature_type == 'logits':
            return learner._network(inputs)["logits"].detach()
        feats = learner._network.convnet(inputs).detach()
        if feature_type == 'rp':
            W_rand = getattr(learner, 'W_rand', None)
            if W_rand is None:
                # fallback: try to fetch from the current FC layer (after replace_fc)
                W_rand = getattr(getattr(learner._network, 'fc', None), 'W_rand', None)
            if W_rand is not None:
                return F.relu(feats @ W_rand)
            # if W_rand is missing, just return raw feats
            return feats
        elif feature_type == 'decorr':
            G = getattr(learner, 'G', None)
            W_rand = getattr(learner, 'W_rand', None)
            W_rand = getattr(getattr(learner._network, 'fc', None), 'W_rand', None)
            feats = F.relu(feats @ W_rand)
            eps = 1000000
            G_inv = torch.linalg.pinv(G.to(device) + eps * torch.eye(G.shape[0], device=device))
            return feats @ G_inv

        return feats

    # 2) feature dimension 파악
    with torch.no_grad():
        sample_x, _ = combined_dataset[0]
        feat_dim = extract_feats(sample_x.unsqueeze(0).to(device)).shape[1]

    # 3) classifier 정의 (simple logistic regression)
    # Dropout이 포함된 binary classifier 정의
    class TEClassifier(torch.nn.Module):
        def __init__(self, input_dim, dropout_prob=0.2):
            super().__init__()
            self.dropout = torch.nn.Dropout(p=dropout_prob)
            self.fc = torch.nn.Linear(input_dim, 1)
        def forward(self, x):
            x = self.dropout(x)
            return self.fc(x).squeeze()

    classifier = TEClassifier(feat_dim, dropout_prob=args.te_dropout).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.te_lr)

    # 4) 학습
    classifier.train()
    for epoch in range(args.te_epochs):
        epoch_loss = 0.0
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.float().to(device)

            with torch.no_grad():  # convnet은 고정
                feats = extract_feats(imgs)

            logits = classifier(feats).squeeze()
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.te_epochs} loss: {epoch_loss/len(loader):.4f}")
    classifier.eval()
    return classifier


# ------------------------------------------------------------------ #
# DataManager mock : just enough for Learner.incremental_train()
# ------------------------------------------------------------------ #
class LoaderDataManager:

    """
    Wraps a single VIL task's train / val dataloaders to look like the
    original utils.data_manager.DataManager.
    Each task is treated as *domain-incremental*: head size == num_classes.
    """
    def __init__(self, loader_pair, num_classes, args=None):
        self._train_loader = loader_pair['train']
        self._val_loader   = loader_pair['val']
        self._num_classes  = num_classes
        self._increments   = [num_classes]       # one "big" task
        
        # develop 모드이면 train loader를 iteration 제한 래퍼로 감싸기
        if args and args.develop:
            self._train_loader = LimitIterationDataloader(self._train_loader, max_iterations=1)
            logging.info("개발 모드: 학습 dataloader iteration을 10회로 제한합니다.")

    @property
    def nb_tasks(self):              # unused, but keep for safety
        return len(self._increments)

    def get_total_classnum(self):
        return self._num_classes

    def get_task_size(self, task):
        return self._increments[task]

    def get_dataset(self, indices, source, mode,
                    appendent=None, ret_data=False):
        # indices / mode 무시하고 loader 속 dataset 바로 반환
        if source == "train":
            ds = self._train_loader.dataset
        elif source == "test":
            ds = self._val_loader.dataset
        else:
            raise ValueError("source must be 'train' or 'test'")
        
        wrapped = IndexedDataset(ds)
        return wrapped if not ret_data else (None, None, wrapped)


def evaluate_till_now(model, loaders, device, task_id,
                      acc_matrix, args, task_experts=None):
    """기존 IL 정확도 + TE 기반 8개 변형 정확도(가중치/마스킹)를 계산합니다.

    변형 이름:
        ORIG,
        TE_<agg>_WEIGHT,
        TE_<agg>_MASK   (agg ∈ args.te_agg_methods)
    """

    # ------------------------------------------------------------------
    # Helper: class→task 매핑 벡터 준비 (길이 = 총 클래스 수)
    # ------------------------------------------------------------------
    total_cls = model._network.fc.out_features  # 현재까지 head 크기
    cls2task = np.zeros(total_cls, dtype=int)
    for t_idx, (lo, hi) in enumerate(model.class_increments):
        cls2task[lo:hi + 1] = t_idx

    # ------------------------------------------------------------------
    # TE 관련 헬퍼 (skip if task_experts 없음)
    # ------------------------------------------------------------------
    if task_experts:
        agg_methods = [m.strip().lower() for m in args.te_agg_methods.split(',')]

        feature_type = args.te_feature.lower()

        def extract_feats(inputs):
            if feature_type == 'logits':
                return model._network(inputs)["logits"].detach()
            feats = model._network.convnet(inputs).detach()
            if feature_type in {'rp', 'decorr'}:
                W_rand = getattr(model, 'W_rand', None)
                if W_rand is None:
                    W_rand = getattr(getattr(model._network, 'fc', None), 'W_rand', None)
                if W_rand is not None:
                    feats = F.relu(feats @ W_rand)
                if feature_type == 'rp':
                    return feats
            if feature_type == 'decorr':
                G = getattr(model, 'G', None)
                eps = 1e6
                G_inv = torch.linalg.pinv(G.to(device) + eps * torch.eye(G.shape[0], device=device))
                feats = feats @ G_inv
                return feats
            return feats

        def _mc_preds(feats):
            mc = args.te_mc_passes
            ps = []
            for cls in task_experts:
                cls.train()
                outs = []
                for _ in range(mc):
                    logit = cls(feats)
                    score = torch.sigmoid(logit) if args.te_score_type == 'sigmoid' else logit
                    outs.append(score)
                outs = torch.stack(outs, dim=-1)  # (B, mc)
                ps.append(outs)
                cls.eval()
            return torch.stack(ps, dim=0)  # (num_te, B, mc)

        def _task_confidence(stacked, agg):
            mean = stacked.mean(-1)  # (num_te, B)
            std  = stacked.std(-1)
            if agg in {'mean', 'max', 'wsum'}:
                conf = mean  # raw mean confidence per TE
            elif agg == 'std':
                conf = -std  # std가 작을수록 ID 가능 ↑
            else:
                raise ValueError(agg)
            return conf  # (num_te, B)

    # ------------------------------------------------------------------
    # 정확도 집계 구조 준비
    # ------------------------------------------------------------------
    variants = ["ORIG"]
    if task_experts:
        for a in agg_methods:
            variants.extend([f"TE_{a.upper()}_WEIGHT", f"TE_{a.upper()}_MASK"])
    correct_tbl = {k: np.zeros(task_id + 1, dtype=int) for k in variants}
    total_tbl   = {k: np.zeros(task_id + 1, dtype=int) for k in variants}

    # ------------------------------------------------------------------
    # 각 과거 task 의 validation 에 대해 평가
    # ------------------------------------------------------------------
    for t in range(task_id + 1):
        loader = loaders[t]['val']
        model._network.eval()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)

                logits = model._network(x)['logits']  # (B, C)
                pred_orig = logits.argmax(1)
                correct_tbl['ORIG'][t] += (pred_orig == y).sum().item()
                total_tbl['ORIG'][t]   += y.size(0)

                if not task_experts:
                    continue  # skip TE variants if none

                feats = extract_feats(x)
                stacked = _mc_preds(feats)  # (num_te, B, mc)

                for agg in agg_methods:
                    conf = _task_confidence(stacked, agg)  # (num_te, B)

                    # weight variant ----------------------------------
                    w = torch.softmax(conf, dim=0)  # (num_te, B)
                    # class-wise weight matrix (B, C)
                    w_expanded = w[cls2task].T  # broadcast via numpy idx → torch idx
                    weighted_logits = logits * w_expanded.to(device)
                    pred_w = weighted_logits.argmax(1)
                    key_w = f"TE_{agg.upper()}_WEIGHT"
                    correct_tbl[key_w][t] += (pred_w == y).sum().item()

                    # mask variant ------------------------------------
                    top_task = conf.argmax(0)  # (B,)
                    # create mask: 1 for classes in top_task else 0
                    mask = torch.zeros_like(weighted_logits)
                    for idx_b, tt in enumerate(top_task):
                        task = tt.item()
                        lo, hi = model.class_increments[task]
                        mask[idx_b, lo:hi+1] = 1.0
                    masked_logits = logits.masked_fill(mask == 0, -1e9)
                    pred_m = masked_logits.argmax(1)
                    key_m = f"TE_{agg.upper()}_MASK"
                    correct_tbl[key_m][t] += (pred_m == y).sum().item()

                for key in variants:
                    if key != 'ORIG':
                        total_tbl[key][t] += y.size(0)

        # after loader loop
    # ------------------------------------------------------------------
    # 정확도 계산 및 기존 acc_matrix 갱신 (ORIG 기준)
    # ------------------------------------------------------------------
    for t in range(task_id + 1):
        acc_matrix[t, task_id] = 100.0 * correct_tbl['ORIG'][t] / max(total_tbl['ORIG'][t], 1)

    # 요약 메트릭 (ORIG)
    A_i    = [np.mean(acc_matrix[:i+1, i]) for i in range(task_id+1)]
    A_last = A_i[-1]
    A_avg  = np.mean(A_i)
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
    else:
        forgetting = 0.0

    msg = (f"[Task {task_id+1:2d}] "
           f"A_last {A_last:.2f} | A_avg {A_avg:.2f} | "
           f"Forgetting {forgetting:.2f}")
    logging.info(msg); print(msg)

    # --- 추가: 변형 정확도 로깅
    for key in variants:
        for t in range(task_id + 1):
            acc = 100.0 * correct_tbl[key][t] / max(total_tbl[key][t], 1)
            logging.info(f"    {key} | Task{t+1}->{task_id+1}: {acc:.2f}%")

    if args.wandb:
        import wandb
        log_data = {k: 100.0 * correct_tbl[k][task_id] / max(total_tbl[k][task_id], 1) for k in variants}
        log_data['A_last'] = A_last
        log_data['A_avg'] = A_avg
        log_data['Forgetting'] = forgetting
        log_data['TASK'] = task_id
        wandb.log(log_data)

    # 원본 accuracy 히트맵 저장 (변형 결과는 저장하지 않음)
    sub_matrix = acc_matrix[:task_id+1, :task_id+1]
    result = np.where(np.triu(np.ones_like(sub_matrix, dtype=bool)), sub_matrix, np.nan)
    save_accuracy_heatmap(result, task_id, args)

    return A_last, A_avg, forgetting

def evaluate_ood(learner, id_datasets, ood_dataset, device, args, task_id=None, task_experts=None):
    """OOD 평가를 위한 통합 함수 (adapter 기반, TE 방식 포함)"""
    learner._network.eval()
    
    ood_method = args.ood_method.upper()

    # 1) 데이터셋 크기 맞추기
    id_size, ood_size = len(id_datasets), len(ood_dataset)
    min_size = min(id_size, ood_size)
    if args.develop:
        min_size = 1000
    if args.ood_develop:
        min_size = args.ood_develop
    if args.verbose:
        print(f"ID dataset size: {id_size}, OOD dataset size: {ood_size}. Using {min_size} samples each for evaluation.")

    id_dataset_aligned = RandomSampleWrapper(id_datasets, min_size, args.seed) if id_size > min_size else id_datasets
    ood_dataset_aligned = RandomSampleWrapper(ood_dataset, min_size, args.seed) if ood_size > min_size else ood_dataset

    id_loader = torch.utils.data.DataLoader(id_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    ood_loader = torch.utils.data.DataLoader(ood_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    from OODdetectors.ood_adapter import SUPPORTED_METHODS, compute_ood_scores

    EXTRA_METHODS = ["TE"]

    if ood_method == "ALL":
        methods = SUPPORTED_METHODS + EXTRA_METHODS
    else:
        # 쉼표로 구분된 메소드들 처리
        methods = [method.strip().upper() for method in ood_method.split(',')]
        # 지원되지 않는 메소드 확인
        unsupported = [m for m in methods if m not in (SUPPORTED_METHODS + EXTRA_METHODS)]
        if unsupported:
            raise ValueError(f"지원되지 않는 OOD 메소드: {unsupported}. 지원되는 메소드: {SUPPORTED_METHODS + EXTRA_METHODS}")

    from sklearn import metrics
    results = {}

    for method in methods:
        if method == "TE":
            # === Uncertainty-aware TE aggregation ===
            if not task_experts:
                raise ValueError("TE 평가를 위해서는 task_experts 리스트가 필요합니다.")

            agg_methods = [m.strip().lower() for m in args.te_agg_methods.split(',')]
            feature_type = args.te_feature.lower()

            def extract_feats(inputs):
                if feature_type == 'logits':
                    return learner._network(inputs)["logits"].detach()
                feats = learner._network.convnet(inputs).detach()
                if feature_type in {'rp', 'decorr'}:
                    W_rand = getattr(learner, 'W_rand', None)
                    if W_rand is None:
                        W_rand = getattr(getattr(learner._network, 'fc', None), 'W_rand', None)
                    if W_rand is not None:
                        feats = F.relu(feats @ W_rand)
                    if feature_type == 'rp':
                        return feats
                if feature_type == 'decorr':
                    G = getattr(learner, 'G', None)
                    eps = 1e6
                    G_inv = torch.linalg.pinv(G.to(device) + eps * torch.eye(G.shape[0], device=device))
                    feats = feats @ G_inv
                    return feats
                return feats

            def _mc_predictions(feats):
                mc_passes = args.te_mc_passes
                preds = []
                for cls in task_experts:
                    cls.train()  # dropout 활성화
                    outs = []
                    for _ in range(mc_passes):
                        logit = cls(feats)
                        score = torch.sigmoid(logit) if args.te_score_type == 'sigmoid' else logit
                        outs.append(score)
                    outs = torch.stack(outs, dim=-1)  # (batch, mc)
                    preds.append(outs)
                    cls.eval()
                return torch.stack(preds, dim=0)  # (num_exp, batch, mc)

            def _aggregate(stacked, agg):
                means = stacked.mean(dim=-1)
                stds = stacked.std(dim=-1)
                if agg == 'max':
                    score = means.max(dim=0).values
                elif agg == 'mean':
                    score = means.mean(dim=0)
                elif agg in {'wsum', 'weighted'}:
                    eps = 1e-6
                    w = 1.0 / (stds + eps)
                    w = w / w.sum(dim=0, keepdim=True)
                    score = (w * means).sum(dim=0)
                elif agg == 'std':
                    score = -stds.mean(dim=0)  # 불확실도가 높을수록 OOD 가능성 ↑
                else:
                    raise ValueError(f"Unknown TE aggregation: {agg}")
                return score

            def _gather_scores(loader):
                collected = {k: [] for k in agg_methods}
                learner._network.eval()
                with torch.no_grad():
                    for inputs, _ in loader:
                        inputs = inputs.to(device)
                        feats = extract_feats(inputs)
                        stacked = _mc_predictions(feats)
                        for agg in agg_methods:
                            s = _aggregate(stacked, agg)
                            conf = 1.0 - s if args.te_score_type == 'sigmoid' and agg != 'std' else -s
                            collected[agg].append(conf.cpu())
                return {k: torch.cat(v, dim=0) for k, v in collected.items()}

            id_scores_dict  = _gather_scores(id_loader)
            ood_scores_dict = _gather_scores(ood_loader)

            for agg in agg_methods:
                method_name = f"TE_{agg.upper()}"
                id_scores  = id_scores_dict[agg]
                ood_scores = ood_scores_dict[agg]
                if args.verbose or args.wandb:
                    hist_path = save_anomaly_histogram(id_scores.numpy(), ood_scores.numpy(), args,
                                                       suffix=method_name.lower(), task_id=task_id)
                    if args.wandb:
                        import wandb
                        wandb.log({f"Anomaly Histogram TASK {task_id}/{method_name}": wandb.Image(hist_path)})

                binary_labels = np.concatenate([np.ones(id_scores.shape[0]), np.zeros(ood_scores.shape[0])])
                all_scores = np.concatenate([id_scores.numpy(), ood_scores.numpy()])
                fpr, tpr, _ = metrics.roc_curve(binary_labels, all_scores, drop_intermediate=False)
                auroc = metrics.auc(fpr, tpr)
                idx_tpr95 = np.abs(tpr - 0.95).argmin()
                fpr_at_tpr95 = fpr[idx_tpr95]
                print(f"[{method_name}]: AUROC {auroc * 100:.2f}% | FPR@TPR95 {fpr_at_tpr95 * 100:.2f}%")
                if args.wandb:
                    import wandb
                    wandb.log({f"{method_name}_AUROC (↑)": auroc * 100,
                               f"{method_name}_FPR@TPR95 (↓)": fpr_at_tpr95 * 100,
                               "TASK": task_id})
                results[method_name] = {"auroc": auroc, "fpr_at_tpr95": fpr_at_tpr95, "scores": all_scores}
            continue  # 다음 method 로
        else:
            # 네트워크 모델을 위한 래퍼 클래스 생성
            class ModelWrapper:
                def __init__(self, network):
                    self.network = network
                    
                def __call__(self, x):
                    return self.network(x)["logits"]
                    
                def eval(self):
                    self.network.eval()
                    
                def zero_grad(self, set_to_none=True):
                    self.network.zero_grad(set_to_none=set_to_none)
                    
            wrapped_model = ModelWrapper(learner._network)
            id_scores, ood_scores = compute_ood_scores(method, wrapped_model, id_loader, ood_loader, device)

        # 시각화 및 로깅
        if args.verbose or args.wandb:
            hist_path = save_anomaly_histogram(id_scores.numpy(), ood_scores.numpy(), args, suffix=method.lower(), task_id=task_id)
            if args.wandb:
                import wandb
                wandb.log({f"Anomaly Histogram TASK {task_id}": wandb.Image(hist_path)})

        binary_labels = np.concatenate([np.ones(id_scores.shape[0]), np.zeros(ood_scores.shape[0])])
        all_scores = np.concatenate([id_scores.numpy(), ood_scores.numpy()])

        fpr, tpr, _ = metrics.roc_curve(binary_labels, all_scores, drop_intermediate=False)
        auroc = metrics.auc(fpr, tpr)
        idx_tpr95 = np.abs(tpr - 0.95).argmin()
        fpr_at_tpr95 = fpr[idx_tpr95]

        print(f"[{method}]: AUROC {auroc * 100:.2f}% | FPR@TPR95 {fpr_at_tpr95 * 100:.2f}%")
        if args.wandb:
            import wandb
            wandb.log({f"{method}_AUROC (↑)": auroc * 100, f"{method}_FPR@TPR95 (↓)": fpr_at_tpr95 * 100, "TASK": task_id})

        results[method] = {"auroc": auroc, "fpr_at_tpr95": fpr_at_tpr95, "scores": all_scores}

    return results

def vil_train(args):
    devices = [torch.device(f'cuda:{d}') if d >= 0 else torch.device('cpu')
               for d in args.device]
    args.device = devices
    seed_everything(args.seed)
    
    # OOD 하이퍼파라미터 업데이트
    update_ood_hyperparams(args)
    
    # develop 모드일 경우 tuned_epoch을 1로 설정
    if args.develop:
        args.tuned_epoch = 1
        logging.info("개발 모드: tuned_epoch을 1로 설정합니다.")
    
    # wandb 초기화
    if args.wandb_run and args.wandb_project:
        import wandb
        import getpass
        
        args.wandb = True
        wandb.init(entity="OODVIL", project=args.wandb_project, name=args.wandb_run, config=args)
        wandb.config.update({"username": getpass.getuser()})
    else:
        args.wandb = False

    loaders, class_mask, domain_list = build_continual_dataloader(args)
    if args.ood_dataset:
        loaders[-1]['ood'] = get_ood_dataset(args.ood_dataset, args)

    # OOD 학습용 데이터셋 준비 (optional)
    # --ood_train_dataset 인자에 여러 개의 데이터셋을 콤마(,)로 나열하면
    # 각 데이터셋을 개별적으로 로드한 뒤 ConcatDataset 으로 합칩니다.
    if args.ood_train_dataset:
        dataset_names = [name.strip() for name in args.ood_train_dataset.split(',') if name.strip()]

        # 'random' 또는 'blur' 옵션은 per-task 단계에서 동적으로 생성하므로 여기서는 None 으로 처리
        if len(dataset_names) == 1 and dataset_names[0].lower() in {"random", "blur"}:
            ood_train_dataset = None
        elif len(dataset_names) == 1:
            ood_train_dataset = get_ood_dataset(dataset_names[0], args)
        else:
            # 여러 데이터셋을 로드한 뒤 하나로 결합하되, 각 데이터셋에서 동일한 개수의 샘플을 사용하도록
            # 가장 작은 데이터셋 크기에 맞추어 RandomSampleWrapper 로 균등 샘플링합니다.
            loaded_datasets = [get_ood_dataset(n, args) for n in dataset_names]

            # 가장 작은 데이터셋 크기 산출
            min_len = min(len(ds) for ds in loaded_datasets)
            balanced_datasets = []
            for idx, ds in enumerate(loaded_datasets):
                if len(ds) > min_len:
                    balanced_datasets.append(RandomSampleWrapper(ds, min_len, args.seed + idx))
                else:
                    balanced_datasets.append(ds)

            ood_train_dataset = ConcatDataset(balanced_datasets)
            logging.info(
                f"Balanced concat OOD train datasets: {dataset_names} | per-dataset samples: {min_len} | total: {len(ood_train_dataset)}"
            )
    else:
        ood_train_dataset = None

    learner = Learner(vars(args))
    learner.is_dil   = True          # domain-IL branch
    learner.dil_init = False         # first task triggers PETL
    logging.info(f"Params: {sum(p.numel() for p in learner._network.parameters()):,}")

    num_tasks  = args.num_tasks
    acc_matrix = np.zeros((num_tasks, num_tasks))
    log_book   = {"A_last": [], "A_avg": [], "Forgetting": [], "Task_Time": []}
    
    total_start_time = time.time()
    task_experts = []  # 각 task 별 binary TE classifier 저장

    for tid in range(num_tasks):
        print(f"{' Training Task ' + str(tid):=^60}")
        task_start_time = time.time()
        
        dm = LoaderDataManager(loaders[tid], args.num_classes, args)

        learner._cur_task            = -1
        learner._known_classes       = 0
        learner._classes_seen_so_far = 0

        learner.incremental_train(dm)      # FULL RanPAC pipeline

        # -----------------------------------------------------------
        # (1) Task expert(TE) classifier 학습
        # -----------------------------------------------------------
        if args.ood_train_dataset:
            id_train_dataset = loaders[tid]['train'].dataset

            ood_flag = args.ood_train_dataset.lower()

            if ood_flag == "random":
                sample_img = id_train_dataset[0][0]
                if isinstance(sample_img, torch.Tensor):
                    img_shape = sample_img.shape
                else:
                    img_shape = T.ToTensor()(sample_img).shape
                num_samples = len(id_train_dataset)
                ood_ds_task = GaussianNoiseDataset(num_samples, img_shape, seed=args.seed + tid)

            elif ood_flag == "blur":
                ood_ds_task = BlurNoiseWrapper(id_train_dataset,
                                               noise_std=args.te_noise_std,
                                               blur_kernel=args.te_blur_kernel,
                                               blur_sigma=args.te_blur_sigma)

            else:
                ood_ds_task = ood_train_dataset  # 사전에 로드된 OOD 데이터셋 사용

            # 샘플 시각화 wandb 로깅 (최대 20장)
            if args.wandb:
                import wandb, torchvision.utils as vutils
                n_vis = min(20, len(ood_ds_task))
                idxs = torch.randint(0, len(ood_ds_task), (n_vis,))
                imgs = []
                for i in idxs:
                    img, _ = ood_ds_task[i]
                    if not isinstance(img, torch.Tensor):
                        img = T.ToTensor()(img)
                    imgs.append(img)
                grid = vutils.make_grid(imgs, nrow=5, normalize=True, value_range=(0,1))
                wandb.log({f"TE_ood_samples/Task{tid}": wandb.Image(grid), "TASK": tid})

            cls = train_te_classifier(learner, id_train_dataset, ood_ds_task, devices[0], args)
            task_experts.append(cls)

        A_last, A_avg, F = evaluate_till_now(
            learner, loaders, devices[0], tid, acc_matrix, args, task_experts
        )
        log_book["A_last"].append(A_last)
        log_book["A_avg"].append(A_avg)
        log_book["Forgetting"].append(F)
        
        learner._network.eval()
        if args.ood_dataset:
            print(f"{' Running OOD Eval after Task '+str(tid):=^60}")
            # ID val dataset 합치기
            id_val_sets = [loaders[t]['val'].dataset for t in range(tid+1)]
            id_datasets = torch.utils.data.ConcatDataset(id_val_sets)
            ood_dataset = loaders[-1]['ood']
            # evaluate_ood() 호출 (원본과 동일한 로직)
            _ = evaluate_ood(learner, id_datasets, ood_dataset, devices[0], args, task_id=tid, task_experts=task_experts)
        else:
            print("OOD 평가를 위한 데이터셋이 지정되지 않았습니다.")

        task_time = time.time() - task_start_time
        log_book["Task_Time"].append(task_time)
        
        msg = f"[Task {tid+1:2d}] 소요 시간: {task_time:.2f}초 ({task_time/60:.2f}분)"
        logging.info(msg)

    total_time = time.time() - total_start_time
    msg = f"\n전체 훈련 소요 시간: {total_time:.2f}초 ({total_time/60:.2f}분)"
    logging.info(msg)
    print(msg)

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(log_book).to_csv("results/vil_metrics_ranpac.csv", index=False)
    print("\n✓ Finished — metrics in  results/vil_metrics_ranpac.csv")

def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",        type=int, default=1)
    p.add_argument("--device",      type=int, nargs='+', default=[0])
    p.add_argument("--model_name",  default="adapter")
    p.add_argument("--convnet_type",default="pretrained_vit_b16_224_adapter")
    p.add_argument("--body_lr",     type=float, default=0.01)
    p.add_argument("--head_lr",     type=float, default=0.01)
    p.add_argument("--weight_decay",type=float, default=5e-4)
    p.add_argument("--min_lr",      type=float, default=1e-6)
    p.add_argument("--use_RP",      action="store_true")
    p.add_argument("--M",           type=int, default=10000)
    p.add_argument("--tuned_epoch", type=int, default=5)

    #vil 
    p.add_argument("--num_tasks",   type=int, default=20)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--data_path",   default="/local_datasets")
    p.add_argument("--shuffle",     action="store_true")
    p.add_argument("--IL_mode",     default="vil", type=str)
    p.add_argument("--dataset",     default="iDigits", type=str)
    p.add_argument("--save",        default="./save", type=str)
    p.add_argument("--ood_dataset", default=None, type=str)
    p.add_argument("--ood_method", default="ALL", type=str)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--develop", action="store_true")
    p.add_argument("--ood_develop", type=int, default=None)
    
    # wandb 관련 인자 추가
    p.add_argument("--wandb_run", type=str, default=None, help="Wandb run name")
    p.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")

    # === OOD method hyper-parameters ===
    p.add_argument('--energy_temperature', type=float, default=1.0, help='Temperature for ENERGY postprocessor')
    # GEN
    p.add_argument('--gen_gamma', type=float, default=0.01, help='Gamma for GEN / PRO_GEN postprocessor')
    p.add_argument('--gen_M', type=int, default=3, help='Top-M probabilities used in GEN / PRO_GEN postprocessor')
    # PRO-GEN
    p.add_argument('--pro_gen_noise_level', type=float, default=5e-4, help='Noise level for PRO_GEN postprocessor')
    p.add_argument('--pro_gen_gd_steps', type=int, default=3, help='Gradient descent steps for PRO_GEN postprocessor')
    # PRO-MSP
    p.add_argument('--pro_msp_temperature', type=float, default=1.0, help='Temperature for PRO_MSP postprocessor')
    p.add_argument('--pro_msp_noise_level', type=float, default=0.003, help='Noise level for PRO_MSP postprocessor')
    p.add_argument('--pro_msp_gd_steps', type=int, default=1, help='Gradient descent steps for PRO_MSP postprocessor')
    # PRO-MSP-T
    p.add_argument('--pro_msp_t_temperature', type=float, default=1.0, help='Temperature for PRO_MSP_T postprocessor')
    p.add_argument('--pro_msp_t_noise_level', type=float, default=0.003, help='Noise level for PRO_MSP_T postprocessor')
    p.add_argument('--pro_msp_t_gd_steps', type=int, default=1, help='Gradient descent steps for PRO_MSP_T postprocessor')
    # PRO-ENT
    p.add_argument('--pro_ent_noise_level', type=float, default=0.0014, help='Noise level for PRO_ENT postprocessor')
    p.add_argument('--pro_ent_gd_steps', type=int, default=2, help='Gradient descent steps for PRO_ENT postprocessor')

    # === Task-specific OOD classifier 관련 인자 ===
    p.add_argument('--ood_train_dataset', type=str, default=None,
                   help='OOD 데이터셋 지정: "random"(가우시안), "blur"(ID 블러+노이즈) 또는 콤마로 구분된 실제 OOD 데이터셋 이름 목록(e.g., "EMNIST,KMNIST")')
    p.add_argument('--te_epochs', type=int, default=1, help='Epochs to train task expert binary classifier')
    p.add_argument('--te_lr', type=float, default=1e-3, help='Learning rate for task expert binary classifier')
    p.add_argument('--te_batch_size', type=int, default=256, help='Batch size for task expert binary classifier')
    p.add_argument('--te_feature', type=str, default='rp', choices=['feat','rp','decorr','logits'],
                   help='Feature type for task expert: raw convnet feat, random-projected (rp), decorrelated Gram (decorr), or logits')
    p.add_argument('--te_score_type', type=str, default='logit', choices=['sigmoid','logit'],
                   help='Score type for TE OOD confidence: use sigmoid probability or raw logit')
    # --- New TE uncertainty hyper-parameters ---
    p.add_argument('--te_agg_methods', type=str, default='max',
                   help='Comma-separated aggregation methods for TE OOD score (max,mean,wsum,std)')
    p.add_argument('--te_mc_passes', type=int, default=10,
                   help='Number of MC-dropout forward passes for TE uncertainty estimation')
    p.add_argument('--te_dropout', type=float, default=0.2,
                   help='Dropout probability used in TE classifier')
    # --- Blur/Noise 하이퍼파라미터 ---
    p.add_argument('--te_noise_std', type=float, default=0.2,
                   help='Blur OOD 샘플에 추가할 가우시안 노이즈 표준편차')
    p.add_argument('--te_blur_kernel', type=int, default=7,
                   help='Blur OOD 샘플에 적용할 GaussianBlur 커널 크기 (홀수)')
    p.add_argument('--te_blur_sigma', type=float, default=3.0,
                   help='Blur OOD 샘플에 적용할 sigma 값(크면 더 강한 블러)')

    # not used but kept for compatibility
    p.add_argument("--epochs",      type=int, default=1)
    p.add_argument("--print_freq",  type=int, default=1)
    p.add_argument("--use_input_norm", action="store_true")
    return p
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    args = get_parser().parse_args()
    set_data_config(args)
    log_file = f"logs/vil_ranpac_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_file),
                  logging.StreamHandler(sys.stdout)]
    )
    vil_train(args)
