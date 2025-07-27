import argparse, datetime, logging, os, sys, time, copy
import numpy as np
import torch, pandas as pd
import random
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

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
# Task-specific OOD classifier 학습 함수
# ------------------------------------------------------------------ #
def train_task_ood_classifier(learner, id_dataset, ood_dataset, device, args):
    """현재 task에 대한 binary OOD classifier(0:ID, 1:OOD)를 학습합니다."""

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
        batch_size=args.ood_cls_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # ------------------------------------------------------------
    # feature extraction helper
    # ------------------------------------------------------------
    feature_type = args.ood_cls_feature.lower()

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
    classifier = torch.nn.Linear(feat_dim, 1).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.ood_cls_lr)

    # 4) 학습
    classifier.train()
    for epoch in range(args.ood_cls_epochs):
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
        print(f"Epoch {epoch+1}/{args.ood_cls_epochs} loss: {epoch_loss/len(loader):.4f}")
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
                      acc_matrix, args):
    for t in range(task_id + 1):
        correct, total = 0, 0
        model._network.eval()
        with torch.no_grad():
            for x, y in loaders[t]['val']:
                x, y = x.to(device), y.to(device)
                pred = model._network(x)['logits'].argmax(1)
                correct += (pred == y).sum().item()
                total   += y.size(0)
        acc_matrix[t, task_id] = 100. * correct / total

    A_i    = [np.mean(acc_matrix[:i+1, i]) for i in range(task_id+1)]
    A_last = A_i[-1]
    A_avg  = np.mean(A_i)
    if task_id > 0:
        forgetting = np.mean(
            (np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id]
        )
    else:
        forgetting = 0.0

    msg = (f"[Task {task_id+1:2d}] "
           f"A_last {A_last:.2f} | A_avg {A_avg:.2f} | "
           f"Forgetting {forgetting:.2f}")
    logging.info(msg); print(msg)
    
    # wandb 로깅 추가
    if args.wandb:
        import wandb
        wandb.log({"A_last (↑)": A_last, "A_avg (↑)": A_avg, "Forgetting (↓)": forgetting, "TASK": task_id})

    sub_matrix = acc_matrix[:task_id+1, :task_id+1]
    result = np.where(np.triu(np.ones_like(sub_matrix, dtype=bool)), sub_matrix, np.nan)
    save_accuracy_heatmap(result, task_id, args)
    return A_last, A_avg, forgetting

# ========================= OOD 평가 함수 개정 ============================ #
#   * 기존 adapter 기반 방법 + 새 TASKCLS 방법 통합                        #
# ====================================================================== #
def evaluate_ood(learner, id_datasets, ood_dataset, device, args, task_id=None, task_classifiers=None):
    """OOD 평가를 위한 통합 함수 (adapter 기반)"""
    learner._network.eval()
    
    # === New unified OOD evaluation (adapter 기반) ===
    ood_method = args.ood_method.upper()

    # 1) 데이터셋 크기 맞추기
    id_size, ood_size = len(id_datasets), len(ood_dataset)
    min_size = min(id_size, ood_size)
    if args.develop:
        min_size = 1000
    if args.verbose:
        print(f"ID dataset size: {id_size}, OOD dataset size: {ood_size}. Using {min_size} samples each for evaluation.")

    id_dataset_aligned = RandomSampleWrapper(id_datasets, min_size, args.seed) if id_size > min_size else id_datasets
    ood_dataset_aligned = RandomSampleWrapper(ood_dataset, min_size, args.seed) if ood_size > min_size else ood_dataset

    id_loader = torch.utils.data.DataLoader(id_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    ood_loader = torch.utils.data.DataLoader(ood_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 2) 평가할 방법 결정
    from OODdetectors.ood_adapter import SUPPORTED_METHODS, compute_ood_scores

    # 새 방법 추가
    EXTRA_METHODS = ["TASKCLS"]

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
        if method == "TASKCLS":
            if not task_classifiers:
                raise ValueError("TASKCLS 평가를 위해서는 task_classifiers 리스트가 필요합니다.")

            # feature 기반 score 계산 함수 정의
            feature_type = args.ood_cls_feature.lower()

            def extract_feats(inputs):
                if feature_type == 'logits':
                    return learner._network(inputs)["logits"].detach()

                # (2) Backbone features
                feats = learner._network.convnet(inputs).detach()  # (B, D)

                # (3) Random projection (shared by 'rp' and first step of 'decorr')
                if feature_type in {'rp', 'decorr'}:
                    W_rand = getattr(learner, 'W_rand', None)
                    if W_rand is None:
                        W_rand = getattr(getattr(learner._network, 'fc', None), 'W_rand', None)
                    if W_rand is not None:
                        feats = F.relu(feats @ W_rand)             # (B, M)
                    if feature_type == 'rp':
                        return feats

                # (4) Decorrelation step (applied after RP)
                if feature_type == 'decorr':
                    G = getattr(learner, 'G', None)
                    eps = 1000000
                    G_inv = torch.linalg.pinv(G.to(device) + eps * torch.eye(G.shape[0], device=device))
                    feats = feats @ G_inv
                    return feats

                # Default: raw features
                return feats

            def _gather_taskcls_scores(loader, label_name):
                all_scores = []
                learner._network.eval()
                # clf_max_counter = np.zeros(len(task_classifiers), dtype=int)
                with torch.no_grad():
                    for inputs, _ in loader:
                        inputs = inputs.to(device)
                        feats = extract_feats(inputs)
                        # 각 classifier 의 OOD 확률 계산 후 max
                        scores = []
                        for cls in task_classifiers:
                            logit = cls(feats).squeeze()
                            score = torch.sigmoid(logit) if args.ood_score_type == "sigmoid" else logit
                            scores.append(score)
                        scores = torch.stack(scores, dim=0)  # (num_cls, batch)
                        max_ood_score, max_idx = scores.max(dim=0)  # (batch,)
                        # clf_max_counter += torch.bincount(max_idx.cpu(), minlength=len(task_classifiers)).numpy()
                        conf = 1.0 - max_ood_score if args.ood_score_type == "sigmoid" else -max_ood_score  # ID confidence
                        all_scores.append(conf.cpu())
                # wandb 로깅: classifier별 max 선택된 횟수
                # if args.wandb:
                #     import wandb
                #     data = [[i, int(clf_max_counter[i])] for i in range(len(task_classifiers))]
                #     table = wandb.Table(data=data, columns=["classifier", f"{label_name}_count"])
                #     wandb.log({f"TASKCLS_{label_name}_max_counts/Task{task_id}": table, "TASK": task_id})
                return torch.cat(all_scores, dim=0)

            id_scores  = _gather_taskcls_scores(id_loader, "ID")
            ood_scores = _gather_taskcls_scores(ood_loader, "OOD")
            # === 추가: TASKCLS 분포 그리드 로깅 ===
            if args.wandb:
                import matplotlib.pyplot as plt
                import wandb

                # 각 task별 ID 데이터 로더 구성
                # ConcatDataset 는 .datasets 리스트를 보유하며, 단일 데이터셋일 경우에는 해당 속성이 없음
                id_task_datasets = id_datasets.datasets
                task_id_loaders = [torch.utils.data.DataLoader(ds,
                                                                batch_size=args.batch_size,
                                                                shuffle=False,
                                                                num_workers=args.num_workers)
                                   for ds in id_task_datasets]

                # OOD 로더를 마지막 열로 추가
                task_id_loaders.append(ood_loader)
                col_names = [f"T{idx+1}" for idx in range(len(id_task_datasets))] + ["OOD"]

                # TE × (tasks+OOD) 점수 수집
                grid_scores = [[None for _ in range(len(task_id_loaders))]
                               for _ in range(len(task_classifiers))]

                learner._network.eval()
                with torch.no_grad():
                    for col_idx, ldr in enumerate(task_id_loaders):
                        for inputs, _ in ldr:
                            inputs = inputs.to(device)
                            feats = extract_feats(inputs)
                            for row_idx, cls in enumerate(task_classifiers):
                                logit = cls(feats).squeeze()
                                score = torch.sigmoid(logit) if args.ood_score_type == "sigmoid" else logit
                                conf  = 1.0 - score if args.ood_score_type == "sigmoid" else -score
                                if grid_scores[row_idx][col_idx] is None:
                                    grid_scores[row_idx][col_idx] = conf.cpu()
                                else:
                                    grid_scores[row_idx][col_idx] = torch.cat([
                                        grid_scores[row_idx][col_idx],
                                        conf.cpu()
                                    ], dim=0)

                # === Grid 히스토그램 시각화 ===
                # (1) 범위 계산 (전 subplot 공통 범위)
                global_min = min([torch.min(x).item() for row in grid_scores for x in row])
                global_max = max([torch.max(x).item() for row in grid_scores for x in row])

                n_rows, n_cols = len(task_classifiers), len(task_id_loaders)
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.2 * n_rows), squeeze=False)

                for r in range(n_rows):
                    for c in range(n_cols):
                        ax = axes[r][c]
                        data_cell = grid_scores[r][c].numpy()
                        ax.hist(data_cell, bins=30, range=(global_min, global_max), color="steelblue", alpha=0.7)
                        # 축 라벨 및 타이틀
                        if r == 0:
                            ax.set_title(col_names[c])
                        if c == 0:
                            ax.set_ylabel(f"TE{r+1}")
                        # x-축 범위 및 틱
                        ax.set_xlim(global_min, global_max)
                        ax.tick_params(axis='x', labelsize=6)
                        ax.set_yticks([])
                        # 셀별 값 범위 텍스트
                        cell_min, cell_max = data_cell.min(), data_cell.max()
                        ax.text(0.5, 0.9, f"[{cell_min:.1f}, {cell_max:.1f}]", transform=ax.transAxes,
                                ha='center', va='top', fontsize=6, color='dimgray')
                plt.tight_layout()
                wandb.log({f"TASKCLS_score_grid/Task{task_id}": wandb.Image(fig), "TASK": task_id})
                plt.close(fig)
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
        if len(dataset_names) == 1:
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
    task_ood_classifiers = []  # 각 task 별 binary OOD classifier 저장

    for tid in range(num_tasks):
        print(f"{' Training Task ' + str(tid):=^60}")
        task_start_time = time.time()
        
        dm = LoaderDataManager(loaders[tid], args.num_classes, args)

        learner._cur_task            = -1
        learner._known_classes       = 0
        learner._classes_seen_so_far = 0

        learner.incremental_train(dm)      # FULL RanPAC pipeline

        # -----------------------------------------------------------
        # (1) Task-specific OOD classifier 학습
        # -----------------------------------------------------------
        if ood_train_dataset is not None:
            id_train_dataset = loaders[tid]['train'].dataset
            cls = train_task_ood_classifier(learner, id_train_dataset, ood_train_dataset, devices[0], args)
            task_ood_classifiers.append(cls)

        A_last, A_avg, F = evaluate_till_now(
            learner, loaders, devices[0], tid, acc_matrix, args
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
            _ = evaluate_ood(learner, id_datasets, ood_dataset, devices[0], args, task_id=tid, task_classifiers=task_ood_classifiers)
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
                   help='Comma-separated OOD dataset list used for training task-specific classifiers (e.g., "EMNIST,KMNIST")')
    p.add_argument('--ood_cls_epochs', type=int, default=1, help='Epochs to train task-specific OOD classifier')
    p.add_argument('--ood_cls_lr', type=float, default=1e-3, help='Learning rate for task-specific OOD classifier')
    p.add_argument('--ood_cls_batch_size', type=int, default=256, help='Batch size for task-specific OOD classifier')
    p.add_argument('--ood_cls_feature', type=str, default='feat', choices=['feat','rp','decorr','logits'],
                   help='Feature type for task OOD classifier: raw convnet feat, random-projected (rp), decorrelated Gram (decorr), or logits')
    p.add_argument('--ood_score_type', type=str, default='sigmoid', choices=['sigmoid','logit'],
                   help='Score type for TASKCLS OOD confidence: use sigmoid probability or raw logit')

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
