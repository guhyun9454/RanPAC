import argparse, datetime, logging, os, sys, time, copy
import numpy as np
import torch, pandas as pd
import random
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from continual_datasets.build_incremental_scenario import build_continual_dataloader
from RanPAC import Learner
from continual_datasets.dataset_utils import set_data_config
from continual_datasets.dataset_utils import get_ood_dataset
from continual_datasets.dataset_utils import RandomSampleWrapper
from utils.acc_heatmap import save_accuracy_heatmap
from utils import save_anomaly_histogram, save_logits_statistics


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
            logging.info("개발 모드: 학습 dataloader iteration을 1회로 제한합니다.")

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

def evaluate_ood(learner, id_datasets, ood_dataset, device, args, task_id=None):
    # learner._network를 사용해 MSP/ENERGY/KL/PBL OOD 지표 계산
    learner._network.eval()
    ood_method = args.ood_method.upper()

    def MSP(logits):
        return F.softmax(logits, dim=1).max(dim=1)[0]

    def ENERGY(logits):
        return torch.logsumexp(logits, dim=1)

    def KL(logits):
        return F.cross_entropy(logits, torch.ones_like(logits) / logits.shape[-1], reduction='none')

    # 1) 데이터셋 크기 맞추기
    id_size  = len(id_datasets)
    ood_size = len(ood_dataset)
    min_size = min(id_size, ood_size)
    if args.develop:
        min_size = 1000
    if args.verbose:
        print(f"ID size: {id_size}, OOD size: {ood_size}, using {min_size} samples each")

    id_aligned  = RandomSampleWrapper(id_datasets, min_size, args.seed) if id_size > min_size else id_datasets
    ood_aligned = RandomSampleWrapper(ood_dataset, min_size, args.seed) if ood_size > min_size else ood_dataset

    id_loader  = torch.utils.data.DataLoader(id_aligned,  batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)
    ood_loader = torch.utils.data.DataLoader(ood_aligned, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)

    # 2) 로짓 및 feature 수집
    id_logits_list, ood_logits_list = [], []
    id_features_list, ood_features_list = [], []
    
    with torch.no_grad():
        for x, _ in id_loader:
            x = x.to(device)
            # logits와 features 둘 다 추출
            outputs = learner._network(x)
            logits = outputs["logits"]
            features = learner._network.convnet(x)  # feature extraction
            
            id_logits_list.append(logits.cpu())
            id_features_list.append(features.cpu())
            
        for x, _ in ood_loader:
            x = x.to(device)
            outputs = learner._network(x)
            logits = outputs["logits"]
            features = learner._network.convnet(x)
            
            ood_logits_list.append(logits.cpu())
            ood_features_list.append(features.cpu())

    id_logits  = torch.cat(id_logits_list,  dim=0)
    ood_logits = torch.cat(ood_logits_list, dim=0)
    id_features = torch.cat(id_features_list, dim=0)
    ood_features = torch.cat(ood_features_list, dim=0)

    # 3) 통계 저장 (선택)
    if args.save:
        save_logits_statistics(id_logits, ood_logits, args, task_id or 0)

    # 4) 레이블 및 평가 준비
    binary_labels = np.concatenate([np.ones(id_logits.shape[0]), np.zeros(ood_logits.shape[0])])
    methods = ["MSP","ENERGY","KL","PBL"] if ood_method=="ALL" else [ood_method]
    results = {}
    
    # 프로토타입 통계 로깅
    if "PBL" in methods and len(learner.prototypes) > 0:
        proto_stats = {c: len(protos) for c, protos in learner.prototypes.items()}
        print(f"[PBL] Prototype statistics: {proto_stats}")
        total_protos = sum(proto_stats.values())
        print(f"[PBL] Total prototypes: {total_protos}, Classes: {len(proto_stats)}")

    for m in methods:
        if m == "MSP":
            id_scores, ood_scores = MSP(id_logits), MSP(ood_logits)
        elif m == "ENERGY":
            id_scores, ood_scores = ENERGY(id_logits), ENERGY(ood_logits)
        elif m == "KL":
            id_scores, ood_scores = KL(id_logits), KL(ood_logits)
        elif m == "PBL":
            # PBL 스코어 계산 (낮을수록 ID-like)
            id_scores = torch.tensor([learner.compute_ood_score(f.cpu().numpy()) for f in id_features])
            ood_scores = torch.tensor([learner.compute_ood_score(f.cpu().numpy()) for f in ood_features])
            # 부호 반전 (ROC 계산을 위해)
            id_scores = -id_scores
            ood_scores = -ood_scores

        if args.verbose:
            save_anomaly_histogram(id_scores.cpu().numpy(),
                                   ood_scores.cpu().numpy(),
                                   args, suffix=m.lower(), task_id=task_id)

        all_scores = torch.cat([id_scores, ood_scores], dim=0).cpu().numpy()
        fpr, tpr, _ = roc_curve(binary_labels, all_scores, drop_intermediate=False)
        auroc = auc(fpr, tpr)
        idx95 = np.abs(tpr - 0.95).argmin()
        fpr95 = fpr[idx95]

        print(f"[{m}] AUROC: {auroc*100:.2f}%, FPR@TPR95: {fpr95*100:.2f}%")
        results[m] = {"auroc": auroc, "fpr_at_tpr95": fpr95, "scores": all_scores}
        
        # wandb 로깅 추가
        if args.wandb:
            import wandb
            wandb.log({f"{m}_AUROC (↑)": auroc * 100, f"{m}_FPR@TPR95 (↓)": fpr95 * 100, "TASK": task_id})

    # t-SNE 시각화 (verbose 모드이거나 PBL 평가 시)
    if (args.verbose or "PBL" in methods) and len(learner.prototypes) > 0:
        # ID 샘플의 레이블 추출을 위해 원본 데이터 다시 로드
        id_labels_list = []
        with torch.no_grad():
            for x, y in id_loader:
                id_labels_list.append(y.cpu().numpy())
        id_labels = np.concatenate(id_labels_list)
        
        visualize_ood_tsne(id_features.numpy(), id_labels, ood_features.numpy(), 
                          learner, args, task_id)

    return results

def visualize_ood_tsne(id_features, id_labels, ood_features, learner, args, task_id):
    """t-SNE visualization for OOD/old ID/new ID distribution with prototypes"""
    print(f"[t-SNE] Starting visualization for Task {task_id}...")
    
    # 1. 저장 디렉토리 생성
    save_dir = os.path.join(args.save, 'tsne_plots')
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. 샘플링 (너무 많으면)
    max_samples_per_type = 1000
    
    # ID 샘플 구분: Task 0이면 모든 것이 new ID, 그 이후는 old ID/new ID 혼재
    # Domain-incremental learning (iDigits)에서는 각 task에서 모든 클래스를 보므로
    # 시간적 순서로 구분: 현재 task에서 평가하는 것은 이전 task들이 old ID
    
    if task_id == 0:
        # 첫 번째 task: 모든 ID가 new ID
        new_id_features = id_features
        new_id_labels = id_labels
        old_id_features = np.empty((0, id_features.shape[1]))
        old_id_labels = np.empty((0,))
    else:
        # 두 번째 task부터: 실제로는 모든 ID가 누적된 것이므로 비율로 나누기
        # 또는 프로토타입 생성 시점을 기준으로 구분
        n_samples = len(id_features)
        # 최근 30%를 new ID로, 나머지를 old ID로 간주 (휴리스틱)
        new_ratio = 0.3
        split_idx = int(n_samples * (1 - new_ratio))
        
        # 랜덤 셔플 후 분할 (더 균등한 분포를 위해)
        indices = np.random.permutation(n_samples)
        old_indices = indices[:split_idx]
        new_indices = indices[split_idx:]
        
        old_id_features = id_features[old_indices]
        old_id_labels = id_labels[old_indices]
        new_id_features = id_features[new_indices]
        new_id_labels = id_labels[new_indices]
    
    # 샘플링
    if len(new_id_features) > max_samples_per_type:
        indices = np.random.choice(len(new_id_features), max_samples_per_type, replace=False)
        new_id_features = new_id_features[indices]
        new_id_labels = new_id_labels[indices]
        
    if len(old_id_features) > max_samples_per_type:
        indices = np.random.choice(len(old_id_features), max_samples_per_type, replace=False)
        old_id_features = old_id_features[indices]
        old_id_labels = old_id_labels[indices]
        
    if len(ood_features) > max_samples_per_type:
        indices = np.random.choice(len(ood_features), max_samples_per_type, replace=False)
        ood_features = ood_features[indices]
    
    # 3. 프로토타입 수집
    proto_centers_list = []
    proto_radii = []
    proto_labels = []
    proto_task_type = []  # 'old' or 'new'
    
    if learner.prototypes:
        for class_id, plist in learner.prototypes.items():
            for center, radius in plist:
                proto_centers_list.append(center.cpu().detach().numpy())
                proto_radii.append(radius.item())
                proto_labels.append(class_id)
                # Task 0이면 모든 프로토타입이 new, 그 이후는 혼재로 간주
                if task_id == 0:
                    proto_task_type.append('new')
                else:
                    # 간단한 휴리스틱: 클래스 ID의 홀짝으로 구분
                    proto_task_type.append('new' if class_id % 2 == 0 else 'old')
    
    if not proto_centers_list:
        print("[t-SNE] No prototypes found. Skipping visualization.")
        return
    
    proto_centers = np.array(proto_centers_list)
    
    # 4. 모든 데이터 결합
    all_features = []
    all_labels = []
    all_types = []
    
    if len(new_id_features) > 0:
        all_features.append(new_id_features)
        all_labels.extend(new_id_labels)
        all_types.extend(['new_id'] * len(new_id_features))
    
    if len(old_id_features) > 0:
        all_features.append(old_id_features)
        all_labels.extend(old_id_labels)
        all_types.extend(['old_id'] * len(old_id_features))
    
    if len(ood_features) > 0:
        all_features.append(ood_features)
        all_labels.extend([-1] * len(ood_features))
        all_types.extend(['ood'] * len(ood_features))
    
    if not all_features:
        print("[t-SNE] No features to visualize.")
        return
        
    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)
    all_types = np.array(all_types)
    
    # 5. t-SNE 실행
    print(f"[t-SNE] Running t-SNE on {len(all_features)} samples + {len(proto_centers)} prototypes...")
    tsne_data = np.vstack([all_features, proto_centers])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=args.seed)
    tsne_results = tsne.fit_transform(tsne_data)
    
    # 결과 분리
    feat_tsne = tsne_results[:-len(proto_centers)]
    proto_tsne = tsne_results[-len(proto_centers):]
    
    # 6. 시각화
    plt.figure(figsize=(24, 16))
    ax = plt.gca()
    
    # 타입별 색상 매핑
    type_colors = {'new_id': 'blue', 'old_id': 'red', 'ood': 'gray'}
    type_alphas = {'new_id': 0.8, 'old_id': 0.6, 'ood': 0.3}
    type_sizes = {'new_id': 25, 'old_id': 20, 'ood': 15}
    
    # 각 타입별로 플롯
    for type_name in ['ood', 'old_id', 'new_id']:  # ood를 먼저 그려서 뒤에 가리도록
        mask = all_types == type_name
        if np.sum(mask) > 0:
            ax.scatter(feat_tsne[mask, 0], feat_tsne[mask, 1], 
                      c=type_colors[type_name], 
                      label=f'{type_name.upper()} ({np.sum(mask)})',
                      alpha=type_alphas[type_name], 
                      s=type_sizes[type_name])
    
    # 프로토타입 그리기
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    scaling_factor = min(x_range, y_range) / 100.0  # 반경 스케일링
    
    for i, (center_2d, radius, class_id, task_type) in enumerate(zip(proto_tsne, proto_radii, proto_labels, proto_task_type)):
        color = 'darkblue' if task_type == 'new' else 'darkred'
        marker = '*' if task_type == 'new' else 'D'
        
        # 프로토타입 중심점
        ax.scatter(center_2d[0], center_2d[1], 
                  c=color, marker=marker, s=400, 
                  edgecolor='black', linewidth=2, zorder=10,
                  label=f'Proto {task_type.title()}' if i == 0 or (i > 0 and proto_task_type[i-1] != task_type) else "")
        
        # 반경 원
        circle = Circle(center_2d, radius * scaling_factor, 
                       color=color, fill=False, linewidth=2, 
                       linestyle='--', alpha=0.7, zorder=8)
        ax.add_patch(circle)
        
        # 클래스 ID 표시
        ax.text(center_2d[0], center_2d[1] + radius * scaling_factor * 1.5, 
               f'C{class_id}', fontsize=10, ha='center', va='bottom',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.title(f't-SNE Visualization: Task {task_id} (Proto: {len(proto_centers)})', fontsize=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # 저장
    plot_path = os.path.join(save_dir, f'task_{task_id}_ood_tsne.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[t-SNE] Saved visualization to {plot_path}")
    
    # wandb 로깅
    if args.wandb:
        import wandb
        wandb.log({f"OOD t-SNE TASK {task_id}": wandb.Image(plot_path)})

def vil_train(args):
    devices = [torch.device(f'cuda:{d}') if d >= 0 else torch.device('cpu')
               for d in args.device]
    args.device = devices
    seed_everything(args.seed)
    
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

    learner = Learner(vars(args))
    learner.is_dil   = True          # domain-IL branch
    learner.dil_init = False         # first task triggers PETL
    logging.info(f"Params: {sum(p.numel() for p in learner._network.parameters()):,}")

    num_tasks  = args.num_tasks
    acc_matrix = np.zeros((num_tasks, num_tasks))
    log_book   = {"A_last": [], "A_avg": [], "Forgetting": [], "Task_Time": []}
    
    total_start_time = time.time()

    for tid in range(num_tasks):
        print(f"{' Training Task ' + str(tid):=^60}")
        task_start_time = time.time()
        
        dm = LoaderDataManager(loaders[tid], args.num_classes, args)

        learner._cur_task            = -1
        learner._known_classes       = 0
        learner._classes_seen_so_far = 0

        learner.incremental_train(dm)      # FULL RanPAC pipeline
        learner.after_task()               # Log prototype statistics

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
            _ = evaluate_ood(learner, id_datasets, ood_dataset, devices[0], args, task_id=tid)
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
    p.add_argument("--ood_method", default="ALL", choices=["MSP","ENERGY","KL","PBL","ALL"])
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--develop", action="store_true")
    
    # PBL (Prototype-Boundary Learning) 관련 인자
    p.add_argument("--pbl_tau_split", type=float, default=150.0, help="Distance threshold to spawn new prototype (will be converted to cosine similarity)")
    p.add_argument("--pbl_alpha", type=float, default=0.1, help="EMA ratio for updating prototype center/radius")
    p.add_argument("--pbl_max_protos", type=int, default=5, help="Maximum number of prototypes per class")
    
    # wandb 관련 인자 추가
    p.add_argument("--wandb_run", type=str, default=None, help="Wandb run name")
    p.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")

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
