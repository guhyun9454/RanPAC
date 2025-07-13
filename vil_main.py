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
from continual_datasets.dataset_utils import RandomSampleWrapper
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

def evaluate_ood(learner, id_datasets, ood_dataset, device, args, task_id=None):
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
    if ood_method == "ALL":
        methods = SUPPORTED_METHODS
    else:
        # 쉼표로 구분된 메소드들 처리
        methods = [method.strip().upper() for method in ood_method.split(',')]
        # 지원되지 않는 메소드 확인
        unsupported = [m for m in methods if m not in SUPPORTED_METHODS]
        if unsupported:
            raise ValueError(f"지원되지 않는 OOD 메소드: {unsupported}. 지원되는 메소드: {SUPPORTED_METHODS}")

    from sklearn import metrics
    results = {}

    for method in methods:
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

        if method == "PSEUDO" and hasattr(learner, "pseudo_processor") and learner.pseudo_processor.trained:
            # 이미 학습된 processor 사용
            processor = learner.pseudo_processor

            def _gather_scores(loader):
                scores = []
                learner._network.eval()
                for inputs, _ in loader:
                    inputs = inputs.to(device)
                    learner._network.zero_grad(set_to_none=True)
                    _, conf = processor.postprocess(learner._network, inputs)
                    scores.append(conf.detach().cpu())
                return torch.cat(scores, dim=0)

            id_scores = _gather_scores(id_loader)
            ood_scores = _gather_scores(ood_loader)
        else:
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

        if args.develop and tid == 5:
            logging.info("개발 모드: 첫 태스크 학습 완료 후 종료합니다.")
            break

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

    # === PSEUDO-OOD classifier hyper-parameters ===
    p.add_argument('--pseudo_eps', type=float, default=0.02, help='FGSM epsilon for pseudo-OOD generation')
    p.add_argument('--pseudo_max_batches', type=int, default=0, help='Number of train batches used to train pseudo-OOD classifier')
    p.add_argument('--pseudo_lr', type=float, default=1e-4, help='Learning rate of pseudo-OOD logistic classifier')
    p.add_argument('--pseudo_epochs', type=int, default=3, help='Training epochs of pseudo-OOD logistic classifier')
    p.add_argument('--pseudo_hidden_dim', type=int, default=128, help='Hidden dimension for pseudo-OOD classifier MLP / RP output size')
    p.add_argument('--pseudo_layers', type=int, default=0, choices=[0,1,2,3], help='Number of layers: 0=random projection, 1=linear, 2, 3 hidden layers')
    p.add_argument('--pseudo_lambda', type=float, default=1e-3, help='Ridge regularization lambda for decorrelation whitening when pseudo_layers=0')

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
