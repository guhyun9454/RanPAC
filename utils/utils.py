import io
import os
import time
from collections import defaultdict, deque
import datetime
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch.nn.functional as F

def save_logits_statistics(id_logits, ood_logits, args, task_id):
    """
    logits의 통계값을 계산하고 시각화하여 저장합니다. 통계값은 출력만 합니다.
    
    Args:
        id_logits: ID 데이터의 logits
        ood_logits: OOD 데이터의 logits
        args: 설정 인자
        task_id: 현재 태스크 ID
    """
    if not os.path.exists(os.path.join(args.save, 'logits_stats')):
        os.makedirs(os.path.join(args.save, 'logits_stats'))
    
    # ID 데이터의 logits 통계
    id_mean = torch.mean(id_logits, dim=0).cpu().numpy()
    id_max = torch.max(id_logits, dim=0)[0].cpu().numpy()
    id_min = torch.min(id_logits, dim=0)[0].cpu().numpy()
    id_std = torch.std(id_logits, dim=0).cpu().numpy()
    
    # OOD 데이터의 logits 통계
    ood_mean = torch.mean(ood_logits, dim=0).cpu().numpy()
    ood_max = torch.max(ood_logits, dim=0)[0].cpu().numpy()
    ood_min = torch.min(ood_logits, dim=0)[0].cpu().numpy()
    ood_std = torch.std(ood_logits, dim=0).cpu().numpy()
    
    # 클래스별 logits 평균 시각화
    plt.figure(figsize=(12, 6))
    x = np.arange(len(id_mean))
    plt.bar(x - 0.2, id_mean, width=0.4, label='ID Mean', alpha=0.7, color='blue')
    plt.bar(x + 0.2, ood_mean, width=0.4, label='OOD Mean', alpha=0.7, color='red')
    plt.xlabel('Class Index')
    plt.ylabel('Mean Logit Value')
    plt.title(f'Task {task_id+1}: ID vs OOD Mean Logit Values')
    plt.legend()
    plt.savefig(os.path.join(args.save, 'logits_stats', f'task{task_id+1}_mean_logits.png'))
    plt.close()
    
    # 클래스별 logits 최대값 시각화
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, id_max, width=0.4, label='ID Max', alpha=0.7, color='blue')
    plt.bar(x + 0.2, ood_max, width=0.4, label='OOD Max', alpha=0.7, color='red')
    plt.xlabel('Class Index')
    plt.ylabel('Max Logit Value')
    plt.title(f'Task {task_id+1}: ID vs OOD Max Logit Values')
    plt.legend()
    plt.savefig(os.path.join(args.save, 'logits_stats', f'task{task_id+1}_max_logits.png'))
    plt.close()
    
    # 클래스별 logits 표준편차 시각화
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, id_std, width=0.4, label='ID Std', alpha=0.7, color='blue')
    plt.bar(x + 0.2, ood_std, width=0.4, label='OOD Std', alpha=0.7, color='red')
    plt.xlabel('Class Index')
    plt.ylabel('Std Logit Value')
    plt.title(f'Task {task_id+1}: ID vs OOD Logit Standard Deviations')
    plt.legend()
    plt.savefig(os.path.join(args.save, 'logits_stats', f'task{task_id+1}_std_logits.png'))
    plt.close()
    
    # Logits 분포 히스토그램 (전체 logits의 분포)
    plt.figure(figsize=(12, 6))
    plt.hist(id_logits.flatten().cpu().numpy(), bins=50, alpha=0.7, label='ID Logits', color='blue')
    plt.hist(ood_logits.flatten().cpu().numpy(), bins=50, alpha=0.7, label='OOD Logits', color='red')
    plt.xlabel('Logit Value')
    plt.ylabel('Frequency')
    plt.title(f'Task {task_id+1}: Distribution of Logit Values')
    plt.legend()
    plt.savefig(os.path.join(args.save, 'logits_stats', f'task{task_id+1}_logits_distribution.png'))
    plt.close()
    
    # 통계값 출력 (txt 파일 저장 대신)
    print(f"\nTask {task_id+1} Logits Statistics")
    print("="*50)
    print("ID Data Statistics:")
    print(f"Mean: {np.mean(id_mean):.4f}")
    print(f"Max: {np.max(id_max):.4f}")
    print(f"Min: {np.min(id_min):.4f}")
    print(f"Std: {np.mean(id_std):.4f}\n")
    
    print("OOD Data Statistics:")
    print(f"Mean: {np.mean(ood_mean):.4f}")
    print(f"Max: {np.max(ood_max):.4f}")
    print(f"Min: {np.min(ood_min):.4f}")
    print(f"Std: {np.mean(ood_std):.4f}")
    print("="*50)

def save_confusion_matrix_plot(confusion_matrix, labels, args, task_id=None):
    # Task 별 폴더 생성
    task_folder = f"task{task_id+1}" if task_id is not None else "latest"
    task_path = os.path.join(args.save, task_folder)
    os.makedirs(task_path, exist_ok=True)
    
    # 파일명 생성 (task 정보 포함)
    file_name = "confusion_matrix"
    if task_id is not None:
        file_name += f"_task{task_id+1}"
    
    save_path = os.path.join(task_path, f"{file_name}.png")
    
    modified_labels = labels.copy()
    modified_labels[-1] = 'ood'
    plt.figure(figsize=(16,12))
    sns.heatmap(confusion_matrix,
                annot=False,
                fmt='d',
                cmap='Blues',
                xticklabels=modified_labels,
                yticklabels=modified_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    title = "Confusion Matrix"
    if task_id is not None:
        title += f" - Task {task_id+1}"
    
    plt.title(title)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def save_anomaly_histogram(id_scores, ood_scores, args, suffix=None, task_id=None):
    # Task 별 폴더 생성
    task_folder = f"task{task_id+1}" if task_id is not None else "latest"
    task_path = os.path.join(args.save, task_folder)
    os.makedirs(task_path, exist_ok=True)
    
    # 파일명 생성 (task 정보 포함)
    file_name = "ood_histogram"
    if task_id is not None:
        file_name += f"_task{task_id+1}"
    if suffix:
        file_name += f"_{suffix}"
    
    save_path = os.path.join(task_path, f"{file_name}.png")
    plt.figure(figsize=(16,12))
    plt.hist(id_scores, bins=30, color='red', alpha=0.6, label='Known (ID) samples')
    plt.hist(ood_scores, bins=30, color='blue', alpha=0.6, label='Unknown (OOD) samples')
    
    title = f"Anomaly Score Histogram"
    if suffix:
        title += f" ({suffix.upper()})"
    if task_id is not None:
        title += f" - Task {task_id+1}"
    
    plt.title(title)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Histogram saved to {save_path}")
    return save_path

def update_ood_hyperparams(args):
    """OOD 탐지 방법들의 하이퍼파라미터를 업데이트합니다."""
    from OODdetectors import ood_adapter as _oa

    # ENERGY
    _oa._DEFAULT_PARAMS.setdefault("ENERGY", {})["temperature"] = args.energy_temperature

    # GEN (공통: GEN, PRO_GEN)
    _oa._DEFAULT_PARAMS.setdefault("GEN", {})["gamma"] = args.gen_gamma
    _oa._DEFAULT_PARAMS.setdefault("GEN", {})["M"] = args.gen_M

    # PRO_GEN
    _oa._DEFAULT_PARAMS.setdefault("PRO_GEN", {})["gamma"] = args.gen_gamma
    _oa._DEFAULT_PARAMS.setdefault("PRO_GEN", {})["M"] = args.gen_M
    _oa._DEFAULT_PARAMS["PRO_GEN"]["noise_level"] = args.pro_gen_noise_level
    _oa._DEFAULT_PARAMS["PRO_GEN"]["gd_steps"] = args.pro_gen_gd_steps

    # RPO_MSP
    _oa._DEFAULT_PARAMS.setdefault("RPO_MSP", {})["temperature"] = args.pro_msp_temperature
    _oa._DEFAULT_PARAMS["RPO_MSP"]["noise_level"] = args.pro_msp_noise_level
    _oa._DEFAULT_PARAMS["RPO_MSP"]["gd_steps"] = args.pro_msp_gd_steps

    # PRO_MSP_T
    _oa._DEFAULT_PARAMS.setdefault("PRO_MSP_T", {})["temperature"] = args.pro_msp_t_temperature
    _oa._DEFAULT_PARAMS["PRO_MSP_T"]["noise_level"] = args.pro_msp_t_noise_level
    _oa._DEFAULT_PARAMS["PRO_MSP_T"]["gd_steps"] = args.pro_msp_t_gd_steps

    # PRO_ENT
    _oa._DEFAULT_PARAMS.setdefault("PRO_ENT", {})["noise_level"] = args.pro_ent_noise_level
    _oa._DEFAULT_PARAMS["PRO_ENT"]["gd_steps"] = args.pro_ent_gd_steps 
 
    # PSEUDO
    _oa._DEFAULT_PARAMS.setdefault("PSEUDO", {})["eps"] = args.pseudo_eps
    _oa._DEFAULT_PARAMS["PSEUDO"]["max_train_batches"] = args.pseudo_max_batches
    _oa._DEFAULT_PARAMS["PSEUDO"]["lr"] = args.pseudo_lr
    _oa._DEFAULT_PARAMS["PSEUDO"]["epochs"] = args.pseudo_epochs
    _oa._DEFAULT_PARAMS["PSEUDO"]["hidden_dim"] = args.pseudo_hidden_dim
    _oa._DEFAULT_PARAMS["PSEUDO"]["layers"] = args.pseudo_layers 