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
    
    # wandb에 로깅 (설정되어 있는 경우)
    if args.wandb:
        import wandb
        
        # wandb 로그 이름 생성
        log_name = f"Anomaly_Histogram"
        if suffix:
            log_name += f"_{suffix.upper()}"
        if task_id is not None:
            log_name += f"_Task_{task_id+1}"
        
        wandb.log({log_name: wandb.Image(plt), "TASK": task_id if task_id is not None else 0})
    
    plt.close()
    print(f"Histogram saved to {save_path}") 