import os
import matplotlib.pyplot as plt
import numpy as np

def save_accuracy_heatmap(mat, task_id, args):

    os.makedirs(args.save, exist_ok=True)
    
    save_path = os.path.join(args.save, f"accuracy_matrix_task{task_id+1}.png")
    plt.figure(figsize=(16,12))
    im = plt.imshow(mat, interpolation='nearest', cmap="Reds", vmin=0, vmax=100)
    plt.xlabel("Trained up to Task")
    plt.ylabel("Evaluated Task")
    plt.colorbar(im, label="Accuracy")
    plt.xticks(np.arange(mat.shape[1]))
    plt.yticks(np.arange(mat.shape[0]))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Accuracy heatmap saved to {save_path}")