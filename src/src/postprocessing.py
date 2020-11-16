import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import pdb 
import os

def plot_train_progress(history, storage_path):
    """plot loss progress based on training/validation history

    Args:
        history: {pd.DataFrame}
        storage_path: {str}
    
    Returns:
        matplotlib plot
    """
    num_metrics = len(history.columns)
    plt.close()
    plt.figure(figsize=(20, 12))
    for metric in range(num_metrics):
        plt.subplot(num_metrics, 1, metric + 1)
        plt.plot(history.iloc[20:, metric])
        plt.xlabel('training steps')
        plt.ylabel(history.columns[metric])

    plt.tight_layout()

    if storage_path is not None:
        if not os.path.exists(storage_path):
            os.makedirs(f"./{storage_path}")
        plt.savefig(f"{storage_path}.png")
