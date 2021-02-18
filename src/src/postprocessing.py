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
        # plt.plot(history.iloc[20:, metric])
        plt.plot(history.iloc[:, metric])
        plt.xlabel('training steps')
        plt.ylabel(history.columns[metric])

    plt.tight_layout()

    if storage_path is not None:
        if not os.path.exists(storage_path):
            os.makedirs(f"./{storage_path}")
        plt.savefig(f"{storage_path}.png")

    
def plot_boxplots(df, path, metrics=['cindex',
                                     'quantile_0.5',
                                     'quantile_0.25',
                                     'quantile_0.75']):
    """
    """
    seeds  = list(df['manual_seed'].unique())
    for metric in metrics:
        plt.close()
        collections = []
        collections = [df[df['manual_seed'] == seed][metric] for seed in seeds]

        fig = plt.figure(1, figsize=(9, 6))
        ax = fig.add_subplot(111)
        bp = ax.boxplot(collections)
        ax.set_xticklabels(seeds)
        ax.set_xlabel("seeds")
        ax.set_title(f"Seed analysis: {metric}")

        fig.savefig(f"{metric}_boxplot.png")


def get_mlflow_results(mlflow_id, path=None):

    if path is None:
        path = f"mlruns/{mlflow_id}"
    else:
        path = f"{path}/mlruns/{mlflow_id}"
    
    runs = [run for run in os.listdir(path) if len(run) == 32 and not run.startswith('performance')]
    frame = pd.DataFrame(columns=['run_name',
                                'experiment_name',
                                'manual_seed',
                                'avg_test_loss',
                                'avg_val_loss',
                                'cindex',
                                'cindex_train',
                                'cindex_test',
                                'cindex_tabular',
                                'quantile_05',
                                'quantile_025',
                                'quantile_075'])

    i = 0 

    for run in runs:
    
        try:
            run_name = open(f'{path}/{run}/params/run_name').read()
        except:
            run_name = "Nan"
        try: 
            experiment_name = open(f'{path}/{run}/params/experiment_name').read()
        except:
            experiment_name = "Nan"
        try:
            manual_seed = open(f'{path}/{run}/params/manual_seed').read()
        except:
            manual_seed = "Nan"

        try: 
            avg_test_loss = open(f'{path}/{run}/metrics/avg_test_loss').read().split()[1]
        except:
            avg_test_loss = 0.0 
        try:
            avg_val_loss = open(f'{path}/{run}/metrics/avg_val_loss').read().split()[1]
        except:
            avg_val_loss = 0.0
        try: 
            cindex = open(f'{path}/{run}/metrics/cindex').read().split()[1]
        except:
            cindex = 0.0 
        try: 
            cindex_train = open(f'{path}/{run}/metrics/cindex_train').read().split()[1]
        except:
            cindex_train = 0.0
        try: 
            cindex_test = open(f'{path}/{run}/metrics/cindex_test').read().split()[1]
        except:
            cindex_test = 0.0
        try: 
            cindex_tabular = open(f'{path}/{run}/metrics/cindex_tabular').read().split()[1]
        except:
            cindex_tabular = 0.0
        try:
            quantile_05 = open(f'{path}/{run}/metrics/quantile_0.5').read().split()[1]
        except:
            quantile_05 = 0.0
        try: 
            quantile_025 = open(f'{path}/{run}/metrics/quantile_0.25').read().split()[1]
        except:
            quantile_025 = 0.0 
        try:
            quantile_075 = open(f'{path}/{run}/metrics/quantile_0.75').read().split()[1]
        except:
            quantile_075 = 0.0
        


        frame.loc[i] = [run_name,
                        experiment_name,
                        manual_seed,
                        avg_test_loss,
                        avg_val_loss,
                        cindex,
                        cindex_train,
                        cindex_test,
                        cindex_tabular,
                        quantile_05,
                        quantile_025,
                        quantile_075]

        i += 1

    frame.to_csv(f"{experiment_name}_runs.csv")
