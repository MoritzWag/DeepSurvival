import os

num_epochs = [150]

for epoch in num_epochs:
    command1 = f"python run.py --config configs/ADNI/baseline.yaml --experiment_name baseline_adni_{epoch} --run_name 0  --max_epochs {epoch} --split 0"
    command2 = f"python run.py --config configs/ADNI/baseline.yaml --experiment_name baseline_adni_{epoch} --run_name 1  --max_epochs {epoch} --split 1"
    command3 = f"python run.py --config configs/ADNI/baseline.yaml --experiment_name baseline_adni_{epoch} --run_name 2  --max_epochs {epoch} --split 2"
    command4 = f"python run.py --config configs/ADNI/baseline.yaml --experiment_name baseline_adni_{epoch} --run_name 3  --max_epochs {epoch} --split 3"
    command5 = f"python run.py --config configs/ADNI/baseline.yaml --experiment_name baseline_adni_{epoch} --run_name 4  --max_epochs {epoch} --split 4"


    os.system(command1)
    os.system(command2)
    os.system(command3)
    os.system(command4)
    os.system(command5)
