import os

num_epochs = [100, 125, 150]

for epoch in num_epochs:
    command1 = f"python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name seed_imgtab3d{epoch} --run_name 0  --max_epochs {epoch} --split 0"
    command2 = f"python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name seed_imgtab3d{epoch} --run_name 1  --max_epochs {epoch} --split 1"
    command3 = f"python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name seed_imgtab3d{epoch} --run_name 2  --max_epochs {epoch} --split 2"
    command4 = f"python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name seed_imgtab3d{epoch} --run_name 3  --max_epochs {epoch} --split 3"
    command5 = f"python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name seed_imgtab3d{epoch} --run_name 4  --max_epochs {epoch} --split 4"


    os.system(command1)
    os.system(command2)
    os.system(command3)
    os.system(command4)
    os.system(command5)
