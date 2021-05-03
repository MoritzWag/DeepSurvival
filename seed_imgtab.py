import os

num_epochs = [150]

for epoch in num_epochs:
    command1 = f"python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name imgtab_adni_orth_2{epoch} --run_name orth_0 --max_epochs {epoch} --split 0 --orthogonalize True"
    command2 = f"python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name imgtab_adni_orth_2{epoch} --run_name orth_1  --max_epochs {epoch} --split 1 --orthogonalize True"
    command3 = f"python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name imgtab_adni_orth_2{epoch} --run_name orth_2  --max_epochs {epoch} --split 2 --orthogonalize True"
    command4 = f"python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name imgtab_adni_orth_2{epoch} --run_name orth_3  --max_epochs {epoch} --split 3 --orthogonalize True"
    command5 = f"python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name imgtab_adni_orth_2{epoch} --run_name orth_4  --max_epochs {epoch} --split 4 --orthogonalize True"

    command6 = f"python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name imgtab_adni_2{epoch} --run_name 0 --max_epochs {epoch} --split 0"
    command7 = f"python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name imgtab_adni_2{epoch} --run_name 1  --max_epochs {epoch} --split 1"
    command8 = f"python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name imgtab_adni_2{epoch} --run_name 2  --max_epochs {epoch} --split 2"
    command9 = f"python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name imgtab_adni_2{epoch} --run_name 3  --max_epochs {epoch} --split 3"
    command10 = f"python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name imgtab_adni_2{epoch} --run_name 4  --max_epochs {epoch} --split 4"

    os.system(command1)
    os.system(command2)
    os.system(command3)
    os.system(command4)
    os.system(command5)
    
    os.system(command6)
    os.system(command7)
    os.system(command8)
    os.system(command9)
    os.system(command10)

