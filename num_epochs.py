import os 

command1 = "python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name num_epochs --run_name epoch50 --max_epochs 50"
command2 = "python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name num_epochs --run_name epoch100 --max_epochs 100"
command3 = "python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name num_epochs --run_name epoch150 --max_epochs 150"
command4 = "python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name num_epochs --run_name epoch200 --max_epochs 200"

os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)