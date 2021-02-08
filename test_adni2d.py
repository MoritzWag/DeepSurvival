import os 

command1 = "python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name validrun --run_name seed1 --manual_seed 1337"
command2 = "python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name validrun --run_name seed2 --manual_seed 1328"
command3 = "python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name validrun --run_name seed3 --manual_seed 9258"
command4 = "python run.py --config configs/ADNI/deepcoxph.yaml --experiment_name validrun --run_name seed4 --manual_seed 1234"

os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)
