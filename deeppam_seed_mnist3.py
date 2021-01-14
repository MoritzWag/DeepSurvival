import os 
import pandas as pd
from src.postprocessing import get_mlflow_results


command1 = "python run.py --config configs/MNIST3D/deeppam.yaml --manual_seed 1337 --experiment_name seed_check_m3_pam --run_name seed1"
command2 = "python run.py --config configs/MNIST3D/deeppam.yaml --manual_seed 1328 --experiment_name seed_check_m3_pam --run_name seed2"
command3 = "python run.py --config configs/MNIST3D/deeppam.yaml --manual_seed 9258 --experiment_name seed_check_m3_pam --run_name seed3"
command4 = "python run.py --config configs/MNIST3D/deeppam.yaml --manual_seed 5687 --experiment_name seed_check_m3_pam --run_name seed4"
command5 = "python run.py --config configs/MNIST3D/deeppam.yaml --manual_seed 3452 --experiment_name seed_check_m3_pam --run_name seed5"


os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)
os.system(command5)


mlruns = os.listdir('mlruns')
latest_mlflow_id = int(max(mlruns))

get_mlflow_results(mlflow_id=latest_mlflow_id)

#df = pd.read_csv("seed_checks_coxph.csv")