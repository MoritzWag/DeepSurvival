import os

command1 = "python run.py --config configs/ADNI/linear.yaml --experiment_name lr001 --run_name lr001 --learning_rate 0.001"
command2 = "python run.py --config configs/ADNI/linear.yaml --experiment_name lr0025 --run_name lr0025 --learning_rate 0.0025"
command3 = "python run.py --config configs/ADNI/linear.yaml --experiment_name lr005 --run_name lr005 --learning_rate 0.005"
command4 = "python run.py --config configs/ADNI/linear.yaml --experiment_name lr0001 --run_name lr0001 --learning_rate 0.0001"
command5 = "python run.py --config configs/ADNI/linear.yaml --experiment_name lr00025 --run_name lr00025 --learning_rate 0.00025"
command6 = "python run.py --config configs/ADNI/linear.yaml --experiment_name lr0005 --run_name lr0005 --learning_rate 0.0005"


os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)
os.system(command5)
os.system(command6)
