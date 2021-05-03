import os

command1 = "python sksurv_train.py  --experiment_name sksurv --run_name sksurv0  --split 0"
command2 = "python sksurv_train.py  --experiment_name sksurv --run_name sksurv1  --split 1" 
command3 = "python sksurv_train.py  --experiment_name sksurv --run_name sksurv2  --split 2"
command4 = "python sksurv_train.py  --experiment_name sksurv --run_name sksurv3  --split 3"
command5 = "python sksurv_train.py  --experiment_name sksurv --run_name sksurv4  --split 4"

os.system(command1)
os.system(command2)
os.system(command3)
os.system(command4)
os.system(command5)
