import os
import numpy as np

def run_exp(shot, query = 15, dim=512, lrate=0.1):
    way = 5
    gpu = 0
    dataname = 'mini'  # mini tiered cub cifar_fs
    modelname = 'wrn28' # res12 wrn28
    the_command = 'python3.6 main.py' \
        + ' --gpu=' + str(gpu) \
        + ' --seed=' + str(10) \
        + ' --model_type=' + modelname \
        + ' --dataset=' + dataname \
        + ' --lr=' + str(lrate) \
        + ' --shot=' + str(shot) \
        + ' --way=' + str(way) \
        + ' --train_query=' + str(query) \
        + ' --val_query=' + str(query) \
        + ' --latentdim=' + str(dim) \
        + ' --metric=' + 'cos' \
        + ' --merge_dstr=' + 'mean' \
        + ' --WEmodel=' + 'glove' \
        + ' --dataset_dir=' + '/data/' + dataname + '/' + modelname \

    os.system(the_command + ' --phase=test')

run_exp(shot=1, query=15, lrate=0.001)
run_exp(shot=5, query=15, lrate=0.001)