import os

def run_exp(lr=0.1, gamma=0.2):
    max_epoch = 100
    shot = 1
    query = 15
    way = 5
    gpu = '1,2'
    dataname = 'mini' # mini tiered cifar_fs cub
    modelname = 'res12'  # res12 wrn28
    the_command = 'python3.6 main.py' \
        + ' --gpu=' + str(gpu) \
        + ' --pre_max_epoch=' + str(max_epoch) \
        + ' --pre_batch_size=' + str(128) \
        + ' --pre_gamma=' + str(gamma) \
        + ' --pre_lr=' + str(lr) \
        + ' --shot=' + str(shot) \
        + ' --way=' + str(way) \
        + ' --train_query=' + str(query) \
        + ' --dataset=' + dataname \
        + ' --model_type=' + modelname \
        + ' --dataset_dir=' + '/data/'+dataname \
        + ' --opt=' + 'SGD' \
        + ' --metric=' + 'cos' \


    os.system(the_command + ' --phase=pre')
    os.system(the_command + ' --phase=preval')

run_exp(lr=0.05, gamma=0.1)