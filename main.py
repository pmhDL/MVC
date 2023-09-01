""" Main function for this repo. """
import argparse
import torch
from utils.misc import pprint
from utils.gpu_tools import set_gpu
from trainer.mvc import MVCTrainer
from trainer.pre import PRETrainer
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--model_type', type=str, default='res12', choices=['res12', 'wrn28'])
    parser.add_argument('--dataset', type=str, default='mini', choices=['mini', 'tiered', 'cifar_fs', 'cub'])  # Dataset
    parser.add_argument('--phase', type=str, default='train', choices=['pre', 'preval', 'test'])
    parser.add_argument('--seed', type=int, default=0)  # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpu', default='1')  # GPU id
    parser.add_argument('--dataset_dir', type=str, default='./data/mini/')  # Dataset folder

    # Parameters for pretain phase
    parser.add_argument('--pre_max_epoch', type=int, default=100)  # Epoch number for pre-train phase
    parser.add_argument('--pre_batch_size', type=int, default=128)  # Batch size for pre-train phase
    parser.add_argument('--pre_lr', type=float, default=0.1)  # Learning rate for pre-train phase
    parser.add_argument('--pre_gamma', type=float, default=0.2)  # Gamma for the pre-train learning rate decay
    parser.add_argument('--pre_step_size', type=int, default=30)  # The number of epochs to reduce the pre-train learning rate
    parser.add_argument('--pre_momentum', type=float, default=0.9)  # Momentum for the optimizer during pre-train
    parser.add_argument('--pre_weight_decay', type=float, default=0.0005)  # Weight decay for the optimizer during pre-train
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_batch', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--ret', type=int, default=3)
    parser.add_argument('--merge_dstr', type=str, default='merge', choices=['mean', 'merge'])
    parser.add_argument('--temprature', type=float, default=1.0)

    # FSL parameters
    parser.add_argument('--scheduler_milestones', default=[60, 80], nargs='+', type=int,
                        help='milestones if using multistep learning rate scheduler')
    parser.add_argument('--nesterov', action='store_true', help='use nesterov for SGD, disable it in default')
    parser.add_argument('--opt', type=str, default='Adam', choices=['SGD', 'Adam'])
    parser.add_argument('--metric', type=str, default='ED', choices=['ED', 'cos'])
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--train_query', type=int, default=15)
    parser.add_argument('--val_query', type=int, default=15)
    parser.add_argument('--latentdim', type=int, default=1024) #512
    parser.add_argument('--cls', type=str, default='kl', choices=['ED', 'cos', 'kl'])
    parser.add_argument('--expandnum', type=int, default=100)
    parser.add_argument('--WEmodel', type=str, default='glove', choices=['glove', 'w2v', 'fasta', 'clip'])


    # Set and print the parameters
    args = parser.parse_args()
    pprint(vars(args))
    # Set the GPU id
    set_gpu(args.gpu)

    # Set manual seed for PyTorch
    if args.seed == 0:
        print('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Start trainer for pre-train, fsl-train or fsl-eval
    if args.phase == 'pre':
        trainer = PRETrainer(args)
        trainer.train()
    elif args.phase == 'preval':
        trainer = PRETrainer(args)
        trainer.eval()
    elif args.phase == 'test':
        trainer = MVCTrainer(args)
        trainer.eval()
    else:
        raise ValueError('Please set correct phase.')