import os.path as osp
import os
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.nets import Model
from utils.misc import Averager, Timer, count_acc, ensure_path, compute_confidence_interval
from tensorboardX import SummaryWriter
from dataloader.dataset_loaderf import DatasetLoader as Dataset
import numpy as np


class MVCTrainer(object):
    def __init__(self, args):
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'mvc')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type])
        args.save_path = pre_base_dir + '/' + save_path1
        ensure_path(args.save_path)

        self.args = args

        # Build pretrain model
        self.model = Model(self.args, mode='mvc')

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def eval(self):
        """ test phase."""
        # Load test set
        test_set = Dataset('test', self.args)
        sampler = CategoriesSampler(test_set.label, 600, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
        # Set test accuracy recorder
        test_acc_record = np.zeros((600,))

        # Set model to eval mode
        self.model.eval()
        self.model.mode = 'mvc'
        # Set accuracy averager
        ave_acc0 = Averager()
        ave_acc1 = Averager()

        # Generate labels
        label = torch.arange(self.args.way).repeat(self.args.val_query)
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)
            label = label.type(torch.LongTensor)

        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                if torch.cuda.is_available():
                    data, _, sem = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                    sem = batch[2]
            p = self.args.shot * self.args.way
            datas, dataq = data[:p], data[p:]
            sem = torch.tensor(sem).type(datas.type())
            sems = sem[:p]

            logit0, logit1 = self.model((datas, label_shot, sems, dataq))


            acc0 = count_acc(logit0, label)
            acc1 = count_acc(logit1, label)
            ave_acc0.add(acc0)
            ave_acc1.add(acc1)

            test_acc_record[i - 1] = acc1

            if i % 100 == 0:
                print('batch {}: {:.2f} {:.2f} '.format(i, ave_acc0.item() * 100, ave_acc1.item() * 100))

        # Calculate the confidence interval, update the logs
        m, pm = compute_confidence_interval(test_acc_record)
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))