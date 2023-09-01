""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch

class DatasetLoader(Dataset):
    """The class to load the dataset"""
    def __init__(self, setname, args):
        THE_PATH = osp.join(args.dataset_dir, 'feat-' + setname + '.npz')
        data0 = np.load(THE_PATH)
        featn = data0['features']
        label = data0['targets']

        if args.WEmodel =='clip':
            path_sem = osp.join(args.dataset_dir.split(args.dataset)[0] + args.dataset,
                                'few-shot-wordemb-' + setname + '-clip.npz')
        elif args.WEmodel =='glove':
            path_sem = osp.join(args.dataset_dir.split(args.dataset)[0] + args.dataset,
                                'few-shot-wordemb-' + setname + '-glove.npz')
        elif args.WEmodel =='w2v':
            path_sem = osp.join(args.dataset_dir.split(args.dataset)[0] + args.dataset,
                                'few-shot-wordemb-' + setname + '-w2v.npz')
        elif args.WEmodel =='fasta':
            path_sem = osp.join(args.dataset_dir.split(args.dataset)[0] + args.dataset,
                                'few-shot-wordemb-' + setname + '-fast.npz')
        sem = np.load(path_sem)['features']

        self.featn = featn
        self.sem = sem
        self.label = label
        self.num_class = len(np.unique(label))

    def __len__(self):
        return len(self.featn)

    def __getitem__(self, i):
        fn = self.featn[i]
        label = self.label[i]
        sem = self.sem[label]
        return fn, label, sem