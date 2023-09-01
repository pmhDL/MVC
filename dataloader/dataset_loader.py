""" Dataloader for all datasets. """
import os.path as osp
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch

class DatasetLoader(Dataset):
    def __init__(self, setname, args):
        self.args = args
        self.setname = setname
        dataset_dir = args.dataset_dir
        backbone_class = args.model_type
        THE_PATH = osp.join(dataset_dir, 'few-shot-' + setname + '.npz')
        data0 = np.load(THE_PATH)
        data = data0['features']
        label = data0['targets']

        self.data = data
        self.label = label
        self.num_class = len(np.unique(label))

        # Transformation
        image_size = 84
        transforms_list = [
            transforms.Resize(92),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]

        # Transformation
        if backbone_class == 'conv':
            self.transform = transforms.Compose(
                transforms_list + [
                    transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                         np.array([0.229, 0.224, 0.225]))
                ])
        elif backbone_class == 'res12':
            self.transform = transforms.Compose(
                transforms_list + [
                    transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                         np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
                ])
        elif backbone_class == 'res18':
            self.transform = transforms.Compose(
                transforms_list + [
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
        elif backbone_class == 'wrn28':
            self.transform = transforms.Compose(
                transforms_list + [
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])

    def __len__(self):
        return len(self.data)

    def rotate_img(self, img, rot):
        if rot == 0:  # 0 degrees rotation
            return img
        elif rot == 90:  # 90 degrees rotation
            return np.flipud(np.transpose(img, (1, 0, 2)))
        elif rot == 180:  # 90 degrees rotation
            return np.fliplr(np.flipud(img))
        elif rot == 270:  # 270 degrees rotation / or -90
            return np.transpose(np.flipud(img), (1, 0, 2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

    def __getitem__(self, i):
        label = self.label[i]
        image = self.data[i]
        if self.setname == 'train':
            cat = [self.transform(Image.fromarray(image.astype('uint8')).convert('RGB'))]
            image_90 = self.transform(Image.fromarray(self.rotate_img(image, 90).astype('uint8')).convert('RGB'))
            image_180 = self.transform(Image.fromarray(self.rotate_img(image, 180).astype('uint8')).convert('RGB'))
            image_270 = self.transform(Image.fromarray(self.rotate_img(image, 270).astype('uint8')).convert('RGB'))
            cat.append(image_90)
            cat.append(image_180)
            cat.append(image_270)
            images = torch.stack(cat, 0)
            return images, torch.ones(4, dtype=torch.long) * int(label), torch.LongTensor([0, 1, 2, 3])
        else:
            images = self.transform(Image.fromarray(image.astype('uint8')).convert('RGB'))
            return images, label