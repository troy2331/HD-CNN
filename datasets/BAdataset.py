import cv2
import json
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

import torch
from torch.utils.data import Dataset

from .dataset import load_class_X_Y, ulna_mask, ulna_expand_mask, radius_mask, pp3_mask, mp3_mask, dp3_mask, carpal_mask, mc1_mask

def padding(img) :
    height, width = img.shape[0:2]
    margin = [np.abs(height - width) // 2, np.abs(height - width) // 2]

    if np.abs(height - width) % 2 != 0:
        margin[0] += 1

    if height < width:
        margin_list = [margin, [0, 0]]
    else:
        margin_list = [[0, 0], margin]

    if len(img.shape) == 3:
        margin_list.append([0, 0])

    img = np.pad(img, margin_list, mode='constant')
    return img

class BADataset(Dataset):
    def __init__(self, args, transforms=None, mode='train'):
        super(BADataset, self).__init__()
        self.args = args
        self.mode = mode
        self.transforms = transforms
        if self.mode == 'train' or self.mode == 'valid':
            self.img_lst, self.label_lst = load_class_X_Y(
                args.data_path, self.mode)

    def __len__(self):
        return len(self.label_lst)

    def __getitem__(self, idx):
        img_path = self.img_lst[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.args.padding == True:
            img = padding(img)

        transformed_img = self.transforms(image=img)['image']
        raw_img = A.Resize(self.args.img_size, self.args.img_size)(
            image=img)['image']

        if self.mode == 'train' or self.mode == 'valid':
            label = torch.tensor(self.label_lst[idx], dtype=torch.long)
            return {'img': transformed_img,
                    'label': label,
                    'raw_img': raw_img,
                    'img_path': img_path}
        elif self.mode == 'test':
            return {'img': img}


def train_transforms(args):
    return A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Rotate(5, p=0.3),
        A.CLAHE(p=0.3),
        A.RandomBrightness(limit=0.5, p=0.3),
        A.RandomContrast(limit=0.5, p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 250), p=1.),
            A.InvertImg(p=1.),
        ], p=0.3),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def test_transforms(args):
    return A.Compose([
        A.Resize(args.img_size, args.img_size),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class BADataset_custom(Dataset):
    def __init__(self, args, transforms=None, mode='train', target_label=None):
        super(BADataset_custom, self).__init__()
        self.args = args
        self.mode = mode
        self.transforms = transforms
        if self.mode == 'train':
            self.df = pd.read_csv(args.train_csv)
            self.img_lst = self.df['img_path']
            self.main_lst = self.df['main']
            self.sub_lst = self.df['sub']
        elif self.mode == 'valid':
            self.df = pd.read_csv(args.valid_csv)
            self.img_lst = self.df['img_path']
            self.main_lst = self.df['main']
            self.sub_lst = self.df['sub']

        self.label = target_label

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, idx):
        img_path = '/home/crescom/BA_code/'+self.img_lst[idx]
        # img_path = self.img_lst[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.args.padding == True:
            img = padding(img)

        transformed_img = self.transforms(image=img)['image']
        raw_img = A.Resize(self.args.img_size, self.args.img_size)(
            image=img)['image']

        if self.mode == 'train' or self.mode == 'valid':
            if self.label == 'main':
                return torch.tensor(self.main_lst[idx], dtype=torch.long)

            elif self.label == 'sub':
                return torch.tensor(self.sub_lst[idx], dtype=torch.long)

            elif self.label == 'total':
                main_label = torch.tensor(self.main_lst[idx], dtype=torch.long)
                sub_label = torch.tensor(self.sub_lst[idx], dtype=torch.long)
                if self.img_lst[idx].split('/')[1] == 'ulna':
                    mask = torch.tensor(ulna_mask[self.main_lst[idx]])
                elif self.img_lst[idx].split('/')[1] == 'radius':
                    mask = torch.tensor(radius_mask[self.main_lst[idx]])
                elif self.img_lst[idx].split('/')[1] == 'pp3':
                    mask = torch.tensor(pp3_mask[self.main_lst[idx]])
                elif self.img_lst[idx].split('/')[1] == 'mp3':
                    mask = torch.tensor(mp3_mask[self.main_lst[idx]])
                elif self.img_lst[idx].split('/')[1] == 'dp3':
                    mask = torch.tensor(dp3_mask[self.main_lst[idx]])
                elif self.img_lst[idx].split('/')[1] == 'mc1':
                    mask = torch.tensor(mc1_mask[self.main_lst[idx]])
                elif self.img_lst[idx].split('/')[1] == 'carpal':
                    mask = torch.tensor(carpal_mask[self.main_lst[idx]])

                return {'img': transformed_img,
                        'main_label': main_label,
                        'sub_label': sub_label,
                        'mask': mask,
                        'raw_img': raw_img}

        elif self.mode == 'test':
            return {'img': transformed_img}


class BADataset_fc(Dataset):
    def __init__(self, json_path, mode='train'):
        super(BADataset_fc).__init__()
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
        self.annos = data['annotations']
        self.mode = mode

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):
        meta = self.annos[idx]
        input_prob = torch.cat([torch.tensor(meta['dp3']), torch.tensor(meta['mp3']), torch.tensor(meta['pp3']), torch.tensor(
            meta['radius']), torch.tensor(meta['ulna']), torch.tensor(meta['carpal']), torch.tensor(meta['mc1']), torch.tensor(meta['hand'])])

        if self.mode == 'train' or self.mode == 'valid':
            return {'prob': input_prob,
                    'label': torch.tensor(meta['age'], dtype=torch.half),
                    'cls': torch.tensor(meta['cls'], dtype=torch.long),
                    'img_path': meta['img_path']}

        else:
            return {'prob': input_prob}
