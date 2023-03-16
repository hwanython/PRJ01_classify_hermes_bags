import os, glob
import cv2
import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode
from torch.utils.data import Dataset
import albumentations as A
import itertools
import pandas as pd

class StratifiedDataSplit:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def train_val_test(self):
        grouped = self.dataframe.groupby('labels')
        datasize = len(self.dataframe)
        groups = [grouped.get_group(x) for x in grouped.groups]

        i = 0
        train, train_size = [groups[i]], len(groups[i])
        while train_size < datasize * 0.8:
            i += 1
            train_size += len(groups[i])
            train.append(groups[i])
        test, test_size = [groups[i]], len(groups[i])
        while test_size <datasize* 0.1:
            i += 1
            test_size += len(groups[i])
            test.append(groups[i])

        valid, valid_size = [groups[i]], len(groups[i])
        while valid_size < datasize * 0.1:
            i += 1
            valid_size += len(groups[i])
            valid.append(groups[i])

        train.extend(groups[i+1:])
        train, valid, test = pd.concat(train), pd.concat(valid), pd.concat(test)
        return train, valid, test


class Cumstomloader(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.normalization = 255
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_filepath = self.dataframe['filepath'][idx]
        label = self.dataframe['labels'][idx]

        # load images
        image = cv2.imread(image_filepath)
        # normalization & tensor
        if self.transform is not None:
            image = self.transform(image=image)['image']

        # image = torch.from_numpy(image)
        label = torch.tensor(label).long()
        image = image.float() / self.normalization

        # TODO: Augmentations
        return image, label
