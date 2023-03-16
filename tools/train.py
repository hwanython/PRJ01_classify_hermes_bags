import os
import time
import datetime
import argparse
import numpy as np
import torch
import cv2
from torch import nn
from torch import optim
import wandb
import torchvision
from torchvision import datasets, transforms, models
import pandas as pd
# from PRJ01_classify_hermes_bags.utils.visual_utils import imshow
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from PRJ01_classify_hermes_bags.utils.data_utils import StratifiedDataSplit, Cumstomloader
from PRJ01_classify_hermes_bags.configs.train_settings import *
from PRJ01_classify_hermes_bags.configs.CLASSES import *
from PRJ01_classify_hermes_bags.utils.op_utils import *
import wandb

wandb.init(project='kangkas-prj01-base')

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(
        f"=====================================================\nSetting ...\n  Device: {device}")

    # read data table
    train_df = pd.read_csv('../datasets/csv/train.csv')
    valid_df = pd.read_csv('../datasets/csv/valid.csv')

    train_transforms = A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
                    p=1.0),
        ToTensorV2()
    ])

    valid_transforms = A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
                    p=1.0),
        ToTensorV2()
    ])

    train_dataset = Cumstomloader(train_df, transform=train_transforms)
    valid_dataset = Cumstomloader(valid_df, transform=valid_transforms)

    print(
        f"Dataset Total: {len(train_dataset) + len(valid_dataset)}\n  train: {len(train_dataset)}\n "
        f" valid: {len(valid_dataset)}\n=====================================================")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CFG["BATCH_SIZE"],
        num_workers=CFG["num_workers"],
        pin_memory=True,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=CFG["BATCH_SIZE"],
        num_workers=CFG["num_workers"],
        pin_memory=True,
        shuffle=False
    )

    # defined model
    net = models.resnet50(pretrained=True)
    net.conv1.in_channels = 3
    # print(net)
    net.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, N_CLS))
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.fc.parameters(), lr=CFG["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10,
                                                           threshold_mode='abs', min_lr=1e-8, verbose=True)

    train(model=net, criterion=criterion, optimizer=optimizer, train_loader=train_dataloader,
          val_loader=valid_dataloader, scheduler=scheduler, device=device)
