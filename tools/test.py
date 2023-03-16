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
from PRJ01_classify_hermes_bags.utils.visual_utils import *


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(
        f"=====================================================\nSetting ...\n  Device: {device}")

    # read data table
    test_df = pd.read_csv('../datasets/csv/test.csv')

    test_transforms = A.Compose([
        A.Resize(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
                    p=1.0),
        ToTensorV2()
    ])

    test_dataset = Cumstomloader(test_df, transform=test_transforms)

    print(
        f"Test Dataset : {len(test_dataset)}\n=====================================================")

    test_dataloader = DataLoader(
        test_dataset,
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
    model_path = r'E:\kangkas\PRJ01_classify_hermes_bags\experiments\model\best_model_6_epoch_val_loss_0.332_val_score_1.000.pth'
    net.load_state_dict(torch.load(model_path), strict=False)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.fc.parameters(), lr=CFG["LEARNING_RATE"])

    test_loss, test_score, outs = test_loop(net, criterion, test_dataloader, device)

    print(
        f'Val Loss : [{test_loss:.5f}] Val Score : [{test_score:.5f}]')
    true_labels = test_df["labels"].to_list()
    pred_labels = outs
    classes = list(CLASSES.keys())
    plot_confusion_matrix(true_labels, pred_labels, classes=classes)
    plot_predict(test_df, pred_labels)
