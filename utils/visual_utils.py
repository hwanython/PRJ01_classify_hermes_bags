import matplotlib.pyplot as plt
import copy
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import glob
from tqdm import tqdm
import itertools
from sklearn.metrics import confusion_matrix
from PRJ01_classify_hermes_bags.configs.CLASSES import *
#######################################################
#                  Visualize Dataset
#         Images are plotted after augmentation
#######################################################

def visualize_augmentations(dataset, idx=0, samples=10, cols=5, random_img=False):
    dataset = copy.deepcopy(dataset)
    # we remove the normalize and tensor conversion from our augmentation pipeline
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols

    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for i in range(samples):
        if random_img:
            idx = np.random.randint(1, len(train_image_paths))
        image, lab = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(idx_to_class[lab])
    plt.tight_layout(pad=1)
    plt.show()



# confusion matrix 사용을 위한 라이브러리


# confusion matrix 그리는 함수

def plot_confusion_matrix(true_labels, pred_labels, classes, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'),
                              normalize=False):
    con_mat = confusion_matrix(true_labels, pred_labels)
    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks = np.arange(len(classes))
    nlabels = []
    for k in range(len(con_mat)):
        n = sum(con_mat[k])
        nlabel = '{0}(n={1})'.format(classes[k],n)
        nlabels.append(nlabel)
    plt.xticks(marks, classes)
    plt.yticks(marks, nlabels)

    thresh = con_mat.max() / 2.
    if normalize:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, '{0}%'.format(con_mat[i, j] * 100 / n), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# 예측 결과를 figure로 확인하는 함수

def plot_predict(dataframe, pred_labels):
    figsize, num =  (4,8), 32
    fig = plt.figure(figsize=(figsize[0], figsize[1]))

    for i in range(num):
        plt.subplot(figsize[0], figsize[1], i + 1)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        # image load
        fn = plt.imread(dataframe["filepath"][i])
        plt.imshow(fn)
        # match labels
        target = dataframe["labels"][i]
        target_bag = [k for k, v in CLASSES.items() if v == target][0]
        pred_bag = [k for k, v in CLASSES.items() if v == pred_labels[i]][0]
        if pred_bag == target_bag:
            color = 'green'
        else:
            color = 'red'
        plt.xlabel("{}".format(pred_bag), color=color)
    plt.show()
