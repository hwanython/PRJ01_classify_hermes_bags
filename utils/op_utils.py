import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
from PRJ01_classify_hermes_bags.configs.train_settings import *

from sklearn import metrics


def train(model, criterion, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    best_score = 0
    best_model = None

    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        for img, label in tqdm(iter(train_loader)):
            img = img.to(device)
            label = label.to(device)

            out = model(img)

            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            out = out.argmax(1)
            # acc = (out == label).float().mean().cpu().detach().numpy()
            train_loss.append(loss.item())

        val_loss, val_score = validation(model, criterion, val_loader, device)
        print(
            f'Epoch [{epoch}], Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] Val Score : [{val_score:.5f}]')

        if scheduler is not None:
            scheduler.step(val_score)

        if best_score < val_score:
            best_score = val_score
            torch.save(model.state_dict(),
                       './experiments/model/best_model_{0}_epoch_val_loss_{1:.3f}_val_score_{2:.3f}.pth'.format(epoch,
                                                                                                                val_loss,
                                                                                                                val_score))
    return best_model


def validation(model, criterion, val_loader, device):
    model.to(device)
    model.eval()
    pred_labels = []
    true_labels = []
    val_loss = []
    with torch.no_grad():
        for img, label in tqdm(iter(val_loader)):
            true_labels += label.tolist()
            img = img.to(device)
            label = label.to(device)

            out = model(img)

            loss = criterion(out, label)

            val_loss.append(loss.item())
            out = out.argmax(1)
            # acc = (out == label).float().mean().cpu().detach().numpy()
            pred_labels += out.tolist()

    val_score = metrics.f1_score(y_true=true_labels, y_pred=pred_labels, average='macro')
    return np.mean(val_loss), val_score


def test_loop(model, criterion, val_loader, device):
    model.to(device)
    model.eval()
    pred_labels = []
    true_labels = []
    test_loss = []
    with torch.no_grad():
        for img, label in tqdm(iter(val_loader)):
            true_labels += label.tolist()
            img = img.to(device)
            label = label.to(device)

            out = model(img)

            loss = criterion(out, label)

            test_loss.append(loss.item())
            out = out.argmax(1)
            # acc = (out == label).float().mean().cpu().detach().numpy()
            pred_labels += out.tolist()

    test_score = metrics.f1_score(y_true=true_labels, y_pred=pred_labels, average='macro')
    return np.mean(test_loss), test_score, pred_labels
