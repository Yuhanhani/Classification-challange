# with data augmentation
# with test loop
# with saving model, adaptive learning rate (learning rate scheduler) & dataset class & saving model

import os
from PIL import Image
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
from torchvision import transforms
import torchvision.models as models
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
from torch import nn
import pandas as pd
import csv
import cv2
import metrics
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import dataset
import data_augmentation

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(device)

# download and read labels and data ------------------------------------------------------------------------------------

pd.options.display.max_rows = 1000

if torch.cuda.is_available():
    annotations_file = open('/well/papiez/users/pea322/C_DR_Grading/Groundtruths/labels.csv')
else:
    annotations_file = open('/Users/mirandazheng/Desktop/folder/labels.csv')

if torch.cuda.is_available():
    img_dir = '/well/papiez/users/pea322/C_DR_Grading/Original_Images/TrainingSet'
else:
    img_dir = '/Users/mirandazheng/Desktop/folder'

if torch.cuda.is_available():
    test_annotations_file = open('/well/papiez/users/pea322/C_DR_Grading/labels.csv')
else:
    test_annotations_file = open('/Users/mirandazheng/Desktop/labels.csv')


transform_train = data_augmentation.data_augmentation_transform().transform('train')
training_dataset = dataset.CustomImageDataset(annotations_file=annotations_file, img_dir=img_dir, transform=transform_train)
length = training_dataset.__len__()


#transform_test = data_augmentation.data_augmentation_transform().transform('test')
test_dataset = dataset.CustomImageDataset(annotations_file=test_annotations_file, img_dir=img_dir) # transform=transform_test


y_true = []
for i in range(length):

    image, label = training_dataset.__getitem__(i)


    if i == 0:
        label = label.item()
        label = torch.tensor([label])
        y_true = label
    else:
        label = label.item()
        label = torch.tensor([label])
        y_true = torch.cat((y_true, label), 0)

print(y_true)
print(type(y_true))


batch_size = 5
train_dataloader = DataLoader(training_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


# use model-------------------------------------------------------------------------------------------------------------
# model = models.resnet152(weights=None, progress=False) # equivalent to parameter = false
# model.fc = nn.Linear(512 * 4, 3)   # can use this to change fully connected -> 1000 to 3

model = models.resnet34(weights=None, progress=False) # equivalent to parameter = false
model.fc = nn.Linear(512 * 1, 3)   # can use this to change fully connected -> 1000 to 3

# ----------------------------------------------------------------------------------------------------------------------
# cannot recover the 20th metrics value because this model is saved after last back propagation, but the metrics are
# calculated before 20th back propagation. can save model also at this position or just keep it and test using
# training data
# path = '/users/papiez/pea322/resnet34_model/5_0.01_adaptive_0.7_15.pth'
# model.load_state_dict(torch.load(path))
# model.eval() batch normalisation makes training descent smoothier, without it, could use smaller learning rate to try
# to see the same effect. Otherwise the loss will explode
# ----------------------------------------------------------------------------------------------------------------------


model = model.to(device)  # move model to GPU


# Define the training loop ---------------------------------------------------------------------------------------------

def train_loop(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):

        X = X.to(device)
        y = y.to(device)

        pred = model(X)

        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch == int(size//batch_size):
            loss, current = loss.item(), batch * batch_size
        else:
            loss, current = loss.item(), batch * len(X)

        print(f'loss:{loss:>7f} [{current:>5d}/{size:>5d}]')


# Define test loop -----------------------------------------------------------------------------------------------------

def test_loop(dataloader, model, loss_fn):

    size = len(dataloader.dataset)
    total_loss = 0
    prediction = torch.empty(batch_size, 3)

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):

            X = X.to(device)
            y = y.to(device)
            #print(X.shape)
            #print(y.shape)

            pred = model(X)

            initial_pred = pred

            loss = loss_fn(pred, y)

            if batch == int(size//batch_size):
                loss, current = loss.item(), batch * batch_size
            else:
                loss, current = loss.item(), batch * len(X)


            print(f'loss:{loss:>7f} [{current:>5d}/{size:>5d}]')


            total_loss = total_loss + loss

            if batch == 0:
                prediction = initial_pred
            else:
                prediction = torch.cat((prediction, pred), 0)

            del X, y

    return prediction, total_loss


# Start to train (an example) ------------------------------------------------------------------------------------------

learning_rate = 0.01 # set the initial learning rate
epochs = 20
gamma = 0.7

loss_fn = nn.CrossEntropyLoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

writer = SummaryWriter()

if torch.cuda.is_available():
    project_path = '/users/papiez/pea322/resnet34_model'
else:
    project_path = '/Users/mirandazheng/PycharmProjects/pythonProject3'

for t in range(epochs):
    print(f'Epoch {t+1}\n train loop---------------------------- ')

    model.train()
    train_loop(train_dataloader, model, loss_fn, optimizer)

    path = os.path.join(project_path, '{}_{}_{}_{}_{}_{}.pth'.format(batch_size, learning_rate, 'adaptive', gamma, (t+1), 'aug.'))
    # '/users/papiez/pea322/resnet34_model.pth'

    torch.save(model.state_dict(), path)

    print(f'Epoch {t+1}\n test loop---------------------------- ')

    model.eval()
    prediction, total_loss = test_loop(test_dataloader, model, loss_fn)
    writer.add_scalars('Losses', {'Losses': total_loss}, t)

    scheduler.step()
    print('epoch={}, learning rate={:.6f}'.format(t+1, optimizer.state_dict()['param_groups'][0]['lr']))

    Softmax_layer = nn.Softmax(dim=1)
    prediction = Softmax_layer(prediction)
    prediction = prediction.to(device)
    y_pred = torch.argmax(prediction, dim=1)

    print(y_pred)
    y_pred = y_pred.detach().cpu() # only move into cpu, can then be transferred into np array

    label_encoder = LabelEncoder()
    y_pred_encoded = label_encoder.fit_transform(y_pred)
    y_pred_encoded = y_pred_encoded.reshape(len(y_pred_encoded), 1)

    onehot_encoder = OneHotEncoder(categories=[[0, 1, 2]], handle_unknown='ignore', sparse=False)
    y_pred_encoded = onehot_encoder.fit_transform(y_pred_encoded)  # onehot encoder only accepts 2d array

    print(y_true)

    kappa_score = metrics.quadratic_weighted_kappa(y_true, y_pred)
    macro_auc = metrics.macro_auc(y_true, y_pred_encoded)
    macro_precision = metrics.marco_precision(y_true, y_pred)
    macro_sensitivity = metrics.marco_sensitivity(y_true, y_pred)
    macro_specificity = metrics.marco_specificity(y_true, y_pred)

    print(total_loss)
    print(kappa_score)
    print(macro_auc)
    print(macro_precision)
    print(macro_sensitivity)
    print(macro_specificity)

writer.close()
print('Done!')