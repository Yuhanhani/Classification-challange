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
# import metrics


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(device)

# download and read labels -----------------------------------------------------------------------------
pd.options.display.max_rows = 1000
# label = pd.read_csv(r'/Users/mirandazheng/Desktop/folder/labels.csv')

if torch.cuda.is_available():
    file = open('/well/papiez/users/pea322/C_DR_Grading/Groundtruths/labels.csv')
else:
    file = open('/Users/mirandazheng/Desktop/folder/labels.csv')

label_csv = csv.reader(file)
label_list = []
for row in label_csv:
    # print(row)
    label_list.append(row)

# print(label_list[1:-1])
label_list = label_list[1:]
x = np.array(label_list)
x = [int(string) for string in x[:, 1]]
x = torch.FloatTensor(x)
x = x.long()
# print(type(x))


# write labels and images into training data ------------------------------------------------------------

training_data = []

if torch.cuda.is_available():
    image_path = sorted(glob.glob('/well/papiez/users/pea322/C_DR_Grading/Original_Images/TrainingSet/*.png'))
else:
    image_path = sorted(glob.glob('/Users/mirandazheng/Desktop/folder/*.png'))

for i, filename in enumerate(image_path):
    # image = Image.open(filename)
    image = cv2.imread(filename)
    # print(i, filename)
    image_bgr = cv2.split(image)   # split the 1D BGR image into three channels
    image_nparray = np.array(image_bgr)
    image_tensor = torch.FloatTensor(image_nparray)

    # training_data.append([transforms.ToTensor()(image), x[i]])
    training_data.append([image_tensor, x[i]])

# print(training_data)

train_dataloader = DataLoader(training_data, batch_size=2)
# print(train_dataloader)



# use model----------------------------------------------------------------------------------------------
model = models.resnet152(weights=None, progress=False) # equivalent to parameter = false
model.fc = nn.Linear(512 * 4, 3)   # can use this to change fully connected -> 1000 to 3
model = model.to(device)  # move model to GPU



# Define the training loop -------------------------------------------------------------------------------

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    prediction = torch.empty(2, 3) # otherwise will induce error when return (unboundlocalerror) when run in gpu
    for batch, (X, y) in enumerate(dataloader):
        # compute prediction and loss
        # print(batch, X, y)
        # X = X.unsqueeze(0)
        # X = X.unsqueeze(0)


        # X = X.expand(2, 3, 1024, 1024)
        # X = cv2.cvtColor(X, cv2.COLOR_GRAY2RGB)

        # move data to GPU (earlier Dataloader object has no attribute 'to', will lead an error)
        X = X.to(device)
        y = y.to(device)
        # print(X.size(), type(X)) # [2,1,1024,1024] -> [batch_size, channels, depth (bit depth), height, width]
        pred = model(X)
        initial_pred = pred
        # print(pred.size())
        # print(y.size())
        # print(pred, y)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch == int(size//2):
            loss, current = loss.item(), batch * 2
        else:
            loss, current = loss.item(), batch * len(X)

        print(f'loss:{loss:>7f} [{current:>5d}/{size:>5d}]')


        if batch == 0:
            prediction = initial_pred
        else:
            prediction = torch.cat((prediction, pred), 0)

    return prediction



# Start to train (an example) ---------------------------------------------------------------------------

learning_rate = 1e-3
epochs = 5

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f'Epoch {t+1}\n ---------------------------- ')
    prediction = train_loop(train_dataloader, model, loss_fn, optimizer)
    # print(prediction)
    # kappa = metrics.quadratic_weighted_kappa(x,pred)
    # print(kappa)
print('Done!')


