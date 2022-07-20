import os
from PIL import Image
import glob
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
#import metrics

# downloading the labels and image
pd.options.display.max_rows = 1000
#label = pd.read_csv(r'/Users/mirandazheng/Desktop/folder/labels.csv')
file = open('/Users/mirandazheng/Desktop/folder/labels.csv')
label_csv = csv.reader(file)
label_list = []
for row in label_csv:
    #print(row)
    label_list.append(row)

#print(label_list[1:-1])
label_list = label_list[1:-1]
x = np.array(label_list)
x = [int(string) for string in x[:, 1]]
x = torch.FloatTensor(x)
x = x.long()
#print(type(x))

training_data = []
for i, filename in enumerate(sorted(glob.glob('/Users/mirandazheng/Desktop/folder/*.png'))):
    image = Image.open(filename)
    #print(i, filename)
    training_data.append([transforms.ToTensor()(image), x[i]])

#print(training_data)

train_dataloader = DataLoader(training_data, batch_size=2)
#print(train_dataloader)

model = models.resnet152(weights=None, progress=False) # equivalent to parameter = false


# Define the training loop
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # compute prediction and loss
        #print(batch, X, y)
        #X = X.unsqueeze(0)
        #X = X.unsqueeze(0)


        X = X.expand(2, 3, 1024, 1024)


        #print(X.size(), type(X)) # [2,1,1024,1024] -> [batch_size, channels, depth (bit depth), height, width]
        pred = model(X)
        #print(pred, y)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        print(f"loss:{loss:>7f} [{current:>5d}/{size:>5d}]")
    return pred


# Start to train (an example)
learning_rate = 1e-3
epochs = 5

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n ---------------------------- ")
    pred = train_loop(train_dataloader, model, loss_fn, optimizer)
    print(pred)
    #kappa = metrics.quadratic_weighted_kappa(x,pred)
    #print(kappa)
print("Done!")


