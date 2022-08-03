import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import csv
import cv2
import numpy as np
import torch

class CustomImageDataset(Dataset):  #inherent from dataset class

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0])
        image = cv2.imread(img_path)

        image_bgr = cv2.split(image)  # split the 1D BGR image into three channels
        image_nparray = np.array(image_bgr)
        image_tensor = torch.FloatTensor(image_nparray) # convert image to tensor

        label = self.img_labels.iloc[index, 1]
        if self.transform:
            image_tensor = self.transform(image_tensor)
        if self.target_transform:
            label = self.target_transform(label)

        y_true = torch.tensor(label)     # convert label to tensor
        #print(image_tensor.shape)
        return image_tensor, y_true   # all tensors


