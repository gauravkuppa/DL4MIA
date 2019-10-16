import numpy as np
import pandas as pd
from PIL import Image
import torchvision
import torch
import os
import random


class CustomDatasetNPY(torch.utils.data.Dataset):

    X = list()
    X_slices = list()
    Y = None

    def __init__(self, path):
        """
        Args:
            path (string): path to root folder [axial]
        """
        self.Y = np.genfromtxt(
            "/home/gauravkuppa24/Documents/datasets/MRNet-v1.0/train-acl.csv",
            delimiter=",",
        )[:1126]
        "

        # removing last four images from dataset
        # they do not add anything to the dataset bc not consistent with dimensions of other images
        for file in os.listdir(path):
            if file.endswith(".npy"):
                if int(file.split(".")[0]) not in list(range(1126, 1130)):
                    self.X.append(np.load(path + "/" + file))

        for img in self.X:
            rand_val = random.randint(0, img.shape[0] - 1)
            self.X_slices.append(img[rand_val])



        self.X_slices = np.asarray(self.X_slices)
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToPILImage(), torchvision.transforms.ToTensor()]
        )
        self.length = len(self.X_slices)

    def __getitem__(self, index):
        """
        Args:
        return (Image, Ground_truth)
        """

        return (self.transforms(self.X_slices[index]), self.Y[index])

    def __len__(self):
        """
        Args: return length
        """
        return self.length
