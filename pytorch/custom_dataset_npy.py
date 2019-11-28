import numpy as np
import pandas as pd
from PIL import Image
import torchvision
import torch
from torch.utils.data import *
import os
import random
import math
from compress_pickle import dump, load

class CustomDatasetNPY(torch.utils.data.Dataset):

    def __init__(self, train_or_valid):
        """
        Args:
            path (string): path to root folder [axial]
        """
        
        train_path_X = "/home/gauravkuppa24/Documents/datasets/MRNet-v1.0/train/coronal"
        valid_path_X = "/home/gauravkuppa24/Documents/datasets/MRNet-v1.0/valid/coronal"
        train_path_Y = "/home/gauravkuppa24/Documents/datasets/MRNet-v1.0/train-meniscus.csv"
        valid_path_Y = "/home/gauravkuppa24/Documents/datasets/MRNet-v1.0/valid-meniscus.csv"
        self.train_or_valid = train_or_valid
        paths = {"train":[train_path_X, train_path_Y],"valid":[valid_path_X, valid_path_Y]}
        x_path, y_path = paths[train_or_valid]
        self.X = list()
        self.X_slices = list()
        self.Y = None
        #dataset processinggit 
        self.Y = np.genfromtxt(y_path, delimiter=",")

        if train_or_valid == "valid":
            self.Y[:,0] = np.subtract(self.Y[:,0], 1130)    
            print(self.Y)
            
        
        # removing last four images from dataset
        # they do not add anything to the dataset bc not consistent with dimensions of other images
        
        for file in os.listdir(x_path):
            if file.endswith(".npy"):
                val = file.split('.')[1][0]
                if val == '_':
                    continue
                if int(file.split(".")[0]) not in list(range(1126, 1130)):
                    print("file:", file)
                    x = np.load(x_path + "/" + file)
                    # print(x.shape)
                    self.X.append(x)
            
        

        # for each (s, 256, 256) tensor, take the middle slice of the tensor and append to X_slices
        for img in self.X:
            x = img[math.floor(img.shape[0] / 2)]
            x = np.asarray(x)
            x = np.expand_dims(x, axis=0)
            self.X_slices.append(x)

        
        self.length = len(self.X_slices)

    def __getitem__(self, index):
        """
        Args:
        return (Image, Ground_truth)
        """
        if isinstance(index, int):
            print("index:", index)
            x_slice = self.X_slices[index]
            x = np.squeeze(x_slice, axis=0)
            pil_image = torchvision.transforms.functional.to_pil_image(x)
            tensor = torchvision.transforms.functional.to_tensor(pil_image)
            triple_tensor = torch.stack(list(tensor) * 3, dim=0)
            # normalize tensor given mean and std dev. from ResNet architecture
            transformed_tensor = torchvision.transforms.functional.normalize(triple_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
            if self.train_or_valid == "valid" and index > 1129:
                y = self.Y[index - 1130]
            else:
                y = self.Y[index]
            
            return (transformed_tensor, y)
        elif isinstance(index, slice):
            returnArr = []
            for val in range(index.start, index.stop):
                x_slice = self.X_slices[val]
                x = np.squeeze(x_slice, axis=0)
                pil_image = torchvision.transforms.functional.to_pil_image(x)
                tensor = torchvision.transforms.functional.to_tensor(pil_image)
                triple_tensor = torch.stack(list(tensor) * 3, dim=0)
                # normalize tensor given mean and std dev. from ResNet architecture
                transformed_tensor = torchvision.transforms.functional.normalize(triple_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                
                if self.train_or_valid == "valid" and val > 1129:
                    y = self.Y[val - 1130]
                else:
                    y = self.Y[val]
                
                returnArr.append((transformed_tensor, y))
            
            return returnArr
            


    def __len__(self):
        """
        Args: return length
        """
        return self.length


def main():
    # Create Dataset and DataLoader for training and validation dataset
    '''train_loader_pickle = open("train_loader.pickle", "wb")
    valid_loader_pickle = open("valid_loader.pickle", "wb")'''
    
    dataset_train = CustomDatasetNPY("train")
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=4, shuffle=True  # , num_workers=4
    )
    dataset_valid = CustomDatasetNPY("valid")
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=4, shuffle=True  # , num_workers=4
    )
    '''dump(train_loader, "train_loader.gz")
    dump(train_loader, "train_loader.pkl")
    dump(valid_loader, "valid_loader.gz")
    
    print(os.path.getsize("train_loader.gz"))
    print(os.path.getsize("train_loader.pkl"))'''

    '''for index, (tensor, ground) in enumerate(valid_loader):
        print("index:", index)
        print("tensor:", tensor.shape)
        print("truth:", ground)'''

    loader_iter = iter(train_loader)
    dataset_iter = iter(dataset_train)

    print(next(loader_iter))
    print("#1:",next(dataset_iter))
    print("#2:",next(dataset_iter))
    print("#3:",next(dataset_iter))
    print("#4:",next(dataset_iter))

    


if __name__ == "__main__":
    main()