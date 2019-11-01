import numpy as np
import pandas as pd
from PIL import Image
import torchvision
import torch
from torch.utils.data import *
import os
import random
import math

# TODO(G): finish this transform code
'''class ToTripleTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = 
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}'''

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
        #dataset processing
        self.Y = np.genfromtxt(y_path, delimiter=",")

        if train_or_valid == "valid":
            self.Y[:,0] = np.subtract(self.Y[:,0], 1130)    
            print(self.Y)
            
        
        # removing last four images from dataset
        # they do not add anything to the dataset bc not consistent with dimensions of other images
        
        for file in os.listdir(x_path):
            if file.endswith(".npy"):
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

        

        
        '''self.transforms1 = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                # torchvision.transforms.RandomGrayscale(),
                torchvision.transforms.ToTensor(),
                torchviion.transforms.Lambda(lambda img: ),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )'''
        #
        self.length = len(self.X_slices)

    def __getitem__(self, index):
        """
        Args:
        return (Image, Ground_truth)
        """
 
        print("index:", index)
        x_slice = self.X_slices[index]
        # print("x slice shape:", x_slice.shape)
        x = np.squeeze(x_slice, axis=0)
        pil_image = torchvision.transforms.functional.to_pil_image(x)
        tensor = torchvision.transforms.functional.to_tensor(pil_image)
        # 
        # print("tensor shape:", tensor.shape)
        # tensor = tensor.type(torch.LongTensor)
        triple_tensor = torch.stack(list(tensor) * 3, dim=0)
        transformed_tensor = torchvision.transforms.functional.normalize(triple_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # print("transformed slice shape:", transformed_tensor.shape)
        # transformed_tensor = np.concatenate(tuple(list(transformed_slice) * 3), axis=0)
        # print("transformed tensor shape:",transformed_tensor.shape)
        if self.train_or_valid == "valid" and index > 1129:
            y = self.Y[index - 1130]
        else:
            y = self.Y[index]

        return (transformed_tensor, y)

    def __len__(self):
        """
        Args: return length
        """
        return self.length


def main():
    # Create Dataset and DataLoader for training and validation dataset
    dataset_train = CustomDatasetNPY("train")
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=4, shuffle=True  # , num_workers=4
    )
    dataset_valid = CustomDatasetNPY("valid")
    test_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=4, shuffle=True  # , num_workers=4
    )

    for index, (tensor, ground) in enumerate(test_loader):
        print("index:", index)
        print("tensor:", tensor.shape)
        print("truth:", ground)

    


if __name__ == "__main__":
    main()