# follow Google coding standards for imports
import copy
import multiprocessing
import os
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn import preprocessing
from torch.optim import lr_scheduler
from zmq.devices import device
from tqdm import tqdm
from custom_dataset_npy import CustomDatasetNPY
from net import ConvNet
import pickle
import matplotlib

multiprocessing.set_start_method("spawn", True)

# try to run resnet34 architecture

# look at first: https://pytorch.org/hub/pytorch_vision_resnet/

# look at second: https://stackoverflow.com/questions/23202132/splitting-an-rgb-image-to-r-g-b-channels-python/23208666
class Model(object):
    """Deep learning model for image recognition.

    Loads dataset and neural network hyperparameters. 
    

    Typical usage example:

    model = Model()
    print(model.dataset_train)

    """

    def __init__(self):
        """Initialize dataset and neural network.

        Traverses train_path and valid_path to load (.npy) images using CustomDataNPY)().
        Creates neural network using Net(). Defines hyperparameters. These hyperparameters may need more work and iteration.
        """
        '''
        # Defines paths for training and validation datasets.
        train_path = "/home/gauravkuppa24/Documents/datasets/MRNet-v1.0/train/coronal"
        valid_path = "/home/gauravkuppa24/Documents/datasets/MRNet-v1.0/valid/axial"
        '''
        # Create Dataset and DataLoader for training and validation dataset
        self.dataset_train = CustomDatasetNPY("train")[0:200]
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset_train, batch_size=30, shuffle=False  # , num_workers=4
        )
        self.dataset_valid = CustomDatasetNPY("valid")[0:25]
        self.valid_loader = torch.utils.data.DataLoader(
            self.dataset_valid, batch_size=30, shuffle=False  # , num_workers=4
        )
        self.dataset_sizes = {'train':len(self.dataset_train), 'valid':len(self.dataset_valid)}
        self.dataloaders = {
            'train': self.train_loader,
            'valid': self.valid_loader
        }
        # Create Neural Network with hyperparameters.
        self.net = ConvNet(2)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=0.01
        )  # how do you know which optim to use when?
        self.criterion = (
            torch.nn.CrossEntropyLoss()
        )  # how do you know which criterion to use? why do we choose cross entropy loss
        self.exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1
        )
        self.device = torch.device("cpu")

    # TODO(G): make a train() method
    def train(self, num_epochs=10):
        net, criterion, optimizer, scheduler = self.net, self.criterion, self.optimizer, self.exp_lr_scheduler
        since = time.time()
        train_loss = []
        train_accuracy = []
        valid_loss = []
        valid_accuracy = []
        self.stats = {"train":[train_loss, train_accuracy],"valid":[valid_loss, valid_accuracy]}
        best_model_wts = copy.deepcopy(net.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)
            
            # Each epoch has a training and validation phase
            for phase in ["train", "valid"]:
                if phase == "train":
                    net.train()  # Set model to training mode
                else:
                    net.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for i in tqdm(range(len(self.dataloaders[phase]))):
                    it = iter(self.dataloaders[phase])
                    inputs, labels = it.next()

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    labels = labels.long()[:,1]

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = net(inputs)
                        _, preds = torch.max(outputs, 1)
                        #print(labels)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double().item() / self.dataset_sizes[phase]
                self.stats[phase][0].append(epoch_loss)
                self.stats[phase][1].append(epoch_acc)
                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )

                # deep copy the model
                if phase == "valid" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(net.state_dict())

            print()

        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Acc: {:4f}".format(best_acc))

        # load best model weights
        net.load_state_dict(best_model_wts)
        return net

    def plot_images(self):
        for imgBunch, groundBunch in self.train_loader:
            print(imgBunch.shape)
            for img in imgBunch:
                print("x", img.shape)
                img = img[2,:,:]
                print("y", img.shape)
                # TODO(g): display img #, ground truth, img index
                plt.imshow(img.view(256, -1), cmap="gray")
                plt.show()

    def plot_results(self, epochs, loss_acc):
        train_loss, train_accuracy, valid_loss, valid_accuracy = loss_acc
        fig = plt.figure(figsize=(20,4))
        ax = fig.add_subplot(1, 2, 1)
        plt.title("Train - Validation Loss")
        plt.plot(list(np.arange(epochs) + 1) , train_loss, label='train')
        plt.plot(list(np.arange(epochs) + 1), valid_loss, label='validation')
        plt.xlabel('num_epochs', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        plt.legend(loc='best')
        
        ax = fig.add_subplot(1, 2, 2)
        plt.title("Train - Validation Accuracy")
        plt.plot(list(np.arange(epochs) + 1) , train_accuracy, label='train')
        plt.plot(list(np.arange(epochs) + 1), valid_accuracy, label='validation')
        plt.xlabel('num_epochs', fontsize=12)
        plt.ylabel('accuracy', fontsize=12)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        plt.legend(loc='best')
        plt.show()

def main():
    # set random seed to 0
    torch.manual_seed(0)
    np.random.seed(0)

    model = Model()
    
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []
    epochs = 25
    
    model.train(num_epochs=epochs)
    train_loss, train_accuracy = model.stats['train']
    valid_loss, valid_accuracy = model.stats['valid']
    
    loss_acc = [train_loss, train_accuracy, valid_loss, valid_accuracy]

    model.plot_results(epochs, loss_acc)
    


if __name__ == "__main__":
    main()