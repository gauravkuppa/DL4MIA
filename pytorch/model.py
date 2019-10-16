# follow Google coding standards for imports
import os
import torch
import torchvision
from custom_dataset_npy import CustomDatasetNPY
from net import Net


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

        # Defines paths for training and validation datasets.
        train_path = "/home/gauravkuppa24/Documents/datasets/MRNet-v1.0/train/axial"
        valid_path = "/home/gauravkuppa24/Documents/datasets/MRNet-v1.0/valid/axial"

        # Create Dataset and DataLoader for training and validation dataset
        self.dataset_train = CustomDatasetNPY(train_path)
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset_train, batch_size=4, shuffle=True, num_workers=4
        )
        self.dataset_valid = CustomDatasetNPY(valid_path)
        self.test_loader = torch.utils.data.DataLoader(
            self.dataset_valid, batch_size=4, shuffle=True, num_workers=4
        )

        # Create Neural Network with hyperparameters.
        self.net = Net()
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=0.01
        )  # how do you know which optim to use when?
        self.criterion = (
            torch.nn.CrossEntropyLoss()
        )  # how do you know which criterion to use? why do we choose cross entropy loss
        self.exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=7, gamma=0.1
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO(G): make a train() method

    # TODO(G): make a evaluate() method

    # TODO(G): make a plot_image() method

    # TODO(G): make a plot_results() method


def main():
    model = Model()
    print(model.dataset_train)


if __name__ == "__main__":
    main()
