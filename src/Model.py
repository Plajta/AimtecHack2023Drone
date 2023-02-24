import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

from torchsummary import summary

device = "cpu"
log_interval = 10

class SmileNet(nn.Module):
    sweep_configuration = {
        #TODO: lets just leave it blank
    }

    config = {
        "Conv1_size": 3,
        "Conv1_in": 3,
        "Conv1_out": 16,

        "Conv2_size": 3,
        "Conv2_in": 16,
        "Conv2_out": 32,

        "Linear1_in": 128,
        "Linear1_out": 64,
        "Linear2_in": 64,
        "Linear2_out": 7,

        "loss": "categorical_crossentropy",
        "lr": 1e-4,
        "metric": "accuracy",
        "epochs": 7

    }

    def __init__(self):
        super(SmileNet).__init__()

        self.Conv1 = nn.Conv2d(3, 16, 3)
        self.Pool1 = nn.MaxPool2d(2)
        self.Conv2 = nn.Conv2d(16, 32, 3)
        self.Pool2 = nn.MaxPool2d(2)

        self.layer1 = nn.Linear(128, 64)
        self.layer2 = nn.Linear(64, 7)

    def Model_init(self):
        self.model = SmileNet().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), self.config["lr"])

    def forward(self, x):
        x = self.Conv1(x)
        x = F.relu(self.Pool1(x))

        x = self.Conv2(x)
        x = F.relu(self.Pool2(x))

        x = F.relu(self.layer1(x))
        x = F.softmax(self.layer2(x))

    def Train(self):
        self.model.train()
        #TODO: dodělat