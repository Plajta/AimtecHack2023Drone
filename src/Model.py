import torch
import torchvision
from torch import nn, flatten
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

from torchsummary import summary

device = "cpu"
log_interval = 10

class SmileNet(nn.Module):
    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
        'parameters': 
        {
            'batch_size': {'values': [16, 32, 64]},
            'epochs': {'values': [5, 10, 15]},
            'lr': {'max': 0.1, 'min': 0.0001}
        },
        'run_cap': 10
    }

    config = {
        "Conv1_size": 3,
        "Conv1_in": 3,
        "Conv1_out": 16,

        "Conv2_size": 3,
        "Conv2_in": 16,
        "Conv2_out": 32,

        "Linear1_in": 38,
        "Linear1_out": 64,
        "Linear2_in": 64,
        "Linear2_out": 5,

        "loss": "categorical_crossentropy",
        "lr": 1e-4,
        "metric": "accuracy",
        "epochs": 20
    }

    def __init__(self):
        super(SmileNet, self).__init__()

        self.Conv1 = nn.Conv2d(3, 16, 3)
        self.Pool1 = nn.MaxPool2d(2)
        self.Conv2 = nn.Conv2d(16, 32, 3)
        self.Pool2 = nn.MaxPool2d(2)
        
        self.dropout = nn.Dropout(0.25)
        self.layer1 = nn.Linear(46208, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, 5)

    def Model_init(self):
        self.model = SmileNet().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), self.config["lr"])

    def forward(self, x):
        x = self.Conv1(x)
        x = F.selu(self.Pool1(x))

        x = self.Conv2(x)
        x = F.selu(self.Pool2(x))

        x = flatten(x, 1)
        x = self.dropout(x)
        x = F.selu(self.layer1(x))
        x = F.selu(self.layer2(x))
        
        pred = F.softmax(self.layer3(x))
        return pred

    def Train(self, epoch, train, wandb):
        self.model.train()
        for idx, (data, target) in enumerate(train):
            correct = 0

            self.optimizer.zero_grad()
            output = self.model(data)
            target = target.to(torch.float32)

            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

            pred = torch.max(output,1)[1]
            correct += (pred == target).sum()

            if idx % log_interval == 0:
                train_loss = loss.item()
                train_accuracy = 100. * correct / len(train.dataset)

                #reset counter
                correct = 0

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, idx * len(data), len(train.dataset), 100. * idx / len(train), train_loss))

                #log to wandb
                wandb.log({"accuracy": train_accuracy, "loss": train_loss})

    def Test(self, test):
        self.model.eval()
        correct = 0
        test_loss = 0

        with torch.no_grad():
            for data, target in test:
                
                output = self.model(data)
                target = target.to(torch.float32)

                test_loss += F.cross_entropy(output, target, size_average=False).item()
                pred = torch.max(output,1)[1]
                correct += (pred == target).sum()

        test_loss /= len(test.dataset)
        test_accuracy = 100. * correct / len(test.dataset)

        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test.dataset), test_accuracy))
        return test_loss, test_accuracy
    
    def Table_validate(self, test_loader, wandb, indexes = None):
        if isinstance(indexes, list):
            test_features, test_labels = next(iter(test_loader))
            print(f"Feature batch shape: {test_features.size()}")
            print(f"Labels batch shape: {test_labels.size()}")

            data = []

            for i in indexes:
                img = test_features[i].squeeze()
                label = test_labels[i]

                output = self.model(test_features[i]) #detect
                pred = int(output.data.max(1, keepdim=True)[1][0][0].tolist())

                data.append([wandb.Image(img), pred, label])
            
            columns=["image", "prediction", "truth"]
            table = wandb.Table(data=data, columns=columns)

            wandb.log({"table": table})