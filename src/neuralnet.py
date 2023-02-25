import torch
import torchsummary
import wandb

import Model
import process_dataset

#setup
train, test = process_dataset.Load_Dataset()

model = Model.SmileNet().to(Model.device)
model.Model_init()

def RunSMILENet():
    #basically a CNN

    model.Test(test)
    for epoch in range(1, model.config["epochs"] + 1):
        model.Train(train)
        loss, acc = model.Test(test)

    model.Table_validate() #TODO: dodělat!

#actual Run
RunSMILENet()