import Wandb
import torch
import Model
import process_dataset
import os

#setup
train, test = process_dataset.Load_Dataset()

model = Model.SmileNet().to(Model.device)
model.Model_init()

def RunSMILENet():
    #basically a CNN
    Wandb.Init("SMILENet", model.config, "SMILENet v1 run:" + str(model.model_iter))

    init_loss, init_acc = model.Test(test)
    Wandb.wandb.log({"val_accuracy": init_acc, "val_loss": init_loss})
    for epoch in range(1, model.config["epochs"] + 1):
        model.Train(epoch, train, Wandb.wandb)
        loss, acc = model.Test(test)

        Wandb.wandb.log({"val_accuracy": acc, "val_loss": loss})
    
    torch.save(model, os.getcwd() + "/model/SMILENet" + str(model.model_iter) + ".pth")

    model.model_iter += 1
    #model.Table_validate(test, Wandb.wandb, [1, 2, 3]) does not work <- update
    Wandb.End()

#actual Run
#Wandb.InitSweep(model.sweep_configuration, "SMILENet", RunSMILENet)
RunSMILENet()