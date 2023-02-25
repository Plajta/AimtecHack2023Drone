import os
import json
import cv2
import random
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F 
import torchvision.transforms as transforms
import albumentations as A
import numpy as np
import torch

#just little stupid and simple script to load training and testing images to the algorithm

class SmileDataset(Dataset):
    def __init__(self, train_or_test):
        self.train_or_test = train_or_test
        self.Load_Train_and_Test()

    def __len__(self):
        return len(self.Train_X) + len(self.Test_X)
    
    def __getitem__(self, idx):
        if self.train_or_test:
            image = self.Train_X[idx]
            labels = open(os.getcwd() + "/data/labels_train.json")
        else:
            image = self.Test_X[idx]
            labels = open(os.getcwd() + "/data/labels_test.json")

        labels = json.load(labels)

        label = labels[str(idx)]
        
        return image, label
    
    def Load_Train_and_Test(self):
        abs_path_train = os.getcwd() + "/data/train"
        abs_path_test = os.getcwd() + "/data/test"

        labels_train = open(os.getcwd() + "/data/labels_train.json")
        labels_test = open(os.getcwd() + "/data/labels_test.json")
        labels_train = json.load(labels_train)
        labels_test = json.load(labels_test)

        self.Train_X = []
        self.Train_y = []

        self.Test_X = []
        self.Test_y = []


        #enumerate training set
        for i, path in enumerate(os.listdir(abs_path_train)):

            if os.path.isfile(os.path.join(abs_path_train, path)):
                #train and test reading i kinda weird (index read is kinda weird), at least
                # i dont have to worry about dataset shuffle
                #index = path.replace("train.png", "")

                Train_img = cv2.imread(abs_path_train + "/" + path)
                Train_img = Convert_To_Tensor(Train_img)
                
                self.Train_X.append(Train_img)
                self.Train_y.append(labels_train[str(i)])

        self.Train_y = F.one_hot(torch.tensor(self.Train_y, dtype=torch.int64), 5)

        #enumerate testing set
        for i, path in enumerate(os.listdir(abs_path_test)):
            
            if os.path.isfile(os.path.join(abs_path_test, path)):
                #train and test reading i kinda weird (index read is kinda weird), at least
                # i dont have to worry about dataset shuffle

                Test_img = cv2.imread(abs_path_test + "/" + path)
                Test_img = Convert_To_Tensor(Test_img)

                self.Test_X.append(Test_img)
                self.Test_y.append(labels_test[str(i)])

        self.Test_y = F.one_hot(torch.tensor(self.Test_y, dtype=torch.int64), 5)

def Random_dataset_inspect(Train_X, Train_y, Test_X, Test_y):
    print("-- TRAIN --")
    idx_train = random.randint(0, len(Train_y))
    print("y:" + str(Train_y[idx_train]))

    print("-- TEST --")
    idx_test = random.randint(0, len(Train_y))
    print("y:" + str(Test_y[idx_test]))

    cv2.imshow("Train X", Train_X[idx_train])
    cv2.imshow("Test X", Test_X[idx_test])
    cv2.waitKey(0)

def Convert_To_Tensor(img):
    transform = transforms.ToTensor()
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    X = transform(img)

    return X
def Load_Dataset():

    #dataset setup
    train = SmileDataset(train_or_test=True)
    test = SmileDataset(train_or_test=False)

    train_loader = DataLoader(
        train, batch_size=1, shuffle=False
    )

    test_loader = DataLoader(
        test, batch_size=1, shuffle=False
    )
    
    return train_loader, test_loader