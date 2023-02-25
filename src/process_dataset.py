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

class SmileDataset(Dataset):
    def __init__(self, train_or_test):
        self.train_or_test = train_or_test
        self.Train_X = []
        self.Train_y = []
        self.Test_X = []
        self.Test_y = []
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

        #enumerate training set
        if self.train_or_test:
            for i, path in enumerate(os.listdir(abs_path_train)):

                if os.path.isfile(os.path.join(abs_path_train, path)):

                    Train_img = cv2.imread(abs_path_train + "/" + path)
                    Train_img = Convert_To_Tensor(Train_img)

                    #Augment 1 training image to 7 images (total train images = 30*7=210)
                    #list of augmentations: Resize (3 imgs), Rotate (3 imgs), Horizontal Flip (1 imgs)
                    
                    Augmented_imgs, Augmented_labels = Augment_img(Train_img, labels_train[str(i)])
                    
                    self.Train_X.extend(Augmented_imgs)
                    self.Train_X.extend(Augmented_labels)

            self.Train_y = F.one_hot(torch.tensor(self.Train_y, dtype=torch.int64), 5)

        #enumerate testing set
        else:
            for i, path in enumerate(os.listdir(abs_path_test)):
                
                if os.path.isfile(os.path.join(abs_path_test, path)):

                    Test_img = cv2.imread(abs_path_test + "/" + path)
                    Test_img = Convert_To_Tensor(Test_img)

                    #Augment 1 testing image to 7 images (total test images = 20*7=140)
                    #list of augmentations: Resize (3 imgs), Rotate (3 imgs), Horizontal Flip (1 imgs)

                    Augmented_imgs, Augmented_labels = Augment_img(Test_img, labels_test[str(i)])
                    
                    self.Test_X.extend(Augmented_imgs)
                    self.Test_y.extend(Augmented_labels)

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
        train, batch_size=1, shuffle=True #dataset shuffle
    )

    test_loader = DataLoader(
        test, batch_size=1, shuffle=True #dataset shuffle
    )
    
    print(len(train_loader))
    return train_loader, test_loader

def Augment_img(Train_img, label):
    Augmented_imgs = []
    Augmented_labels = [label] * 8

    for i in range(3):
        resize = random.randint(40, 120)
        Resize_transform = transforms.Resize(size=(resize, resize))
        Augmented_imgs.append(Resize_transform(Train_img))

    for i in range(3):
        rotate = random.randint(45, 125)
        Rotate_transform = transforms.RandomRotation(rotate)
        Augmented_imgs.append(Rotate_transform(Train_img))

    Flip_transform = transforms.RandomHorizontalFlip(1)
    Augmented_imgs.append(Flip_transform(Train_img))
    Augmented_imgs.append(Train_img) #append original image

    return Augmented_imgs, Augmented_labels