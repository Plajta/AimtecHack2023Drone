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
from PIL import Image

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
            label = self.Train_y[idx]
        else:
            image = self.Test_X[idx]
            label = self.Test_y[idx]
        
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
            for path in os.listdir(abs_path_train):
                if os.path.isfile(os.path.join(abs_path_train, path)):

                    Train_img = cv2.imread(abs_path_train + "/" + path)
                    Train_img[np.all(Train_img == (0, 0, 255), axis=-1)] = (255,255,255)
                    Train_img = cv2.cvtColor(Train_img, cv2.COLOR_BGR2GRAY)
                    Train_img = Convert_To_Tensor(Train_img)

                    #Augment 1 training image to more images (total train images = 30*7=210)
                    #list of augmentations: Resize (10 imgs), Rotate (10 imgs), Horizontal Flip (1 imgs)
                    path = path.replace("train.png", "")
                    Augmented_imgs, Augmented_labels = Augment_img(Train_img, labels_train[path], (120, 120, 60))
                    
                    self.Train_X.extend(Augmented_imgs)
                    self.Train_y.extend(Augmented_labels)

            self.Train_y = F.one_hot(torch.tensor(self.Train_y, dtype=torch.int64), 4)

        #enumerate testing set
        else:
            for path in os.listdir(abs_path_test):     
                if os.path.isfile(os.path.join(abs_path_test, path)):
                    
                    #image conversion
                    Test_img = cv2.imread(abs_path_test + "/" + path)
                    Test_img[np.all(Test_img == (0, 0, 255), axis=-1)] = (255,255,255)
                    Test_img = cv2.cvtColor(Test_img, cv2.COLOR_BGR2GRAY)
                    Test_img = Convert_To_Tensor(Test_img)

                    #Augment 1 testing image to more images
                    #list of augmentations: Resize (10 imgs), Rotate (10 imgs), Horizontal Flip (1 imgs)
                    path = path.replace("test.png", "")
                    Augmented_imgs, Augmented_labels = Augment_img(Test_img, labels_test[path], (120, 120, 60))

                    self.Test_X.extend(Augmented_imgs)
                    self.Test_y.extend(Augmented_labels)

            self.Test_y = F.one_hot(torch.tensor(self.Test_y, dtype=torch.int64), 4)

def Random_dataset_inspect(Train_X, Train_y, Test_X, Test_y):
    print("-- TRAIN --")
    idx_train = random.randint(0, len(Train_y))
    print("y:" + str(Train_y[idx_train]))

    print("-- TEST --")
    idx_test = random.randint(0, len(Train_y))
    print("y:" + str(Test_y[idx_test]))

def Convert_To_Tensor(img):
    transform = transforms.ToTensor()
    
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    X = transform(img).to(torch.float32)

    return X

def Load_Dataset():

    #dataset setup
    train = SmileDataset(train_or_test=True)
    test = SmileDataset(train_or_test=False)

    train_loader = DataLoader(
        train, batch_size=32, shuffle=True #dataset shuffle
    )

    test_loader = DataLoader(
        test, batch_size=32, shuffle=True #dataset shuffle
    )

    print(len(train_loader), len(test_loader))
    
    return train_loader, test_loader

def Augment_img(Train_img, label, len_transforms):
    Augmented_imgs = []
    Augmented_labels = [label] * (sum(len_transforms) + 1)

    for i in range(len_transforms[0]):
        resize = random.randint(60, 120)
        Resize_transform = transforms.Resize(size=(resize, resize))
        Resize2_transform = transforms.Resize(size=(160, 160))
        Convert_transform = transforms.ToPILImage()
        Convert2_transform = transforms.ToTensor()

        resized_img = Resize_transform(Train_img)
        PILImage = Convert_transform(resized_img)
        pad = int((160-resize) / 2)
        PILImage = add_margin(PILImage, pad)
        resized_img = Convert2_transform(PILImage)
        resized_img = Resize2_transform(resized_img) #just in case

        Augmented_imgs.append(resized_img)

    for i in range(len_transforms[1]):
        rotate = random.randint(45, 125)
        Rotate_transform = transforms.RandomRotation(rotate)
        Augmented_imgs.append(Rotate_transform(Train_img).to(torch.float32))

    for i in range(len_transforms[2]):
        Flip_transform = transforms.RandomHorizontalFlip(1)
        Augmented_imgs.append(Flip_transform(Train_img).to(torch.float32))
    
    Augmented_imgs.append(Train_img) #append original image

    return Augmented_imgs, Augmented_labels

def add_margin(pil_img, pad):
    width, height = pil_img.size
    new_width = width + 2 * pad
    new_height = height + 2 * pad
    result = Image.new(pil_img.mode, (new_width, new_height), 0)
    result.paste(pil_img, (pad, pad))
    return result