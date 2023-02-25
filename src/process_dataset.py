import os
import json
import cv2
import random
import torch

#just little stupid and simple script to load training and testing images to the algorithm

def Load_Train_and_Test():
    abs_path = os.getcwd() + "/data/"

    labels = open(abs_path + "labels.json")
    labels = json.load(labels)

    Train_X = []
    Train_y = []

    Test_X = []
    Test_y = []

    for i, path in enumerate(os.listdir(abs_path)):
        
        if os.path.isfile(os.path.join(abs_path, path)):
            #train and test reading i kinda weird (indexes are read differently), at least
            # i dont have to worry about dataset shuffle

            if "train.png" in path:
                index = path.replace("train.png", "")

                Train_img = cv2.imread(abs_path + path)
                Train_img = Convert_To_Tensor(Train_img)
                
                Train_X.append(Train_img)
                Train_y.append(labels[index])

            elif "test.png" in path:
                index = path.replace("test.png", "")

                Test_img = cv2.imread(abs_path + path)
                Test_img = Convert_To_Tensor(Test_img)

                Test_X.append(Test_img)
                Test_y.append(labels[index])

    return Train_X, Train_y, Test_X, Test_y

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

def Convert_To_Tensor(img): #TODO: This may be useless, but in case of adding also something else to conversion
    X = torch.from_numpy(img)
    print(X.shape)
    return X
