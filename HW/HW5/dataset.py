import torch
from torchvision import transforms
import torchvision.transforms as transforms
import torch.utils.data as data
import os, glob
import pickle, json
import numpy as np
import random, collections
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
import copy
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------

class AirlinePassengerData(data.Dataset):
    """
        Dataloader to generate dataset
    """

    def __init__(self, data_dir="./", data_file = None ,transforms=None, args=None):
        """
            function to initialize dataset.
        """
        self.transform = transformation_functions(args)

        self.df_data = pd.read_csv(data_dir + data_file)
        self.data = self.df_data.values[:,1]
        self.data.astype(float)

        self.scaler = MinMaxScaler()

        self.data = np.array([self.data])

        self.data_scaled = self.scaler.fit_transform(self.data.T).T.squeeze()

        self.data = np.squeeze(self.data)

        self.data_trunc = self.data[:-1]


    def __getitem__(self, id):
        """
            Returns a data_item and corresponding label 
        """
        x1 = 0. # 
        x2 = 0. # previous months value

        if (id==1):
            x2 = self.data_scaled[id - 1]


        else:
            x1 = self.data_scaled[id - 2]
            x2 = self.data_scaled[id - 1]

        x3 = self.data_scaled[id] # current months value

        data_item = np.array([[x1, x2, x3]])

        label = np.array([self.data_scaled[id+1]])

        
        
        data_item = self.transform(data_item)
        label = torch.FloatTensor(label)
        data_item = data_item.squeeze()

       #print("*******", id , "--" ,label.shape, label)

        return data_item, label

    def __len__(self):
        return self.data_trunc.shape[0]

    def _inverse_scaler_(self, X):
        return self.scaler.inverse_transform(X)




def get_loader(batch_size, shuffle, num_workers, drop_last=False, args=None):
    '''
        data loader function
    '''
    dataset = AirlinePassengerData(data_dir=args.data_dir,data_file = args.data_file ,args=args)
    dataset_size = len(dataset)

    test_set_size = int(dataset_size * args.data_split)
    train_set_size = dataset_size - test_set_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_set_size, test_set_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
        num_workers = num_workers, shuffle = args.shuffle)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
        num_workers = num_workers, shuffle = args.shuffle)


    full_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),num_workers=num_workers, shuffle=False)

    print()

    print("*** Dataset Stats ***\n")

    print("Full dataset size : ", len(dataset))
    print("Train data size : ", len(train_dataset))
    print("Test data size : ", len(test_dataset))
    print("---------- --------- ----------")
    print()

    return dataset, full_loader, train_loader, test_loader


"""
   Auxilary data functions
"""

def transformation_functions(args, mode="train"):

    transforms_list = [transforms.ToTensor()]

    return transforms.Compose(transforms_list)
