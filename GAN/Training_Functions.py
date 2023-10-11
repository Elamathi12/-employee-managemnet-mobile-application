import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Subset, TensorDataset
from torch.utils.data.dataloader import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

def split_indices(root_dir, val_pct):

    data_files = os.listdir(root_dir) #retrieve a list of 500 csv file names inside the "E:/August/Datasets" 
    data_size = len(data_files) #assigning variable data_size to find the length of files. Here it is 10,000
    # print(data_size, "length of datasize")

    val_size = int(data_size * val_pct) #validation percentage * length of datafiles will return number of files used for validation 
    idxs = np.random.permutation(data_size) #permutating for a range of len of data files / shuffling data and creating random subsets

    """Using slicing operation on a list of indices to split """
    val_indices = idxs[: val_size]   #spliting the idxs array for validation - index 0 to val_size for validation
    train_indices = idxs[val_size :] #splitting the indxs array for training - val_size to end of the indices for training
    print("Train_indices", len(train_indices))
    print("Val_indices", len(val_indices))

    return train_indices, val_indices #returs training indices and validation indices 

def data_to_tensor(file):

    dataset = pd.read_csv(file, sep=',') #loading data from a CSV file using pandas.read_csv

    """~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ REAL_DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ """
    real_data = dataset.iloc[:, 0]  # extracting the temperature column at index 3 from the loaded dataset
    real_array = np.array(real_data, dtype='float32')  # the extracted temperature column is converted into a Numpy array
    real_array = real_array.reshape((1,1000))
    real_normalized_array = real_array / 1000 # Normalization of real data dividing it by maximum temperature of real_data
    real_tensor = torch.from_numpy(real_normalized_array)

    """ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONDITION_DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

    condition_data = dataset.iloc[:, 1]
    condition_array = np.array(condition_data, dtype='float32') 
    condition_array = condition_array.reshape((1,1000)) 
    condition_normalized_array = condition_array / 1000 # Normalization of condition data dividing it by maximum temperature of real_data
    condition_tensor = torch.from_numpy(condition_normalized_array)

    return real_tensor, condition_tensor #returning the training tensor(temperature values)

class Create_dataset(Dataset):  #creating class named Create_dataset

    def __init__(self, root_dir): #initialization function with a directory as an input

        self.train_dir = root_dir #initializing the input directory
        self.train_files = os.listdir(self.train_dir)  #retrieve a list of 500 csv file names inside the root_dir, where root_dir here is "E:/August/Datasets" 
        """self.train_files contains random .csv files, for example [dataset7032.csv, dataset1287.csv, dataset6479.csv, dataset3444.csv]"""


    def __len__(self): #function to find the length of the training files

        return len(self.train_files) #len(files) gives the number of files, here it is 500
      

    def __getitem__(self, index):   #reads a file at the given input index and constructs full directory path

        file = self.train_files[index] #assigning variable for the variable at the given index
        file_path = os.path.join(self.train_dir, file) #constructs full directory path using os.path.join to provide inside the data_to_tensor() function
       
        real_tensor, condition_tensor = data_to_tensor(file_path) #data_to_tensor function is called which returns a training tensor (temperature values)

        return real_tensor, condition_tensor #returns the training tensor values(temperature)
    

def get_dataloader(training_datadirectory): #input will be a file containing multiple training .csv files
        
    dataset = Create_dataset(root_dir = training_datadirectory) #accessing a particular .csv file by calling the Create_dataset class
    print(dataset, "DATASET FROM CREATE DATASET CLASS")
    train_data, val_data = split_indices(root_dir = training_datadirectory, val_pct = 0.1)  #accessing the training and validation datasets

    print(len(train_data), "length of train data")  

    """SubsetRandomSampler's purpose is to randomly sample a subset of indices  
    from a given list of indices. This subset can then be used to load inside the DataLoader function"""
    train_sampler = SubsetRandomSampler(train_data) #train_sampler to load inside DataLoader

    
    train_loader = DataLoader(  #creating data loader for training data
        dataset = dataset,
        batch_size = 100,
        num_workers = 2, #assigning number of workers that are used to load data in parallel, 
        #increasing num_workers will speeds up data loading and training by utilizing multiple CPU cores
        sampler = train_sampler
        )
    
    print(len(train_loader), "length of train loader")

    val_sampler = SubsetRandomSampler(val_data) #validation sampler to load inside DataLoader function

    train_val_loader = DataLoader(dataset=dataset,  # creating data loader for validation data
                                  batch_size= 100,
                                  num_workers=2,
                                  sampler=val_sampler)

    test_loader = DataLoader(dataset = dataset, #creating data loader for testing data using training data
                    batch_size = 1,
                    num_workers = 2,
                    sampler = val_sampler)

    return train_loader, train_val_loader, test_loader