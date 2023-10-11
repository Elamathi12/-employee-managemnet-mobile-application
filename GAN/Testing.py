import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from Generator_Model import Generator
import torch.nn as nn
import random
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from Training_Functions import get_dataloader



Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, train_val_loader, test_loader = get_dataloader(training_datadirectory = "D:/Mathi/PROJECTS/GAN/Datasets")


def calculate_MSE(actual, predicted):
    mse_loss = nn.MSELoss()
    mse = mse_loss(predicted, actual)
    print("Mean Square Error : /n")
    print(mse)
    return mse


def denormalize_actual( actual):
    max_temp = 1000
    denormalize_actual = actual * max_temp    
    return denormalize_actual

def denormalize_predicted(predicted):
    max_temp = 1000
    denormalize_predicted = predicted * max_temp    
    return denormalize_predicted

def testing(gen, test_data_loader):

    testing_mse_list = []
    testing_dnormMSE_list = []

    for idx, (real, condition) in enumerate(test_data_loader):

        real = real.to(Device)
        condition = condition.to(Device)

        actual = torch.tensor(real)
        actual = real.clone().detach().cpu()
        predicted = gen(condition).cpu().detach()
        predicted = torch.tensor(predicted)

        denormalize_actual_result = denormalize_actual(actual = real.clone().detach().cpu())
        d_predicted = gen(condition).cpu().detach()
        denormalize_predicted_result = denormalize_predicted(d_predicted)  # Denormalize predicted data
        mse = calculate_MSE(actual=actual, predicted=predicted).cpu()
        denorm_mse = calculate_MSE(actual=denormalize_actual_result, predicted=denormalize_predicted_result).cpu()


        testing_mse_list.append(mse)
        testing_dnormMSE_list.append(denorm_mse)
    
    

    print(testing_mse_list)     

    mse_tensor = torch.tensor(testing_mse_list)
    mean = mse_tensor.mean()
    dnorm_mse = torch.tensor(testing_dnormMSE_list)
    dnorm_mse_mean = dnorm_mse.mean()

    print(mean.item(), " is the mean value for the list of mse values")
    print(dnorm_mse_mean.item(), " is the mean value for the list of denormalized mse values")

    #Visualizing

    x_axis = len(testing_mse_list)
    # Create a range of epochs for the x-axis
    testing_epoch_range = list(range(1, x_axis + 1))
    # Create a figure and axis
    plt.figure(figsize=(10, 6))
    plt.plot(testing_epoch_range, testing_mse_list, label='MSE(Testing)', marker='o')    
    # Set labels and title
    plt.xlabel('Index')
    plt.ylabel('MSE')   
    plt.title('Testing Visualizer')
    plt.legend()
    # Show the plot
    plt.show()
    plt.plot(testing_epoch_range, testing_dnormMSE_list, label='Denormalized MSE(Testing)', marker='o')    
    return mean.item()


def testing_module(test_loader, gen_pth_file_path):

    #Testing
    test = True
    model_generator = Generator(input_shape=(1,1,1000))  # Instantiate your generator model

    print("--------------------------------------Testing module----------------------------------")
    gen = torch.load(gen_pth_file_path)  # Load the saved state dictionary
    model_generator.load_state_dict(gen, strict=False)
    print("Model Loaded")
    model_generator = model_generator.to(Device)
    mean_list = []
       
    ############### TEST LOADER #############

    if test == True:

        print("Entered test == true")
        mean = testing(gen = model_generator, test_data_loader = test_loader)
        print("Model Testing has been executed successfully")        
        num_files_to_visualize = int(input("Enter the number of files you want to visualize: "))
        plotting(gen=model_generator, test_data_loader=test_loader,num_files_to_visualize=num_files_to_visualize)
        print("plotted successfully")
    mean_list.append(mean)

    print("#################### LIST OF MEAN FOR EVERY EPOCHS ####################")
    print(mean_list)

def plotting(gen, test_data_loader,num_files_to_visualize):

    total_samples = 100
    # Randomly select num_files_to_visualize
    files_to_visualize = random.sample(range(total_samples), num_files_to_visualize)


    for idx, (real, condition) in enumerate(test_data_loader):
        if idx in files_to_visualize:            
            real = real.to(Device)
            condition = condition.to(Device)

            actual = denormalize_actual(real.clone().detach().cpu())  # Denormalize actual data
            predicted = gen(condition).cpu().detach()
            predicted = denormalize_predicted(predicted)  # Denormalize predicted data

            # Visualize the selected samples
            plt.figure(figsize=(10, 6))
            plt.plot(actual[0][0], label='Actual Data', marker='o')
            plt.plot(predicted[0][0], label='Generated Data', marker='o')
            plt.xlabel('Index')
            plt.ylabel('Numeric_values')
            plt.title(f'Real vs Generated Data (Sample {idx})')
            plt.legend()
            plt.show()

#########TESTING#########

if __name__ == "__main__":

    filename = "WASSERSTEIN GAN MODEL  for 30 epochs_generator.pth"
    pathname = "D:/Mathi/PROJECTS/GAN/save_model/"

    Generator_filepath_to_load = pathname + filename

    testing_module(test_loader = test_loader, gen_pth_file_path = Generator_filepath_to_load)