import torch
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
import os
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import matplotlib.pyplot as plt
import numpy as np
from Generator_Model import Generator
from Discriminator_Model import Discriminator
from Training_Functions import get_dataloader


if __name__ == '__main__' :


    model_generator = Generator(input_shape=(1,1,1000))  
    model_discriminator = Discriminator(input_shape=(1,1,1000))

    #Linking CUDA to the model
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(torch.cuda.is_available(),"cuda availbaility")
    generator = model_generator.to(Device)
    discriminator = model_discriminator.to(Device)

    # Optimizers
    generator_optimizer = optim.Adam(model_generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    discriminator_optimizer = optim.Adam(model_discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    generator_loss = nn.L1Loss()   
    discriminator_loss = nn.BCELoss()

    epochs = 20

    g_loss_list = []
    d_loss_list = []
    mse_list = []

    # function to retrieve training data loaders and samplers.
    train_loader, train_val_loader, test_loader = get_dataloader(training_datadirectory = "D:/Mathi/PROJECTS/GAN/Datasets")

    print(len(train_loader), "Shape of train loader")

    title = 'GAN MODEL ' + ' for ' + str(epochs) + ' epochs' 

    def calculate_RMSE(actual,predicted):
        rmse = torch.sqrt(torch.mean((actual-predicted)**2))
        print("Root Mean Square Error : \n")
        print(rmse)
        return rmse
    
    def calculate_MSE(actual, predicted):
        calculate_mse = nn.MSELoss()
        mse = calculate_mse(actual, predicted)
        print("Mean Square Error : \n")
        print(mse)        
        return mse


    def Train_discriminator(condition, data, discriminator, generator, loss_function):

        batch_size = data.size(0)
        discriminator_optimizer.zero_grad()

        real_labels = torch.ones(batch_size, 1,1,device=Device) 
        real_outputs = discriminator(data)
        d_real_loss = loss_function(real_outputs, real_labels)   
       
        fake_data = generator(condition)       
        fake_labels = torch.zeros(batch_size,1,1,device=Device)
        fake_outputs = discriminator(fake_data.detach())
        d_fake_loss = loss_function(fake_outputs, fake_labels)

        
        d_loss = d_real_loss + d_fake_loss
        print(d_loss, "discriminator_losss")
        d_loss.backward()

        return fake_data , d_loss


    def Train_generator(fake_data, data, discriminator, generator, loss_function):

        batch_size = data.size(0)
        generator_optimizer.zero_grad()
        gen_labels = torch.ones(batch_size, 1,1,device = Device)
        gen_labels = gen_labels.to(Device) 
        gen_outputs = discriminator(fake_data)
        g_loss1= loss_function(data,fake_data)

        g_loss = g_loss1                                                                                                                                                                                                          
        print(g_loss, "generator  loss \n")
        g_loss.backward()
        return g_loss


    def save_models(generator, discriminator, save_path = "D:/Mathi/PROJECTS/GAN/save_model"):

        torch.save(generator.state_dict(), os.path.join(save_path, title+'_generator.pth'))
        torch.save(discriminator.state_dict(), os.path.join(save_path, title+'_discriminator.pth')) #to load and send to testing 
        print("Models saved successfully")



    for epoch in range(1, (epochs+1)):
        
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~EPOCH",epoch,"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

        ############### TRAIN LOADER #############
        #print(len(train_loader), "shape of train loader")
      
        for batch_idx, (real, fake) in enumerate(train_loader):

            print("----------------------BATCH", (batch_idx + 1),"---------------------- \n")

            real = real.to(Device)
            fake = fake.to(Device)

            fake_data, d_loss  = Train_discriminator(condition = fake,
                                                     data=real,
                                                     discriminator = model_discriminator, 
                                                     generator = model_generator,
                                                     loss_function = discriminator_loss)
            
            discriminator_optimizer.step()

            fake_data = fake_data.to(Device)

            g_loss= Train_generator(fake_data = fake_data,
                                    data = real,
                                    discriminator = model_discriminator,
                                    generator = model_generator,               
                                    loss_function= generator_loss)
            
            generator_optimizer.step()



        d_loss_list.append(d_loss)
        g_loss_list.append(g_loss)
        
        actual = torch.tensor(real).clone().detach().cpu()
        predicted = model_generator(fake)
        predicted = torch.tensor(predicted).clone().cpu().detach()
        mse = calculate_MSE(actual=actual, predicted=predicted)
        mse_list.append(mse.cpu())

        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}".format(epoch, epochs, g_loss, d_loss))


    generator_losses = g_loss_list
    g_list = [tensor.item() for tensor in generator_losses]
    print(g_list, " generator losses list")

    discriminator_losses = d_loss_list
    d_list = [tensor.item() for tensor in discriminator_losses]
    print(d_list," discriminator loss list ")

    print(mse_list, "List of MSE")

    x_label = len(g_list)
    # Create a range of epochs for the x-axis
    epoch_range = list(range(1, x_label + 1))

    # Create a figure and axis
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_range, g_list, label='Generator Loss', marker='o')
    plt.plot(epoch_range, d_list, label='Discriminator Loss', marker='o')
    plt.plot(epoch_range, mse_list, label = "MSE", marker = 'o')
    # Set labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()

    # Show the plot
    plt.show()
    # Save models
    save_models(model_generator, model_discriminator)