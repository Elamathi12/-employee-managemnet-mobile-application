import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as tf
from PIL import Image
from critic_model import Critic
from Generator_model import Generator
import multiprocessing
multiprocessing.set_start_method('spawn')
from torchvision.utils import make_grid, save_image
import os
torch.manual_seed(2021)
np.random.seed(2021)
cuda = True if torch.cuda.is_available() else False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Parameters
n_epochs = 100
batch_size = 128
lr_c = 0.0005
lr_g = 0.0002
b1 = 0.5
b2 = 0.999
latent_dim = 32
img_size = 28
n_critic = 2
g_loss_list= []
c_loss_list = []

def calc_gradient_penalty(netD, real, fake, device):
    epsilon = torch.rand(real.size(0), 1, 1, 1).to(device)
    interpolation = epsilon * real + (1 - epsilon) * fake
    interpolation = interpolation.to(device).requires_grad_(True)

    LAMBDA = 10 # gradient penalty coefficient
    disc_interpolated = netD(interpolation)
    gradients = torch.autograd.grad(outputs=disc_interpolated, inputs=interpolation,
                                   grad_outputs=torch.ones(disc_interpolated.size()).to(device),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty


    

def get_latent_condition(batch):
    z = torch.zeros(batch, 32)
    nn.init.normal_(z)
    return z

def show_results(g, epoch_done, output_directory='D:/Elamathi/Projects/PROJECTS/WGAN_GP/Figures/generated_images_MNIST'):
    os.makedirs(output_directory, exist_ok=True)
    z = get_latent_condition(100).to(device)
    fake = g(z).detach().cpu().numpy()
    fig, axs = plt.subplots(10, 10,figsize=(10,10))
    fig.suptitle('Generated images after {} epochs'.format(epoch_done))
    for i in range(10):
        for j in range(10):
            axs[i, j].imshow(fake[i*10+j, 0] * std + mean, cmap='gray')
            axs[i, j].axis('off')
    plt.savefig(os.path.join(output_directory, 'generated_images_epoch_{}.png'.format(epoch_done)))
    plt.close()



def get_dataloader(trainset, img_size, batch_size):
    """
    trainset: numpy array 
    img_size: resize the given train set to the desired size
    batch_size: size of the batches
   
    """
    new_trainset = torch.from_numpy(trainset.astype(np.float32)).view(-1, 1, img_size, img_size)
    return torch.utils.data.DataLoader(new_trainset, batch_size=batch_size,num_workers=0, shuffle=True)

root_path =  "D:/Elamathi/Projects/PROJECTS/WGAN_GP/MNIST"
dataset = torchvision.datasets.FashionMNIST(root_path, train=True, download=True)
# discard labels in dataset
data = np.array([np.array(img) for img, _ in dataset])
mean, std = data.mean(), data.std()
data = (data - mean) / std

img_size = 28  
batch_size = 128


dataloader = get_dataloader(data, img_size, batch_size)

generator = Generator()
critic = Critic()


if cuda:
    generator.cuda()
    critic.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(b1,b2))
optimizer_C = torch.optim.Adam(critic.parameters(), lr=lr_c, betas=(b1,b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# Set device (CPU or GPU)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # --device cuda:0 if torch.cuda.is_available() else cpu
generated_images = []
# Training loop
for epoch in range(1,(n_epochs+1)):  
    for i, (imgs) in enumerate(dataloader):
        # Configure input
        real_imgs = imgs.to(device)  # No need for Variable() and Tensor() in newer PyTorch versions

        # --------------------- TRAINING CRITIC ------------------------------------------------------------#

        optimizer_C.zero_grad()

        # Sample noise as generator input
        z = torch.randn(imgs.shape[0], latent_dim).to(device)  

        # Generate a batch of images
        fake_imgs = generator(z).detach()

        # Adversarial loss
        loss_D = -torch.mean(critic(real_imgs)) + torch.mean(critic(fake_imgs))

        loss_D.backward()

        grad_loss = calc_gradient_penalty(critic, real_imgs.detach(), fake_imgs.detach(), device)
        grad_loss.backward()

        optimizer_C.step()

        c_loss_list.append(-loss_D.item())

        if i % n_critic != 0:
            continue

        # --------------------------------- TRAINING GENERATOR ---------------------------------------#
 

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = torch.randn(imgs.shape[0], latent_dim).to(device)  

        # Generate a batch of images
        gen_imgs = generator(z)

        # Adversarial loss
        loss_G = -torch.mean(critic(gen_imgs))

        loss_G.backward()
        optimizer_G.step()

        g_loss_list.append(-loss_G.item())

        batches_done = epoch * len(dataloader) + i
        
        print("[Epoch %d/%d] [Batch %d] [C loss: %f] [G loss: %f]" % (epoch, n_epochs, i, c_loss_list[-1], -loss_G.item()))

    
    show_results(generator, epoch )