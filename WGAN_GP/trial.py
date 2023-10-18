import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt 
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as tf
from PIL import Image
from Discriminator_model import Critic
from Generator_model import Generator



torch.manual_seed(2021)
np.random.seed(2021)
cuda = True if torch.cuda.is_available() else False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Parameters
n_epochs = 80
batch_size = 256
lr_c = 0.0005
lr_g = 0.0002
b1 = 0.5
b2 = 0.999
n_cpu = 2
latent_dim = 64
img_size = 28
channels = 1
sample_interval = 400
im_show_interval = 5
n_critic = 2
g_his, fake_his, real_his = [], [], []
d_his = []

def calc_gradient_penalty(netD, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, 1).to(device)
    interpolation = alpha * real + (1 - alpha) * fake
    interpolation = interpolation.to(device).requires_grad_(True)

    LAMBDA = 10
    disc_interpolated = netD(interpolation)
    gradients = torch.autograd.grad(outputs=disc_interpolated, inputs=interpolation,
                                   grad_outputs=torch.ones(disc_interpolated.size()).to(device),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty

class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=True):
        super(MinibatchDiscrimination, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        torch.nn.init.normal_(self.T, 0, 1)

    def forward(self, x):
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)
        M_T = M.permute(1, 0, 2, 3)
        norm = torch.abs(M - M_T).sum(3)
        expnorm = torch.exp(-norm)
        o_b = expnorm.sum(0) - 1

        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x
    

def get_latent_noise(batch):
    z = torch.zeros(batch, 64)
    nn.init.normal_(z)
    return z

def show_results(g, epoch_done):
    z = get_latent_noise(100).to(device)
    fake = g(z).detach().cpu().numpy()
    fig, axs = plt.subplots(10, 10,figsize=(6,6))
    fig.suptitle('Generated images after {} epochs'.format(epoch_done))
    for i in range(10):
        for j in range(10):
            axs[i, j].imshow(fake[i*10+j, 0] * std + mean, cmap='gray')
            axs[i, j].axis('off')
    plt.show()
def get_dataloader(trainset, img_size, batch_size, n_cpu):
    """
    trainset: numpy array 
    img_size: resize the given train set to the desired size
    batch_size: size of the batches
    n_cpu: number of cpu threads to use during batch generation
    """
    new_trainset = torch.from_numpy(trainset.astype(np.float32)).view(-1, 1, img_size, img_size)
    return torch.utils.data.DataLoader(new_trainset, batch_size=batch_size,num_workers=n_cpu, shuffle=True)

root_path =  "D:/Elamathi/Projects/PROJECTS/WGAN_GP/FashionMNIST"
dataset = torchvision.datasets.FashionMNIST(root_path, train=True, download=True)
# discard labels in dataset
data = np.array([np.array(img) for img, _ in dataset])
mean, std = data.mean(), data.std()
data = (data - mean) / std

img_size = 28  
batch_size = 256
n_cpu = 2

dataloader = get_dataloader(data, img_size, batch_size, n_cpu)

# Loss function
adversarial_loss = torch.nn.BCELoss()


generator = Generator()
critic = Critic()


if cuda:
    generator.cuda()
    critic.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(b1,b2))
optimizer_C = torch.optim.Adam(critic.parameters(), lr=lr_c, betas=(b1,b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


# Set device (CPU or GPU)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # --device cuda:0 if torch.cuda.is_available() else cpu

# Training loop
for epoch in range(n_epochs):  # --n_epochs 80
    for i, (imgs) in enumerate(dataloader):
        # Configure input
        real_imgs = imgs.to(device)  # No need for Variable() and Tensor() in newer PyTorch versions

        # --------------------- TRAINING CRITIC ------------------------------------------------------------#

        optimizer_C.zero_grad()

        # Sample noise as generator input
        z = torch.randn(imgs.shape[0], latent_dim).to(device)  # --latent_dim 64

        # Generate a batch of images
        fake_imgs = generator(z).detach()

        # Adversarial loss
        loss_D = -torch.mean(critic(real_imgs)) + torch.mean(critic(fake_imgs))

        loss_D.backward()

        grad_loss = calc_gradient_penalty(critic, real_imgs.detach(), fake_imgs.detach(), device)
        grad_loss.backward()

        optimizer_C.step()

        d_his.append(-loss_D.item())

        if i % n_critic != 0:
            continue

        # --------------------------------- TRAINING GENERATOR ---------------------------------------#
 

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = torch.randn(imgs.shape[0], latent_dim).to(device)  # --latent_dim 64

        # Generate a batch of images
        gen_imgs = generator(z)

        # Adversarial loss
        loss_G = -torch.mean(critic(gen_imgs))

        loss_G.backward()
        optimizer_G.step()

        g_his.append(-loss_G.item())

        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:  # --sample_interval 400
            print("[Epoch %d/%d] [Batch %d] [C loss: %f] [G loss: %f]"
                  % (epoch, n_epochs, batches_done, d_his[-1], -loss_G.item()))

    if epoch % im_show_interval == 0:  # --im_show_interval 5
        show_results(generator, epoch + 1)
