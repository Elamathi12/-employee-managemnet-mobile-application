import torchvision.transforms as transforms
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from IPython.display import clear_output



from generator import Generator
from critic import Critic

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class Hyperparameters(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


hp = Hyperparameters(
    n_epochs=1000,
    batch_size=64,
    lr=0.00005,
    n_cpu=8,
    latent_dim=100,
    img_size=32,
    channels=1,
    n_critic=25,
    clip_value=0.005,
    sample_interval=400,
)

root_path =  "D:/Mathi/PROJECTS/WGAN/FashionMNIST"
""" The Fashion-MNIST dataset contains 60,000 training images (and 10,000 test images) of fashion and clothing items,
taken from 10 classes. Each image is a standardized 28×28 size in grayscale (784 total pixels). """

train_dataloader = torch.utils.data.DataLoader(
    datasets.FashionMNIST(
        root_path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(hp.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        ),
    ),
    batch_size=hp.batch_size,
    shuffle=True,
)

img_shape = (hp.channels, hp.img_size, hp.img_size)

generator = Generator(img_shape, hp.latent_dim)
critic = Critic(img_shape)

optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=hp.lr)
optimizer_D = torch.optim.RMSprop(critic.parameters(), lr=hp.lr)


# initialise weights 
generator._init_weights()
critic._init_weights()



def train():

    visualize_epoch = [500,550,600,650,700,750,800,850,900,950,1000]
    current_epoch = 0
    for epoch in range(1,(hp.n_epochs+1)):
        for i, (imgs, _) in enumerate(train_dataloader):

            # labelling real and fake
            real= Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            #########################
            #  Train Critic
            #########################

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], hp.latent_dim))))

            # Generate a batch of images
            # fake_imgs = generator(z).detach()
            fake_imgs = generator(img_shape, z).detach()

            """ The math for the loss functions for the critic and generator is:
                Critic Loss: D(x) - D(G(z))
                -D(x) + D(G(z))
                Generator Loss: D(G(z))
                Now for the Critic Loss, as per the Paper, we have to maximize the expression.
                So, arithmetically, maximizing an expression, means minimizing the -ve of that expression
                i.e. -(D(x) - D(G(z))) which is -D(x) + D(G(z)) i.e. -D(real_imgs) + D(G(real_imgs))
            """
            d_loss = -torch.mean(critic(real_imgs)) + torch.mean(critic(fake_imgs))

            d_loss.backward()
            optimizer_D.step()

            """ Clip weights of critic to avoid vanishing/exploding gradients in the
            critic/critic. """
            for p in critic.parameters():
                p.data.clamp_(-hp.clip_value, hp.clip_value)

            """ In WGAN, we update Critic more than Generator
            Train the generator every n_critic iterations
            we need to increase training iterations of the critic so that it works to
            approximate the real distribution sooner.
            """
            if i % hp.n_critic == 0:
                #########################
                #  Train Generators
                #########################
                optimizer_G.zero_grad()

                # Generate a batch of images
                fake_images_from_generator = generator(img_shape, z)
                # Adversarial loss
                g_loss = -torch.mean(critic(fake_images_from_generator))

                g_loss.backward()
                optimizer_G.step()

            ##################
            #  Log Progress
            ##################

            batches_done = epoch * len(train_dataloader) + i
            
            if batches_done % hp.sample_interval == 0:
                clear_output()
                print(f"Epoch:{epoch}:It{i}:DLoss{d_loss.item()}:GLoss{g_loss.item()}")

                if epoch == visualize_epoch[current_epoch]:
                    # Generate and visualize images
                    generated_imgs = generator(img_shape,z).detach()
                    visualise_output(generated_imgs.data[:50], 10, 10)
                    current_epoch += 1

                if current_epoch == len(visualize_epoch):
                    break


train()