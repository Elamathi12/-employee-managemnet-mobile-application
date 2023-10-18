import torch
from torch import nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

torch.manual_seed(0) 

def plot_images_from_tensor(image_tensor, num_images=25, size=(1, 28, 28)):
    # Normalize the image tensor to [0, 1]
    image_tensor = (image_tensor + 1) / 2

    # Detach the tensor from its computation graph and move it to the CPU
    img_detached = image_tensor.detach().cpu()

    # Create a grid of images using the make_grid function from torchvision.utils
    image_grid = make_grid(img_detached[:num_images], nrow=5)

    # Plot the grid of images
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

""" The reason for doing "image_grid.permute(1, 2, 0)"

PyTorch modules processing image data expect tensors in the format C × H × W.

Whereas PILLow and Matplotlib expect image arrays in the format H × W × C

So to use them with matplotlib you need to reshape it
to put the channels as the last dimension:

I could have used permute() method as well like below
"np.transpose(npimg, (1, 2, 0))"

------------------

Tensor.detach() is used to detach a tensor from the current computational graph. It returns a new tensor that doesn't require a gradient.

When we don't need a tensor to be traced for the gradient computation, we detach the tensor from the current computational graph.

We also need to detach a tensor when we need to move the tensor from GPU to CPU.

"""

def make_grad_hook():
    """
    Constructs a gradient hook for Conv2d and ConvTranspose2d layers in a model.

    The hook function 'grad_hook' checks if the passed module 'm' is an instance of nn.Conv2d or nn.ConvTranspose2d.
    If it is, the function appends the gradient of the module's weights to the 'gradients_list' list.

    The function 'make_grad_hook' returns this list along with the hook function.

    Returns:
        gradients_list (list): A list to store the gradients of the weights of the Conv2d and ConvTranspose2d layers.
        grad_hook (function): A function that, when called with a module as an argument, checks if the module is an
                              instance of Conv2d or ConvTranspose2d and, if so, appends the gradient of the module's
                              weights to 'gradients_list'.
    """

    # List to store the gradients
    gradients_list = []

    def grad_hook(m, grad_input, grad_output):
        """
        The hook function that is applied on each module 'm'. If the module is an instance of Conv2d or
        ConvTranspose2d, the function appends the gradient of the module's weights to the 'gradients_list' list.

        Args:
            module (nn.Module): The module on which the hook function is applied.
            grad_input (tuple): Gradients of the inputs.
            grad_output (tuple): Gradients of the outputs.

        Returns:
            None. The function operates in-place on 'gradients_list'.
        """
        # Check if the module is an instance of Conv2d or ConvTranspose2d
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            # If it is, append the gradient of the module's weights to 'gradients_list'
            gradients_list.append(m.weight.grad)

    return gradients_list, grad_hook

def weights_init(m):
    """
    Initialize the weights of the ConvTranspose2d and Conv2d layers with appropriate weights for the Generator,
    and Conv2d layers for the Critic.

    Args:
        m (nn.Module): The module to be initialized.

    Returns:
        None. The function operates in-place on 'm'.
    """
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
        # Initialize the weights using a normal distribution with mean 0.0 and standard deviation 0.02
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    if isinstance(m, nn.BatchNorm2d):
        # Initialize BatchNorm2d layer's weights with a normal distribution with mean 1.0 and standard deviation 0.02,
        # and biases with a constant value of 0
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_noise(n_samples, z_dim, device="cpu"):
    """
    Generate random noise to be used as input to the Generator.

    Args:
        n_samples (int): Number of noise samples to generate.
        z_dim (int): Dimension of the noise vector.
        device (str): Device to place the generated noise tensor.

    Returns:
        noise (torch.Tensor): Random noise tensor of shape (n_samples, z_dim).
    """
    return torch.randn(n_samples, z_dim, device=device)


##############################
# Generator Loss Calculation
##############################
"""
#### Generator Loss = -[average critic score on fake images]

Generator Loss: D(G(z))

The generator tries to maximize this function. In other words,
It tries to maximize the discriminator's output for its fake instances. In these functions: """


def get_gen_loss(critic_fake_prediction):
    gen_loss = -1.0 * torch.mean(critic_fake_prediction)
    return gen_loss


# UNIT TEST of Generator Loss Calculation
assert torch.isclose(get_gen_loss(torch.tensor(1.0)), torch.tensor(-1.0))

assert torch.isclose(get_gen_loss(torch.rand(10000)), torch.tensor(-0.5), 0.05)

print("Success!")

##############################
# Critic Loss Calculation
##############################
def get_crit_loss(critic_fake_prediction, crit_real_pred, gp, c_lambda):

    """The math for the loss functions for the critic and generator is:

    Critic Loss: D(x) - D(G(z))

    Now for the Critic Loss, as per the Paper, we have to maximize the above expression.
    So, arithmetically, maximizing an expression, means minimizing the -ve of that expression
    c_LAMBDA = (gradient penalty coefficient) is a hyperparameter that controls the influence of the gradient penalty term in the critic loss, 
    balancing regularization and adversarial learning in the WGAN-GP approach.
    i.e. -(D(x) - D(G(z)))
    i.e. -D(x) + D(G(z))
    i.e. -D(real_imgs) + D(G(real_imgs))
    i.e. -D(real_imgs) + D(fake_imgs)
    """
    crit_loss = (torch.mean(critic_fake_prediction) - torch.mean(crit_real_pred) + c_lambda * gp)
    return crit_loss


# UNIT TEST of Critic Loss Calculation
assert torch.isclose(
    get_crit_loss(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0), 0.1), torch.tensor(-0.7),
)
assert torch.isclose(
    get_crit_loss(torch.tensor(20.0), torch.tensor(-20.0), torch.tensor(2.0), 10),torch.tensor(60.0),
)

print("Success!")


