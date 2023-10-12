from torchvision.utils import make_grid
import torch.nn.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm

def to_img(x):
    x = x.clamp(0, 1)
    return x


def visualise_output(images, x, y):
    """
    Function to visualize a grid of images.

    Parameters:
    - images (torch.Tensor): Tensor containing the images to visualize.
    - x (int): Number of images per row in the grid.
    - y (int): Number of images per column in the grid.

    Returns:
    None
    """
    with torch.no_grad():
        # Move the images to CPU if they are on a GPU device
        images = images.cpu()

        # Convert the images to the correct format for visualization
        images = to_img(images)

        # Convert the tensor grid to a numpy array
        np_imagegrid = make_grid(images, x, y).numpy()

        # Set the figure size for the plot
        plt.figure(figsize=(20, 20))

        # Transpose the image grid to the correct format and display it
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()

        return visualise_output