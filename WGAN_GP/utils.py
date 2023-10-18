import numpy as np
import math
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
from trial import *

cuda = True if torch.cuda.is_available() else False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'






