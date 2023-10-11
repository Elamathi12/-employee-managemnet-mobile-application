import torch
import torch.nn as nn

class WassersteinLoss(nn.Module):
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, y_true,y_pred):
        print("~~~~~~~~~~~~~~~~~~entered wasserstein loss~~~~~~~~~~~~~~~~~~~~~~~~ ")
        return torch.mean(y_true*y_pred)